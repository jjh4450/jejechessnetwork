"""
Parquet 파일에서 체스 데이터를 로드하는 PyTorch Dataset 클래스
"""

import torch
from torch.utils.data import IterableDataset, get_worker_info
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyarrow.parquet as pq
from typing import List, Tuple, Optional, Iterator
import random

def load_sample_from_parquet(df_row):
    """
    Parquet 파일의 한 행을 원래 형태로 복원
    
    Args:
        df_row: pandas DataFrame의 한 행
        
    Returns:
        (state, policy, mask, value) 튜플
    """
    state = np.array(df_row['state'], dtype=np.float32).reshape(18, 8, 8)
    policy = int(df_row['policy'])
    mask = np.array(df_row['mask'], dtype=np.float32)
    value = float(df_row['value'])
    
    return state, policy, mask, value


def get_parquet_file_info(parquet_dir: str, file_pattern: str = "chess_samples_*.parquet") -> Tuple[List[Path], List[int]]:
    """
    Parquet 파일 리스트와 각 파일의 샘플 수를 반환
    
    Args:
        parquet_dir: parquet 파일이 있는 디렉토리
        file_pattern: 파일 패턴
        
    Returns:
        (parquet_files, file_lengths) 튜플
    """
    parquet_dir = Path(parquet_dir)
    parquet_files = sorted(parquet_dir.glob(file_pattern))
    
    if not parquet_files:
        raise ValueError(f"Parquet 파일을 찾을 수 없습니다: {parquet_dir}/{file_pattern}")
    
    file_lengths = []
    print(f"Parquet 파일 정보 수집 중... ({len(parquet_files)}개 파일)")
    for file_path in tqdm(parquet_files, desc="파일 스캔"):
        parquet_file = pq.ParquetFile(file_path)
        length = parquet_file.metadata.num_rows
        file_lengths.append(length)
    
    total_samples = sum(file_lengths)
    print(f"총 파일 수: {len(parquet_files)}, 총 샘플 수: {total_samples:,}")
    
    return parquet_files, file_lengths


def split_files_by_ratio(
    parquet_files: List[Path], 
    file_lengths: List[int], 
    train_ratio: float = 0.9,
    shuffle: bool = False,
    seed: int = 42
) -> Tuple[List[Path], List[Path], int, int]:
    """
    파일 리스트를 train/val로 분할 (샘플 수를 고려하여 균형 맞춤)
    
    Args:
        parquet_files: parquet 파일 경로 리스트
        file_lengths: 각 파일의 샘플 수 리스트
        train_ratio: 학습 데이터 비율 (기본 0.9 = 90%)
        shuffle: 파일 순서를 섞을지 여부 (기본 False - 순서 유지)
        seed: 랜덤 시드 (shuffle=True일 때 사용)
        
    Returns:
        (train_files, val_files, train_samples, val_samples) 튜플
    """
    if len(parquet_files) != len(file_lengths):
        raise ValueError("parquet_files와 file_lengths의 길이가 다릅니다.")
    
    if not parquet_files:
        raise ValueError("파일 리스트가 비어있습니다.")
    
    # 파일과 길이를 쌍으로 묶기
    file_info = list(zip(parquet_files, file_lengths))
    
    # 필요시 셔플
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(file_info)
    
    total_samples = sum(file_lengths)
    target_train_samples = int(total_samples * train_ratio)
    
    # 파일 단위로 분할 (샘플 수 기준)
    train_files = []
    val_files = []
    current_train_samples = 0
    
    for file_path, length in file_info:
        if current_train_samples < target_train_samples:
            train_files.append(file_path)
            current_train_samples += length
        else:
            val_files.append(file_path)
    
    # val_files가 비어있으면 마지막 train 파일을 val로 이동
    if not val_files and len(train_files) > 1:
        last_file = train_files.pop()
        val_files.append(last_file)
    
    train_samples = sum(l for f, l in file_info if f in train_files)
    val_samples = sum(l for f, l in file_info if f in val_files)
    
    print(f"\n파일 단위 분할 완료:")
    print(f"  Train: {len(train_files)}개 파일, {train_samples:,}개 샘플 ({train_samples/total_samples*100:.1f}%)")
    print(f"  Val: {len(val_files)}개 파일, {val_samples:,}개 샘플 ({val_samples/total_samples*100:.1f}%)")
    
    return train_files, val_files, train_samples, val_samples


class ParquetChessDataset(IterableDataset):
    """
    Parquet 파일에서 체스 데이터를 로드하는 Dataset
    
    특징:
    - 파일을 순차적으로 읽어 I/O 효율 극대화 (10-100배 향상)
    - shuffle=True: Shuffle Buffer로 메모리 내에서 데이터 셔플 (학습용)
    - shuffle=False: 순차 읽기 (검증/테스트용)
    - 에폭마다 파일 순서를 섞어 다양성 확보
    - DataLoader의 num_workers > 0 지원 (파일 분할)
    
    주의: IterableDataset은 len()을 지원하지 않습니다.
    대신 estimated_length 속성을 사용하세요.
    """
    
    def __init__(
        self,
        parquet_files: List[Path],
        shuffle: bool = True,
        buffer_size: int = 100000,
        shuffle_files: bool = True,
        seed: int = 42,
        verbose: bool = True
    ):
        """
        Args:
            parquet_files: parquet 파일 경로 리스트
            shuffle: 데이터 셔플 여부 (기본 True)
                    - True: Shuffle Buffer 사용 (학습용)
                    - False: 순차 읽기 (검증/테스트용)
            buffer_size: Shuffle Buffer 크기 (기본 100,000 샘플)
                        - 클수록 셔플 품질 향상, 메모리 사용량 증가
                        - 파일 1개 크기(50,000) * 2 정도 권장
                        - shuffle=False일 때는 무시됨
            shuffle_files: 에폭마다 파일 순서를 섞을지 여부 (기본 True)
                          - shuffle=False일 때는 무시됨
            seed: 랜덤 시드
            verbose: 진행 상황 출력 여부
        """
        super().__init__()
        self.parquet_files = [Path(f) for f in parquet_files]
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.shuffle_files = shuffle_files if shuffle else False
        self.seed = seed
        self.verbose = verbose
        self.epoch = 0
        
        if not self.parquet_files:
            raise ValueError("파일 리스트가 비어있습니다.")
        
        # 총 샘플 수 계산 (메타데이터만 읽기)
        self.file_lengths = []
        if verbose:
            print(f"Parquet 파일 정보 수집 중... ({len(self.parquet_files)}개 파일)")
        for file_path in self.parquet_files:
            pf = pq.ParquetFile(file_path)
            self.file_lengths.append(pf.metadata.num_rows)
        
        self.estimated_length = sum(self.file_lengths)
        if verbose:
            print(f"총 샘플 수: {self.estimated_length:,}")
            if shuffle:
                print(f"모드: Shuffle Buffer ({buffer_size:,} 샘플, {buffer_size / self.estimated_length * 100:.2f}%)")
            else:
                print(f"모드: 순차 읽기 (검증용)")
    
    def set_epoch(self, epoch: int):
        """에폭 설정 (파일 순서 셔플에 사용)"""
        self.epoch = epoch
    
    def _get_file_indices_for_worker(self) -> List[int]:
        """현재 워커가 처리할 파일 인덱스 반환"""
        worker_info = get_worker_info()
        file_indices = list(range(len(self.parquet_files)))
        
        # 파일 순서 셔플 (shuffle=True일 때만)
        if self.shuffle_files:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(file_indices)
        
        # 멀티 워커인 경우 파일 분할
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # 각 워커에게 파일 분배
            file_indices = [idx for i, idx in enumerate(file_indices) if i % num_workers == worker_id]
        
        return file_indices
    
    def _load_samples_from_file(self, file_path: Path) -> Iterator[Tuple]:
        """파일에서 샘플을 순차적으로 읽기"""
        df = pd.read_parquet(file_path, engine='pyarrow')
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            state = np.array(row['state'], dtype=np.float32).reshape(18, 8, 8)
            policy = int(row['policy'])
            mask = np.array(row['mask'], dtype=np.float32)
            value = float(row['value'])
            
            yield (
                torch.from_numpy(state).float(),
                torch.tensor(policy, dtype=torch.long),
                torch.from_numpy(mask).float(),
                torch.tensor(value, dtype=torch.float32),
            )
    
    def _iter_with_shuffle(self, file_indices: List[int]) -> Iterator[Tuple]:
        """Shuffle Buffer를 사용한 데이터 스트리밍"""
        buffer = []
        worker_id = get_worker_info().id if get_worker_info() else 0
        rng = random.Random(self.seed + self.epoch + worker_id)
        
        # 파일별로 순차 읽기
        for file_idx in file_indices:
            file_path = self.parquet_files[file_idx]
            
            # 파일에서 샘플 읽어서 버퍼에 추가
            for sample in self._load_samples_from_file(file_path):
                buffer.append(sample)
                
                # 버퍼가 가득 차면 랜덤하게 하나 꺼내서 반환
                if len(buffer) >= self.buffer_size:
                    idx = rng.randint(0, len(buffer) - 1)
                    yield buffer[idx]
                    # 마지막 요소를 해당 위치로 이동하고 pop (O(1))
                    buffer[idx] = buffer[-1]
                    buffer.pop()
        
        # 남은 버퍼 데이터 셔플 후 반환
        rng.shuffle(buffer)
        for sample in buffer:
            yield sample
    
    def _iter_sequential(self, file_indices: List[int]) -> Iterator[Tuple]:
        """순차적으로 데이터 스트리밍 (셔플 없음)"""
        for file_idx in file_indices:
            file_path = self.parquet_files[file_idx]
            yield from self._load_samples_from_file(file_path)
    
    def __iter__(self) -> Iterator[Tuple]:
        """데이터 스트리밍"""
        file_indices = self._get_file_indices_for_worker()
        
        if self.shuffle:
            yield from self._iter_with_shuffle(file_indices)
        else:
            yield from self._iter_sequential(file_indices)
