"""
Parquet 파일에서 체스 데이터를 로드하는 PyTorch Dataset 클래스

v2: pyarrow 배치 스트리밍으로 재작성 (pandas 제거, 성능 대폭 향상)
"""

import torch
from torch.utils.data import IterableDataset, get_worker_info
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa
from typing import List, Tuple, Optional, Iterator
import random


def load_sample_from_parquet(row_dict):
    """
    Parquet 파일의 한 행(dict)을 원래 형태로 복원
    
    Args:
        row_dict: dict 형태의 한 행 {'state': [...], 'policy': int, ...}
        
    Returns:
        (state, policy, mask, value) 튜플
    """
    state = np.array(row_dict['state'], dtype=np.float32).reshape(18, 8, 8)
    policy = int(row_dict['policy'])
    mask = np.array(row_dict['mask'], dtype=np.float32)
    value = float(row_dict['value'])
    
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
) -> Tuple[List[Path], List[Path], List[int], List[int], int, int]:
    """
    파일 리스트를 train/val로 분할 (샘플 수를 고려하여 균형 맞춤)
    
    Args:
        parquet_files: parquet 파일 경로 리스트
        file_lengths: 각 파일의 샘플 수 리스트
        train_ratio: 학습 데이터 비율 (기본 0.9 = 90%)
        shuffle: 파일 순서를 섞을지 여부 (기본 False - 순서 유지)
        seed: 랜덤 시드 (shuffle=True일 때 사용)
        
    Returns:
        (train_files, val_files, train_lengths, val_lengths, train_samples, val_samples) 튜플
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
    train_lengths = []
    val_files = []
    val_lengths = []
    current_train_samples = 0
    
    for file_path, length in file_info:
        if current_train_samples < target_train_samples:
            train_files.append(file_path)
            train_lengths.append(length)
            current_train_samples += length
        else:
            val_files.append(file_path)
            val_lengths.append(length)
    
    # val_files가 비어있으면 마지막 train 파일을 val로 이동
    if not val_files and len(train_files) > 1:
        last_file = train_files.pop()
        last_length = train_lengths.pop()
        val_files.append(last_file)
        val_lengths.append(last_length)
    
    train_samples = sum(train_lengths)
    val_samples = sum(val_lengths)
    
    print(f"\n파일 단위 분할 완료:")
    print(f"  Train: {len(train_files)}개 파일, {train_samples:,}개 샘플 ({train_samples/total_samples*100:.1f}%)")
    print(f"  Val: {len(val_files)}개 파일, {val_samples:,}개 샘플 ({val_samples/total_samples*100:.1f}%)")
    
    return train_files, val_files, train_lengths, val_lengths, train_samples, val_samples


class ParquetChessDataset(IterableDataset):
    """
    Parquet 파일에서 체스 데이터를 로드하는 Dataset
    
    특징:
    - 파일을 순차적으로 읽어 I/O 효율 극대화 (10-100배 향상)
    - shuffle_files=True: 에폭마다 파일 순서를 섞어 학습 다양성 확보
    - DataLoader의 num_workers > 0 지원 (파일 분할)
    - 첫 배치 즉시 반환 (지연 없음)
    
    주의: IterableDataset은 len()을 지원하지 않습니다.
    대신 estimated_length 속성을 사용하세요.
    """
    
    def __init__(
        self,
        parquet_files: List[Path],
        file_lengths: Optional[List[int]] = None,
        shuffle_files: bool = True,
        seed: int = 42,
        verbose: bool = True
    ):
        """
        Args:
            parquet_files: parquet 파일 경로 리스트
            file_lengths: 각 파일의 샘플 수 리스트 (None이면 자동 계산, 성능 최적화를 위해 미리 계산해서 전달 권장)
            shuffle_files: 에폭마다 파일 순서를 섞을지 여부 (기본 True)
                          - True: 에폭마다 파일 순서 셔플로 학습 다양성 확보
                          - False: 파일 순서 고정 (검증/테스트용)
            seed: 랜덤 시드
            verbose: 진행 상황 출력 여부
        """
        super().__init__()
        self.parquet_files = [Path(f) for f in parquet_files]
        self.shuffle_files = shuffle_files
        self.seed = seed
        self.verbose = verbose
        self.epoch = 0
        
        if not self.parquet_files:
            raise ValueError("파일 리스트가 비어있습니다.")
        
        # 총 샘플 수 계산 (file_lengths가 제공되면 스캔 생략)
        if file_lengths is not None:
            if len(file_lengths) != len(self.parquet_files):
                raise ValueError(f"file_lengths 길이({len(file_lengths)})와 parquet_files 길이({len(self.parquet_files)})가 다릅니다.")
            self.file_lengths = file_lengths
            if verbose:
                print(f"파일 길이 정보 사용 (스캔 생략, {len(self.parquet_files)}개 파일)")
        else:
            self.file_lengths = []
            if verbose:
                print(f"Parquet 파일 정보 수집 중... ({len(self.parquet_files)}개 파일)")
            for file_path in self.parquet_files:
                pf = pq.ParquetFile(file_path)
                self.file_lengths.append(pf.metadata.num_rows)
        
        self.estimated_length = sum(self.file_lengths)
        if verbose:
            print(f"총 샘플 수: {self.estimated_length:,}")
            mode_str = "파일 순서 셔플" if shuffle_files else "순차 읽기"
            print(f"모드: {mode_str}")
    
    def set_epoch(self, epoch: int):
        """에폭 설정 (파일 순서 셔플에 사용)"""
        self.epoch = epoch
    
    def _get_file_indices_for_worker(self) -> List[int]:
        """현재 워커가 처리할 파일 인덱스 반환"""
        worker_info = get_worker_info()
        file_indices = list(range(len(self.parquet_files)))
        
        # 파일 순서 셔플 (shuffle_files가 True일 때)
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
        """
        파일에서 샘플을 pyarrow 배치 스트리밍으로 읽기
        
        pandas 대신 pyarrow iter_batches를 사용해 성능 대폭 향상:
        - 메모리 효율: 전체 파일을 한 번에 로드하지 않음
        - CPU 효율: 컬럼 단위 벡터 연산으로 파이썬 루프 오버헤드 최소화
        """
        parquet_file = pq.ParquetFile(file_path)
        
        # 배치 단위로 읽기 (기본 1024행, 파일 크기에 따라 조정 가능)
        for batch in parquet_file.iter_batches(batch_size=2048):
            # pyarrow 배열을 numpy로 한 번에 변환
            batch_size = batch.num_rows
            
            # state: list of lists → numpy 배열로 벡터 변환
            states_list = batch.column('state').to_pylist()
            states = np.array(states_list, dtype=np.float32).reshape(batch_size, 18, 8, 8)
            
            # policy: 스칼라 컬럼
            policies = batch.column('policy').to_numpy().astype(np.int64)
            
            # mask: list of lists → numpy 배열
            masks_list = batch.column('mask').to_pylist()
            masks = np.array(masks_list, dtype=np.float32)
            
            # value: 스칼라 컬럼
            values = batch.column('value').to_numpy().astype(np.float32)
            
            # 배치 내 각 샘플을 yield (torch 변환은 여기서)
            for i in range(batch_size):
                yield (
                    torch.from_numpy(states[i].copy()),
                    torch.tensor(policies[i], dtype=torch.long),
                    torch.from_numpy(masks[i].copy()),
                    torch.tensor(values[i], dtype=torch.float32),
                )
    
    def __iter__(self) -> Iterator[Tuple]:
        """데이터 스트리밍"""
        file_indices = self._get_file_indices_for_worker()
        
        for file_idx in file_indices:
            file_path = self.parquet_files[file_idx]
            yield from self._load_samples_from_file(file_path)
