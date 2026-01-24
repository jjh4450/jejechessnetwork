"""
Parquet 파일에서 체스 데이터를 로드하는 PyTorch Dataset 클래스
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


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


class ParquetChessDataset(Dataset):
    """
    Parquet 파일에서 체스 데이터를 로드하는 Dataset
    
    여러 parquet 파일을 하나의 Dataset으로 처리합니다.
    """
    
    def __init__(self, parquet_dir, file_pattern="chess_samples_*.parquet", cache_files=False):
        """
        Args:
            parquet_dir: parquet 파일이 있는 디렉토리
            file_pattern: 파일 패턴
            cache_files: True면 파일을 메모리에 캐시 (메모리 사용량 증가)
        """
        self.parquet_dir = Path(parquet_dir)
        self.parquet_files = sorted(self.parquet_dir.glob(file_pattern))
        self.cache_files = cache_files
        
        if not self.parquet_files:
            raise ValueError(f"Parquet 파일을 찾을 수 없습니다: {parquet_dir}/{file_pattern}")
        
        # 각 파일의 행 수를 미리 계산
        self.file_lengths = []
        self.cumulative_lengths = [0]
        self.file_cache = {} if cache_files else None
        
        print(f"Parquet 파일 로드 중... ({len(self.parquet_files)}개 파일)")
        for file_path in tqdm(self.parquet_files, desc="파일 인덱싱"):
            df = pd.read_parquet(file_path)
            length = len(df)
            self.file_lengths.append(length)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
            
            if cache_files:
                self.file_cache[file_path] = df
        
        self.total_length = self.cumulative_lengths[-1]
        print(f"총 샘플 수: {self.total_length:,}")
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # 어떤 파일에 속하는지 찾기
        file_idx = 0
        for i, cum_len in enumerate(self.cumulative_lengths[1:], 1):
            if idx < cum_len:
                file_idx = i - 1
                break
        
        # 파일 내 상대 인덱스
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        # 파일 로드
        file_path = self.parquet_files[file_idx]
        if self.cache_files:
            df = self.file_cache[file_path]
        else:
            df = pd.read_parquet(file_path)
        
        row = df.iloc[local_idx]
        
        # 데이터 복원
        state, policy, mask, value = load_sample_from_parquet(row)
        
        return (
            torch.from_numpy(state).float(),
            torch.tensor(policy, dtype=torch.long),
            torch.from_numpy(mask).float(),
            torch.tensor(value, dtype=torch.float32),
        )
