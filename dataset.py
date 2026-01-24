"""
PyTorch Dataset 클래스

체스 전처리 샘플을 PyTorch에서 사용할 수 있는 형태로 제공합니다.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple


class ChessDataset(Dataset):
    """
    체스 데이터셋
    
    각 샘플은 (state, policy, mask, value)로 구성됩니다.
    """
    
    def __init__(self, samples: List[Tuple[np.ndarray, int, np.ndarray, float]]):
        """
        Args:
            samples: 전처리된 샘플 리스트
                    [(state, policy, mask, value), ...]
        """
        self.samples = samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: 샘플 인덱스
            
        Returns:
            (state, policy, mask, value) 튜플
            - state: (18, 8, 8) float32 tensor
            - policy: () long tensor (액션 인덱스)
            - mask: (4096,) float32 tensor
            - value: () float32 tensor
        """
        state, policy, mask, value = self.samples[idx]
        
        return (
            torch.from_numpy(state).float(),
            torch.tensor(policy, dtype=torch.long),
            torch.from_numpy(mask).float(),
            torch.tensor(value, dtype=torch.float32),
        )
