"""
ChessDataset 클래스의 pytest 테스트

dataset.py의 ChessDataset 클래스가 올바르게 동작하는지 확인합니다.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import ChessDataset


@pytest.fixture
def sample_data():
    """테스트용 샘플 데이터 생성"""
    samples = []
    for i in range(100):
        state = np.random.randn(18, 8, 8).astype(np.float32)
        policy = np.random.randint(0, 4096)
        mask = np.random.randint(0, 2, size=4096).astype(np.float32)
        value = np.random.uniform(-1, 1)
        samples.append((state, policy, mask, value))
    return samples


@pytest.fixture
def single_sample():
    """단일 샘플 데이터"""
    state = np.ones((18, 8, 8), dtype=np.float32) * 0.5
    policy = 42
    mask = np.zeros(4096, dtype=np.float32)
    mask[42] = 1.0  # policy 위치만 1
    value = 0.75
    return [(state, policy, mask, value)]


# ===== 기본 동작 테스트 =====

def test_dataset_creation(sample_data):
    """ChessDataset - 생성 확인"""
    dataset = ChessDataset(sample_data)
    assert len(dataset) == 100, f"Expected 100 samples, got {len(dataset)}"


def test_dataset_empty():
    """ChessDataset - 빈 데이터셋 생성"""
    dataset = ChessDataset([])
    assert len(dataset) == 0, "Empty dataset should have length 0"


def test_dataset_single_sample(single_sample):
    """ChessDataset - 단일 샘플 데이터셋"""
    dataset = ChessDataset(single_sample)
    assert len(dataset) == 1


# ===== __getitem__ 테스트 =====

def test_getitem_returns_tuple(sample_data):
    """ChessDataset - __getitem__이 4-튜플 반환"""
    dataset = ChessDataset(sample_data)
    result = dataset[0]
    
    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 4, "Result should have 4 elements"


def test_getitem_tensor_types(sample_data):
    """ChessDataset - 반환값이 모두 Tensor인지 확인"""
    dataset = ChessDataset(sample_data)
    state, policy, mask, value = dataset[0]
    
    assert isinstance(state, torch.Tensor), "State should be a Tensor"
    assert isinstance(policy, torch.Tensor), "Policy should be a Tensor"
    assert isinstance(mask, torch.Tensor), "Mask should be a Tensor"
    assert isinstance(value, torch.Tensor), "Value should be a Tensor"


def test_getitem_shapes(sample_data):
    """ChessDataset - 반환 텐서 shape 확인"""
    dataset = ChessDataset(sample_data)
    state, policy, mask, value = dataset[0]
    
    assert state.shape == (18, 8, 8), f"State shape mismatch: {state.shape}"
    assert policy.shape == (), f"Policy should be scalar, got {policy.shape}"
    assert mask.shape == (4096,), f"Mask shape mismatch: {mask.shape}"
    assert value.shape == (), f"Value should be scalar, got {value.shape}"


def test_getitem_dtypes(sample_data):
    """ChessDataset - 반환 텐서 dtype 확인"""
    dataset = ChessDataset(sample_data)
    state, policy, mask, value = dataset[0]
    
    assert state.dtype == torch.float32, f"State dtype: {state.dtype}"
    assert policy.dtype == torch.int64, f"Policy dtype: {policy.dtype}"
    assert mask.dtype == torch.float32, f"Mask dtype: {mask.dtype}"
    assert value.dtype == torch.float32, f"Value dtype: {value.dtype}"


def test_getitem_values_preserved(single_sample):
    """ChessDataset - 값이 올바르게 보존되는지 확인"""
    expected_state, expected_policy, expected_mask, expected_value = single_sample[0]
    dataset = ChessDataset(single_sample)
    state, policy, mask, value = dataset[0]
    
    # 값 비교
    assert torch.allclose(state, torch.from_numpy(expected_state)), "State values mismatch"
    assert policy.item() == expected_policy, f"Policy mismatch: {policy.item()} != {expected_policy}"
    assert torch.allclose(mask, torch.from_numpy(expected_mask)), "Mask values mismatch"
    assert abs(value.item() - expected_value) < 1e-6, f"Value mismatch: {value.item()} != {expected_value}"


def test_getitem_all_indices(sample_data):
    """ChessDataset - 모든 인덱스 접근 가능"""
    dataset = ChessDataset(sample_data)
    
    for i in range(len(dataset)):
        result = dataset[i]
        assert result is not None, f"Index {i} returned None"
        assert len(result) == 4, f"Index {i} returned wrong length"


def test_getitem_negative_index(sample_data):
    """ChessDataset - 음수 인덱스 접근"""
    dataset = ChessDataset(sample_data)
    
    # Python 리스트처럼 음수 인덱스 지원 확인
    last_sample = dataset[-1]
    assert last_sample is not None, "Negative index should work"


# ===== DataLoader 통합 테스트 =====

def test_with_dataloader(sample_data):
    """ChessDataset - DataLoader와 함께 사용"""
    dataset = ChessDataset(sample_data)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    total_samples = 0
    for batch in loader:
        states, policies, masks, values = batch
        
        batch_size = states.shape[0]
        assert batch_size <= 16, f"Batch size should be <= 16, got {batch_size}"
        
        assert states.shape == (batch_size, 18, 8, 8), f"States shape: {states.shape}"
        assert policies.shape == (batch_size,), f"Policies shape: {policies.shape}"
        assert masks.shape == (batch_size, 4096), f"Masks shape: {masks.shape}"
        assert values.shape == (batch_size,), f"Values shape: {values.shape}"
        
        total_samples += batch_size
    
    assert total_samples == 100, f"Expected 100 samples, got {total_samples}"


def test_with_dataloader_shuffle(sample_data):
    """ChessDataset - DataLoader 셔플 모드"""
    dataset = ChessDataset(sample_data)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    batches = list(loader)
    assert len(batches) == 10, f"Expected 10 batches, got {len(batches)}"


def test_with_dataloader_drop_last(sample_data):
    """ChessDataset - DataLoader drop_last 옵션"""
    # 100개 샘플, batch_size=30, drop_last=True -> 3배치 * 30 = 90개
    dataset = ChessDataset(sample_data)
    loader = DataLoader(dataset, batch_size=30, shuffle=False, drop_last=True)
    
    total_samples = sum(batch[0].shape[0] for batch in loader)
    assert total_samples == 90, f"Expected 90 samples with drop_last, got {total_samples}"


# ===== Edge Cases =====

def test_dataset_with_edge_values():
    """ChessDataset - 극단값 처리"""
    state = np.full((18, 8, 8), 1.0, dtype=np.float32)  # 최대값
    policy = 4095  # 최대 인덱스
    mask = np.ones(4096, dtype=np.float32)  # 모두 1
    value = 1.0  # 최대값
    
    samples = [(state, policy, mask, value)]
    dataset = ChessDataset(samples)
    
    s, p, m, v = dataset[0]
    assert p.item() == 4095
    assert v.item() == 1.0


def test_dataset_with_negative_value():
    """ChessDataset - 음수 value 처리"""
    state = np.zeros((18, 8, 8), dtype=np.float32)
    policy = 0
    mask = np.zeros(4096, dtype=np.float32)
    mask[0] = 1.0
    value = -1.0  # 최소값
    
    samples = [(state, policy, mask, value)]
    dataset = ChessDataset(samples)
    
    _, _, _, v = dataset[0]
    assert v.item() == -1.0


def test_dataset_with_zero_value():
    """ChessDataset - 0 value (무승부) 처리"""
    state = np.zeros((18, 8, 8), dtype=np.float32)
    policy = 100
    mask = np.zeros(4096, dtype=np.float32)
    value = 0.0  # 무승부
    
    samples = [(state, policy, mask, value)]
    dataset = ChessDataset(samples)
    
    _, _, _, v = dataset[0]
    assert v.item() == 0.0
