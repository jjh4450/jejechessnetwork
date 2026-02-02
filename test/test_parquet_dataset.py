"""
ParquetChessDataset 테스트

Parquet 데이터셋 클래스와 유틸리티 함수들이 올바르게 동작하는지 확인합니다.
"""

import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from parquet_dataset import (
    load_sample_from_parquet,
    get_parquet_file_info,
    split_files_by_ratio,
    ParquetChessDataset,
)


@pytest.fixture
def temp_parquet_dir():
    """임시 Parquet 파일 디렉토리 생성"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # 테스트 후 정리
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_parquet_files(temp_parquet_dir):
    """테스트용 Parquet 파일 생성 (3개 파일, 각 100개 샘플)"""
    files = []
    samples_per_file = 100
    
    for i in range(3):
        # 샘플 데이터 생성
        data = {
            'state': [np.random.randn(18 * 8 * 8).astype(np.float32).tolist() 
                     for _ in range(samples_per_file)],
            'policy': [np.random.randint(0, 4096) for _ in range(samples_per_file)],
            'mask': [np.random.randint(0, 2, size=4096).astype(np.float32).tolist() 
                    for _ in range(samples_per_file)],
            'value': [np.random.uniform(-1, 1) for _ in range(samples_per_file)],
        }
        
        df = pd.DataFrame(data)
        file_path = temp_parquet_dir / f"chess_samples_{i:04d}.parquet"
        df.to_parquet(file_path, index=False, engine='pyarrow')
        files.append(file_path)
    
    return files


@pytest.fixture
def sample_df_row():
    """테스트용 DataFrame 행 생성"""
    state = np.random.randn(18 * 8 * 8).astype(np.float32)
    policy = 42
    mask = np.random.randint(0, 2, size=4096).astype(np.float32)
    value = 0.5
    
    df = pd.DataFrame({
        'state': [state.tolist()],
        'policy': [policy],
        'mask': [mask.tolist()],
        'value': [value],
    })
    
    return df.iloc[0], state, policy, mask, value


# ===== load_sample_from_parquet 테스트 =====

def test_load_sample_from_parquet_shapes(sample_df_row):
    """load_sample_from_parquet - 반환값 shape 확인"""
    row, expected_state, expected_policy, expected_mask, expected_value = sample_df_row
    
    state, policy, mask, value = load_sample_from_parquet(row)
    
    assert state.shape == (18, 8, 8), f"State shape mismatch: {state.shape}"
    assert mask.shape == (4096,), f"Mask shape mismatch: {mask.shape}"
    assert isinstance(policy, int), f"Policy should be int, got {type(policy)}"
    assert isinstance(value, float), f"Value should be float, got {type(value)}"


def test_load_sample_from_parquet_values(sample_df_row):
    """load_sample_from_parquet - 값 복원 확인"""
    row, expected_state, expected_policy, expected_mask, expected_value = sample_df_row
    
    state, policy, mask, value = load_sample_from_parquet(row)
    
    assert policy == expected_policy, f"Policy mismatch: {policy} != {expected_policy}"
    assert abs(value - expected_value) < 1e-6, f"Value mismatch: {value} != {expected_value}"
    assert np.allclose(state.flatten(), expected_state), "State values mismatch"
    assert np.allclose(mask, expected_mask), "Mask values mismatch"


def test_load_sample_from_parquet_dtype(sample_df_row):
    """load_sample_from_parquet - dtype 확인"""
    row, _, _, _, _ = sample_df_row
    
    state, policy, mask, value = load_sample_from_parquet(row)
    
    assert state.dtype == np.float32, f"State dtype should be float32, got {state.dtype}"
    assert mask.dtype == np.float32, f"Mask dtype should be float32, got {mask.dtype}"


# ===== get_parquet_file_info 테스트 =====

def test_get_parquet_file_info_returns_correct_count(temp_parquet_dir, sample_parquet_files):
    """get_parquet_file_info - 파일 수 확인"""
    files, lengths = get_parquet_file_info(str(temp_parquet_dir))
    
    assert len(files) == 3, f"Expected 3 files, got {len(files)}"
    assert len(lengths) == 3, f"Expected 3 lengths, got {len(lengths)}"


def test_get_parquet_file_info_correct_lengths(temp_parquet_dir, sample_parquet_files):
    """get_parquet_file_info - 파일별 샘플 수 확인"""
    files, lengths = get_parquet_file_info(str(temp_parquet_dir))
    
    for length in lengths:
        assert length == 100, f"Expected 100 samples per file, got {length}"


def test_get_parquet_file_info_sorted(temp_parquet_dir, sample_parquet_files):
    """get_parquet_file_info - 파일 정렬 확인"""
    files, _ = get_parquet_file_info(str(temp_parquet_dir))
    
    file_names = [f.name for f in files]
    assert file_names == sorted(file_names), "Files should be sorted"


def test_get_parquet_file_info_empty_dir(temp_parquet_dir):
    """get_parquet_file_info - 빈 디렉토리 예외 처리"""
    with pytest.raises(ValueError, match="Parquet 파일을 찾을 수 없습니다"):
        get_parquet_file_info(str(temp_parquet_dir))


# ===== split_files_by_ratio 테스트 =====

def test_split_files_by_ratio_default(sample_parquet_files):
    """split_files_by_ratio - 기본 분할 (9:1)"""
    lengths = [100, 100, 100]  # 총 300개
    
    train_files, val_files, train_lengths, val_lengths, train_samples, val_samples = split_files_by_ratio(
        sample_parquet_files, lengths, train_ratio=0.9
    )
    
    assert len(train_files) + len(val_files) == 3, "All files should be assigned"
    assert len(train_lengths) == len(train_files), "train_lengths should match train_files"
    assert len(val_lengths) == len(val_files), "val_lengths should match val_files"
    assert train_samples + val_samples == 300, "All samples should be assigned"


def test_split_files_by_ratio_balanced(sample_parquet_files):
    """split_files_by_ratio - 50:50 분할"""
    lengths = [100, 100, 100]
    
    train_files, val_files, train_lengths, val_lengths, train_samples, val_samples = split_files_by_ratio(
        sample_parquet_files, lengths, train_ratio=0.5
    )
    
    # 파일 단위 분할이므로 정확히 50%가 아닐 수 있음
    assert len(train_files) >= 1, "Should have at least 1 train file"
    assert len(val_files) >= 1, "Should have at least 1 val file"
    assert len(train_lengths) == len(train_files), "train_lengths should match train_files"
    assert len(val_lengths) == len(val_files), "val_lengths should match val_files"


def test_split_files_by_ratio_shuffle(sample_parquet_files):
    """split_files_by_ratio - 셔플 적용 확인"""
    lengths = [100, 100, 100]
    
    # 같은 시드로 두 번 호출
    train1, val1, _, _, _, _ = split_files_by_ratio(
        sample_parquet_files, lengths, shuffle=True, seed=42
    )
    train2, val2, _, _, _, _ = split_files_by_ratio(
        sample_parquet_files, lengths, shuffle=True, seed=42
    )
    
    # 같은 결과여야 함
    assert train1 == train2, "Same seed should produce same result"
    assert val1 == val2, "Same seed should produce same result"


def test_split_files_by_ratio_no_empty_val(sample_parquet_files):
    """split_files_by_ratio - val이 비어있지 않음 확인"""
    lengths = [100, 100, 100]
    
    # train_ratio=1.0으로 해도 val이 최소 1개 있어야 함
    train_files, val_files, train_lengths, val_lengths, _, _ = split_files_by_ratio(
        sample_parquet_files, lengths, train_ratio=0.99
    )
    
    assert len(val_files) >= 1, "Val should have at least 1 file"
    assert len(val_lengths) >= 1, "val_lengths should have at least 1 element"


def test_split_files_by_ratio_length_mismatch(sample_parquet_files):
    """split_files_by_ratio - 길이 불일치 예외 처리"""
    lengths = [100, 100]  # 파일 3개인데 길이 2개
    
    with pytest.raises(ValueError, match="길이가 다릅니다"):
        split_files_by_ratio(sample_parquet_files, lengths)


def test_split_files_by_ratio_empty_files():
    """split_files_by_ratio - 빈 파일 리스트 예외 처리"""
    with pytest.raises(ValueError, match="비어있습니다"):
        split_files_by_ratio([], [])


# ===== ParquetChessDataset 테스트 =====

def test_dataset_creation(sample_parquet_files):
    """ParquetChessDataset - 생성 확인"""
    dataset = ParquetChessDataset(
        parquet_files=sample_parquet_files,
        shuffle_files=True,
        verbose=False
    )
    
    assert dataset.estimated_length == 300, f"Expected 300 samples, got {dataset.estimated_length}"
    assert len(dataset.parquet_files) == 3, f"Expected 3 files, got {len(dataset.parquet_files)}"


def test_dataset_iteration_with_shuffle_files(sample_parquet_files):
    """ParquetChessDataset - 파일 셔플 모드 반복 확인"""
    dataset = ParquetChessDataset(
        parquet_files=sample_parquet_files,
        shuffle_files=True,
        verbose=False,
        batch_size=10  # 테스트용 배치 크기
    )
    
    total_samples = 0
    for batch in dataset:
        states, _, _, _ = batch
        total_samples += states.shape[0]
    
    assert total_samples == 300, f"Expected 300 samples, got {total_samples}"


def test_dataset_iteration_sequential(sample_parquet_files):
    """ParquetChessDataset - 순차 모드 반복 확인"""
    dataset = ParquetChessDataset(
        parquet_files=sample_parquet_files,
        shuffle_files=False,
        verbose=False,
        batch_size=10  # 테스트용 배치 크기
    )
    
    total_samples = 0
    for batch in dataset:
        states, _, _, _ = batch
        total_samples += states.shape[0]
    
    assert total_samples == 300, f"Expected 300 samples, got {total_samples}"


def test_dataset_sample_format(sample_parquet_files):
    """ParquetChessDataset - 배치 형식 확인"""
    dataset = ParquetChessDataset(
        parquet_files=sample_parquet_files,
        shuffle_files=False,
        verbose=False,
        batch_size=32  # 테스트용 배치 크기
    )
    
    for batch in dataset:
        states, policies, masks, values = batch
        
        assert isinstance(states, torch.Tensor), "States should be a tensor"
        assert isinstance(policies, torch.Tensor), "Policies should be a tensor"
        assert isinstance(masks, torch.Tensor), "Masks should be a tensor"
        assert isinstance(values, torch.Tensor), "Values should be a tensor"
        
        batch_size = states.shape[0]
        assert states.shape == (batch_size, 18, 8, 8), f"States shape mismatch: {states.shape}"
        assert policies.shape == (batch_size,), f"Policies shape mismatch: {policies.shape}"
        assert masks.shape == (batch_size, 4096), f"Masks shape mismatch: {masks.shape}"
        assert values.shape == (batch_size,), f"Values shape mismatch: {values.shape}"
        
        assert states.dtype == torch.float32, f"States dtype: {states.dtype}"
        assert policies.dtype == torch.int64, f"Policies dtype: {policies.dtype}"
        assert masks.dtype == torch.float32, f"Masks dtype: {masks.dtype}"
        assert values.dtype == torch.float32, f"Values dtype: {values.dtype}"
        
        break  # 첫 배치만 확인


def test_dataset_set_epoch(sample_parquet_files):
    """ParquetChessDataset - set_epoch 동작 확인 (파일 순서 셔플)"""
    dataset = ParquetChessDataset(
        parquet_files=sample_parquet_files,
        shuffle_files=True,
        seed=42,
        verbose=False,
        batch_size=10  # 테스트용 배치 크기
    )
    
    # epoch 0
    dataset.set_epoch(0)
    samples_epoch0 = []
    for batch in dataset:
        _, policies, _, _ = batch
        samples_epoch0.extend(policies.tolist())
    
    # epoch 1
    dataset.set_epoch(1)
    samples_epoch1 = []
    for batch in dataset:
        _, policies, _, _ = batch
        samples_epoch1.extend(policies.tolist())
    
    # 다른 epoch은 다른 순서여야 함 (파일 순서가 다르게 됨)
    assert samples_epoch0 != samples_epoch1, "Different epochs should produce different order"


def test_dataset_with_dataloader(sample_parquet_files):
    """ParquetChessDataset - DataLoader와 함께 사용"""
    from torch.utils.data import DataLoader
    
    dataset = ParquetChessDataset(
        parquet_files=sample_parquet_files,
        shuffle_files=True,
        verbose=False,
        batch_size=32  # Dataset이 반환할 배치 크기
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=None,  # Dataset이 이미 배치를 반환
        collate_fn=lambda x: x[0],  # 배치가 이미 완성되어 있으므로 첫 번째 요소만 반환
        shuffle=False
    )
    
    total_samples = 0
    for batch in loader:
        states, policies, masks, values = batch
        
        batch_size = states.shape[0]
        assert batch_size <= 32, f"Batch size should be <= 32, got {batch_size}"
        assert states.shape == (batch_size, 18, 8, 8), f"States shape: {states.shape}"
        assert policies.shape == (batch_size,), f"Policies shape: {policies.shape}"
        assert masks.shape == (batch_size, 4096), f"Masks shape: {masks.shape}"
        assert values.shape == (batch_size,), f"Values shape: {values.shape}"
        
        total_samples += batch_size
    
    assert total_samples == 300, f"Expected 300 samples, got {total_samples}"


def test_dataset_empty_files():
    """ParquetChessDataset - 빈 파일 리스트 예외 처리"""
    with pytest.raises(ValueError, match="비어있습니다"):
        ParquetChessDataset(parquet_files=[], verbose=False)


def test_dataset_shuffle_files_flag(sample_parquet_files):
    """ParquetChessDataset - shuffle_files 플래그 확인"""
    dataset_shuffle = ParquetChessDataset(
        parquet_files=sample_parquet_files,
        shuffle_files=True,
        verbose=False
    )
    
    dataset_no_shuffle = ParquetChessDataset(
        parquet_files=sample_parquet_files,
        shuffle_files=False,
        verbose=False
    )
    
    assert dataset_shuffle.shuffle_files == True, "shuffle_files should be True"
    assert dataset_no_shuffle.shuffle_files == False, "shuffle_files should be False"


def test_dataset_deterministic_with_seed(sample_parquet_files):
    """ParquetChessDataset - 시드로 재현성 확인"""
    # 같은 시드, 같은 epoch
    dataset1 = ParquetChessDataset(
        parquet_files=sample_parquet_files,
        shuffle_files=True,
        seed=42,
        verbose=False,
        batch_size=10  # 테스트용 배치 크기
    )
    dataset1.set_epoch(0)
    
    dataset2 = ParquetChessDataset(
        parquet_files=sample_parquet_files,
        shuffle_files=True,
        seed=42,
        verbose=False,
        batch_size=10  # 테스트용 배치 크기
    )
    dataset2.set_epoch(0)
    
    samples1 = []
    for batch in dataset1:
        _, policies, _, _ = batch
        samples1.extend(policies.tolist())
    
    samples2 = []
    for batch in dataset2:
        _, policies, _, _ = batch
        samples2.extend(policies.tolist())
    
    assert samples1 == samples2, "Same seed and epoch should produce same result"
