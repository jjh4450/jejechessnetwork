"""
전처리 함수들의 pytest 테스트

전처리 함수들이 올바르게 동작하는지 확인합니다.
"""

import pytest
import chess
import numpy as np
from preprocessing import (
    board_to_tensor,
    legal_move_mask,
    move_to_action_index,
    action_index_to_move,
    game_result_to_value,
)


@pytest.fixture
def initial_board():
    """초기 체스 보드 생성"""
    return chess.Board()


def test_board_to_tensor_shape_and_dtype(initial_board):
    """보드 텐서 변환 - shape와 dtype 확인"""
    x = board_to_tensor(initial_board)
    
    assert x.shape == (18, 8, 8), f"Shape mismatch: {x.shape}"
    assert x.dtype == np.float32, f"Dtype mismatch: {x.dtype}"


def test_legal_move_mask_shape(initial_board):
    """합법 수 마스크 - shape 확인"""
    mask = legal_move_mask(initial_board)
    
    assert mask.shape == (4096,), f"Mask shape mismatch: {mask.shape}"


def test_legal_move_mask_count(initial_board):
    """합법 수 마스크 - 초기 보드의 합법 수 개수 확인"""
    mask = legal_move_mask(initial_board)
    legal_count = mask.sum()
    
    assert 15 <= legal_count <= 25, f"Unexpected legal move count: {legal_count}"


def test_move_to_action_index_roundtrip(initial_board):
    """수 변환 테스트 - move ↔ action_index 변환"""
    legal_moves = list(initial_board.legal_moves)
    
    if not legal_moves:
        pytest.skip("No legal moves available")
    
    test_move = legal_moves[0]
    action_idx = move_to_action_index(test_move)
    reconstructed_move = action_index_to_move(action_idx)
    
    assert test_move.from_square == reconstructed_move.from_square
    assert test_move.to_square == reconstructed_move.to_square


@pytest.mark.parametrize("result,side,expected", [
    ("1-0", chess.WHITE, 1.0),
    ("1-0", chess.BLACK, -1.0),
    ("0-1", chess.WHITE, -1.0),
    ("0-1", chess.BLACK, 1.0),
    ("1/2-1/2", chess.WHITE, 0.0),
    ("1/2-1/2", chess.BLACK, 0.0),
])
def test_game_result_to_value(result, side, expected):
    """게임 결과 변환 테스트"""
    value = game_result_to_value(result, side)
    assert abs(value - expected) < 1e-6, f"Value mismatch: {value} != {expected}"


def test_white_king_position(initial_board):
    """White King 위치 확인"""
    x = board_to_tensor(initial_board)
    white_king_square = initial_board.king(chess.WHITE)
    
    if white_king_square is None:
        pytest.skip("White king not found")
    
    row = 7 - (white_king_square // 8)
    col = white_king_square % 8
    king_channel = 5  # White King은 채널 5
    
    assert x[king_channel, row, col] == 1.0, "White King not found at expected position"


def test_board_to_tensor_channels(initial_board):
    """보드 텐서 채널 구성 확인"""
    x = board_to_tensor(initial_board)
    
    # 채널 수 확인
    assert x.shape[0] == 18, "Expected 18 channels"
    
    # 각 채널이 올바른 범위에 있는지 확인
    assert np.all(x >= 0), "All values should be >= 0"
    assert np.all(x <= 1), "All values should be <= 1"


def test_legal_move_mask_values(initial_board):
    """합법 수 마스크 값 확인 (0 또는 1만 있어야 함)"""
    mask = legal_move_mask(initial_board)
    
    assert np.all((mask == 0) | (mask == 1)), "Mask should only contain 0 or 1"
    assert mask.dtype == np.float32, "Mask should be float32"
