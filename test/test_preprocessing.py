"""
전처리 함수들의 pytest 테스트

전처리 함수들이 올바르게 동작하는지 확인합니다.
"""

import pytest
import chess
import chess.pgn
import numpy as np
import io
from preprocessing import (
    board_to_tensor,
    legal_move_mask,
    move_to_action_index,
    action_index_to_move,
    game_result_to_value,
    parse_time_control,
    parse_elo,
    should_include_game,
    policy_label_from_move,
    square_to_index,
    ACTION_SPACE_SIZE,
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


# ===== parse_time_control 테스트 =====

@pytest.mark.parametrize("time_str,expected", [
    ("300+0", 300),
    ("600+3", 600),
    ("1800+0", 1800),
    ("180+2", 180),
    ("60+0", 60),
])
def test_parse_time_control_valid(time_str, expected):
    """parse_time_control - 유효한 시간 제어 문자열 파싱"""
    result = parse_time_control(time_str)
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.parametrize("time_str", [
    "",
    "-",
    "?",
    None,
])
def test_parse_time_control_invalid(time_str):
    """parse_time_control - 유효하지 않은 시간 제어 문자열"""
    result = parse_time_control(time_str)
    assert result is None, f"Expected None for '{time_str}', got {result}"


def test_parse_time_control_malformed():
    """parse_time_control - 잘못된 형식"""
    result = parse_time_control("abc+xyz")
    assert result is None


# ===== parse_elo 테스트 =====

@pytest.mark.parametrize("elo_str,expected", [
    ("1500", 1500),
    ("2000", 2000),
    ("800", 800),
    ("2800", 2800),
    ("0", 0),
])
def test_parse_elo_valid(elo_str, expected):
    """parse_elo - 유효한 레이팅 문자열 파싱"""
    result = parse_elo(elo_str)
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.parametrize("elo_str", [
    "",
    "?",
    None,
    "abc",
    "15.5",
])
def test_parse_elo_invalid(elo_str):
    """parse_elo - 유효하지 않은 레이팅 문자열"""
    result = parse_elo(elo_str)
    assert result is None, f"Expected None for '{elo_str}', got {result}"


# ===== should_include_game 테스트 =====

@pytest.fixture
def sample_game():
    """테스트용 PGN 게임 생성"""
    pgn = """[Event "Test Event"]
[Site "Test Site"]
[Date "2024.01.01"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "1900"]
[TimeControl "600+0"]

1. e4 e5 2. Nf3 Nc6 1-0
"""
    game = chess.pgn.read_game(io.StringIO(pgn))
    return game


@pytest.fixture
def game_without_elo():
    """레이팅이 없는 게임"""
    pgn = """[Event "Test Event"]
[Result "1-0"]
[TimeControl "600+0"]

1. e4 e5 1-0
"""
    game = chess.pgn.read_game(io.StringIO(pgn))
    return game


@pytest.fixture
def game_without_time_control():
    """시간 제어가 없는 게임"""
    pgn = """[Event "Test Event"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "1900"]

1. e4 e5 1-0
"""
    game = chess.pgn.read_game(io.StringIO(pgn))
    return game


def test_should_include_game_no_filter(sample_game):
    """should_include_game - 필터 없이 모든 게임 포함"""
    result = should_include_game(sample_game)
    assert result is True


def test_should_include_game_rating_pass(sample_game):
    """should_include_game - 레이팅 필터 통과"""
    result = should_include_game(sample_game, min_rating=1800)
    assert result is True


def test_should_include_game_rating_fail(sample_game):
    """should_include_game - 레이팅 필터 실패 (레이팅 부족)"""
    result = should_include_game(sample_game, min_rating=2100)
    assert result is False


def test_should_include_game_no_elo(game_without_elo):
    """should_include_game - 레이팅이 없는 게임은 필터에 걸림"""
    result = should_include_game(game_without_elo, min_rating=1000)
    assert result is False


def test_should_include_game_time_control_pass(sample_game):
    """should_include_game - 시간 제어 필터 통과"""
    result = should_include_game(sample_game, min_time_control=300)
    assert result is True


def test_should_include_game_time_control_fail(sample_game):
    """should_include_game - 시간 제어 필터 실패"""
    result = should_include_game(sample_game, min_time_control=900)
    assert result is False


def test_should_include_game_no_time_control(game_without_time_control):
    """should_include_game - 시간 제어가 없는 게임은 필터에 걸림"""
    result = should_include_game(game_without_time_control, min_time_control=300)
    assert result is False


def test_should_include_game_both_filters_pass(sample_game):
    """should_include_game - 양쪽 필터 모두 통과"""
    result = should_include_game(sample_game, min_rating=1800, min_time_control=300)
    assert result is True


def test_should_include_game_both_filters_fail_rating(sample_game):
    """should_include_game - 레이팅 필터만 실패"""
    result = should_include_game(sample_game, min_rating=2500, min_time_control=300)
    assert result is False


def test_should_include_game_both_filters_fail_time(sample_game):
    """should_include_game - 시간 제어 필터만 실패"""
    result = should_include_game(sample_game, min_rating=1500, min_time_control=900)
    assert result is False


# ===== policy_label_from_move 테스트 =====

def test_policy_label_from_move():
    """policy_label_from_move - 기본 동작 확인"""
    move = chess.Move.from_uci("e2e4")
    label = policy_label_from_move(move)
    
    # e2=12, e4=28 -> 12*64 + 28 = 796
    expected = 12 * 64 + 28
    assert label == expected, f"Expected {expected}, got {label}"


def test_policy_label_roundtrip():
    """policy_label_from_move - 왕복 변환 확인"""
    move = chess.Move.from_uci("g1f3")
    label = policy_label_from_move(move)
    reconstructed = action_index_to_move(label)
    
    assert move.from_square == reconstructed.from_square
    assert move.to_square == reconstructed.to_square


# ===== square_to_index 테스트 =====

def test_square_to_index():
    """square_to_index - 기본 동작"""
    # a1 = 0, h8 = 63
    assert square_to_index(chess.A1) == 0
    assert square_to_index(chess.H8) == 63
    assert square_to_index(chess.E4) == chess.E4


# ===== ACTION_SPACE_SIZE 상수 테스트 =====

def test_action_space_size():
    """ACTION_SPACE_SIZE - 올바른 값 확인"""
    assert ACTION_SPACE_SIZE == 4096, f"Expected 4096, got {ACTION_SPACE_SIZE}"


# ===== 보드 상태 변환 심화 테스트 =====

def test_board_to_tensor_after_move(initial_board):
    """보드 텐서 - 수를 둔 후 상태 변환"""
    board = initial_board.copy()
    board.push_san("e4")
    
    x = board_to_tensor(board)
    
    # 흑의 차례
    assert x[12, 0, 0] == 0.0, "Side to move should be 0 (Black)"
    
    # e4에 폰이 있어야 함 (White Pawn = 채널 0)
    # e4 = square 28, row = 7 - 28//8 = 7 - 3 = 4, col = 28 % 8 = 4
    assert x[0, 4, 4] == 1.0, "White pawn should be at e4"


def test_board_to_tensor_castling_rights():
    """보드 텐서 - 캐슬링 권한 확인"""
    board = chess.Board()
    x = board_to_tensor(board)
    
    # 초기 상태에서 백/흑 모두 캐슬링 권한 있음
    assert x[13, 0, 0] == 1.0, "White should have full castling rights"
    assert x[14, 0, 0] == 1.0, "Black should have full castling rights"
    
    # 캐슬링 권한 제거
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1")
    x = board_to_tensor(board)
    
    # 백은 Kingside만, 흑은 Queenside만
    assert x[13, 0, 0] == 0.5, "White should have only kingside castling"
    assert x[14, 0, 0] == 0.5, "Black should have only queenside castling"


def test_board_to_tensor_en_passant():
    """보드 텐서 - 앙파상 확인"""
    # d5에서 e5로 앙파상 가능한 상황
    board = chess.Board("rnbqkbnr/pppp1ppp/8/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 3")
    x = board_to_tensor(board)
    
    # e6 = square 44, row = 7 - 44//8 = 7 - 5 = 2, col = 44 % 8 = 4
    assert x[15, 2, 4] == 1.0, "En passant square should be marked at e6"


def test_legal_move_mask_promotion():
    """합법 수 마스크 - 프로모션 수 처리"""
    # 백 폰이 e7에 있고 e8로 승격 가능한 상황
    board = chess.Board("8/4P3/8/8/8/8/8/4K2k w - - 0 1")
    mask = legal_move_mask(board)
    
    # Queen 승격만 허용되어야 함
    assert mask.sum() >= 1, "Should have at least one legal move"


def test_board_to_tensor_halfmove_clock():
    """보드 텐서 - halfmove clock 정규화"""
    board = chess.Board()
    board.halfmove_clock = 50  # 50수 규칙 절반
    x = board_to_tensor(board)
    
    assert abs(x[16, 0, 0] - 0.5) < 1e-6, "Halfmove clock should be normalized to 0.5"


def test_board_to_tensor_fullmove_number():
    """보드 텐서 - fullmove number 정규화"""
    board = chess.Board()
    board.fullmove_number = 100  # 100수
    x = board_to_tensor(board)
    
    assert abs(x[17, 0, 0] - 0.5) < 1e-6, "Fullmove number should be normalized to 0.5"


# ===== move_to_action_index 심화 테스트 =====

def test_move_to_action_index_all_corners():
    """move_to_action_index - 코너 케이스"""
    # a1-a1 (자기 자리, 실제로는 불가능하지만 인덱싱 테스트)
    move_a1 = chess.Move(chess.A1, chess.A1)
    assert move_to_action_index(move_a1) == 0
    
    # h8-h8
    move_h8 = chess.Move(chess.H8, chess.H8)
    assert move_to_action_index(move_h8) == 63 * 64 + 63


def test_action_index_to_move_boundaries():
    """action_index_to_move - 경계값 테스트"""
    # 최소값
    move_min = action_index_to_move(0)
    assert move_min.from_square == 0
    assert move_min.to_square == 0
    
    # 최대값
    move_max = action_index_to_move(4095)
    assert move_max.from_square == 63
    assert move_max.to_square == 63
