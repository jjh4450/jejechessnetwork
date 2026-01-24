"""
체스 데이터 전처리 모듈

.pgn.zst 파일을 스트리밍으로 처리하여
(state, policy, mask, value) 샘플을 생성합니다.
"""

import zstandard as zstd
import chess
import chess.pgn
import numpy as np
from typing import List, Tuple, Optional


# 액션 공간: from-square × to-square = 64 × 64 = 4096
ACTION_SPACE_SIZE = 64 * 64


def square_to_index(square: chess.Square) -> int:
    """체스 보드의 square를 0-63 인덱스로 변환"""
    return square


def move_to_action_index(move: chess.Move) -> int:
    """
    체스 수를 액션 인덱스로 변환
    
    액션 인덱스 = from_square * 64 + to_square
    
    Args:
        move: chess.Move 객체
        
    Returns:
        액션 인덱스 (0~4095)
    """
    from_sq = square_to_index(move.from_square)
    to_sq = square_to_index(move.to_square)
    return from_sq * 64 + to_sq


def action_index_to_move(action_idx: int) -> chess.Move:
    """
    액션 인덱스를 체스 수로 변환
    
    Args:
        action_idx: 액션 인덱스 (0~4095)
        
    Returns:
        chess.Move 객체
    """
    from_sq = action_idx // 64
    to_sq = action_idx % 64
    return chess.Move(from_sq, to_sq)


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    체스 보드를 (18, 8, 8) 텐서로 변환
    
    채널 구성:
    0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    12: Side to move (1 if white, 0 if black)
    13: White castling rights (Kingside + Queenside)
    14: Black castling rights (Kingside + Queenside)
    15: En passant square (1 if exists, 0 otherwise)
    16: Halfmove clock (normalized)
    17: Fullmove count (normalized)
    
    Args:
        board: chess.Board 객체
        
    Returns:
        (18, 8, 8) numpy array (float32)
    """
    tensor = np.zeros((18, 8, 8), dtype=np.float32)
    
    # Piece channels (0-11) - 딕셔너리로 O(1) 조회 (최적화)
    piece_to_idx = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    for square in chess.SQUARES:
        row = 7 - (square // 8)  # 체스 보드는 아래에서 위로 (0=rank8, 7=rank1)
        col = square % 8
        
        piece = board.piece_at(square)
        if piece:
            piece_type_idx = piece_to_idx[piece.piece_type]  # O(1) 조회
            if piece.color == chess.WHITE:
                tensor[piece_type_idx, row, col] = 1.0
            else:
                tensor[piece_type_idx + 6, row, col] = 1.0
    
    # Side to move (12)
    tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    
    # Castling rights (13-14)
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] += 0.5
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, :, :] += 0.5
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[14, :, :] += 0.5
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[14, :, :] += 0.5
    
    # En passant square (15)
    if board.ep_square is not None:
        ep_row = 7 - (board.ep_square // 8)
        ep_col = board.ep_square % 8
        tensor[15, ep_row, ep_col] = 1.0
    
    # Halfmove clock (16) - normalized to [0, 1] (max 100)
    tensor[16, :, :] = min(board.halfmove_clock / 100.0, 1.0)
    
    # Fullmove count (17) - normalized to [0, 1] (max 200)
    tensor[17, :, :] = min(board.fullmove_number / 200.0, 1.0)
    
    return tensor


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """
    합법 수 마스크 생성
    
    Args:
        board: chess.Board 객체
        
    Returns:
        (4096,) numpy array (1=합법, 0=불법)
    """
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    
    for move in board.legal_moves:
        # 프로모션은 Queen 승격만 허용 (경량화)
        if move.promotion is not None and move.promotion != chess.QUEEN:
            continue
        
        action_idx = move_to_action_index(move)
        mask[action_idx] = 1.0
    
    return mask


def policy_label_from_move(move: chess.Move) -> int:
    """
    실제 기보의 수를 정책 라벨(액션 인덱스)로 변환
    
    Args:
        move: chess.Move 객체
        
    Returns:
        액션 인덱스 (0~4095)
    """
    return move_to_action_index(move)


def game_result_to_value(result: str, side_to_move: chess.Color) -> float:
    """
    게임 결과를 가치 라벨로 변환
    
    Args:
        result: PGN 결과 문자열 ("1-0", "0-1", "1/2-1/2", "*")
        side_to_move: 현재 수를 둘 차례 (WHITE or BLACK)
        
    Returns:
        가치 라벨: 백 승=+1, 무승부=0, 흑 승=-1
    """
    if result == "1-0":  # 백 승
        return 1.0 if side_to_move == chess.WHITE else -1.0
    elif result == "0-1":  # 흑 승
        return -1.0 if side_to_move == chess.WHITE else 1.0
    elif result == "1/2-1/2":  # 무승부
        return 0.0
    else:  # "*" (진행 중)
        return 0.0


def parse_time_control(time_control_str: str) -> Optional[int]:
    """
    시간 제어 문자열을 초 단위로 파싱
    
    Args:
        time_control_str: PGN TimeControl 헤더 (예: "300+0", "600+3", "1800+0")
        
    Returns:
        초 단위 시간 (초기 시간), 파싱 실패 시 None
    """
    if not time_control_str or time_control_str == "-" or time_control_str == "?":
        return None
    
    try:
        # "300+0" 형식 파싱
        parts = time_control_str.split("+")
        initial_time = int(parts[0])
        return initial_time
    except (ValueError, IndexError):
        return None


def parse_elo(elo_str: str) -> Optional[int]:
    """
    레이팅 문자열을 정수로 파싱
    
    Args:
        elo_str: PGN Elo 헤더 (예: "1500", "2000")
        
    Returns:
        레이팅 정수, 파싱 실패 시 None
    """
    if not elo_str or elo_str == "?":
        return None
    
    try:
        return int(elo_str)
    except ValueError:
        return None


def should_include_game(
    game: chess.pgn.Game,
    min_rating: Optional[int] = None,
    min_time_control: Optional[int] = None
) -> bool:
    """
    게임이 필터 조건을 만족하는지 확인
    
    Args:
        game: chess.pgn.Game 객체
        min_rating: 최소 레이팅 (양쪽 플레이어 모두 이 값 이상이어야 함)
        min_time_control: 최소 시간 제어 (초 단위, 초기 시간)
        
    Returns:
        조건을 만족하면 True
    """
    # 레이팅 필터
    if min_rating is not None:
        white_elo = parse_elo(game.headers.get("WhiteElo", "?"))
        black_elo = parse_elo(game.headers.get("BlackElo", "?"))
        
        # 양쪽 플레이어 모두 레이팅이 있어야 하고, 최소값 이상이어야 함
        if white_elo is None or black_elo is None:
            return False
        if white_elo < min_rating or black_elo < min_rating:
            return False
    
    # 시간 제어 필터
    if min_time_control is not None:
        time_control_str = game.headers.get("TimeControl", "?")
        time_control = parse_time_control(time_control_str)
        
        if time_control is None or time_control < min_time_control:
            return False
    
    return True


def extract_samples_from_pgn_zst(
    zst_path: str,
    max_games: Optional[int] = None,
    max_samples: Optional[int] = None,
    min_rating: Optional[int] = None,
    min_time_control: Optional[int] = None
) -> List[Tuple[np.ndarray, int, np.ndarray, float]]:
    """
    .pgn.zst 파일에서 샘플 추출
    
    Args:
        zst_path: .pgn.zst 파일 경로
        max_games: 최대 처리 게임 수 (None이면 전체)
        max_samples: 최대 샘플 수 (None이면 제한 없음)
        min_rating: 최소 레이팅 (양쪽 플레이어 모두 이 값 이상)
        min_time_control: 최소 시간 제어 (초 단위, 초기 시간)
        
    Returns:
        샘플 리스트: [(state, policy, mask, value), ...]
        - state: (18, 8, 8) numpy array
        - policy: 액션 인덱스 (int)
        - mask: (4096,) 합법 수 마스크
        - value: 가치 라벨 (float)
    """
    samples = []
    skipped_games = 0
    import io
    
    with open(zst_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as binary_reader:
            # 바이너리 스트림을 텍스트 스트림으로 변환
            text_reader = io.TextIOWrapper(binary_reader, encoding='utf-8', errors='ignore')
            game_count = 0
            
            while True:
                game = chess.pgn.read_game(text_reader)
                if game is None:
                    break
                
                # 필터링 조건 확인
                if not should_include_game(game, min_rating, min_time_control):
                    skipped_games += 1
                    game_count += 1
                    continue
                
                board = game.board()
                result = game.headers.get("Result", "*")
                
                # 게임의 각 수마다 샘플 생성
                for move in game.mainline_moves():
                    # 현재 포지션의 상태
                    state = board_to_tensor(board)
                    
                    # 정책 라벨 (실제 기보의 수)
                    policy = policy_label_from_move(move)
                    
                    # 합법 수 마스크
                    mask = legal_move_mask(board)
                    
                    # 가치 라벨 (게임 결과)
                    value = game_result_to_value(result, board.turn)
                    
                    samples.append((state, policy, mask, value))
                    
                    # 다음 수로 진행
                    board.push(move)
                    
                    # 샘플 수 제한 체크
                    if max_samples and len(samples) >= max_samples:
                        return samples
                
                game_count += 1
                if max_games and game_count >= max_games:
                    break
    
    return samples
