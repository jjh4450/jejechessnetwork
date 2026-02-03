"""
fast_chess.py 테스트 - python-chess와 비교
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np
from typing import Set

from fast_chess import (
    # 상수
    WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    ACTION_SPACE_SIZE,
    # 유틸리티
    get_bit, set_bit, clear_bit, pop_count, get_lsb,
    square_to_rank, square_to_file,
    action_index_to_squares, squares_to_action_index,
    # 공격 테이블
    KNIGHT_ATTACKS, KING_ATTACKS, PAWN_ATTACKS,
    # State 함수
    create_initial_state, get_piece_at, get_turn, get_castling, get_ep_square,
    # 수 생성
    generate_legal_moves,
    # 체크 탐지
    is_in_check,
    # 게임 상태
    make_move, is_checkmate, is_stalemate, is_game_over,
    # RL 인터페이스
    board_to_tensor_fast, legal_move_mask_fast,
)


# =============================================================================
# 유틸리티 함수
# =============================================================================

def python_chess_to_action_idx(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square

def get_python_chess_legal_moves_set(board: chess.Board) -> Set[int]:
    moves = set()
    for move in board.legal_moves:
        if move.promotion and move.promotion != chess.QUEEN:
            continue
        moves.add(python_chess_to_action_idx(move))
    return moves

def get_fast_chess_legal_moves_set(state) -> Set[int]:
    return set(generate_legal_moves(state))


# =============================================================================
# 테스트: 초기 위치
# =============================================================================

class TestInitialPosition:
    
    def test_initial_position_pieces(self):
        state = create_initial_state()
        board = chess.Board()
        
        for square in range(64):
            piece = board.piece_at(square)
            fast_piece_type, fast_color = get_piece_at(state, square)
            
            if piece is None:
                assert fast_piece_type == 0, f"Square {square}: mismatch"
            else:
                assert fast_piece_type == piece.piece_type, f"Square {square}: piece type"
                expected_color = 0 if piece.color == chess.WHITE else 1
                assert fast_color == expected_color, f"Square {square}: color"
    
    def test_initial_turn(self):
        state = create_initial_state()
        assert get_turn(state) == WHITE
    
    def test_initial_castling_rights(self):
        state = create_initial_state()
        assert get_castling(state) == 0b1111
    
    def test_initial_ep_square(self):
        state = create_initial_state()
        assert get_ep_square(state) == -1


# =============================================================================
# 테스트: 수 생성 (초기 위치)
# =============================================================================

class TestMoveGenerationInitial:
    
    def test_initial_legal_moves_count(self):
        state = create_initial_state()
        board = chess.Board()
        
        fast_moves = generate_legal_moves(state)
        python_moves = [m for m in board.legal_moves if not m.promotion or m.promotion == chess.QUEEN]
        
        assert len(fast_moves) == len(python_moves), \
            f"Count: fast={len(fast_moves)}, python={len(python_moves)}"
    
    def test_initial_legal_moves_exact(self):
        state = create_initial_state()
        board = chess.Board()
        
        fast_moves = get_fast_chess_legal_moves_set(state)
        python_moves = get_python_chess_legal_moves_set(board)
        
        assert fast_moves == python_moves, "Initial legal moves mismatch"


# =============================================================================
# 테스트: 수 실행
# =============================================================================

class TestMakeMove:
    
    def test_e2e4_move(self):
        state = create_initial_state()
        board = chess.Board()
        
        action_idx = squares_to_action_index(12, 28)
        make_move(state, action_idx)
        board.push(chess.Move(12, 28))
        
        assert get_turn(state) == BLACK
        assert board.turn == chess.BLACK
        assert get_ep_square(state) == 20
        assert board.ep_square == 20
        
        piece_type, color = get_piece_at(state, 28)
        assert piece_type == PAWN
        assert color == WHITE
    
    def test_multiple_moves_position(self):
        state = create_initial_state()
        board = chess.Board()
        
        moves = [(12, 28), (52, 36), (6, 21), (57, 42)]
        
        for from_sq, to_sq in moves:
            action_idx = squares_to_action_index(from_sq, to_sq)
            make_move(state, action_idx)
            board.push(chess.Move(from_sq, to_sq))
        
        for square in range(64):
            piece = board.piece_at(square)
            fast_piece_type, fast_color = get_piece_at(state, square)
            
            if piece is None:
                assert fast_piece_type == 0
            else:
                assert fast_piece_type == piece.piece_type


# =============================================================================
# 테스트: 특수 수
# =============================================================================

class TestSpecialMoves:
    
    def test_castling_available(self):
        state = create_initial_state()
        board = chess.Board()
        
        moves = [(12, 28), (52, 36), (6, 21), (57, 42), (5, 26), (61, 34)]
        
        for from_sq, to_sq in moves:
            make_move(state, squares_to_action_index(from_sq, to_sq))
            board.push(chess.Move(from_sq, to_sq))
        
        fast_moves = get_fast_chess_legal_moves_set(state)
        python_moves = get_python_chess_legal_moves_set(board)
        
        castling_move = squares_to_action_index(4, 6)
        
        assert castling_move in python_moves, "python-chess: no castling"
        assert castling_move in fast_moves, "fast_chess: no castling"


# =============================================================================
# 테스트: 체크 탐지
# =============================================================================

class TestCheck:
    
    def test_check_detection(self):
        state = create_initial_state()
        board = chess.Board()
        
        moves = [(12, 28), (52, 36), (3, 39), (57, 42), (5, 26), (62, 45)]
        
        for from_sq, to_sq in moves:
            make_move(state, squares_to_action_index(from_sq, to_sq))
            board.push(chess.Move(from_sq, to_sq))
        
        assert not board.is_check()
        assert not is_in_check(state, get_turn(state))


# =============================================================================
# 테스트: 게임 종료
# =============================================================================

class TestGameOver:
    
    def test_initial_not_game_over(self):
        state = create_initial_state()
        assert not is_game_over(state)
    
    def test_scholars_mate(self):
        state = create_initial_state()
        board = chess.Board()
        
        moves = [(12, 28), (52, 36), (3, 39), (57, 42), (5, 26), (62, 45), (39, 53)]
        
        for from_sq, to_sq in moves:
            make_move(state, squares_to_action_index(from_sq, to_sq))
            board.push(chess.Move(from_sq, to_sq))
        
        assert board.is_checkmate()
        assert is_checkmate(state)
        assert is_game_over(state)


# =============================================================================
# 테스트: RL 인터페이스
# =============================================================================

class TestRLInterface:
    
    def test_board_to_tensor_shape(self):
        state = create_initial_state()
        tensor = board_to_tensor_fast(state)
        
        assert tensor.shape == (18, 8, 8)
        assert tensor.dtype == np.float32
    
    def test_legal_move_mask_shape(self):
        state = create_initial_state()
        mask = legal_move_mask_fast(state)
        
        assert mask.shape == (4096,)
        assert mask.dtype == np.float32
    
    def test_legal_move_mask_values(self):
        state = create_initial_state()
        mask = legal_move_mask_fast(state)
        
        assert np.all((mask == 0) | (mask == 1))
        assert np.sum(mask) == 20


# =============================================================================
# 테스트: 랜덤 게임
# =============================================================================

class TestRandomGames:
    
    def test_random_game_consistency(self):
        import random
        random.seed(42)
        
        state = create_initial_state()
        board = chess.Board()
        
        max_moves = 100
        for move_num in range(max_moves):
            if board.is_game_over():
                break
            
            fast_moves = get_fast_chess_legal_moves_set(state)
            python_moves = get_python_chess_legal_moves_set(board)
            
            only_fast = fast_moves - python_moves
            only_python = python_moves - fast_moves
            
            if only_fast or only_python:
                print(f"\nMove {move_num + 1}: FEN: {board.fen()}")
                if only_fast:
                    print(f"  fast only: {only_fast}")
                if only_python:
                    print(f"  python only: {only_python}")
            
            assert fast_moves == python_moves, f"Move {move_num + 1}: mismatch"
            
            if not python_moves:
                break
            
            action_idx = random.choice(list(python_moves))
            from_sq, to_sq = action_index_to_squares(action_idx)
            
            piece = board.piece_at(from_sq)
            promo = None
            if piece and piece.piece_type == chess.PAWN:
                to_rank = to_sq // 8
                if (piece.color == chess.WHITE and to_rank == 7) or \
                   (piece.color == chess.BLACK and to_rank == 0):
                    promo = chess.QUEEN
            
            make_move(state, action_idx)
            board.push(chess.Move(from_sq, to_sq, promotion=promo))


# =============================================================================
# 테스트 실행
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("fast_chess.py 테스트 시작")
    print("=" * 60)
    
    try:
        print("\n[1] 초기 위치 테스트...")
        test = TestInitialPosition()
        test.test_initial_position_pieces()
        test.test_initial_turn()
        test.test_initial_castling_rights()
        test.test_initial_ep_square()
        print("    ✓ 통과")
        
        print("\n[2] 수 생성 테스트...")
        test = TestMoveGenerationInitial()
        test.test_initial_legal_moves_count()
        print("    ✓ 통과")
        
        print("\n[3] 수 실행 테스트...")
        test = TestMakeMove()
        test.test_e2e4_move()
        test.test_multiple_moves_position()
        print("    ✓ 통과")
        
        print("\n[4] RL 인터페이스 테스트...")
        test = TestRLInterface()
        test.test_board_to_tensor_shape()
        test.test_legal_move_mask_shape()
        test.test_legal_move_mask_values()
        print("    ✓ 통과")
        
        print("\n[5] 게임 종료 테스트...")
        test = TestGameOver()
        test.test_initial_not_game_over()
        test.test_scholars_mate()
        print("    ✓ 통과")
        
        print("\n[6] 랜덤 게임 테스트...")
        test = TestRandomGames()
        test.test_random_game_consistency()
        print("    ✓ 통과")
        
        print("\n" + "=" * 60)
        print("모든 테스트 통과!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
