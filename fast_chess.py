"""
Numba 기반 비트보드 체스 엔진 (최적화 버전)

GameState를 numpy 배열로 표현하여 최대 성능을 달성합니다.
state[0:12] = 12개 비트보드
state[12] = turn
state[13] = castling_rights
state[14] = ep_square
state[15] = halfmove_clock
state[16] = fullmove_number
"""

import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList
from typing import Tuple

# =============================================================================
# 상수 정의
# =============================================================================

PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

WHITE = 0
BLACK = 1

ACTION_SPACE_SIZE = 64 * 64
STATE_SIZE = 17  # 12 bitboards + 5 state variables

# 64비트 마스크
MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)

# 파일/랭크 마스크
FILE_A = np.uint64(0x0101010101010101)
FILE_H = np.uint64(0x8080808080808080)
RANK_1 = np.uint64(0xFF)
RANK_8 = np.uint64(0xFF00000000000000)

# =============================================================================
# 비트보드 유틸리티 함수
# =============================================================================

@njit(cache=True)
def get_bit(bitboard, square):
    return (bitboard >> np.uint64(square)) & np.uint64(1)

@njit(cache=True)
def set_bit(bitboard, square):
    return bitboard | (np.uint64(1) << np.uint64(square))

@njit(cache=True)
def clear_bit(bitboard, square):
    return bitboard & ~(np.uint64(1) << np.uint64(square))

@njit(cache=True)
def pop_count(bitboard):
    count = 0
    bb = bitboard
    while bb:
        count += 1
        bb &= bb - np.uint64(1)
    return count

@njit(cache=True)
def get_lsb(bitboard):
    if bitboard == 0:
        return -1
    lsb = bitboard & (~bitboard + np.uint64(1))
    index = 0
    while lsb > np.uint64(1):
        lsb >>= np.uint64(1)
        index += 1
    return index

@njit(cache=True)
def square_to_rank(square):
    return square >> 3

@njit(cache=True)
def square_to_file(square):
    return square & 7

@njit(cache=True)
def rank_file_to_square(rank, file):
    return (rank << 3) + file

# =============================================================================
# 공격 테이블 (사전 계산)
# =============================================================================

@njit(cache=True)
def _init_knight_attacks():
    attacks = np.zeros(64, dtype=np.uint64)
    knight_offsets = np.array([17, 15, 10, 6, -17, -15, -10, -6], dtype=np.int64)
    
    for sq in range(64):
        rank = sq >> 3
        file = sq & 7
        bb = np.uint64(0)
        
        for offset in knight_offsets:
            to_sq = sq + offset
            if 0 <= to_sq < 64:
                to_rank = to_sq >> 3
                to_file = to_sq & 7
                if abs(to_rank - rank) <= 2 and abs(to_file - file) <= 2:
                    bb |= np.uint64(1) << np.uint64(to_sq)
        attacks[sq] = bb
    return attacks

@njit(cache=True)
def _init_king_attacks():
    attacks = np.zeros(64, dtype=np.uint64)
    king_offsets = np.array([8, -8, 1, -1, 9, 7, -9, -7], dtype=np.int64)
    
    for sq in range(64):
        rank = sq >> 3
        file = sq & 7
        bb = np.uint64(0)
        
        for offset in king_offsets:
            to_sq = sq + offset
            if 0 <= to_sq < 64:
                to_rank = to_sq >> 3
                to_file = to_sq & 7
                if abs(to_rank - rank) <= 1 and abs(to_file - file) <= 1:
                    bb |= np.uint64(1) << np.uint64(to_sq)
        attacks[sq] = bb
    return attacks

@njit(cache=True)
def _init_pawn_attacks():
    attacks = np.zeros((2, 64), dtype=np.uint64)
    
    for sq in range(64):
        file = sq & 7
        
        if sq < 56:
            if file > 0:
                attacks[WHITE, sq] |= np.uint64(1) << np.uint64(sq + 7)
            if file < 7:
                attacks[WHITE, sq] |= np.uint64(1) << np.uint64(sq + 9)
        
        if sq > 7:
            if file > 0:
                attacks[BLACK, sq] |= np.uint64(1) << np.uint64(sq - 9)
            if file < 7:
                attacks[BLACK, sq] |= np.uint64(1) << np.uint64(sq - 7)
    
    return attacks

KNIGHT_ATTACKS = _init_knight_attacks()
KING_ATTACKS = _init_king_attacks()
PAWN_ATTACKS = _init_pawn_attacks()

# =============================================================================
# 슬라이딩 피스 공격
# =============================================================================

@njit(cache=True)
def get_rook_attacks(square, occupied):
    attacks = np.uint64(0)
    rank = square >> 3
    file = square & 7
    
    for r in range(rank + 1, 8):
        s = (r << 3) + file
        attacks |= np.uint64(1) << np.uint64(s)
        if occupied & (np.uint64(1) << np.uint64(s)):
            break
    
    for r in range(rank - 1, -1, -1):
        s = (r << 3) + file
        attacks |= np.uint64(1) << np.uint64(s)
        if occupied & (np.uint64(1) << np.uint64(s)):
            break
    
    for f in range(file + 1, 8):
        s = (rank << 3) + f
        attacks |= np.uint64(1) << np.uint64(s)
        if occupied & (np.uint64(1) << np.uint64(s)):
            break
    
    for f in range(file - 1, -1, -1):
        s = (rank << 3) + f
        attacks |= np.uint64(1) << np.uint64(s)
        if occupied & (np.uint64(1) << np.uint64(s)):
            break
    
    return attacks

@njit(cache=True)
def get_bishop_attacks(square, occupied):
    attacks = np.uint64(0)
    rank = square >> 3
    file = square & 7
    
    r, f = rank + 1, file + 1
    while r < 8 and f < 8:
        s = (r << 3) + f
        attacks |= np.uint64(1) << np.uint64(s)
        if occupied & (np.uint64(1) << np.uint64(s)):
            break
        r += 1
        f += 1
    
    r, f = rank - 1, file + 1
    while r >= 0 and f < 8:
        s = (r << 3) + f
        attacks |= np.uint64(1) << np.uint64(s)
        if occupied & (np.uint64(1) << np.uint64(s)):
            break
        r -= 1
        f += 1
    
    r, f = rank + 1, file - 1
    while r < 8 and f >= 0:
        s = (r << 3) + f
        attacks |= np.uint64(1) << np.uint64(s)
        if occupied & (np.uint64(1) << np.uint64(s)):
            break
        r += 1
        f -= 1
    
    r, f = rank - 1, file - 1
    while r >= 0 and f >= 0:
        s = (r << 3) + f
        attacks |= np.uint64(1) << np.uint64(s)
        if occupied & (np.uint64(1) << np.uint64(s)):
            break
        r -= 1
        f -= 1
    
    return attacks

@njit(cache=True)
def get_queen_attacks(square, occupied):
    return get_rook_attacks(square, occupied) | get_bishop_attacks(square, occupied)

# =============================================================================
# State 배열 함수
# state[0:12] = bitboards (uint64)
# state[12] = turn, state[13] = castling, state[14] = ep, state[15] = half, state[16] = full
# =============================================================================

@njit(cache=True)
def create_initial_state():
    """초기 상태 생성"""
    state = np.zeros(17, dtype=np.uint64)
    
    # 백 피스
    state[0] = np.uint64(0xFF00)  # 폰
    state[1] = np.uint64(0x42)    # 나이트
    state[2] = np.uint64(0x24)    # 비숍
    state[3] = np.uint64(0x81)    # 룩
    state[4] = np.uint64(0x08)    # 퀸
    state[5] = np.uint64(0x10)    # 킹
    
    # 흑 피스
    state[6] = np.uint64(0xFF000000000000)
    state[7] = np.uint64(0x4200000000000000)
    state[8] = np.uint64(0x2400000000000000)
    state[9] = np.uint64(0x8100000000000000)
    state[10] = np.uint64(0x0800000000000000)
    state[11] = np.uint64(0x1000000000000000)
    
    state[12] = np.uint64(WHITE)  # turn
    state[13] = np.uint64(0b1111)  # castling
    state[14] = np.uint64(0xFFFFFFFFFFFFFFFF)  # ep_square (-1 as uint64 max)
    state[15] = np.uint64(0)  # halfmove
    state[16] = np.uint64(1)  # fullmove
    
    return state

@njit(cache=True)
def get_turn(state):
    return np.int64(state[12])

@njit(cache=True)
def get_castling(state):
    return np.int64(state[13])

@njit(cache=True)
def get_ep_square(state):
    ep = state[14]
    if ep == np.uint64(0xFFFFFFFFFFFFFFFF):
        return np.int64(-1)
    return np.int64(ep)

@njit(cache=True)
def get_all_pieces(state, color):
    offset = color * 6
    return state[offset] | state[offset+1] | state[offset+2] | state[offset+3] | state[offset+4] | state[offset+5]

@njit(cache=True)
def get_all_pieces_both(state):
    return get_all_pieces(state, WHITE) | get_all_pieces(state, BLACK)

@njit(cache=True)
def get_piece_at(state, square):
    """(piece_type, color) 또는 (0, -1)"""
    sq_bit = np.uint64(1) << np.uint64(square)
    for piece_idx in range(12):
        if state[piece_idx] & sq_bit:
            piece_type = (piece_idx % 6) + 1
            color = 0 if piece_idx < 6 else 1
            return piece_type, color
    return 0, -1

# =============================================================================
# 액션 인덱스 변환
# =============================================================================

@njit(cache=True)
def action_index_to_squares(action_idx):
    return action_idx // 64, action_idx % 64

@njit(cache=True)
def squares_to_action_index(from_sq, to_sq):
    return from_sq * 64 + to_sq

# =============================================================================
# 공격 탐지
# =============================================================================

@njit(cache=True)
def is_square_attacked(state, square, attacker_color, occupied):
    offset = attacker_color * 6
    
    pawns = state[offset + PAWN - 1]
    defender_color = 1 - attacker_color
    if pawns & PAWN_ATTACKS[defender_color, square]:
        return True
    
    knights = state[offset + KNIGHT - 1]
    if knights & KNIGHT_ATTACKS[square]:
        return True
    
    king = state[offset + KING - 1]
    if king & KING_ATTACKS[square]:
        return True
    
    bishops = state[offset + BISHOP - 1]
    queens = state[offset + QUEEN - 1]
    if (bishops | queens) & get_bishop_attacks(square, occupied):
        return True
    
    rooks = state[offset + ROOK - 1]
    if (rooks | queens) & get_rook_attacks(square, occupied):
        return True
    
    return False

@njit(cache=True)
def is_in_check(state, color):
    king = state[color * 6 + KING - 1]
    if king == 0:
        return False
    king_square = get_lsb(king)
    occupied = get_all_pieces_both(state)
    return is_square_attacked(state, king_square, 1 - color, occupied)

# =============================================================================
# 수 생성
# =============================================================================

@njit(cache=True)
def generate_pawn_moves(state, moves_array, move_count):
    color = get_turn(state)
    occupied = get_all_pieces_both(state)
    enemy = get_all_pieces(state, 1 - color)
    pawns = state[color * 6 + PAWN - 1]
    ep_square = get_ep_square(state)
    
    not_occupied = ~occupied
    
    if color == WHITE:
        single_push = (pawns << np.uint64(8)) & not_occupied
        double_push = ((pawns & np.uint64(0xFF00)) << np.uint64(16)) & not_occupied & (not_occupied << np.uint64(8))
        
        temp = single_push & ~RANK_8
        while temp:
            to_sq = get_lsb(temp)
            moves_array[move_count] = (to_sq - 8) * 64 + to_sq
            move_count += 1
            temp = clear_bit(temp, to_sq)
        
        temp = single_push & RANK_8
        while temp:
            to_sq = get_lsb(temp)
            moves_array[move_count] = (to_sq - 8) * 64 + to_sq
            move_count += 1
            temp = clear_bit(temp, to_sq)
        
        while double_push:
            to_sq = get_lsb(double_push)
            moves_array[move_count] = (to_sq - 16) * 64 + to_sq
            move_count += 1
            double_push = clear_bit(double_push, to_sq)
        
        pawns_not_a = pawns & ~FILE_A
        pawns_not_h = pawns & ~FILE_H
        
        attacks_left = (pawns_not_a << np.uint64(7)) & enemy
        while attacks_left:
            to_sq = get_lsb(attacks_left)
            moves_array[move_count] = (to_sq - 7) * 64 + to_sq
            move_count += 1
            attacks_left = clear_bit(attacks_left, to_sq)
        
        attacks_right = (pawns_not_h << np.uint64(9)) & enemy
        while attacks_right:
            to_sq = get_lsb(attacks_right)
            moves_array[move_count] = (to_sq - 9) * 64 + to_sq
            move_count += 1
            attacks_right = clear_bit(attacks_right, to_sq)
        
        if ep_square >= 0:
            ep_bit = np.uint64(1) << np.uint64(ep_square)
            if (pawns_not_a << np.uint64(7)) & ep_bit:
                moves_array[move_count] = (ep_square - 7) * 64 + ep_square
                move_count += 1
            if (pawns_not_h << np.uint64(9)) & ep_bit:
                moves_array[move_count] = (ep_square - 9) * 64 + ep_square
                move_count += 1
    
    else:  # BLACK
        single_push = (pawns >> np.uint64(8)) & not_occupied
        double_push = ((pawns & np.uint64(0xFF000000000000)) >> np.uint64(16)) & not_occupied & (not_occupied >> np.uint64(8))
        
        temp = single_push & ~RANK_1
        while temp:
            to_sq = get_lsb(temp)
            moves_array[move_count] = (to_sq + 8) * 64 + to_sq
            move_count += 1
            temp = clear_bit(temp, to_sq)
        
        temp = single_push & RANK_1
        while temp:
            to_sq = get_lsb(temp)
            moves_array[move_count] = (to_sq + 8) * 64 + to_sq
            move_count += 1
            temp = clear_bit(temp, to_sq)
        
        while double_push:
            to_sq = get_lsb(double_push)
            moves_array[move_count] = (to_sq + 16) * 64 + to_sq
            move_count += 1
            double_push = clear_bit(double_push, to_sq)
        
        pawns_not_a = pawns & ~FILE_A
        pawns_not_h = pawns & ~FILE_H
        
        attacks_right = (pawns_not_h >> np.uint64(7)) & enemy
        while attacks_right:
            to_sq = get_lsb(attacks_right)
            moves_array[move_count] = (to_sq + 7) * 64 + to_sq
            move_count += 1
            attacks_right = clear_bit(attacks_right, to_sq)
        
        attacks_left = (pawns_not_a >> np.uint64(9)) & enemy
        while attacks_left:
            to_sq = get_lsb(attacks_left)
            moves_array[move_count] = (to_sq + 9) * 64 + to_sq
            move_count += 1
            attacks_left = clear_bit(attacks_left, to_sq)
        
        if ep_square >= 0:
            ep_bit = np.uint64(1) << np.uint64(ep_square)
            if (pawns_not_h >> np.uint64(7)) & ep_bit:
                moves_array[move_count] = (ep_square + 7) * 64 + ep_square
                move_count += 1
            if (pawns_not_a >> np.uint64(9)) & ep_bit:
                moves_array[move_count] = (ep_square + 9) * 64 + ep_square
                move_count += 1
    
    return move_count

@njit(cache=True)
def generate_knight_moves(state, moves_array, move_count):
    color = get_turn(state)
    knights = state[color * 6 + KNIGHT - 1]
    own_pieces = get_all_pieces(state, color)
    
    while knights:
        from_sq = get_lsb(knights)
        attacks = KNIGHT_ATTACKS[from_sq] & ~own_pieces
        
        while attacks:
            to_sq = get_lsb(attacks)
            moves_array[move_count] = from_sq * 64 + to_sq
            move_count += 1
            attacks = clear_bit(attacks, to_sq)
        
        knights = clear_bit(knights, from_sq)
    
    return move_count

@njit(cache=True)
def generate_bishop_moves(state, moves_array, move_count):
    color = get_turn(state)
    bishops = state[color * 6 + BISHOP - 1]
    own_pieces = get_all_pieces(state, color)
    occupied = get_all_pieces_both(state)
    
    while bishops:
        from_sq = get_lsb(bishops)
        attacks = get_bishop_attacks(from_sq, occupied) & ~own_pieces
        
        while attacks:
            to_sq = get_lsb(attacks)
            moves_array[move_count] = from_sq * 64 + to_sq
            move_count += 1
            attacks = clear_bit(attacks, to_sq)
        
        bishops = clear_bit(bishops, from_sq)
    
    return move_count

@njit(cache=True)
def generate_rook_moves(state, moves_array, move_count):
    color = get_turn(state)
    rooks = state[color * 6 + ROOK - 1]
    own_pieces = get_all_pieces(state, color)
    occupied = get_all_pieces_both(state)
    
    while rooks:
        from_sq = get_lsb(rooks)
        attacks = get_rook_attacks(from_sq, occupied) & ~own_pieces
        
        while attacks:
            to_sq = get_lsb(attacks)
            moves_array[move_count] = from_sq * 64 + to_sq
            move_count += 1
            attacks = clear_bit(attacks, to_sq)
        
        rooks = clear_bit(rooks, from_sq)
    
    return move_count

@njit(cache=True)
def generate_queen_moves(state, moves_array, move_count):
    color = get_turn(state)
    queens = state[color * 6 + QUEEN - 1]
    own_pieces = get_all_pieces(state, color)
    occupied = get_all_pieces_both(state)
    
    while queens:
        from_sq = get_lsb(queens)
        attacks = get_queen_attacks(from_sq, occupied) & ~own_pieces
        
        while attacks:
            to_sq = get_lsb(attacks)
            moves_array[move_count] = from_sq * 64 + to_sq
            move_count += 1
            attacks = clear_bit(attacks, to_sq)
        
        queens = clear_bit(queens, from_sq)
    
    return move_count

@njit(cache=True)
def generate_king_moves(state, moves_array, move_count):
    color = get_turn(state)
    king = state[color * 6 + KING - 1]
    own_pieces = get_all_pieces(state, color)
    occupied = get_all_pieces_both(state)
    castling = get_castling(state)
    attacker = 1 - color
    
    if king == 0:
        return move_count
    
    king_square = get_lsb(king)
    attacks = KING_ATTACKS[king_square] & ~own_pieces
    
    while attacks:
        to_sq = get_lsb(attacks)
        moves_array[move_count] = king_square * 64 + to_sq
        move_count += 1
        attacks = clear_bit(attacks, to_sq)
    
    # 캐슬링
    if color == WHITE:
        if (castling & 0b1000 and 
            not (occupied & np.uint64(0x60)) and
            king_square == 4 and
            not is_square_attacked(state, 4, attacker, occupied) and
            not is_square_attacked(state, 5, attacker, occupied) and
            not is_square_attacked(state, 6, attacker, occupied)):
            moves_array[move_count] = 4 * 64 + 6
            move_count += 1
        
        if (castling & 0b0100 and
            not (occupied & np.uint64(0x0E)) and
            king_square == 4 and
            not is_square_attacked(state, 4, attacker, occupied) and
            not is_square_attacked(state, 3, attacker, occupied) and
            not is_square_attacked(state, 2, attacker, occupied)):
            moves_array[move_count] = 4 * 64 + 2
            move_count += 1
    else:
        if (castling & 0b0010 and
            not (occupied & np.uint64(0x6000000000000000)) and
            king_square == 60 and
            not is_square_attacked(state, 60, attacker, occupied) and
            not is_square_attacked(state, 61, attacker, occupied) and
            not is_square_attacked(state, 62, attacker, occupied)):
            moves_array[move_count] = 60 * 64 + 62
            move_count += 1
        
        if (castling & 0b0001 and
            not (occupied & np.uint64(0x0E00000000000000)) and
            king_square == 60 and
            not is_square_attacked(state, 60, attacker, occupied) and
            not is_square_attacked(state, 59, attacker, occupied) and
            not is_square_attacked(state, 58, attacker, occupied)):
            moves_array[move_count] = 60 * 64 + 58
            move_count += 1
    
    return move_count

@njit(cache=True)
def generate_pseudo_legal_moves(state, moves_array):
    move_count = 0
    move_count = generate_pawn_moves(state, moves_array, move_count)
    move_count = generate_knight_moves(state, moves_array, move_count)
    move_count = generate_bishop_moves(state, moves_array, move_count)
    move_count = generate_rook_moves(state, moves_array, move_count)
    move_count = generate_queen_moves(state, moves_array, move_count)
    move_count = generate_king_moves(state, moves_array, move_count)
    return move_count

# =============================================================================
# 수 실행
# =============================================================================

@njit(cache=True)
def make_move(state, action_idx):
    """수 실행 (state 직접 수정)"""
    from_sq, to_sq = action_index_to_squares(action_idx)
    color = get_turn(state)
    piece_type, _ = get_piece_at(state, from_sq)
    
    if piece_type == 0:
        return
    
    # 캡처 처리 (이동 전!)
    captured_type, captured_color = get_piece_at(state, to_sq)
    if captured_type > 0 and captured_color != color:
        captured_idx = captured_color * 6 + (captured_type - 1)
        state[captured_idx] = clear_bit(state[captured_idx], to_sq)
        state[15] = np.uint64(0)  # halfmove reset
    elif piece_type == PAWN:
        state[15] = np.uint64(0)
    else:
        state[15] += np.uint64(1)
    
    # 피스 이동
    piece_idx = color * 6 + (piece_type - 1)
    state[piece_idx] = clear_bit(state[piece_idx], from_sq)
    state[piece_idx] = set_bit(state[piece_idx], to_sq)
    
    # 프로모션
    if piece_type == PAWN:
        to_rank = to_sq >> 3
        if (color == WHITE and to_rank == 7) or (color == BLACK and to_rank == 0):
            state[piece_idx] = clear_bit(state[piece_idx], to_sq)
            queen_idx = color * 6 + (QUEEN - 1)
            state[queen_idx] = set_bit(state[queen_idx], to_sq)
    
    # 앙파상 캡처
    ep_square = get_ep_square(state)
    if piece_type == PAWN and to_sq == ep_square:
        if color == WHITE:
            ep_captured_sq = to_sq - 8
        else:
            ep_captured_sq = to_sq + 8
        enemy_pawn_idx = (1 - color) * 6 + (PAWN - 1)
        state[enemy_pawn_idx] = clear_bit(state[enemy_pawn_idx], ep_captured_sq)
    
    # 앙파상 칸 업데이트
    state[14] = np.uint64(0xFFFFFFFFFFFFFFFF)  # -1
    if piece_type == PAWN:
        from_rank = from_sq >> 3
        to_rank = to_sq >> 3
        if abs(to_rank - from_rank) == 2:
            if color == WHITE:
                state[14] = np.uint64(from_sq + 8)
            else:
                state[14] = np.uint64(from_sq - 8)
    
    # 캐슬링 처리
    castling = get_castling(state)
    if piece_type == KING:
        if color == WHITE:
            castling &= 0b0011
        else:
            castling &= 0b1100
        state[13] = np.uint64(castling)
        
        # 캐슬링 실행 (룩 이동)
        if color == WHITE and from_sq == 4:
            if to_sq == 6:
                state[ROOK - 1] = clear_bit(state[ROOK - 1], 7)
                state[ROOK - 1] = set_bit(state[ROOK - 1], 5)
            elif to_sq == 2:
                state[ROOK - 1] = clear_bit(state[ROOK - 1], 0)
                state[ROOK - 1] = set_bit(state[ROOK - 1], 3)
        elif color == BLACK and from_sq == 60:
            if to_sq == 62:
                state[6 + ROOK - 1] = clear_bit(state[6 + ROOK - 1], 63)
                state[6 + ROOK - 1] = set_bit(state[6 + ROOK - 1], 61)
            elif to_sq == 58:
                state[6 + ROOK - 1] = clear_bit(state[6 + ROOK - 1], 56)
                state[6 + ROOK - 1] = set_bit(state[6 + ROOK - 1], 59)
    
    # 룩 이동 시 캐슬링 권한 제거
    if piece_type == ROOK:
        if color == WHITE:
            if from_sq == 0:
                castling &= ~0b0100
            elif from_sq == 7:
                castling &= ~0b1000
        else:
            if from_sq == 56:
                castling &= ~0b0001
            elif from_sq == 63:
                castling &= ~0b0010
        state[13] = np.uint64(castling)
    
    # 턴 변경
    state[12] = np.uint64(1 - color)
    if color == BLACK:
        state[16] += np.uint64(1)

# =============================================================================
# Legal 수 생성
# =============================================================================

@njit(cache=True)
def generate_legal_moves(state):
    """모든 Legal 수 생성 (배열 복사로 최적화)"""
    moves_array = np.zeros(256, dtype=np.int64)
    pseudo_count = generate_pseudo_legal_moves(state, moves_array)
    
    legal_moves = np.zeros(256, dtype=np.int64)
    legal_count = 0
    original_color = get_turn(state)
    
    for i in range(pseudo_count):
        action_idx = moves_array[i]
        temp_state = state.copy()  # 빠른 배열 복사!
        make_move(temp_state, action_idx)
        
        if not is_in_check(temp_state, original_color):
            legal_moves[legal_count] = action_idx
            legal_count += 1
    
    return legal_moves[:legal_count]

# =============================================================================
# 게임 상태 체크
# =============================================================================

@njit(cache=True)
def is_checkmate(state):
    if not is_in_check(state, get_turn(state)):
        return False
    moves = generate_legal_moves(state)
    return len(moves) == 0

@njit(cache=True)
def is_stalemate(state):
    if is_in_check(state, get_turn(state)):
        return False
    moves = generate_legal_moves(state)
    return len(moves) == 0

@njit(cache=True)
def is_draw(state):
    if state[15] >= np.uint64(100):
        return True
    
    total = 0
    for i in range(12):
        total += pop_count(state[i])
    if total <= 2:
        return True
    
    return False

@njit(cache=True)
def is_game_over(state):
    return is_checkmate(state) or is_stalemate(state) or is_draw(state)

@njit(cache=True)
def get_result(state):
    """1.0=백승, 0.0=흑승, 0.5=무승부, -1=진행중"""
    if is_checkmate(state):
        if get_turn(state) == WHITE:
            return 0.0
        else:
            return 1.0
    if is_stalemate(state) or is_draw(state):
        return 0.5
    return -1.0

# =============================================================================
# RL 인터페이스
# =============================================================================

@njit(cache=True)
def board_to_tensor_fast(state):
    """State를 (18, 8, 8) 텐서로 변환"""
    tensor = np.zeros((18, 8, 8), dtype=np.float32)
    
    for piece_idx in range(12):
        bb = state[piece_idx]
        for sq in range(64):
            if bb & (np.uint64(1) << np.uint64(sq)):
                rank = sq >> 3
                file = sq & 7
                tensor[piece_idx, rank, file] = 1.0
    
    if get_turn(state) == WHITE:
        tensor[12, :, :] = 1.0
    
    castling = get_castling(state)
    if castling & 0b1000:
        tensor[13, :, :] = 1.0
    if castling & 0b0100:
        tensor[14, :, :] = 1.0
    if castling & 0b0010:
        tensor[15, :, :] = 1.0
    if castling & 0b0001:
        tensor[16, :, :] = 1.0
    
    ep = get_ep_square(state)
    if ep >= 0:
        tensor[17, ep >> 3, ep & 7] = 1.0
    
    return tensor

@njit(cache=True)
def legal_move_mask_fast(state):
    """합법 수 마스크 생성 (4096,)"""
    mask = np.zeros(4096, dtype=np.float32)
    moves = generate_legal_moves(state)
    for i in range(len(moves)):
        mask[moves[i]] = 1.0
    return mask

# =============================================================================
# Python 래퍼 (호환성)
# =============================================================================

class GameState:
    """이전 버전 호환 래퍼"""
    def __init__(self):
        self._state = create_initial_state()
    
    @property
    def bitboards(self):
        return self._state[:12]
    
    @property
    def turn(self):
        return int(self._state[12])
    
    @property
    def castling_rights(self):
        return int(self._state[13])
    
    @property
    def ep_square(self):
        return get_ep_square(self._state)
    
    @property
    def halfmove_clock(self):
        return int(self._state[15])
    
    @property
    def fullmove_number(self):
        return int(self._state[16])
    
    def get_all_pieces_both(self):
        return get_all_pieces_both(self._state)
    
    def copy(self):
        new_gs = GameState()
        new_gs._state = self._state.copy()
        return new_gs

# 호환성 함수들
def copy_state(gs):
    return gs.copy()

def reset_state(gs):
    gs._state = create_initial_state()

class FastChessBoard:
    """python-chess 호환 래퍼"""
    
    def __init__(self):
        self.state = create_initial_state()
    
    @property
    def turn(self):
        return get_turn(self.state) == WHITE
    
    @property
    def legal_moves(self):
        moves = generate_legal_moves(self.state)
        return [FastChessMove(m) for m in moves]
    
    def push(self, move):
        action_idx = move.from_square * 64 + move.to_square
        make_move(self.state, action_idx)
    
    def is_game_over(self):
        return is_game_over(self.state)
    
    def is_checkmate(self):
        return is_checkmate(self.state)
    
    def is_stalemate(self):
        return is_stalemate(self.state)
    
    def result(self):
        r = get_result(self.state)
        if r == 1.0:
            return "1-0"
        elif r == 0.0:
            return "0-1"
        elif r == 0.5:
            return "1/2-1/2"
        return "*"
    
    def reset(self):
        self.state = create_initial_state()
    
    def copy(self):
        new_board = FastChessBoard()
        new_board.state = self.state.copy()
        return new_board
    
    def piece_at(self, square):
        piece_type, color = get_piece_at(self.state, square)
        if piece_type == 0:
            return None
        return (piece_type, color)

class FastChessMove:
    def __init__(self, action_idx_or_tuple):
        if isinstance(action_idx_or_tuple, tuple):
            self.from_square, self.to_square = action_idx_or_tuple
        else:
            self.from_square = action_idx_or_tuple // 64
            self.to_square = action_idx_or_tuple % 64
        self.promotion = None
    
    def __repr__(self):
        files = "abcdefgh"
        ranks = "12345678"
        from_str = files[self.from_square & 7] + ranks[self.from_square >> 3]
        to_str = files[self.to_square & 7] + ranks[self.to_square >> 3]
        return f"Move.from_uci('{from_str}{to_str}')"
