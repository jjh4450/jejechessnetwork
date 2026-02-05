"""
Monte Carlo Tree Search (MCTS) 알고리즘 구현

AlphaZero 스타일의 MCTS로, 신경망 기반 평가와 UCB1 선택을 사용합니다.
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import fast_chess as fc


class MCTSNode:
    """MCTS 트리의 노드"""
    
    def __init__(self, prior: float = 0.0, parent: Optional['MCTSNode'] = None):
        """
        Args:
            prior: 부모 노드에서 이 노드로 오는 액션의 사전 확률 P(s,a)
            parent: 부모 노드
        """
        self.prior = prior  # P(s,a): Policy Network의 사전 확률
        self.parent = parent
        self.children: Dict[int, 'MCTSNode'] = {}  # action -> child node
        
        self.visit_count = 0  # N(s,a): 방문 횟수
        self.value_sum = 0.0  # W(s,a): 누적 가치
        self.is_expanded = False
    
    @property
    def q_value(self) -> float:
        """Q(s,a) = W(s,a) / N(s,a): 평균 가치"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visit_count: int, c_puct: float = 1.5) -> float:
        """
        UCB1 스코어 계산
        
        UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))
        
        Args:
            parent_visit_count: 부모 노드의 방문 횟수
            c_puct: 탐색-착취 균형 상수
        
        Returns:
            UCB 스코어
        """
        exploration = c_puct * self.prior * np.sqrt(parent_visit_count) / (1 + self.visit_count)
        return self.q_value + exploration
    
    def select_child(self, c_puct: float = 1.5) -> Tuple[int, 'MCTSNode']:
        """
        UCB1 기준으로 최적 자식 노드 선택
        
        Returns:
            (action, child_node) 튜플
        """
        best_score = float('-inf')
        best_action = -1
        best_child = None
        
        for action, child in self.children.items():
            score = child.ucb_score(self.visit_count, c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def expand(self, action_probs: np.ndarray, legal_mask: np.ndarray):
        """
        노드 확장: 모든 합법 수에 대해 자식 노드 생성
        
        Args:
            action_probs: (4096,) Policy Network의 확률 분포
            legal_mask: (4096,) 합법 수 마스크
        """
        self.is_expanded = True
        
        # 합법 수에 대해서만 자식 노드 생성
        legal_actions = np.where(legal_mask > 0.5)[0]
        
        for action in legal_actions:
            if action not in self.children:
                self.children[action] = MCTSNode(prior=action_probs[action], parent=self)
    
    def backup(self, value: float):
        """
        역전파: 루트까지 거슬러 올라가며 값 업데이트
        
        Args:
            value: 리프 노드의 평가 값 (현재 플레이어 관점)
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # 상대방 관점으로 전환
            node = node.parent
    
    def get_visit_counts(self) -> np.ndarray:
        """
        모든 자식 노드의 방문 횟수를 반환
        
        Returns:
            (4096,) 방문 횟수 배열
        """
        visit_counts = np.zeros(4096, dtype=np.float32)
        for action, child in self.children.items():
            visit_counts[action] = child.visit_count
        return visit_counts


class MCTS:
    """Monte Carlo Tree Search 엔진"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_simulations: int = 400,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature: float = 1.0,
    ):
        """
        Args:
            model: 정책/가치 신경망
            device: 연산 장치
            num_simulations: 시뮬레이션 횟수 (최소 400 권장)
            c_puct: UCB 탐색 상수
            dirichlet_alpha: 루트 노드 노이즈의 Dirichlet 파라미터
            dirichlet_epsilon: 노이즈 혼합 비율
            temperature: 행동 선택 온도
        """
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
    
    @torch.no_grad()
    def _evaluate(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        신경망으로 상태 평가
        
        Args:
            state: fast_chess 상태 배열
        
        Returns:
            (policy_probs, value) 튜플
        """
        # 상태를 텐서로 변환
        state_tensor = torch.from_numpy(fc.board_to_tensor_fast(state)).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(fc.legal_move_mask_fast(state)).unsqueeze(0).to(self.device)
        
        # 신경망 추론
        self.model.eval()
        policy_logits, value = self.model(state_tensor, mask_tensor)
        
        # 정책을 확률로 변환
        policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
        value = value.item()
        
        return policy_probs, value
    
    def _add_dirichlet_noise(self, node: MCTSNode, legal_mask: np.ndarray):
        """
        루트 노드에 Dirichlet 노이즈 추가 (탐색 다양성 확보)
        
        Args:
            node: 루트 노드
            legal_mask: 합법 수 마스크
        """
        legal_actions = list(node.children.keys())
        if not legal_actions:
            return
        
        # Dirichlet 노이즈 생성
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
        
        # 노이즈를 Prior에 혼합
        for i, action in enumerate(legal_actions):
            child = node.children[action]
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * noise[i]
    
    def search(self, state: np.ndarray, add_noise: bool = True) -> Tuple[np.ndarray, float]:
        """
        MCTS 탐색 수행
        
        Args:
            state: 현재 게임 상태 (fast_chess 배열)
            add_noise: 루트에 Dirichlet 노이즈 추가 여부
        
        Returns:
            (action_probs, root_value) 튜플
            - action_probs: (4096,) 방문 횟수 기반 확률 분포
            - root_value: 루트 노드의 평균 가치
        """
        # 합법 수 마스크
        legal_mask = fc.legal_move_mask_fast(state)
        
        if legal_mask.sum() == 0:
            return np.zeros(4096, dtype=np.float32), 0.0
        
        # 루트 노드 초기화
        root = MCTSNode()
        policy_probs, value = self._evaluate(state)
        root.expand(policy_probs, legal_mask)
        
        # 루트에 Dirichlet 노이즈 추가
        if add_noise:
            self._add_dirichlet_noise(root, legal_mask)
        
        # MCTS 시뮬레이션 수행
        for _ in range(self.num_simulations):
            node = root
            sim_state = state.copy()
            search_path = [node]
            
            # Selection: 리프 노드까지 내려가기
            while node.is_expanded and node.children:
                action, node = node.select_child(self.c_puct)
                fc.make_move(sim_state, action)
                search_path.append(node)
            
            # 게임 종료 체크
            if fc.is_game_over(sim_state):
                result = fc.get_result(sim_state)
                # 현재 턴 플레이어 관점에서 가치 계산
                turn = fc.get_turn(sim_state)
                if result == 1.0:  # 백 승
                    value = 1.0 if turn == fc.BLACK else -1.0
                elif result == 0.0:  # 흑 승
                    value = 1.0 if turn == fc.WHITE else -1.0
                else:  # 무승부
                    value = 0.0
            else:
                # Expansion & Evaluation
                sim_legal_mask = fc.legal_move_mask_fast(sim_state)
                if sim_legal_mask.sum() > 0:
                    policy_probs, value = self._evaluate(sim_state)
                    node.expand(policy_probs, sim_legal_mask)
                else:
                    value = 0.0
            
            # Backpropagation
            node.backup(value)
        
        # 방문 횟수 기반 정책 생성
        visit_counts = root.get_visit_counts()
        
        # Temperature 적용
        if self.temperature == 0:
            # Greedy 선택
            action_probs = np.zeros(4096, dtype=np.float32)
            best_action = np.argmax(visit_counts)
            action_probs[best_action] = 1.0
        else:
            # Temperature 기반 확률
            visit_counts_temp = np.power(visit_counts, 1.0 / self.temperature)
            total = visit_counts_temp.sum()
            if total > 0:
                action_probs = visit_counts_temp / total
            else:
                action_probs = np.zeros(4096, dtype=np.float32)
        
        return action_probs, root.q_value
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> Tuple[int, np.ndarray, float]:
        """
        MCTS 탐색 후 행동 선택
        
        Args:
            state: 현재 게임 상태
            add_noise: 탐색 시 노이즈 추가 여부
        
        Returns:
            (action, action_probs, value) 튜플
            - action: 선택된 행동 인덱스
            - action_probs: MCTS 방문 횟수 기반 확률 분포 (학습 타겟)
            - value: 루트 노드 가치 추정
        """
        action_probs, value = self.search(state, add_noise)
        
        # 확률적 샘플링
        if self.temperature > 0 and action_probs.sum() > 0:
            action = np.random.choice(4096, p=action_probs)
        else:
            action = np.argmax(action_probs)
        
        return action, action_probs, value


def play_game_with_mcts(
    model: torch.nn.Module,
    device: torch.device,
    num_simulations: int = 400,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    temperature_threshold: int = 30,
    max_moves: Optional[int] = None,
) -> Tuple[List[Dict], float, int]:
    """
    MCTS를 사용하여 Self-play 게임 진행
    
    Args:
        model: 정책/가치 신경망
        device: 연산 장치
        num_simulations: MCTS 시뮬레이션 횟수
        c_puct: UCB 탐색 상수
        temperature: 초기 탐색 온도
        temperature_threshold: 이 수 이후에는 temperature=0 (greedy)
        max_moves: 최대 수 (None이면 제한 없음)
    
    Returns:
        (trajectory, result, num_moves) 튜플
        - trajectory: 각 스텝의 정보 (state, mask, mcts_probs, action, turn)
        - result: 게임 결과 (1.0=백승, -1.0=흑승, 0.0=무승부)
        - num_moves: 총 수
    """
    state = fc.create_initial_state()
    trajectory = []
    
    mcts = MCTS(
        model=model,
        device=device,
        num_simulations=num_simulations,
        c_puct=c_puct,
        temperature=temperature,
    )
    
    move_num = 0
    while True:
        # 게임 종료 체크
        if fc.is_game_over(state):
            break
        
        if max_moves is not None and move_num >= max_moves:
            break
        
        # Temperature 조정 (후반부에는 greedy)
        if move_num >= temperature_threshold:
            mcts.temperature = 0.0
        
        # 상태 텐서 저장 (학습용)
        state_tensor = torch.from_numpy(fc.board_to_tensor_fast(state))
        mask_tensor = torch.from_numpy(fc.legal_move_mask_fast(state))
        
        if mask_tensor.sum() == 0:
            break
        
        # MCTS 탐색 및 행동 선택
        action, mcts_probs, value = mcts.select_action(state, add_noise=True)
        
        # Trajectory 저장 (MCTS 확률을 학습 타겟으로 사용)
        trajectory.append({
            'state': state_tensor,
            'mask': mask_tensor,
            'mcts_probs': torch.from_numpy(mcts_probs),  # 학습 타겟
            'action': action,
            'value': value,
            'turn': fc.get_turn(state) == fc.WHITE,  # True=백
        })
        
        # 수 실행
        fc.make_move(state, action)
        move_num += 1
    
    # 결과 계산
    result_val = fc.get_result(state)
    if result_val == 1.0:  # 백 승
        result = 1.0
    elif result_val == 0.0:  # 흑 승
        result = -1.0
    else:  # 무승부
        result = 0.0
    
    return trajectory, result, len(trajectory)


def play_games_batch_with_mcts(
    model: torch.nn.Module,
    device: torch.device,
    num_games: int,
    num_simulations: int = 400,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    temperature_threshold: int = 30,
    max_moves: Optional[int] = None,
) -> Tuple[List[List[Dict]], List[float], Dict]:
    """
    MCTS를 사용하여 여러 게임을 Self-play로 진행
    
    Args:
        model: 정책/가치 신경망
        device: 연산 장치
        num_games: 총 게임 수
        num_simulations: MCTS 시뮬레이션 횟수
        c_puct: UCB 탐색 상수
        temperature: 초기 탐색 온도
        temperature_threshold: greedy 전환 수
        max_moves: 게임당 최대 수
    
    Returns:
        (all_trajectories, results, stats) 튜플
    """
    all_trajectories = []
    results = []
    total_moves = 0
    white_wins = 0
    black_wins = 0
    draws = 0
    
    for _ in range(num_games):
        trajectory, result, num_moves = play_game_with_mcts(
            model=model,
            device=device,
            num_simulations=num_simulations,
            c_puct=c_puct,
            temperature=temperature,
            temperature_threshold=temperature_threshold,
            max_moves=max_moves,
        )
        
        all_trajectories.append(trajectory)
        results.append(result)
        total_moves += num_moves
        
        if result > 0:
            white_wins += 1
        elif result < 0:
            black_wins += 1
        else:
            draws += 1
    
    stats = {
        'avg_moves': total_moves / max(len(results), 1),
        'white_wins': white_wins,
        'black_wins': black_wins,
        'draws': draws,
    }
    
    return all_trajectories, results, stats


# =============================================================================
# Opponent Pool
# =============================================================================

class OpponentPool:
    """
    상대 모델 풀 관리
    
    SL 모델을 고정 포함하고, 이전 Best 모델들을 유지합니다.
    Self-play 시 랜덤하게 상대를 선택합니다.
    """
    
    def __init__(
        self,
        sl_model: torch.nn.Module,
        device: torch.device,
        max_opponents: int = 5,
        sl_model_weight: float = 0.3,
    ):
        """
        Args:
            sl_model: 지도학습 모델 (고정, 항상 포함)
            device: 연산 장치
            max_opponents: 최대 상대 모델 수 (SL 모델 제외)
            sl_model_weight: SL 모델 선택 확률 가중치
        """
        self.device = device
        self.max_opponents = max_opponents
        self.sl_model_weight = sl_model_weight
        
        # SL 모델 고정 저장
        self.sl_model = sl_model
        self.sl_model.eval()
        for param in self.sl_model.parameters():
            param.requires_grad = False
        
        # 이전 Best 모델들 (state_dict 형태로 저장)
        self.opponent_states: List[Dict] = []
        self.opponent_win_rates: List[float] = []
    
    def add_opponent(self, model: torch.nn.Module, win_rate: float):
        """
        새로운 상대 모델을 풀에 추가
        
        Args:
            model: 추가할 모델
            win_rate: 해당 모델의 승률
        """
        state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if len(self.opponent_states) >= self.max_opponents:
            # 가장 낮은 승률의 모델 제거
            min_idx = np.argmin(self.opponent_win_rates)
            if win_rate > self.opponent_win_rates[min_idx]:
                self.opponent_states.pop(min_idx)
                self.opponent_win_rates.pop(min_idx)
            else:
                return  # 추가하지 않음
        
        self.opponent_states.append(state_dict)
        self.opponent_win_rates.append(win_rate)
    
    def get_random_opponent(self, model_class: type) -> torch.nn.Module:
        """
        랜덤하게 상대 모델 선택
        
        Args:
            model_class: 모델 클래스 (예: ChessCNN)
        
        Returns:
            선택된 상대 모델
        """
        # SL 모델 선택 확률 계산
        num_opponents = len(self.opponent_states)
        
        if num_opponents == 0:
            return self.sl_model
        
        # 가중치 계산: SL 모델은 sl_model_weight, 나머지는 균등 분배
        total_weight = self.sl_model_weight + (1 - self.sl_model_weight)
        sl_prob = self.sl_model_weight / total_weight
        other_prob = (1 - self.sl_model_weight) / (total_weight * num_opponents)
        
        probs = [sl_prob] + [other_prob] * num_opponents
        probs = np.array(probs) / np.sum(probs)  # 정규화
        
        choice = np.random.choice(num_opponents + 1, p=probs)
        
        if choice == 0:
            return self.sl_model
        else:
            # 선택된 상대 모델 로드
            opponent = model_class(num_channels=256).to(self.device)
            state_dict = {k: v.to(self.device) for k, v in self.opponent_states[choice - 1].items()}
            opponent.load_state_dict(state_dict)
            opponent.eval()
            for param in opponent.parameters():
                param.requires_grad = False
            return opponent
    
    def __len__(self):
        return len(self.opponent_states) + 1  # +1 for SL model


def play_game_vs_opponent(
    current_model: torch.nn.Module,
    opponent_model: torch.nn.Module,
    device: torch.device,
    num_simulations: int = 400,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    temperature_threshold: int = 30,
    current_is_white: bool = True,
    max_moves: Optional[int] = None,
) -> Tuple[List[Dict], float, int]:
    """
    상대 모델과 대결하며 Self-play 진행 (현재 모델의 trajectory만 수집)
    
    Args:
        current_model: 학습 중인 모델
        opponent_model: 상대 모델
        device: 연산 장치
        num_simulations: MCTS 시뮬레이션 횟수
        c_puct: UCB 탐색 상수
        temperature: 초기 탐색 온도
        temperature_threshold: greedy 전환 수
        current_is_white: 현재 모델이 백인지 여부
        max_moves: 최대 수
    
    Returns:
        (trajectory, result, num_moves) 튜플
        - trajectory: 현재 모델의 스텝만 포함
        - result: 게임 결과 (현재 모델 관점)
    """
    state = fc.create_initial_state()
    trajectory = []
    
    current_mcts = MCTS(
        model=current_model,
        device=device,
        num_simulations=num_simulations,
        c_puct=c_puct,
        temperature=temperature,
    )
    
    opponent_mcts = MCTS(
        model=opponent_model,
        device=device,
        num_simulations=num_simulations // 2,  # 상대는 시뮬레이션 수 절반
        c_puct=c_puct,
        temperature=0.5,  # 상대는 더 greedy하게
    )
    
    move_num = 0
    while True:
        if fc.is_game_over(state):
            break
        
        if max_moves is not None and move_num >= max_moves:
            break
        
        is_white_turn = fc.get_turn(state) == fc.WHITE
        is_current_turn = (is_white_turn == current_is_white)
        
        # Temperature 조정
        if move_num >= temperature_threshold:
            current_mcts.temperature = 0.0
            opponent_mcts.temperature = 0.0
        
        state_tensor = torch.from_numpy(fc.board_to_tensor_fast(state))
        mask_tensor = torch.from_numpy(fc.legal_move_mask_fast(state))
        
        if mask_tensor.sum() == 0:
            break
        
        if is_current_turn:
            # 현재 모델 턴: MCTS 탐색 및 trajectory 저장
            action, mcts_probs, value = current_mcts.select_action(state, add_noise=True)
            
            trajectory.append({
                'state': state_tensor,
                'mask': mask_tensor,
                'mcts_probs': torch.from_numpy(mcts_probs),
                'action': action,
                'value': value,
                'turn': is_white_turn,
            })
        else:
            # 상대 모델 턴
            action, _, _ = opponent_mcts.select_action(state, add_noise=False)
        
        fc.make_move(state, action)
        move_num += 1
    
    # 결과 계산 (현재 모델 관점)
    result_val = fc.get_result(state)
    if result_val == 0.5:  # 무승부
        result = 0.0
    elif current_is_white:
        result = 1.0 if result_val == 1.0 else -1.0
    else:
        result = 1.0 if result_val == 0.0 else -1.0
    
    return trajectory, result, move_num


def play_games_vs_opponent_pool(
    current_model: torch.nn.Module,
    opponent_pool: OpponentPool,
    model_class: type,
    device: torch.device,
    num_games: int,
    num_simulations: int = 400,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    temperature_threshold: int = 30,
    max_moves: Optional[int] = None,
) -> Tuple[List[List[Dict]], List[float], Dict]:
    """
    Opponent Pool을 사용하여 여러 게임을 진행
    
    Args:
        current_model: 학습 중인 모델
        opponent_pool: 상대 모델 풀
        model_class: 모델 클래스
        device: 연산 장치
        num_games: 총 게임 수
        num_simulations: MCTS 시뮬레이션 횟수
        c_puct: UCB 탐색 상수
        temperature: 초기 탐색 온도
        temperature_threshold: greedy 전환 수
        max_moves: 게임당 최대 수
    
    Returns:
        (all_trajectories, results, stats) 튜플
    """
    all_trajectories = []
    results = []
    total_moves = 0
    wins = 0
    losses = 0
    draws = 0
    
    for game_num in range(num_games):
        # 랜덤 상대 선택
        opponent = opponent_pool.get_random_opponent(model_class)
        
        # 번갈아가며 백/흑 플레이
        current_is_white = (game_num % 2 == 0)
        
        trajectory, result, num_moves = play_game_vs_opponent(
            current_model=current_model,
            opponent_model=opponent,
            device=device,
            num_simulations=num_simulations,
            c_puct=c_puct,
            temperature=temperature,
            temperature_threshold=temperature_threshold,
            current_is_white=current_is_white,
            max_moves=max_moves,
        )
        
        all_trajectories.append(trajectory)
        results.append(result)
        total_moves += num_moves
        
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            draws += 1
    
    stats = {
        'avg_moves': total_moves / max(len(results), 1),
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': (wins + draws * 0.5) / max(len(results), 1),
    }
    
    return all_trajectories, results, stats
