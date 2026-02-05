"""
학습 및 검증 유틸리티 함수

train_epoch: 한 에폭 학습
validate: 검증 (확장된 메트릭 포함)
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


def compute_masked_entropy(policy_logits, mask, device):
    """
    합법 수에 대해서만 정규화된 엔트로피를 계산합니다.
    
    Args:
        policy_logits: (batch_size, 4096) 또는 (4096,) 정책 로짓
        mask: (batch_size, 4096) 또는 (4096,) 합법 수 마스크
        device: 연산 장치
    
    Returns:
        entropy: (batch_size,) 또는 scalar 엔트로피 값
    """
    # 배치 차원 처리
    if policy_logits.dim() == 1:
        policy_logits = policy_logits.unsqueeze(0)
        mask = mask.unsqueeze(0) if mask.dim() == 1 else mask
        squeeze_output = True
    else:
        squeeze_output = False
    
    # 합법 수에 대해서만 softmax 적용
    masked_logits = policy_logits.clone()
    masked_logits[~mask.bool()] = float('-inf')
    probs = torch.softmax(masked_logits, dim=-1)
    
    # 합법 수에 대해서만 엔트로피 계산 (마스크 적용)
    masked_probs = probs * mask.float()  # 합법 수 확률만 유지
    # 합법 수 확률을 정규화 (합법 수 확률의 합으로 나눔)
    legal_prob_sum = masked_probs.sum(dim=-1, keepdim=True)  # (batch_size, 1)
    # 합법 수가 0개인 경우를 방지
    legal_prob_sum = torch.clamp(legal_prob_sum, min=1e-10)
    normalized_probs = masked_probs / legal_prob_sum  # 정규화된 합법 수 확률
    
    # 엔트로피 계산: -sum(p * log(p)), log(0) 방지
    log_probs = torch.log(normalized_probs + 1e-10)
    entropy = -(normalized_probs * log_probs).sum(dim=-1)
    
    # NaN 방지: 합법 수가 0개이거나 1개인 경우 처리
    num_legal = mask.sum(dim=-1) if mask.dim() > 1 else mask.sum()
    entropy = torch.where(
        (num_legal <= 1) | torch.isnan(entropy),
        torch.tensor(0.0, device=device),
        entropy
    )
    
    if squeeze_output:
        entropy = entropy.squeeze(0)
    
    return entropy


def get_masked_log_probs(policy_logits, mask):
    """
    마스킹된 log 확률 분포를 반환합니다.
    
    Args:
        policy_logits: (batch_size, 4096) 또는 (4096,) 정책 로짓
        mask: (batch_size, 4096) 또는 (4096,) 합법 수 마스크
    
    Returns:
        log_probs: (batch_size, 4096) 또는 (4096,) log 확률 분포
    """
    # 배치 차원 처리
    if policy_logits.dim() == 1:
        policy_logits = policy_logits.unsqueeze(0)
        mask = mask.unsqueeze(0) if mask.dim() == 1 else mask
        squeeze_output = True
    else:
        squeeze_output = False
    
    # 불법 수 마스킹
    masked_logits = policy_logits.clone()
    masked_logits[~mask.bool()] = float('-inf')
    
    # Log softmax 계산
    log_probs = F.log_softmax(masked_logits, dim=-1)
    
    if squeeze_output:
        log_probs = log_probs.squeeze(0)
    
    return log_probs


def compute_legal_prob_mass(policy_logits, mask):
    """
    합법 수에 할당된 확률 질량을 계산합니다.
    
    Args:
        policy_logits: (batch_size, 4096) 또는 (4096,) 정책 로짓
        mask: (batch_size, 4096) 또는 (4096,) 합법 수 마스크
    
    Returns:
        legal_prob_mass: (batch_size,) 또는 scalar 합법 수 확률 질량
    """
    # 배치 차원 처리
    if policy_logits.dim() == 1:
        policy_logits = policy_logits.unsqueeze(0)
        mask = mask.unsqueeze(0) if mask.dim() == 1 else mask
        squeeze_output = True
    else:
        squeeze_output = False
    
    # 마스크 적용 전 softmax
    raw_probs = torch.softmax(policy_logits, dim=-1)
    legal_prob_mass = (raw_probs * mask.float()).sum(dim=-1)
    
    if squeeze_output:
        legal_prob_mass = legal_prob_mass.squeeze(0)
    
    return legal_prob_mass


def compute_kl_divergence(teacher_logits, student_logits, mask):
    """
    Teacher Model과 Student Model 간의 KL Divergence를 계산합니다.
    
    KL Divergence: D_KL(P_teacher || P_student) = sum(P_teacher * log(P_teacher / P_student))
    
    Args:
        teacher_logits: (batch_size, 4096) 또는 (4096,) Teacher Model의 정책 로짓
        student_logits: (batch_size, 4096) 또는 (4096,) Student Model의 정책 로짓
        mask: (batch_size, 4096) 또는 (4096,) 합법 수 마스크
    
    Returns:
        kl_div: (batch_size,) 또는 scalar KL Divergence 값
    """
    # 배치 차원 처리
    if teacher_logits.dim() == 1:
        teacher_logits = teacher_logits.unsqueeze(0)
        student_logits = student_logits.unsqueeze(0)
        mask = mask.unsqueeze(0) if mask.dim() == 1 else mask
        squeeze_output = True
    else:
        squeeze_output = False
    
    # 합법 수에 대해서만 마스킹
    teacher_masked_logits = teacher_logits.clone()
    teacher_masked_logits[~mask.bool()] = float('-inf')
    
    student_masked_logits = student_logits.clone()
    student_masked_logits[~mask.bool()] = float('-inf')
    
    # Teacher Model: 확률 분포로 변환 (target)
    teacher_probs = F.softmax(teacher_masked_logits, dim=-1)
    
    # Student Model: Log 확률로 변환 (input)
    student_log_probs = F.log_softmax(student_masked_logits, dim=-1)
    
    # KL Divergence 계산: F.kl_div(input=log_probs, target=probs, reduction='batchmean')
    # D_KL(target || input) = sum(target * log(target / input))
    # = sum(target * log(target)) - sum(target * log(input))
    # = sum(target * log(target)) - sum(target * input_log_probs)
    kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='none', log_target=False)
    kl_div = kl_div.sum(dim=-1)  # (batch_size,)
    
    # NaN 방지: 합법 수가 0개인 경우 처리
    num_legal = mask.sum(dim=-1) if mask.dim() > 1 else mask.sum()
    kl_div = torch.where(
        (num_legal <= 0) | torch.isnan(kl_div),
        torch.tensor(0.0, device=teacher_logits.device),
        kl_div
    )
    
    if squeeze_output:
        kl_div = kl_div.squeeze(0)
    
    return kl_div


def train_epoch(model, dataloader, optimizer, policy_loss_fn, value_loss_fn, 
                policy_weight, value_weight, device, scaler=None, use_amp=False,
                total_steps=None):
    """
    한 에폭 학습 (AMP 지원)
    
    Args:
        model: ChessCNN 모델
        dataloader: 학습 데이터 로더
        optimizer: 옵티마이저
        policy_loss_fn: 정책 손실 함수 (CrossEntropyLoss)
        value_loss_fn: 가치 손실 함수 (MSELoss)
        policy_weight: 정책 손실 가중치
        value_weight: 가치 손실 가중치
        device: 디바이스
        scaler: torch.amp.GradScaler (AMP 사용 시)
        use_amp: AMP 활성화 여부
        total_steps: tqdm 진행률 표시용 총 스텝 수
    
    Returns:
        dict: 손실 메트릭 (policy_loss, value_loss, total_loss)
    """
    model.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    
    # CUDA 사용 시 비동기 전송
    non_blocking = torch.cuda.is_available()
    
    pbar = tqdm(dataloader, desc="학습 중", total=total_steps)
    for states, policies, masks, values in pbar:
        # CUDA 사용 시 비동기 전송으로 속도 향상
        states = states.to(device, non_blocking=non_blocking)
        policies = policies.to(device, non_blocking=non_blocking)
        masks = masks.to(device, non_blocking=non_blocking)
        values = values.to(device, non_blocking=non_blocking)
        
        optimizer.zero_grad(set_to_none=True)
        
        # AMP: autocast로 forward/loss를 float16으로 수행
        with torch.amp.autocast('cuda', enabled=use_amp):
            policy_logits, value_pred = model(states, masks)
            policy_loss = policy_loss_fn(policy_logits, policies)
            value_loss = value_loss_fn(value_pred.squeeze(), values)
            loss = policy_weight * policy_loss + value_weight * value_loss
        
        # AMP: scaler로 backward (gradient scaling)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()
        
        # 통계
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_loss += loss.item()
        num_batches += 1
        
        # 진행 상황 업데이트
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'policy': f'{policy_loss.item():.4f}',
            'value': f'{value_loss.item():.4f}'
        })
    
    return {
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches,
        'total_loss': total_loss / num_batches
    }


def validate(model, dataloader, policy_loss_fn, value_loss_fn, 
             policy_weight, value_weight, device, use_amp=False, total_steps=None):
    """
    검증 (AMP 지원) - 확장된 메트릭 포함
    
    Args:
        model: ChessCNN 모델
        dataloader: 검증 데이터 로더
        policy_loss_fn: 정책 손실 함수 (CrossEntropyLoss)
        value_loss_fn: 가치 손실 함수 (MSELoss)
        policy_weight: 정책 손실 가중치
        value_weight: 가치 손실 가중치
        device: 디바이스
        use_amp: AMP 활성화 여부
        total_steps: tqdm 진행률 표시용 총 스텝 수
    
    Returns:
        dict: 검증 메트릭
            - policy_loss, value_loss, total_loss: 손실
            - accuracy, top3_acc, top5_acc, top10_acc: Top-k 정확도 (마스크 미적용)
            - masked_acc: Top-1 정확도 (마스크 적용)
            - mrr, avg_rank: MRR 및 평균 순위 (마스크 미적용)
            - entropy: 정책 엔트로피 (합법 수 기준)
            - legal_prob_mass: 합법 수 확률 질량
    """
    model.eval()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    
    # 기본 정확도 (마스크 미적용)
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    correct_top10 = 0
    total_predictions = 0
    
    # 마스크 적용 정확도
    correct_masked_top1 = 0
    
    # MRR / 평균 rank
    total_reciprocal_rank = 0.0
    total_rank = 0.0
    
    # 엔트로피 / 확률 질량
    total_entropy = 0.0
    total_legal_prob_mass = 0.0
    
    # CUDA 사용 시 비동기 전송
    non_blocking = torch.cuda.is_available()
    
    with torch.no_grad():
        for states, policies, masks, values in tqdm(dataloader, desc="검증 중", total=total_steps):
            states = states.to(device, non_blocking=non_blocking)
            policies = policies.to(device, non_blocking=non_blocking)
            masks = masks.to(device, non_blocking=non_blocking)
            values = values.to(device, non_blocking=non_blocking)
            
            batch_size = states.size(0)
            
            # AMP: autocast로 forward를 float16으로 수행
            with torch.amp.autocast('cuda', enabled=use_amp):
                policy_logits, value_pred = model(states, masks)
                policy_loss = policy_loss_fn(policy_logits, policies)
                value_loss = value_loss_fn(value_pred.squeeze(), values)
                loss = policy_weight * policy_loss + value_weight * value_loss
            
            # =========================================
            # 1. Top-k 정확도 (마스크 미적용)
            # =========================================
            _, top10_indices = policy_logits.topk(10, dim=1)
            top1_pred = top10_indices[:, 0]
            top3_pred = top10_indices[:, :3]
            top5_pred = top10_indices[:, :5]
            
            correct_top1 += (top1_pred == policies).sum().item()
            correct_top3 += (top3_pred == policies.unsqueeze(1)).any(dim=1).sum().item()
            correct_top5 += (top5_pred == policies.unsqueeze(1)).any(dim=1).sum().item()
            correct_top10 += (top10_indices == policies.unsqueeze(1)).any(dim=1).sum().item()
            
            # =========================================
            # 2. 마스크 적용 후 Top-1 정확도
            # =========================================
            masked_logits = policy_logits.clone()
            masked_logits[~masks.bool()] = float('-inf')
            masked_top1 = masked_logits.argmax(dim=1)
            correct_masked_top1 += (masked_top1 == policies).sum().item()
            
            # =========================================
            # 3. 평균 Rank / MRR (마스크 미적용)
            # =========================================
            # 정답보다 높은 logit 개수 = rank - 1
            sorted_indices = policy_logits.argsort(dim=1, descending=True)
            ranks = (sorted_indices == policies.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
            total_rank += ranks.float().sum().item()
            total_reciprocal_rank += (1.0 / ranks.float()).sum().item()
            
            # =========================================
            # 4. 정책 엔트로피 (합법 수 기준)
            # =========================================
            entropy = compute_masked_entropy(policy_logits, masks, device)
            total_entropy += entropy.sum().item()
            
            # =========================================
            # 5. 합법 수 확률 질량 (마스크 적용 전 softmax)
            # =========================================
            legal_prob_mass = compute_legal_prob_mass(policy_logits, masks)
            total_legal_prob_mass += legal_prob_mass.sum().item()
            
            total_predictions += batch_size
            
            # 통계
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1
    
    n = total_predictions if total_predictions > 0 else 1
    
    return {
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches,
        'total_loss': total_loss / num_batches,
        # 기본 정확도
        'accuracy': correct_top1 / n,
        'top3_acc': correct_top3 / n,
        'top5_acc': correct_top5 / n,
        'top10_acc': correct_top10 / n,
        # 마스크 적용 정확도
        'masked_acc': correct_masked_top1 / n,
        # MRR / 평균 rank
        'mrr': total_reciprocal_rank / n,
        'avg_rank': total_rank / n,
        # 엔트로피 / 확률 질량
        'entropy': total_entropy / n,
        'legal_prob_mass': total_legal_prob_mass / n,
    }


# =============================================================================
# A2C 강화학습 함수
# =============================================================================

def compute_a2c_loss(model, ref_model, trajectories, results, device,
                      value_loss_weight=0.5, entropy_bonus=0.01, kl_penalty=0.01):
    """
    A2C 손실을 계산합니다.
    
    Args:
        model: 학습 중인 ChessCNN 모델
        ref_model: Reference 모델 (Pre-trained, 고정)
        trajectories: 게임 trajectory 리스트 (각 게임의 step 리스트)
        results: 각 게임의 결과 (1.0=백승, -1.0=흑승, 0.0=무승부)
        device: 연산 장치
        value_loss_weight: Value Loss 가중치 (기본값: 0.5)
        entropy_bonus: 엔트로피 보너스 (기본값: 0.01)
        kl_penalty: KL Divergence 패널티 (기본값: 0.01)
    
    Returns:
        total_loss: 전체 손실
        policy_loss: 정책 손실
        value_loss: 가치 손실
        entropy: 평균 엔트로피
        kl_div: 평균 KL Divergence
    """
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl_div = 0.0
    num_steps = 0
    
    for game_idx, trajectory in enumerate(trajectories):
        game_result = results[game_idx]
        
        for step in trajectory:
            state = step['state'].unsqueeze(0).to(device)
            mask = step['mask'].unsqueeze(0).to(device)
            action = step['action']
            turn = step['turn']  # True=백, False=흑
            
            # 보상 계산: 백 기준 결과를 해당 턴 플레이어 관점으로 변환
            if turn:  # 백 차례
                reward = game_result
            else:  # 흑 차례
                reward = -game_result
            
            # 현재 모델 추론
            policy_logits, value_pred = model(state, mask)
            
            # Reference 모델 추론 (고정)
            with torch.no_grad():
                ref_policy_logits, _ = ref_model(state, mask)
            
            # Advantage: R - V(s)
            advantage = reward - value_pred.squeeze()
            
            # Policy Loss: -log π(a|s) × A(s,a)
            log_probs = get_masked_log_probs(policy_logits, mask)
            action_log_prob = log_probs[0, action]
            policy_loss = -action_log_prob * advantage.detach()
            
            # Value Loss: (V(s) - R)²
            value_loss = (value_pred.squeeze() - reward) ** 2
            
            # Entropy Bonus
            entropy = compute_masked_entropy(policy_logits, mask, device)
            
            # KL Divergence
            kl_div = compute_kl_divergence(ref_policy_logits, policy_logits, mask)
            
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy.mean()
            total_kl_div += kl_div.mean()
            num_steps += 1
    
    if num_steps == 0:
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero, zero, zero
    
    # 평균 계산
    avg_policy_loss = total_policy_loss / num_steps
    avg_value_loss = total_value_loss / num_steps
    avg_entropy = total_entropy / num_steps
    avg_kl_div = total_kl_div / num_steps
    
    # 전체 손실: Policy + c1*Value - c2*Entropy + β*KL
    total_loss = (avg_policy_loss + 
                  value_loss_weight * avg_value_loss - 
                  entropy_bonus * avg_entropy + 
                  kl_penalty * avg_kl_div)
    
    return total_loss, avg_policy_loss, avg_value_loss, avg_entropy, avg_kl_div


def train_step(model, ref_model, optimizer, trajectories, results, device,
               value_loss_weight=0.5, entropy_bonus=0.01, kl_penalty=0.01, 
               max_grad_norm=1.0, scaler=None, use_amp=False):
    """
    한 번의 A2C 학습 스텝을 수행합니다.
    
    Args:
        model: 학습 중인 ChessCNN 모델
        ref_model: Reference 모델 (Pre-trained, 고정)
        optimizer: 옵티마이저
        trajectories: 게임 trajectory 리스트
        results: 각 게임의 결과
        device: 연산 장치
        value_loss_weight: Value Loss 가중치
        entropy_bonus: 엔트로피 보너스
        kl_penalty: KL Divergence 패널티
        max_grad_norm: Gradient Clipping 임계값
        scaler: AMP용 GradScaler (선택사항)
        use_amp: AMP 사용 여부
    
    Returns:
        dict: 학습 메트릭
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    # AMP 사용 시
    with torch.amp.autocast('cuda', enabled=use_amp):
        total_loss, policy_loss, value_loss, entropy, kl_div = compute_a2c_loss(
            model, ref_model, trajectories, results, device,
            value_loss_weight, entropy_bonus, kl_penalty
        )
    
    # Backward
    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'policy_loss': policy_loss.item() if hasattr(policy_loss, 'item') else policy_loss,
        'value_loss': value_loss.item() if hasattr(value_loss, 'item') else value_loss,
        'entropy': entropy.item() if hasattr(entropy, 'item') else entropy,
        'kl_div': kl_div.item() if hasattr(kl_div, 'item') else kl_div,
    }


def select_action(policy_logits, mask, temperature=1.0):
    """
    Temperature 기반 확률적 샘플링으로 액션을 선택합니다.
    
    Args:
        policy_logits: (4096,) 또는 (batch, 4096) 정책 로짓
        mask: (4096,) 또는 (batch, 4096) 합법 수 마스크
        temperature: 탐색 온도 (1.0=정책 그대로, >1.0=더 랜덤, <1.0=더 Greedy)
    
    Returns:
        action: 선택된 액션 인덱스
        log_prob: log π(a|s)
        probs: 확률 분포
    """
    # 1. 불법 수 마스킹
    masked_logits = policy_logits.clone()
    if masked_logits.dim() == 1:
        masked_logits[~mask.bool()] = float('-inf')
    else:
        masked_logits[~mask.bool()] = float('-inf')
    
    # 2. Temperature 적용
    if temperature != 1.0:
        masked_logits = masked_logits / temperature
    
    # 3. Softmax로 확률 계산
    probs = F.softmax(masked_logits, dim=-1)
    
    # 4. 확률적 샘플링
    if probs.dim() == 1:
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[action] + 1e-10)
    else:
        action = torch.multinomial(probs, 1).squeeze(-1)
        log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)).squeeze(-1) + 1e-10)
    
    return action, log_prob, probs


def play_single_game(model, device, temperature=1.0, max_moves=None):
    """
    단일 Self-play 게임을 진행합니다.
    
    Args:
        model: ChessCNN 모델
        device: 연산 장치
        temperature: 탐색 온도
        max_moves: 최대 수 (None이면 체스 규칙에 따른 종료만 사용)
    
    Returns:
        trajectory: 게임 trajectory
        result: 게임 결과 (1.0=백승, -1.0=흑승, 0.0=무승부)
        num_moves: 총 수
    """
    import fast_chess as fc
    
    state = fc.create_initial_state()
    trajectory = []
    
    model.eval()
    
    move_num = 0
    while True:
        # 게임 종료 체크 (체스 규칙: 체크메이트, 스테일메이트, 50수 규칙, 기물 부족)
        if fc.is_game_over(state):
            break
        
        # max_moves 제한 체크 (설정된 경우에만)
        if max_moves is not None and move_num >= max_moves:
            break
        
        # 상태 텐서 변환
        state_tensor = torch.from_numpy(fc.board_to_tensor_fast(state))
        mask_tensor = torch.from_numpy(fc.legal_move_mask_fast(state))
        
        # 합법 수가 없으면 종료
        if mask_tensor.sum() == 0:
            break
        
        # 모델 추론
        with torch.no_grad():
            state_input = state_tensor.unsqueeze(0).to(device)
            mask_input = mask_tensor.unsqueeze(0).to(device)
            policy_logits, value_pred = model(state_input, mask_input)
        
        # 액션 선택
        action, log_prob, _ = select_action(
            policy_logits.squeeze(0), mask_tensor.to(device), temperature
        )
        
        # Trajectory 저장
        trajectory.append({
            'state': state_tensor,
            'mask': mask_tensor,
            'action': action,
            'log_prob': log_prob.cpu(),
            'value': value_pred.item(),
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


def play_games_batch(model, device, num_games, temperature=1.0, num_envs=1, max_moves=None):
    """
    여러 게임을 Self-play로 진행합니다.
    
    Args:
        model: ChessCNN 모델
        device: 연산 장치
        num_games: 총 게임 수
        temperature: 탐색 온도
        num_envs: 동시 진행 게임 수 (배치 크기)
        max_moves: 게임당 최대 수 (None이면 체스 규칙에 따른 종료만 사용)
    
    Returns:
        all_trajectories: 모든 게임의 trajectory 리스트
        results: 각 게임의 결과
        stats: 통계 정보
    """
    import fast_chess as fc
    
    all_trajectories = []
    results = []
    total_moves = 0
    white_wins = 0
    black_wins = 0
    draws = 0
    
    model.eval()
    
    if num_envs <= 1:
        # 순차 실행
        for _ in range(num_games):
            trajectory, result, num_moves = play_single_game(
                model, device, temperature, max_moves
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
    else:
        # 배치 GPU 추론 (Vectorized Environment)
        games_played = 0
        
        while games_played < num_games:
            batch_size = min(num_envs, num_games - games_played)
            
            # 게임 상태 초기화
            states = [fc.create_initial_state() for _ in range(batch_size)]
            trajectories = [[] for _ in range(batch_size)]
            active = [True] * batch_size
            
            move_num = 0
            while True:
                # 활성 게임 확인
                active_indices = [i for i, a in enumerate(active) if a]
                if not active_indices:
                    break
                
                # max_moves 제한 체크 (설정된 경우에만)
                if max_moves is not None and move_num >= max_moves:
                    break
                
                # 배치 텐서 준비
                batch_states = []
                batch_masks = []
                
                for i in active_indices:
                    state_tensor = torch.from_numpy(fc.board_to_tensor_fast(states[i]))
                    mask_tensor = torch.from_numpy(fc.legal_move_mask_fast(states[i]))
                    batch_states.append(state_tensor)
                    batch_masks.append(mask_tensor)
                
                batch_states = torch.stack(batch_states).to(device)
                batch_masks = torch.stack(batch_masks).to(device)
                
                # 배치 추론
                with torch.no_grad():
                    policy_logits, value_preds = model(batch_states, batch_masks)
                
                # 각 게임에 액션 적용
                for batch_idx, game_idx in enumerate(active_indices):
                    if not active[game_idx]:
                        continue
                    
                    mask = batch_masks[batch_idx]
                    
                    # 합법 수가 없으면 종료
                    if mask.sum() == 0:
                        active[game_idx] = False
                        continue
                    
                    # 액션 선택
                    action, log_prob, _ = select_action(
                        policy_logits[batch_idx], mask, temperature
                    )
                    
                    # Trajectory 저장
                    trajectories[game_idx].append({
                        'state': batch_states[batch_idx].cpu(),
                        'mask': mask.cpu(),
                        'action': action,
                        'log_prob': log_prob.cpu(),
                        'value': value_preds[batch_idx].item(),
                        'turn': fc.get_turn(states[game_idx]) == fc.WHITE,
                    })
                    
                    # 수 실행
                    fc.make_move(states[game_idx], action)
                    
                    # 게임 종료 체크 (체스 규칙: 체크메이트, 스테일메이트, 50수 규칙, 기물 부족)
                    if fc.is_game_over(states[game_idx]):
                        active[game_idx] = False
                
                move_num += 1
            
            # 결과 수집
            for i in range(batch_size):
                result_val = fc.get_result(states[i])
                if result_val == 1.0:
                    result = 1.0
                    white_wins += 1
                elif result_val == 0.0:
                    result = -1.0
                    black_wins += 1
                else:
                    result = 0.0
                    draws += 1
                
                all_trajectories.append(trajectories[i])
                results.append(result)
                total_moves += len(trajectories[i])
            
            games_played += batch_size
    
    stats = {
        'avg_moves': total_moves / max(len(results), 1),
        'white_wins': white_wins,
        'black_wins': black_wins,
        'draws': draws,
    }
    
    return all_trajectories, results, stats


def evaluate_vs_opponent(current_model, opponent_model, device, num_games=10, temperature=0.5):
    """
    모델 평가를 위해 상대 모델과 대결합니다.
    
    Args:
        current_model: 현재 학습 중인 모델
        opponent_model: 상대 모델 (이전 최고 모델)
        device: 연산 장치
        num_games: 평가용 대결 게임 수
        temperature: 탐색 온도 (낮을수록 Greedy)
    
    Returns:
        win_rate: 승률 (0.0 ~ 1.0, 무승부 = 0.5점)
    """
    import fast_chess as fc
    
    current_model.eval()
    opponent_model.eval()
    
    wins = 0
    draws = 0
    
    for game_num in range(num_games):
        # 번갈아가며 백/흑 플레이
        current_is_white = (game_num % 2 == 0)
        
        state = fc.create_initial_state()
        
        # 체스 규칙에 따른 종료까지 진행
        while not fc.is_game_over(state):
            # 현재 턴 모델 결정
            is_white_turn = (fc.get_turn(state) == fc.WHITE)
            if is_white_turn == current_is_white:
                model = current_model
            else:
                model = opponent_model
            
            # 상태 텐서
            state_tensor = torch.from_numpy(fc.board_to_tensor_fast(state))
            mask_tensor = torch.from_numpy(fc.legal_move_mask_fast(state))
            
            if mask_tensor.sum() == 0:
                break
            
            # 추론
            with torch.no_grad():
                state_input = state_tensor.unsqueeze(0).to(device)
                mask_input = mask_tensor.unsqueeze(0).to(device)
                policy_logits, _ = model(state_input, mask_input)
            
            # 액션 선택
            action, _, _ = select_action(
                policy_logits.squeeze(0), mask_tensor.to(device), temperature
            )
            
            fc.make_move(state, action)
        
        # 결과 확인
        result = fc.get_result(state)
        
        if result == 0.5:  # 무승부
            draws += 1
        elif current_is_white:
            if result == 1.0:  # 백 승 = 현재 모델 승
                wins += 1
        else:
            if result == 0.0:  # 흑 승 = 현재 모델 승
                wins += 1
    
    # 승률 계산 (무승부 = 0.5점)
    win_rate = (wins + draws * 0.5) / num_games
    
    return win_rate


def check_weight_diff(model1, model2):
    """
    두 모델의 가중치 차이를 계산합니다.
    
    Args:
        model1: 첫 번째 모델
        model2: 두 번째 모델
    
    Returns:
        total_diff: 가중치 L2 차이의 합
    """
    total_diff = 0.0
    
    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        diff = (param1 - param2).pow(2).sum().item()
        total_diff += diff
    
    return total_diff


def create_tensorboard_writer(run_type: str, base_dir: str = "models/tensorboard") -> SummaryWriter:
    """
    타임스탬프 기반 TensorBoard SummaryWriter 생성
    
    Args:
        run_type: 실행 타입 (예: "cnn", "rl")
        base_dir: 기본 로그 디렉토리
    
    Returns:
        SummaryWriter 인스턴스
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(base_dir) / f"{run_type}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


# =============================================================================
# AlphaZero 스타일 MCTS 기반 강화학습 함수
# =============================================================================

def compute_alphazero_loss(model, ref_model, trajectories, results, device,
                           value_loss_weight=1.0, entropy_bonus=0.01, kl_penalty=0.1):
    """
    AlphaZero 스타일 손실을 계산합니다.
    
    Policy Loss: CrossEntropy(π_network, π_mcts)
    - MCTS 방문 횟수 분포를 정답으로 사용
    
    Value Loss: MSE(V_network, z)
    - 게임 결과를 타겟으로 사용
    
    Args:
        model: 학습 중인 ChessCNN 모델
        ref_model: Reference 모델 (Pre-trained, 고정) - KL 페널티용
        trajectories: 게임 trajectory 리스트 (MCTS 확률 포함)
        results: 각 게임의 결과 (1.0=백승, -1.0=흑승, 0.0=무승부)
        device: 연산 장치
        value_loss_weight: Value Loss 가중치 (기본값: 1.0)
        entropy_bonus: 엔트로피 보너스 (기본값: 0.01)
        kl_penalty: KL Divergence 패널티 (기본값: 0.1)
    
    Returns:
        total_loss: 전체 손실
        policy_loss: 정책 손실
        value_loss: 가치 손실
        entropy: 평균 엔트로피
        kl_div: 평균 KL Divergence
    """
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl_div = 0.0
    num_steps = 0
    
    for game_idx, trajectory in enumerate(trajectories):
        game_result = results[game_idx]
        
        for step in trajectory:
            state = step['state'].unsqueeze(0).to(device)
            mask = step['mask'].unsqueeze(0).to(device)
            mcts_probs = step['mcts_probs'].unsqueeze(0).to(device)  # MCTS 타겟
            turn = step['turn']  # True=백, False=흑
            
            # 타겟 가치: 해당 턴 플레이어 관점
            if turn:  # 백 차례
                target_value = game_result
            else:  # 흑 차례
                target_value = -game_result
            
            # 현재 모델 추론
            policy_logits, value_pred = model(state, mask)
            
            # Reference 모델 추론 (고정)
            with torch.no_grad():
                ref_policy_logits, _ = ref_model(state, mask)
            
            # Policy Loss: CrossEntropy with MCTS target
            # -sum(π_mcts * log(π_network))
            log_probs = get_masked_log_probs(policy_logits, mask)
            policy_loss = -(mcts_probs * log_probs).sum(dim=-1).mean()
            
            # Value Loss: MSE
            target_tensor = torch.tensor([target_value], device=device, dtype=torch.float32)
            value_loss = F.mse_loss(value_pred.squeeze(), target_tensor.squeeze())
            
            # Entropy Bonus
            entropy = compute_masked_entropy(policy_logits, mask, device)
            
            # KL Divergence (Reference 모델과의 거리)
            kl_div = compute_kl_divergence(ref_policy_logits, policy_logits, mask)
            
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy.mean()
            total_kl_div += kl_div.mean()
            num_steps += 1
    
    if num_steps == 0:
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero, zero, zero
    
    # 평균 계산
    avg_policy_loss = total_policy_loss / num_steps
    avg_value_loss = total_value_loss / num_steps
    avg_entropy = total_entropy / num_steps
    avg_kl_div = total_kl_div / num_steps
    
    # 전체 손실: Policy + c1*Value - c2*Entropy + β*KL
    total_loss = (avg_policy_loss + 
                  value_loss_weight * avg_value_loss - 
                  entropy_bonus * avg_entropy + 
                  kl_penalty * avg_kl_div)
    
    return total_loss, avg_policy_loss, avg_value_loss, avg_entropy, avg_kl_div


def train_step_mcts(model, ref_model, optimizer, trajectories, results, device,
                    value_loss_weight=1.0, entropy_bonus=0.01, kl_penalty=0.1,
                    max_grad_norm=1.0, scaler=None, use_amp=False):
    """
    한 번의 AlphaZero 스타일 학습 스텝을 수행합니다.
    
    MCTS 방문 횟수 분포를 정책 타겟으로 사용합니다.
    
    Args:
        model: 학습 중인 ChessCNN 모델
        ref_model: Reference 모델 (Pre-trained, 고정)
        optimizer: 옵티마이저
        trajectories: 게임 trajectory 리스트 (MCTS 확률 포함)
        results: 각 게임의 결과
        device: 연산 장치
        value_loss_weight: Value Loss 가중치
        entropy_bonus: 엔트로피 보너스
        kl_penalty: KL Divergence 패널티
        max_grad_norm: Gradient Clipping 임계값
        scaler: AMP용 GradScaler (선택사항)
        use_amp: AMP 사용 여부
    
    Returns:
        dict: 학습 메트릭
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    # AMP 사용 시
    with torch.amp.autocast('cuda', enabled=use_amp):
        total_loss, policy_loss, value_loss, entropy, kl_div = compute_alphazero_loss(
            model, ref_model, trajectories, results, device,
            value_loss_weight, entropy_bonus, kl_penalty
        )
    
    # Backward
    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'policy_loss': policy_loss.item() if hasattr(policy_loss, 'item') else policy_loss,
        'value_loss': value_loss.item() if hasattr(value_loss, 'item') else value_loss,
        'entropy': entropy.item() if hasattr(entropy, 'item') else entropy,
        'kl_div': kl_div.item() if hasattr(kl_div, 'item') else kl_div,
    }
