"""
학습 및 검증 유틸리티 함수

train_epoch: 한 에폭 학습
validate: 검증 (확장된 메트릭 포함)
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm


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
