"""
체스 CNN 모델 정의

지도학습 후 강화학습의 정책 신경망으로 사용됩니다.
"""
import torch
import torch.nn as nn


class ChessCNN(nn.Module):
    """
    체스 CNN 모델
    
    Policy Head와 Value Head를 가진 구조로,
    지도학습 후 강화학습의 정책 신경망으로 사용됩니다.
    
    Args:
        num_channels: 백본 CNN의 채널 수 (기본값: 256)
    
    입력:
        x: (batch, 18, 8, 8) 체스 보드 상태
        mask: (batch, 4096) 합법 수 마스크 (선택사항)
    
    출력:
        policy_logits: (batch, 4096) 정책 로짓
        value: (batch, 1) 가치 예측 (-1 ~ 1)
    """
    
    def __init__(self, num_channels=256):
        super().__init__()
        
        # 입력: (batch, 18, 8, 8)
        # 공통 CNN 백본
        self.conv_layers = nn.Sequential(
            # 첫 번째 블록
            nn.Conv2d(18, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 두 번째 블록
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 세 번째 블록
            nn.Conv2d(128, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        
        # Policy Head (4096개 액션)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4096),
        )
        
        # Value Head (1개 출력)
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, 18, 8, 8) 입력 텐서
            mask: (batch, 4096) 합법 수 마스크 (선택사항)
        
        Returns:
            policy_logits: (batch, 4096) 정책 로짓
            value: (batch, 1) 가치 예측
        """
        # 공통 백본
        features = self.conv_layers(x)
        
        # Policy Head
        policy_logits = self.policy_head(features)
        
        # Mask 적용 (불법 수 제거)
        if mask is not None:
            illegal_mask = mask < 0.5
            policy_logits = policy_logits.masked_fill(illegal_mask, float('-inf'))
        
        # Value Head
        value = self.value_head(features)
        
        return policy_logits, value


def load_model(checkpoint_path, device='cuda', num_channels=256):
    """
    체크포인트에서 모델 로드
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        device: 디바이스 ('cuda' 또는 'cpu')
        num_channels: 모델 채널 수
    
    Returns:
        model: 로드된 모델
        checkpoint: 체크포인트 딕셔너리
    """
    model = ChessCNN(num_channels=num_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint
