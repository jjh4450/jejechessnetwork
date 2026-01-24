"""
체스 데이터 전처리 사용 예제

.pgn.zst 파일에서 샘플을 추출하고 PyTorch DataLoader로 사용하는 예제입니다.
"""

import torch
from torch.utils.data import DataLoader
from preprocessing import extract_samples_from_pgn_zst
from dataset import ChessDataset


def main():
    # 데이터 파일 경로
    zst_path = "data/lichess_db_standard_rated_2016-12.pgn.zst"
    
    print("=" * 60)
    print("체스 데이터 전처리 예제")
    print("=" * 60)
    
    # 샘플 추출 (테스트용으로 작은 수만)
    print(f"\n1. .pgn.zst 파일에서 샘플 추출 중...")
    print(f"   파일: {zst_path}")
    print(f"   제한: 최대 10게임 또는 1000샘플")
    
    samples = extract_samples_from_pgn_zst(
        zst_path=zst_path,
        max_games=10,
        max_samples=1000
    )
    
    print(f"   추출된 샘플 수: {len(samples)}")
    
    if len(samples) == 0:
        print("   ⚠️  샘플이 없습니다. 파일 경로를 확인하세요.")
        return
    
    # 첫 번째 샘플 확인
    state, policy, mask, value = samples[0]
    print(f"\n2. 샘플 구조 확인")
    print(f"   State shape: {state.shape}")
    print(f"   Policy (액션 인덱스): {policy}")
    print(f"   Mask shape: {mask.shape}, 합법 수: {int(mask.sum())}")
    print(f"   Value: {value}")
    
    # Dataset 생성
    print(f"\n3. PyTorch Dataset 생성")
    dataset = ChessDataset(samples)
    print(f"   Dataset 크기: {len(dataset)}")
    
    # DataLoader 생성
    print(f"\n4. DataLoader 생성")
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # Windows에서는 0 권장
    )
    
    # 배치 확인
    print(f"\n5. 배치 로드 테스트")
    for batch_idx, (states, policies, masks, values) in enumerate(dataloader):
        print(f"   배치 {batch_idx + 1}:")
        print(f"     States: {states.shape}")
        print(f"     Policies: {policies.shape}")
        print(f"     Masks: {masks.shape}")
        print(f"     Values: {values.shape}")
        
        if batch_idx >= 2:  # 처음 3개 배치만 출력
            break
    
    print(f"\n" + "=" * 60)
    print("✅ 예제 실행 완료!")
    print("=" * 60)
    print(f"\n사용법:")
    print(f"  - 전체 데이터셋: max_games=None, max_samples=None")
    print(f"  - 배치 크기 조정: DataLoader(batch_size=64)")
    print(f"  - 학습 루프에서 사용:")
    print(f"    for states, policies, masks, values in dataloader:")
    print(f"        # 모델 학습 코드")


if __name__ == "__main__":
    main()
