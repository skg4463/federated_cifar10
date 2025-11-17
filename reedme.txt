# 연합학습 시뮬레이션 (mainv2.7.py)

## 1. 프로젝트 개요
- CIFAR-10 기반 연합학습 실험 코드.
- 위원회(Committee) 평가를 통한 기여도 가중 집계(FedContrib)와 악의적 클라이언트 시뮬레이션 지원.
- Flower(FLWR) 시뮬레이터를 활용한 CPU/GPU 병렬 실행.

## 2. 주요 기능
1. **기여도 기반 통합(FedContrib)**  
   - 위원회가 각 클라이언트 모델을 독립 평가(EWMA 적용) 후 가중치를 동적으로 조정.  
   - `--csf` 옵션으로 기여도 차이를 승수 형태로 증폭 가능.

2. **악의적 노드 시뮬레이션**  
   - `--ns`: 학습 없이 랜덤 파라미터 제출(랜덤 노이즈 공격).  
   - `--df N K`: 데이터 라벨 변조(포이즈닝) 공격, 공격자 수 N, 각자 K개 클래스 변조.  
   - 오염 클래스를 추적해 정확도 변화를 별도로 기록·시각화.

3. **위원회 스킵 (FedAvg 모드)**  
   - `--s` 사용 시 위원회 평가 생략, 모든 로컬 모델을 동일 가중치로 평균.

4. **결과 분석 및 시각화**  
   - 라운드별 글로벌 정확도, 기여도 가중치, 오염 클래스 정확도, 시간 통계를 PNG/텍스트로 저장.  
   - 결과 폴더명에 실행 옵션 자동 기록.

5. **재현성 보장**  
   - 고정 랜덤 시드로 실험 반복 시 동일 분할 및 결과 재현.

## 3. 시스템 요구 사항
- Python 3.9 이상 권장
- 주요 라이브러리: `torch`, `torchvision`, `flwr`, `numpy`, `pandas`, `matplotlib`
- GPU 사용 시 CUDA 환경 필요 (기본 설정은 GPU 자동 탐지)

## 4. 설치 방법
```
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # 파일이 없으면 수동으로 의존성 설치
```

## 5. 데이터 준비
- 실행 시 CIFAR-10 데이터가 `./data` 폴더로 자동 다운로드.

## 6. 실행 방법
```
python mainv2.7.py [옵션...]
```

### 자주 사용하는 실행 예시
```
# 기본 설정 (위원회 평가 + 기여도 기반 통합)
python mainv2.7.py

# 위원회 평가 스킵 (FedAvg), 기여도 배수 1.5
python mainv2.7.py --s --csf 1.5

# 노이즈 공격자 2명, 데이터 포이즈닝 3명(각 2개 클래스 변조)
python mainv2.7.py --ns 2 --df 3 2
```

## 7. 명령줄 옵션 요약
| 옵션 | 설명 |
|------|------|
| `--s` | 위원회 평가 스킵 → 모든 클라이언트 동일 가중치(표준 FedAvg). |
| `--ns [N]` | 랜덤 노이즈 공격자 수. 값 미지정 시 1명으로 설정. |
| `--df N K` | 데이터 포이즈닝 공격자 N명, 각자 K개 클래스 라벨 변조. |
| `--csf F` | 기여도 가중치 승수(기본 1.5). |
| `--iid` *(옵션 추가 시)* | 훈련 데이터셋을 IID로 분할하여 각 클라이언트에 배정. |

※ `--df` 사용 시 K 미지정 → 기본 1개 클래스 변조.

## 8. 출력 및 로그
- 실행 시 `MMDD_HHMM_v2.7_R{라운드수}_[옵션]` 형태의 폴더 생성.
- 저장 파일:
  - `federated_learning_results_v2.txt`: 데이터 분할 현황, 정확도/가중치/시간 테이블, 실행 파라미터.
  - `federated_learning_accuracy_v2.png`: 글로벌 정확도 추이.
  - `federated_learning_weights_v2.png`: 라운드별 클라이언트 가중치 변화.
  - `federated_learning_poisoned_accuracy_v2.png`: (포이즈닝 시) 오염 클래스 정확도.
  - `federated_learning_full_graph_v2.png`: 전체 정확도 vs. 오염 클래스 정확도 비교.

## 9. 코드 구조 개요
- `prepare_datasets()`: CIFAR-10 분할 (기본 Non-IID Dirichlet, 위원회 테스트는 균등 분할).
- `CifarClient`, `NoiseClient`, `DataPoisoningClient`: 정상/악성 클라이언트 구현.
- `FedContrib`: 위원회 평가 → 기여도 계산 → 가중 평균 집계 전략.
- `save_results()`: 로그 파일 및 시각화 이미지 생성.
- `main()`: CLI 옵션 파싱, 클라이언트/전략 구성, Flower 시뮬레이션 실행.

## 10. 활용 팁
- GPU 자원이 부족하면 `client_resources` 설정을 조정해 동시에 실행될 클라이언트 수를 제한.
- 악의적 노드 수 합계가 `NUM_CLIENTS`(기본 10)를 초과하지 않도록 주의.
- `--iid` 옵션 사용 시, `prepare_datasets`에서 IID 분할 로직이 구현되어 있어야 함.
- 로그/그래프 기반으로 악성 노드 영향도 및 기여도 변화를 분석할 수 있음.