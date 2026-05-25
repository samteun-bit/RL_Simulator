# RL Autonomous Driving Simulator

자율주행 자동차가 강화학습(PPO)으로 목적지까지 스스로 주행하는 3D 시뮬레이터.

## 화면 구성

```
┌─────────────────────────┬─────────────────────────┐
│   Top View + Raycasts   │     Driver Camera        │
│                         │                          │
│  추적 카메라 (뒤에서)    │  자동차 전방 카메라 시점 │
│  레이캐스트 센서 시각화  │  실제 운전자 눈높이 시점 │
└─────────────────────────┴─────────────────────────┘
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 학습

```bash
python main.py --mode train
```

옵션:
- `--timesteps 2000000` : 총 학습 스텝 수 (기본값: 2,000,000)
- `--envs 8` : 병렬 환경 수 (기본값: 8)

학습 진행상황은 TensorBoard로 확인:
```bash
tensorboard --logdir tensorboard_logs
```

### 데모 (학습된 모델 시각화)

```bash
python main.py --mode demo
```

옵션:
- `--model models/best/best_model.zip` : 모델 경로 (기본값)

학습 전에도 데모 실행 가능 → 랜덤 행동으로 동작

## 환경 설계

| 항목 | 내용 |
|------|------|
| 트랙 | 직사각형 오벌 (외벽 80×50m, 내벽 50×24m) |
| 관측 | 16개 레이캐스트 + 속도 + 목표 방향 + 목표 거리 (총 19차원) |
| 행동 | 조향 [-1, 1] + 스로틀 [0, 1] (연속 행동 공간) |
| 보상 | 트랙 진행도 × 100 + 목표 도달 +50 - 충돌 -5 - 시간패널티 |
| 알고리즘 | PPO (stable-baselines3) |
| 네트워크 | MLP 256×256 (policy + value) |

## 파일 구조

```
RLSimulator/
├── main.py                      # 진입점
├── src/
│   ├── environment/
│   │   ├── car_env.py           # Gymnasium 환경
│   │   ├── car.py               # 자동차 운동학 모델
│   │   └── track.py             # 트랙 기하 + 레이캐스트
│   ├── rendering/
│   │   └── simulator_3d.py      # Panda3D 3D 렌더러
│   └── training/
│       └── train.py             # PPO 학습 스크립트
├── models/                      # 저장된 모델
└── requirements.txt
```

## 키 조작

- `ESC` : 종료
