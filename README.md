# Noise Flow 주차 경로 생성 모델

초기 경로 없이 가우시안 노이즈에서 출발해 주차 제어입력 시퀀스를 생성하는 조건부 Flow Matching 모델의 코드, 실험 결과, 학습된 가중치입니다. 조건은 장면(점유 격자를 사각형·경계 선분으로 바꾼 기하 토큰)과 시작·목표 자세뿐입니다. 모델이 낸 제어입력(기어 구간별 길이, 조향·속도 B-spline 계수 64개)은 공통 보정(목표 연결, 가속도·조향 속도 재시간화)을 거쳐 운동학적 자전거 모델로 적분해 경로가 되고, 충돌·제어 한계·목표 허용오차(0.1 m, 5°) 검증을 통과한 후보 중 비용이 가장 낮은 경로를 고릅니다.

비용함수: J = T + 0.5∫(a² + v²δ̇²)dt (T는 정지 조향 시간을 포함한 주행 시간).

## 결과 요약 (평가 장면 500개)

| 방법 | 성공률 | GT 대비 J 평균 | 추론 시간 평균 (중앙값) |
|---|---|---|---|
| HA* 단독 | 98.0% | 4.073 | 3.30 s (2.39 s) |
| HA* + IPOPT | 97.6% | 1.066 | 9.30 s (7.50 s) |
| **Noise Flow (후보 64개, 상위 3개 구조)** | 92.8% | **1.007** | **0.92 s (0.89 s)** |
| OCP 기준해 (GT) | 100% | 1.000 | 3.88 s (2.18 s) |

GT 대비 J는 네 방법이 모두 성공한 453개 장면 기준입니다. 시간은 500개 장면(실패 포함) 기준이며, HA* + IPOPT는 HA* 탐색과 IPOPT 시간의 합(각각 단일 프로세스), Noise Flow는 입력 인코딩부터 최종 선택까지(RTX 4060 Laptop GPU, 후보 검증 CPU 8개 프로세스), OCP(GT)는 데이터셋 GT와 같은 OCP 코드를 저장된 초기 경로에서 다시 풀어 잰 값으로 초기 경로를 만든 시간은 포함하지 않습니다. 전체 수치는 `experiments/results/exp2_four_method_comparison_N64K3.json`에 있습니다.

후보 16개 설정에서는 성공률 88.4%, 평균 추론 시간 0.31 s였습니다(`experiments/results/exp2_noise_flow_bspline64_epoch20.jsonl`).

## 구성

| 경로 | 내용 |
|---|---|
| `scripts/k4_flow_solver/noise_flow.py` | 모델(장면·태스크 인코더, 구조 헤드, Flow 벡터장, bspline64 및 grid64/128/513 표현), CFM 학습 쌍, 초기화 |
| `scripts/k4_flow_solver/noise_flow_solver.py` | 추론기: 상위 K개 구조에 후보 N개 배정 → 생성 → 보정 → 적분 → 검증 → 최소 J 선택 |
| `scripts/train_noise_flow.py` | 학습 스크립트 (FM + 구조 CE + 물리 보조손실) |
| `scripts/k4_flow_solver/` 나머지, `scripts/*.py` | 공통 수치 모듈(적분·비용·검증), 보정, 기준선(HA*, OCP) 코드 |
| `tools/run_noise_flow_eval.py` | 500개 장면 평가와 시간-비용 스윕 |
| `tools/time_gt_ocp.py` | OCP(GT) 방식 계산 시간 측정 |
| `tools/compute_grid_normalization.py` | grid 표현의 정규화 통계 계산 |
| `tools/export_noise_flow_viewer_data.py` | 경로 비교 뷰어 데이터 생성 |
| `tests/` | 단위 테스트 (데이터가 없으면 데이터가 필요한 테스트는 건너뜀) |
| `experiments/eval_uids.json` | 평가 장면 500개 목록 (학습·모델 선택에 사용하지 않음) |
| `experiments/results/` | 평가 결과 (`exp2_*`: Noise Flow, `offline_methods.jsonl`·`timing_*.jsonl`: 기준선과 시간 측정) |
| `experiments/viewer_noise_flow/` | 경로·속도 비교 뷰어 (`python -m http.server`로 열기) |
| `run-logs/noise_flow_gt_bspline64_v1/best.pt` | 최종 가중치 (20 epoch, 검증 장면으로 선택) |
| `run-logs/noise_flow_gt_bspline64_v1/*.json*, *.log` | 학습 설정, 에폭별 지표, 학습 로그 |

## 데이터

학습 데이터와 장면 파일은 용량 때문에 포함하지 않았습니다. 원본 릴리스(`kang-jun-mo12/k4-whole-flow-gt`, 태그 `k4-whole-flow-gt-20260922`)를 받아 `data/` 폴더를 이 저장소 최상위에 두면 됩니다.

```bash
python tools/download_release.py --destination ./downloaded
# downloaded/k4_whole_flow_gt_20260922/data 를 ./data 로 옮기거나 연결
```

필요한 경로: `data/k4_numeric_flow_full_v1_20260917/`(학습 83,996개, 검증 2,048개), `data/flow_scenes/<uid>/planner_input.npz`. 처음부터 다시 학습하려면 같은 릴리스의 초기화 가중치 `run-logs/k4_spline_projected_j_v1_20260920/epoch_10.pt`도 필요합니다. HA* + IPOPT 평가에는 HA* 경로 캐시가 추가로 필요합니다.

## 실행

환경: Python 3.11, PyTorch 2.11 (CUDA 12.8), CasADi 3.8.1 (IPOPT 3.14.19). 패키지는 `requirements-release.txt`.

```bash
# 테스트
python -m unittest tests.test_noise_flow tests.test_noise_grid_flow

# 추론 (한 장면)
PYTHONPATH=scripts python -c "
from k4_flow_solver.noise_flow_solver import NoiseFlowSolver
from k4_flow_solver.prepare import load_scene
s = NoiseFlowSolver('run-logs/noise_flow_gt_bspline64_v1/best.pt', device='cuda', validation_workers=8)
s.set_goal_tolerance(0.1, 0.0872665)
trajectory, report = s.plan(load_scene('data/flow_scenes/<uid>'), candidates=64, top_k=3, steps=16)
print(report['status'], report['selected_J'], report['timings'])
s.close()"

# 평가 (500개 장면, 결과를 새 폴더에 저장)
python tools/run_noise_flow_eval.py config --candidates 64 --steps 16 --top-k 3 --out-dir runs/eval_n64

# 학습
PYTHONPATH=scripts python scripts/train_noise_flow.py --output run-logs/noise_flow_reproduce
```

Windows PowerShell에서는 `PYTHONPATH=scripts` 대신 `$env:PYTHONPATH='scripts'`를 먼저 실행합니다.

## 주의

- OCP 기준해(GT)는 전역 최적해가 보장되지 않습니다. 제안 방법의 J가 GT보다 낮은 장면은 대부분 더 적은 기어 전환을 고른 경우입니다.
- 평가 장면은 검증 분할에서 고정 시드로 뽑았고 학습과 모델 선택에는 쓰지 않았지만, 독립된 테스트 분할은 아닙니다.
- `exp2_noise_flow_bspline64_epoch20_sweep_partial.jsonl`의 N=16·4 step 세 조합은 측정 중 전력 제한으로 시간이 오염되었습니다.
