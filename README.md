# Seoul Bike Demand — Fourier Regression with Linear Algebra

서울 공공자전거 시간별 대여량을 **24시간 및 12시간 주기의 삼각함수**로 모델링하고,
행렬의 최소제곱법으로 계수를 구한 뒤 미관측 날짜의 오차를 비교하는 탐구 프로젝트입니다.

## 검증된 데이터

- **출처:** UCI Machine Learning Repository, [Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand).
- **DOI:** https://doi.org/10.24432/C5F62R
- **이용 조건:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — 출처를 명시해야 합니다.
- **기간:** 2017-12-01 ~ 2018-11-30
- **원본:** 8,760행, 14개 열(시간별 대여 건수 및 날씨·공휴일 등).
- **전처리:** `Functioning Day = No` 295행 제외 → 8,465행.
- **보존 자료:** [data/SeoulBikeData.csv](data/SeoulBikeData.csv). UCI의 공개 자료를 동일한 이름으로 배포한 [공개 GitHub 사본](https://github.com/Itskindastrange/Seoul-Bike-Sharing-Demand-UCI-Dataset)에서 복사했습니다. 데이터 내용은 수정하지 않았습니다.

## 분석한 수학적 모델

시간 `t ∈ {0, …, 23}`, 실제 시간당 대여 건수 `y`, 주말 여부 `w ∈ {0, 1}`.

- **Baseline:** 학습 데이터 전체의 평균 대여량만 예측.
- **M1 (24h):** `ŷ = a0 + a1 sin(2πt/24) + b1 cos(2πt/24)`.
- **M2 (24h + 12h):** M1에 `a2 sin(4πt/24) + b2 cos(4πt/24)` 추가.
- **M3 (weekend interaction):** M2에 `w(c0 + c1 sin(2πt/24) + d1 cos(2πt/24) + c2 sin(4πt/24) + d2 cos(4πt/24))` 추가.

각 행은 한 시각의 삼각함수값으로 이루어진 설계행렬 `X`를 만들고,
`min ||Xβ-y||²`를 푸는 **선형 최소제곱법**(`numpy.linalg.lstsq`)으로 회귀계수를 얻습니다.
삼각함수가 포함되어 있어도 **계수에 대해 선형**인 모델입니다.

## 실행 (Windows PowerShell / macOS / Linux)

Python 3.10 이상이 필요합니다.

```powershell
git clone https://github.com/aether0xff-afk/LinearAlgebra_Seoul_bike.git
cd LinearAlgebra_Seoul_bike
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.run_analysis
python -m unittest discover -s tests -v
```

macOS/Linux에서는 가상환경 활성화 명령을 `source .venv/bin/activate`로 바꾸세요.
PowerShell 실행 정책에서 활성화를 차단하면 `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`,
`.\.venv\Scripts\python.exe -m src.run_analysis`처럼 가상환경 파이썬을 직접 호출할 수 있습니다.

옵션: `python -m src.run_analysis --data data/SeoulBikeData.csv --out results --seed 42`

### 자동 생성 결과

- `results/metrics.csv` — 학습/검증/테스트 MAE, RMSE, R².
- `results/coefficients.csv` — 학습 데이터에서 추정된 각 삼각함수의 계수.
- `results/hourly_means.csv` — 전체 정상 운영 기록의 시각별 표본수와 평균.
- `results/predictions.csv` — 각 원자료 행의 실제값/모델 예측값(실행 시 생성).
- `results/test_residuals_by_hour.csv` — 테스트 잔차의 시각별 평균.
- `results/figures/*.png` — 데이터 경향·주중/주말 곡선·MAE 비교·잔차 그래프(실행 시 생성).
- `results/summary.md` — 실행 결과의 간단한 표와 개선율.

일부 표 및 SVG 그래프는 저장소에 미리 제공됩니다. PNG와 행별 예측치는 명령 실행으로 재생성됩니다.

## 테스트 데이터 결과 (seed=42)

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Mean baseline | 508.296 | 633.756 | -0.000 |
| 24h | 430.716 | 565.230 | 0.204 |
| 24h + 12h | 414.978 | 541.564 | 0.270 |
| 24h + 12h + weekend interaction | **405.584** | **528.761** | **0.304** |

- **24h → 24h+12h MAE 감소:** 약 **3.65%**.
- **시간대별 전체 평균:** 오전 8시 약 1,050건, 오후 18시 약 1,554건.
- 수치 단위: 대여 건수/시간. 작을수록 MAE/RMSE가 좋습니다.
- **중요:** 이 수치는 **원자료가 존재하는 1년 안에서 보지 않은 날짜**를 맞히는 실험입니다. 다른 연도 예측 결과가 아닙니다.

## 평가 설계와 한계

날짜별로 `season × weekend` 층을 구분하고, `date|seed`의 FNV-1a 해시로
정렬해 각 층의 약 70%를 훈련, 15%를 검증, 나머지를 테스트로 배정합니다.
같은 날짜의 24시간 기록이 서로 다른 데이터셋에 섞이지 않습니다.

**실제 관측 353일** 가운데 훈련 243일(5,832시간), 검증 47일(1,121시간),
테스트 63일(1,512시간)입니다. 일부 날짜는 미운영 시간이 있어 24개보다 적습니다.

- 시간 이외의 날씨·강수·계절·이벤트 차이를 모델 식에는 직접 사용하지 않았습니다.
- 공휴일과 강우 등 변수는 오차 분석과 후속 개선에 활용할 수 있습니다.
- 주기함수는 대여량을 음수로 예측할 수도 있으므로 운영용 예측 시스템으로 쓰려면 제약을 검토해야 합니다.
- 더 복잡한 모델의 학습 오차는 낮아지더라도 일반화 오차가 반드시 개선되지는 않습니다.
- 순수한 미래 예측을 평가하려면 **시간순 분할 또는 다른 연도 자료**가 추가로 필요합니다.

자세한 식과 보고서 형식은 [report/analysis.md](report/analysis.md)에 정리했습니다.
