# Apple Stock Prediction & Sentiment Analysis Pipeline

이 저장소는 애플(AAPL) 주가 데이터와 GPT로 생성된 뉴스 감성 점수(Sentiment Scores)를 분석하는 End-to-End 머신러닝 워크플로우를 제공합니다.

전통적인 기술적 지표와 자연어 기반의 감성 분석을 결합하여, 자동화된 데이터 전처리, 다중 모델 회귀 분석, 감성 기반 분류 분석, 그리고 하이퍼파라미터 튜닝을 수행해 **상위 5개의 최적 예측 모델 조합**을 찾아냅니다.

---

## 📑 목차 (Table of Contents)

- [Phase 1: 전처리 (Preprocessing)](#phase-1-전처리-preprocessing)
- [Phase 2: 회귀 분석 (Regression Analysis)](#phase-2-회귀-분석-regression-analysis)
- [Phase 3: 분류 및 감성 지표 비교 (Classification & Sentiment Comparison)](#phase-3-분류-및-감성-지표-비교-classification--sentiment-comparison)
- [Phase 4: Top 5 최적 조합 탐색 (Combination Search)](#phase-4-top-5-최적-조합-탐색-combination-search)

---

## Phase 1: 전처리 (Preprocessing)

**Raw Data:** 애플의 주력 제품인 아이폰이 처음 출시된 2007년 이후의 주요 기사들을 수집한 데이터셋입니다.

### Step 1: 데이터 초기화
1. 데이터를 날짜순으로 정렬하고 타임스탬프를 `UTC`로 통일합니다.
2. 2019년 이전 데이터의 히스토리컬 노이즈를 제거하기 위해 **2020년 이후의 데이터만 추출**합니다.
3. 데이터셋의 기본 정보(형태, 컬럼 리스트, 데이터 타입, 기초 통계량)를 출력하여 점검합니다.

### Step 2: Dirty Data (불량 데이터) 정제
- **중복 제거:** 완전히 동일한 중복 데이터와, 기사 제목(`Title`)을 기준으로 부분 중복된 데이터를 제거합니다.
- **OHLC 논리 검사:** 가격 데이터의 논리적 오류를 검증합니다 (예: 고가(`High`)는 저가(`Low`)보다 크거나 같아야 함).
- **음수값 검사:** 주가 및 거래량에 음수가 포함되지 않았는지 확인합니다.
- **희소/불필요 컬럼 제거:** 빈도수가 극도로 적은 컬럼(`Dividends`, `Stock Splits`)과 GPT가 이미 정량화한 텍스트 컬럼(`Title`, `Link`, `Source`, `gpt_summary`)을 제거합니다.

### Step 3: GPT 감성 지표 정제
중복되는 긍정/부정 감성 점수를 하나의 `signed_sentiment_score`로 통합하여, 점수의 부호(+/-)만으로 감성의 방향성을 명확히 나타내도록 가공합니다.

### Step 4: 이상치 탐지 (Outlier Detection)
다음 주요 지표들에 대해 이상치를 점검합니다:
`Open`, `High`, `Low`, `Close`, `Volume`, `ATR`, `ADX`, `MACD_Signal`.
*참고: 금융 시장에서 발생하는 이상치는 시스템 오류가 아닌 실제 거시경제적 충격(Event)을 반영하는 경우가 많으므로, 이 파이프라인에서는 삭제하지 않고 그대로 보존합니다.*

### Step 5: 희소 카테고리 통합
원-핫 인코딩 시 발생하는 차원의 저주(Curse of Dimensionality)를 방지하기 위해, 출현 빈도가 10회 미만인 `gpt_event_type` 카테고리들을 `Other`로 통합합니다.

### Step 6: 파생 변수 생성 (Feature Engineering)
모델이 시장 패턴을 더 잘 학습할 수 있도록 도메인 특화 기술적 지표들을 생성합니다:
1. `Daily_Volatility`: `(High - Low) / Open` (일일 주가 변동성)
2. `Buy_Pressure` & `Sell_Pressure`: 꼬리(Shadows)를 기반으로 장중 매수/매도 압력을 측정
3. `Close_Change_Rate`: 전일 대비 종가 등락률(%)
4. `BB_Position`: 볼린저 밴드 내 현재 주가의 위치 `(Close - Lower) / (Upper - Lower)`
5. `Volume_Change_Rate`: 전일 대비 거래량 증감률(%)

### Step 7: Target 라벨 및 시장 상태(Bull/Bear) 생성
- **Target (1/0):** 내일 종가가 오늘 종가보다 엄격히 높으면 `1`(상승), 아니면 `0`(하락/보합)으로 설정합니다.
- **시장 상태 (MA_20 & Bull_Bear):** 최근 20일 이동평균선(`MA_20`)을 계산합니다. 현재 종가가 이보다 높으면 `1`(강세장/Bull), 낮으면 `0`(약세장/Bear)으로 라벨링합니다.

### Step 8: 결측치(Missing Values) 처리
정의된 Feature들 중 `NaN`(결측치)이 하나라도 포함된 행은 모두 제거합니다.

### Step 9: 데이터 시각화
파이프라인이 자동으로 진단용 그래프 3종을 생성하여 로컬에 저장합니다:
- **시계열 차트 (Time Series):** AAPL 종가 흐름 그래프
- **Target 분포 (Bar Chart):** 상승(Up)/하락(Down) 빈도에 대한 클래스 불균형 점검
- **상관관계 히트맵 (Correlation Heatmap):** 가격, 기술적 지표, GPT 감성 점수를 포함한 모든 숫자형 변수들 간의 상관관계를 시각화

### Step 10 & 11: 최종 Feature 선택 및 원-핫 인코딩
이동평균, 모멘텀, 변동성, GPT 감성 지표 등 최적의 숫자형 변수를 선택하고, 카테고리형 변수(`gpt_event_type`, `gpt_sentiment_direction`)에는 원-핫 인코딩(Boolean → Integer)을 적용합니다. 이후 데이터 누수(Leakage) 방지를 위해 Feature 기준 중복 행을 제거합니다.

### Step 12 & 13: Train/Test 분할 및 스케일링
- **분할 (Split):** 시계열 특성을 반영하여 가장 최근 3개월의 데이터를 온전한 `Test` 세트로 분리합니다.
- **스케일링 (Scaling):** 날짜 컬럼과 이진 타겟(0/1)은 스케일링에서 제외됩니다. 다음 3가지 데이터셋을 생성합니다:
  1. `Raw Data` (스케일링 없음)
  2. `Standard Scaler` (정규분포 가정, 이상치에 다소 민감)
  3. `Robust Scaler` (사분위수(IQR)를 활용하여 급등/급락 등 극단적 이상치에 강건함)

### Step 14 & 15: 최종 데이터셋 내보내기 및 품질 검수
Train/Test 용도로 분할 및 스케일링된 총 7개의 `.csv` 파일을 로컬에 저장합니다. 마지막으로 결측치나 중복 데이터가 없는지, 타겟 비율이 적절한지 최종 산출물 품질(QA)을 검증합니다.

---

## Phase 2: 회귀 분석 (Regression Analysis)

회귀 모듈은 내일의 '정확한 주가(Closing Price)' 수치를 예측한 뒤, 이를 기반으로 주가의 등락 방향을 유추합니다.

- **사용된 데이터셋:** `Raw Data`, `Robust Scaled`, `Standard Scaled`
- **평가 알고리즘:** 
  - 선형 회귀 (Linear Regression)
  - 랜덤 포레스트 (Random Forest - 기본 및 max_depth=3 제한)
  - 의사결정나무 (Decision Tree)
- **시각화 분석 창 (팝업):**
  - **Prediction Error (RMSE) 바 차트:** 막대가 낮을수록 모델의 가격 예측 오차가 적음을 뜻합니다.
  - **Directional Accuracy 바 차트:** 모델이 주가의 오르고 내리는 방향을 맞출 확률을 나타냅니다. 점선(50% Boundary)을 넘어야 유의미합니다.

---

## Phase 3: 분류 및 감성 지표 비교 (Classification & Sentiment Comparison)

회귀 분석과 달리, 분류 모듈은 구체적 가격을 배제하고 오직 시장의 방향인 `상승(1)` 또는 `하락(0)` 여부만을 예측하는 데 특화되어 있습니다. 이 파트는 **GPT 뉴스 감성 지표**가 모델 성능을 얼마나 향상시키는지 중점적으로 테스트합니다.

- **A/B 테스트 그룹:**
  - `Version A (Base)`: 순수 기술적 지표 및 재무 데이터만 사용
  - `Version B (+Sentiment)`: 기술적 지표에 GPT 감성 점수를 추가로 사용
- **교차 검증 (Cross Validation):** K-Fold 검증(`StratifiedKFold`)과 하이퍼파라미터 튜닝(`GridSearchCV`)을 결합하여 과적합을 방지하면서 `DecisionTreeClassifier`의 가장 완벽한 트리를 구축합니다.
- **시각화 분석 창 3종 (팝업):**
  1. **Performance Metrics:** 전체 기간, 강세장(Bull), 약세장(Bear) 각각에 대해 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1-Score를 꼼꼼히 비교하는 바 차트.
  2. **Confusion Matrices:** 모델이 상승을 하락으로(또는 그 반대로) 잘못 짚어낸 횟수를 색상 짙기로 보여주는 오차 행렬 히트맵.
  3. **ROC & P-R Curves:** 팽팽하게 당겨진 곡선 아래 면적(AUC, AP)이 넓을수록 분류 모델이 건강하게 작동함을 증명합니다.

---

## Phase 4: Top 5 최적 조합 탐색 (Combination Search)

파이프라인의 하이라이트입니다. 프로그램이 이전 단계들에서 활용했던 모든 기법과 재료들을 동적으로 교차 결합하여 수많은 파이프라인을 새롭게 구동해 봅니다.

- **탐색 범위 (Search Space):**
  - `Features`: [기본 기술적 지표 vs 감성이 포함된 풀 지표]
  - `Scalers`: [Standard vs Robust vs 스케일링 없음]
  - `Algorithms`: [DecisionTree vs RandomForest]
  - `Hyperparameters`: 깊이(Depth), 분할 조건, 트리 갯수 등
- **최종 출력 (Output):** 터미널 콘솔 화면에 **정확도(Accuracy)**를 기준으로 엄격히 랭크된 **Top 5 최고 조합**이 출력됩니다. 산출된 1등 조합의 설정값(파라미터)은 여러분의 실전 트레이딩 전략이나 후속 연구에 즉시 투입 가능한 레시피가 됩니다.

---

*문서 끝. 통합 파이프라인을 직접 구동하려면 터미널에서 `python single_top_level_func.py` 명령어를 실행하고, 화면에 순차적으로 나타나는 시각화 차트들을 확인하세요.*
