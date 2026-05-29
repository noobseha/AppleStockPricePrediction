## PreProcessing 단계

Raw data : Apple 사의 주력 제품인 <i Phone>이 출시된 2007년 부터의 주요 기사들을 모아 놓은 데이터 셋

### Step1 : 데이터 확인

1. 데이터를 date 순으로 정렬하고, datetime형식으로 변환 후 UTC를 통일합니다.
2. 2019년 이전 데이터는 수가 적어 노이즈 가능성이 존재하므로, 2020년 이후의 데이터만 추출합니다.
3. 데이터의 기본 정보를 출력합니다. 출력되는 정보는 다음과 같습니다:
    1. 행과 열의 개수
    2. 전체 Column list
    3. 각 column의 data type
    4. 각 column의 기본 통계량

### Step2 : Dirty Data 처리

1. 완전 중복 제거 및 Title 기준 부분 중복 제거 

![image.png](attachment:00b14d77-3d4d-43fd-9546-c208038ad7f4:image.png)

→ 위 예시(421~437번 레코드)와 같이, 동일한 데이터로써 완전 중복된 데이터들을 삭제하고,

→ 동일 기사이지만 조금씩 다르게 수집된 데이터들을 골라 1건을 제외하고 모두 삭제합니다.

1. OHLC 논리 검사
    
    Open,High,Low,Close (시가,고가,저가,종가) 에 대한 수학적 논리 검사를 실시합니다.
    
    ex) 고가가 저가보다 낮으면 논리적 오류가 발생합니다.
    
2. 음수값 검사
    
    주가 및 거래량은 0보다 작은 음수가 될 수 없습니다. 데이터에 음수가 포함되어있는지 검사합니다.
    
3. Sparse(희소한) Column 선별 후 제거
    - Dividends(배당금) Column의 경우 66개의 행에만 데이터가 존재합니다.
    - Stock Split (주식 분할)은 단 3개의 행에만 데이터가 존재했습니다.
4. 불필요한 카테고리형 Column 제거
    
    Title, Link, Source, gpt_summary는 이미 텍스트를 분석하여 정량화 해둔 숫자 점수 Column이 존재하므로 제거합니다.
    

#### Step3 : GPT Sentiment Feature 정리

GPT에 의해 분석된 기사 자료에 대한 감성 평가 지표를 가공합니다.

중복되는 Features (예 - gpt_positive_score / gpt_negative_score) 을 +/- 로 한번에 표기합니다.

 

#### Step4 : 이상치 탐지 및 분석

주가 정보 및 ATR, ADX, MACD Signal 같은 핵심 기술적 지표 8개를 대상으로 시행됩니다.

```
'Open','High','Low','Close','Volume','ATR','ADX','MACD_Signal'
```

주식 시장에서 발생하는 이상치는 시스템 오류가 아닌 실제 시장의 거대한 충격과 전전(Event)을 반영하므로, 의미가 있다고 판단하여 최종적으로 이상치로 분류하지 않았습니다.

#### Step5 : 희소 카테고리 통합

`gpt_event_type` 컬럼은 GPT가 뉴스의 성격을 분류한 카테고리형 데이터이므로, 머신러닝 모델에 이 데이터를 넣기 위해 One-Hot Encoding을 통해 숫자로 변환해야 합니다.

이때, 출현 빈도가 극도로 낮은 카테고리는 차원의 저주, 과적합 등의 문제를 일으킵니다. 따라서, 희소 카테고리는 Other로 묶어 통일합니다.

해당 프로그램의 경우, 빈도 기준을 10으로 설정하여 기준에 부합되지 않는 카테고리는 모두 Other로 통일하였습니다.

#### Step6 : 파생 변수 생성 - Feature Engineering

모델이 주가 패턴을 더 잘 학습할 수 있도록 유의미한 features를  제작합니다.

1. Daily_Volatility = (HIGH - LOW) / OPEN 
    
    → 일일 주가 변동성 지표입니다. 과거와 현재 주가 Gap이 있는데, 이때 변동폭을 공평하게 비교할 수 있습니다.
    
2. Buy Pressure & Sell Pressure
    1. Buy Pressure = HIGH - max(OPEN, CLOSE)
        
        장중에 매수세(Buyers)가 주가를 고가까지 강하게 밀어 올렸던 총 수치를 측정
        
    2. Sell Pressure = min(OPEN, CLOSE) - LOW
        
         장중에 매도세(Sellers)가 주가를 저가까지 얼마나 강하게 끌어내렸었는지 측정
        
3. Close_Change_Rate = df['Close'].pct_change()
    
    전일 종가 대비 오늘 종가가 몇 % 상승했거나 하락했는지 나타냄
    
4. BB_Position = (Close - BB_lower) / (BB_Upper - BB_lower)
현재 주가가 볼린저 밴드 안에서 어느 수준에 위치해 있는지 비율로 수치화
5. Volume_Change_Rate = df['Volume'].pct_change()
    
    어제 거래량 대비 오늘 거래량이 몇 %나 증가/감소했는지를 나타냄.
    

→ 위 6개의 변수를 추가함으로써, "오늘은 어제보다 거래량이 2배 터졌고, 장 중에 매수세가 윗꼬리를 길게 만들었으며, 볼린저 밴드 상단에 바짝 붙은 과열 상태다” 와 같은 추측을 가능하게 합니다.

#### Step7:  Target Label + Bull/Bear 생성

- 일일 종가를 추출하여, 다음날 종가와 비교합니다. 이때, 익일 종가가 더 높을 경우,
    
    Target의 값이 1이되며, 전날과 비교하여 같거나 낮을 경우, 0이 됩니다.
    
- MA_20(20일 이동평균선) 파생 변수를 생성합니다. 최근 20거래일동안의 평균 종가입니다.
    
    이 변수는 단기/중기적 추세를 가르는 중요한 지표입니다.
    
- 만약 현재 종가가 MA_20보다 높으면 새롭게 생성될 Bull_Bear 변수는 1로 정의되며(강세장),
    
    MA_20보다 낮으면 0(약세장)으로 정의됩니다.
    
    → 두 변수(MA_20, Bull_Bear) 를 원본 데이터에 merge합니다.
    

#### Step8: 결측치 처리

Column에 부합하는 값이 존재하지 않는 행을 삭제합니다.

#### Step9: 데이터 시각화

다음과 같은 시각적 그래프가 출력됩니다:

- AAPL 종가 시계열 그래프 : 시간에 따른 애플 주가의 종가를 나타낸 Line Plot입니다.
- Target Feature 분포 막대 그래프 : data imbalance를 점검하기 위해 상승장과 하락장의 밸런스를 확인할 수 있습니다.
- Correlation Heatmap : 데이터에 포함된 모든 숫자 데이터(주가, 기술적 지표, GPT 감성 점수 등)들 간의 상관계수(Correlation)를 색상 격자로 나타낸 지도입니다. 각 Feature들의 상관 관계를 한눈에 비교하여 Target 변수와 가장 관련도가 높은 Feature를 탐색할 수 있습니다.

#### Step10: 최종 Feature 선택

 다음과 같은 Feature가 생성됩니다:

```
# 기본 주가
'Open','High','Low','Close','Volume'

# 이동평균
'SMA_50','SMA_200','EMA_50','EMA_200','MA_20'

# 추세 / 모멘텀
'RSI','MACD','MACD_Signal','MACD_Hist','ADX'

# 변동성
'ATR','BB_Upper','BB_Middle','BB_Lower','BB_Position'

# 파생 변수
'Daily_Volatility','Buy_Pressure','Sell_Pressure','Close_Change_Rate','Volume_Change_Rate'

# GPT 감성
'gpt_sentiment_score','signed_sentiment_score','gpt_relevance_to_apple','gpt_importance_score'

# 범주형
'gpt_event_type','gpt_sentiment_direction'

# 시장 상태
'Bull_Bear'
```

#### Step11: One-Hot Coding

- 카테고리형 데이터인  gpt_event_type , gpt_sentiment_direction을 모델이 이해할 수 있도록
    
    수치형 데이터로 변환합니다.
    
- Pandas에서는 `pd.get_dummies`를 실행하면 결과 데이터가 Boolean 형식으로 반환되는 경우가 있는데, 이를 수치형으로 변환합니다.
- Feature 기준 중복 데이터를 제거합니다. 날짜가 달라도 fields 값이 동일하다면 동일 데이터로 간주하고 한 데이터만 남깁니다. Overfitting을 방지하기 위함입니다.
- 또한, 다중공선성(Dummy Variable Trap) 문제를 방지하기 위해 `drop_first=True` 옵션을 적용하여 첫 번째 카테고리형 데이터를 제외합니다.

#### Step 12: Train/Test Split

최근 3개월의 애플 주가를 Test Case로 간주하고 분리하고, 나머지 데이터는 Train case로 분류합니다.

#### Step 13: Scaling

날짜, 정답 label(Target, Bull_Bear), gpt 지표 Feature들은 숫자의 크기에 의미를 두는 Feature가 아니므로 Scaling에서 제외합니다. 

이 함수는 두 가지 Scaling을 사용하여 데이터에 적용합니다:

1. Standard Scaler - Outlier에 민감하여 일반 baseline 모델용으로 사용합니다.
2. Robust Scaler - COVID-19 등 이상치에 대응하여, 급등 및 거래량 폭증 영향을 완화합니다.

#### Step 14: 최종 데이터 저장

index들을 재정렬 후,

test_dataset_raw.csv

train_dataset_raw.csv

baseline_dataset_final.csv

test_dataset_robust_scaled.csv

test_dataset_standard_scaled.csv

train_dataset_robust_scaled.csv

train_dataset_standard_scaled.csv

총 7개의 csv파일을 저장합니다.

#### Step 15: 최종 품질 검증

최종적으로 모든 데이터 파일에 대하여

결측치, 중복 검사를 시행하고, Target 비율과 Feature 개수를 확인하며 전처리 과정을 마무리합니다. 

---

## 모델 및 평가 단계

Pre-processing 이후 진행되는 회귀 분석(Regression) 및 분류 분석(Classification) 모델 학습, 그리고 최적의 파이프라인 조합을 도출하는 평가 방법에 대한 내용입니다.

#### 회귀 분석 파트 (Regression Analysis)

주식의 '내일 종가'를 구체적인 수치로 예측 후 주가가 오를지 내릴지를 판별합니다.

1. 스케일링 비교
정규화되지 않은 Raw Data, 이상치에 둔감한 Robust Scaling, 표준 정규분포를 따르는 Standard Scaling 3가지 데이터셋에 대해 모델을 구동합니다.
2. 활용 알고리즘
선형 회귀(Linear Regression), 랜덤 포레스트(Random Forest 기본 및 max_depth=3 제한), 단일 트리의 의사결정나무(Decision Tree)가 사용되었습니다.
3. 시각화
    1. Prediction Error (RMSE) 바 차트 : 막대가 낮을수록 모델이 예측한 가격이 실제 가격과 오차가 적음을 의미합니다.
    2. Directional Accuracy 바 차트 :  오를지 내릴지 방향을 맞춘 확률입니다. 50% Boundary 점선에 가까워야 의미 있는 성능을 낸 것으로 간주합니다.
    

#### 감성 분류 분석 파트 (Classification Sentiment)

분류 모델은 주식의 특정 가격을 예측하지 않고 오직 상승(1)과 하락(0) 자체를 판별하는 것에 특화되어 있습니다. 기존 재무/차트 데이터에 chatGPT를 통해 추출한 뉴스 감성 지표(Sentiment)를 추가했을 때 성능이 어떻게 변하는지 관찰합니다.

1. 비교 그룹
    
    순수 기술적 지표로 구성된 Version A (Base) vs 감성 지표가 포함된 Version B (+Sentiment)
    
2. 교차 검증 - GridSearchCV 사용
    
    데이터를 쪼개어 번갈아 시험을 치르는 K-Fold 검증과 모델 파라미터를 미세조정하는 방식을 결합하여, 과적합을 방지하고 최상의 조건으로 트리를 구성합니다.
    
3. 시각화 ( Ver A vs Ver B)
    1.  Performance Metrics
        
        전체, 상승장, 하락장 기간에 대해 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1-Score를 비교하는 바 차트입니다.
        
    2. Confusion Matrices
        
        모델이 상승을 하락으로, 하락을 상승으로 잘못 예측한 경우가 각각 몇 번인지 색상 격자로 보여주는 히트맵입니다.
        
    3. ROC & P-R Curves
        
        그래프가 좌측 및 우측 상단으로 팽팽하게 당겨져 곡선 아래 면적(AUC, AP)이 넓을수록 분류 모델이 적절히 작동함을 의미합니다.
        

#### 최종 Top 5 조합 산출 (Combination Search)

모든 개별 분석 파트가 끝나면, 프로그램은 자동으로 앞서 쓰였던 재료들을 교차 융합하여 새로운 파이프라인 조합을 동적으로 탐색합니다.

조합 후보군:

[스케일러 종류 (Standard/Robust/None)]

[특성 지표(Base vs Base+Sentiment)] 

[머신러닝 알고리즘 (DecisionTree vs RandomForest)] 

[알고리즘별 하이퍼파라미터들]

가장 최우선 지표인 Accuracy를 기준으로, 콘솔 화면에 최상위 1위부터 5위까지 모델의 구체적인 세팅 방법과 파라미터 값이 출력됩니다.

→ 출력된 Top 5의 세팅값을 투자 전략이나 후속 분석에 적용할 수 있습니다.
