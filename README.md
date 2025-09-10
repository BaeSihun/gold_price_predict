# 금 가격 예측 머신러닝 프로젝트

## 프로젝트 개요

2010년부터 2025년까지의 다양한 경제 지표를 활용하여 금 가격을 예측하는 머신러닝 프로젝트입니다. 
13개의 핵심 경제 및 금융 지표를 통합하여 XGBoost, RandomForest 등의 머신러닝 알고리즘과 LSTM, GRU 딥러닝 모델을 비교 분석을 수행합니다.

## 프로젝트 구조

```
MLproject/
├── data/                       # 데이터 관련 폴더
│   ├── gold_data/             # 원본 데이터 파일들 (12개 CSV)
│   │   ├── 01_gld_monthly_mean.csv          # 금 가격 (타겟 변수)
│   │   ├── 02_m2_money_supply_2010_2025.csv # 미국 M2 통화공급량
│   │   ├── 03_us_cpi.csv                    # 미국 소비자물가지수
│   │   ├── 04_volume.csv                    # 금 거래량
│   │   ├── 05_gold_reserves_2010_2025.csv   # 주요국 금 보유량
│   │   ├── 06_exchange_rates_2010_2025.csv  # 환율 (USD/CNY, USD/EUR)
│   │   ├── 08.SLV_DF.csv                    # 은 가격 (Silver ETF)
│   │   ├── 09_dollar_index.csv              # 달러 지수 (DXY)
│   │   ├── 10.snp_monthly.csv               # S&P 500 지수
│   │   ├── 11_nasdaq.csv                    # NASDAQ 지수
│   │   ├── 12.prices_monthly.csv            # 채권 수익률
│   │   └── 13_vix_fear_index_2010_2025.csv  # VIX 공포 지수
│   ├── gold_data.ipynb        # 데이터 전처리 및 결합 노트북
│   └── gold_data.csv          # 전처리된 통합 데이터셋
├── LSTM_GRU.ipynb            # 딥러닝 모델 (LSTM, GRU)
├── RANDOM_FOREST.ipynb       # Random Forest 머신러닝 모델
├── XGBOOST.ipynb             # XGBoost 머신러닝 모델
└── README.md                  # 프로젝트 문서 (현재 파일)
```

## 주요 특징

- **포괄적 데이터**: 13개 경제/금융 지표 통합 (2010-2025, 186개월)
- **다양한 알고리즘**: XGBoost, RandomForest, LSTM, GRU 성능 비교
- **특성 엔지니어링**: 래그 변수, 이동평균, 변화율 등 시계열 특성 생성
- **시계열 모델링**: 12개월 시퀀스 기반 딥러닝 예측
- **성능 최적화**: Optuna를 활용한 하이퍼파라미터 튜닝
- **미래 예측**: 2025년까지의 금 가격 예측 
- **시각화**: 예측 결과 및 특성 중요도 상세 분석

## 데이터셋 설명

### 통합 데이터셋 (gold_data.csv)
- **기간**: 2010년 1월 ~ 2025년 6월 (186개월)
- **변수**: 21개 특성 변수 + 1개 타겟 변수 (총 22개 컬럼)
- **구조**: 날짜, 금가격, 거시경제지표, 금융시장지표, 시장심리지표

### 주요 변수 설명 (실제 컬럼명 기준)

#### 1. 기본 정보
- **Date_YYYY_MM**: 날짜 (YYYY-MM 형식)
- **Gold_Price**: 월별 평균 금 가격 (USD/oz) - **타겟 변수**

#### 2. 거시경제 지표
- **US_M2_Money_Supply**: 미국 M2 통화공급량 (조 달러)
- **US_CPI**: 미국 소비자물가지수 (인플레이션 지표)

#### 3. 금 시장 지표
- **Gold_Volume**: 금 거래량 (시장 활동성)
- **Gold_Reserves_United States**: 미국 중앙은행 금 보유량
- **Gold_Reserves_Germany**: 독일 중앙은행 금 보유량
- **Gold_Reserves_Italy**: 이탈리아 중앙은행 금 보유량
- **Gold_Reserves_France**: 프랑스 중앙은행 금 보유량
- **Gold_Reserves_Russia**: 러시아 중앙은행 금 보유량
- **Gold_Reserves_China**: 중국 중앙은행 금 보유량
- **Gold_Reserves_Switzerland**: 스위스 중앙은행 금 보유량
- **Gold_Reserves_Japan**: 일본 중앙은행 금 보유량
- **Gold_Reserves_India**: 인도 중앙은행 금 보유량
- **Gold_Reserves_Netherlands**: 네덜란드 중앙은행 금 보유량

#### 4. 환율 및 화폐 지표
- **Exchange_Rate_USD_CNY**: 달러-위안 환율
- **Exchange_Rate_USD_EUR**: 달러-유로 환율
- **Dollar_Index**: 달러 지수 (DXY)

#### 5. 금융시장 지표
- **Silver_Price**: 은 가격 (귀금속 시장 동향)
- **SP500**: S&P 500 지수 (미국 주식시장)
- **NASDAQ**: NASDAQ 지수 (기술주 시장)
- **Bond_Rate**: 채권 수익률 (무위험 수익률)

#### 6. 시장 심리 지표
- **VIX_Fear_Index**: VIX 공포 지수 (시장 변동성)

## 데이터 선정 이유 및 전처리

### 데이터 선정 근거

#### 1. 거시경제 지표 선정 이유
- **M2 통화공급량**: 통화 공급량 증가는 인플레이션 압력을 높여 금 가격 상승 요인으로 작용
- **소비자물가지수 (CPI)**: 인플레이션 상승 시 실물자산인 금에 대한 수요 증가
- **달러 지수 (DXY)**: 달러 강세 시 금 가격 하락, 달러 약세 시 금 가격 상승의 역상관 관계

#### 2. 금융시장 지표 선정 이유
- **S&P 500 & NASDAQ**: 주식시장 불안정성 증가 시 안전자산인 금으로 자금 이동
- **채권 수익률**: 실질금리 하락 시 무이자 자산인 금의 매력도 증가
- **은 가격**: 귀금속 시장의 동반 상승/하락 패턴 분석
- **VIX 공포 지수**: 시장 불안 심리 증가 시 금 가격 상승 경향

#### 3. 환율 지표 선정 이유
- **USD/CNY 환율**: 중국은 세계 최대 금 소비국으로 위안화 가치 변동이 금 수요에 직접 영향
- **USD/EUR 환율**: 유럽중앙은행의 통화정책과 금 가격 간 상관관계 분석

#### 4. 금 시장 특화 지표
- **금 거래량**: 시장 유동성과 투자 관심도를 나타내는 선행 지표
- **주요국 금 보유량**: 중앙은행의 금 보유량 변화는 장기 금 가격 트렌드에 영향

### 데이터 전처리 과정

#### 1. 데이터 수집 및 통합
```python
# 12개 개별 CSV 파일을 하나의 통합 데이터셋으로 결합
# 기간: 2010년 1월 ~ 2025년 6월 (186개월)
# 최종 형태: 186 rows × 23 columns (Date + 21 features + Gold_Price)
```

#### 2. 시간 정렬 및 결측치 처리
- **시간 정렬**: 모든 데이터를 월별 기준으로 정렬하여 시계열 연속성 확보
- **결측치 처리**: 
  - 선형 보간법 (Linear Interpolation) 적용
  - 월별 데이터의 경우 전후 값의 평균으로 보완
  - 결측률 5% 미만 유지

#### 3. 데이터 형식 표준화
- **날짜 형식**: YYYY-MM-DD 표준 형식으로 통일
- **수치 형식**: 모든 경제 지표를 float64 타입으로 표준화
- **단위 통일**: 
  - 금 가격: USD/oz
  - 통화공급량: 조 달러 (Trillion USD)
  - 지수: 기준년도 100 기준

#### 4. 특성 엔지니어링 준비
- **시차 변수 생성**: 1개월, 3개월, 6개월 지연 변수 생성
- **변화율 계산**: 월간 성장률 (Month-over-Month) 계산
- **이동평균**: 3개월, 6개월 이동평균 및 표준편차 계산
- **주기성 인코딩**: 월별 계절성을 sin/cos 변환으로 표현

### 전처리 결과물
- **최종 데이터셋**: `data/gold_data.csv`
- **데이터 차원**: 186 × 22 (결측치 제거 후)
- **타겟 변수**: Gold_Price (연속형)
- **특성 변수**: 21개

## 데이터 처리 및 모델링

### 노트북 파일 설명

#### 1. data/gold_data.ipynb
- **목적**: 12개 원본 CSV 파일을 하나로 통합
- **입력**: gold_data/ 폴더의 01번~13번 CSV 파일들
- **출력**: gold_data.csv (186행 × 22열 통합 데이터셋)
- **주요 기능**:
  - 각 CSV 파일 자동 로딩 (01-13번 파일)
  - 날짜 기준 데이터 표준화 및 병합
  - 결측치 처리 및 데이터 검증
  - 최종 통합 데이터셋 생성 및 저장

#### 2. LSTM_GRU.ipynb
- **목적**: 딥러닝 기반 시계열 예측
1. ##### LSTM
- **주요 기능**:
  - 금 가격 시계열 데이터(gold_data.csv)를 기반으로 지난 12개월 데이터를 입력으로 금 가격 예측
  - `MinMaxScaler`로 데이터 정규화 후 시퀀스 데이터 생성
  - LSTM 신경망(64 units)을 활용한 회귀 모델 구성
  - 훈련/테스트 데이터 분리(80:20) 후 모델 학습 및 검증
  - 실제 값 vs 예측 값 시각화로 결과 비교
  
- **LSTM 회귀 결과**
  | 지표(Metric) | 값(Value) |
  | :-:  | :-: |
  | R² Score  | 0.9091  |
  | RMSE  | 141.8155|
  | MAE | 111.8357 |
<img width="1189" height="590" alt="Image" src="https://github.com/user-attachments/assets/9852720a-e440-4dfb-9ffd-f730e6928168" />

2. ##### GRU
- **주요 기능**:
    - 월별 금 가격 데이터를 활용하여 지난 12개월 데이터를 입력으로 금 가격 예측
    - `MinMaxScaler`로 데이터 정규화 후 시퀀스 데이터 생성
    - GRU 신경망(64 units) 기반 회귀 모델 구성
    - 훈련/테스트 데이터 분리(80:20) 후 학습
    - 성능 평가 지표 출력: R² Score, RMSE, MAE
    - 실제 금 가격 vs 예측 금 가격 시각화로 결과 비교

- **GRU 회귀 결과**
    | 지표(Metric) | 값(Value) |
    | :-:  | :-: |
    | R² Score  | 0.9504  |
    | RMSE  | 104.7133 |
    | MAE | 78.9009 |
<img width="1189" height="590" alt="Image" src="https://github.com/user-attachments/assets/1c9453eb-3e3d-4bfd-9368-51c210a0405a" />

#### 3. RANDOM_FOREST.ipynb
- **목적**: Random Forest 모델 개발 및 분석
- **주요 기능**:
  - **다양한 파생 변수(피처 엔지니어링)**를 생성하여 금 가격 예측
  - **RandomForest(RandomForestRegressor)**를 활용한 회귀 모델 구성(Optuna로 하이퍼파라미터 최적화)
  - StandardScaler로 데이터 정규화 후 Lag, 이동 평균, 변동률 등의 특성(Feature)을 입력으로 사용
  - 훈련/테스트 데이터 분리(80:20) 후 모델 학습 및 검증
  - 실제 값 vs 예측 값 시각화로 결과 비교
 
  - **RANDOMFOREST 회귀 결과**
    | 지표(Metric) | 값(Value) |
    | :-:  | :-: |
    | R² Score  | 0.9973  |
    | RMSE  | 24.31 |
    | MAE | 16.99 |
    
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/ab47a0fa-ffda-4265-9f17-9dca05ba1181" />

#### 4. XGBOOST.ipynb
- **목적**: XGBoost 모델 상세 분석 및 최적화
- **주요 기능**:
  - **XGBoost(XGBRegressor)**를 활용한 회귀 모델 구성 (Optuna로 하이퍼파라미터 최적화)
  - 훈련/테스트 데이터 분리(80:20) 후 모델 학습 및 검증
  - 실제 값 vs 예측 값 시각화로 결과 비교
- **XGBOOST 회귀 결과**
  | 지표(Metric) | 값(Value) |
  | :-:  | :-: |
  | R² Score  | 0.9826  |
  | RMSE  | 61.94 |
  | MAE | 47.48 |
  
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/15f6b750-faef-4a5c-bb2c-0214e00d22e4" />
