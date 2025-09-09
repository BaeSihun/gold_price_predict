# 금 가격 예측 머신러닝 프로젝트

## 프로젝트 개요

2010년부터 2025년까지의 다양한 경제 지표를 활용하여 금 가격을 예측하는 머신러닝 모델을 개발하는 프로젝트입니다. 총 12개의 핵심 경제 및 금융 지표를 통합하여 포괄적인 금 가격 예측 시스템을 구축합니다.

## 프로젝트 구조

```
MLproject/
├── data/
│   ├── gold_data/               # 원본 데이터 파일들
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
│   ├── gold_data.ipynb         # 데이터 전처리 및 결합 노트북
│   └── gold_data.csv           # 전처리된 통합 데이터셋
├── predict_gold.ipynb          # 머신러닝 모델 개발 및 예측
├── .gitignore                  # Git 버전 관리 설정
├── .venv/                      # Python 가상 환경
└── README.md                   # 프로젝트 문서
```

## 데이터셋 설명

### 통합 데이터셋 (gold_data.csv)
- **기간**: 2010년 1월 ~ 2025년 1월 (180+ 월별 관측값)
- **변수**: 22개 특성 변수 + 1개 타겟 변수
- **구조**: 날짜, 금가격, 거시경제지표, 금융시장지표, 시장심리지표

### 주요 변수 설명

#### 1. 타겟 변수
- **Gold_Price**: 월별 평균 금 가격 (USD/oz)

#### 2. 거시경제 지표
- **US_M2_Money_Supply**: 미국 M2 통화공급량 (조 달러)
- **US_CPI**: 미국 소비자물가지수 (인플레이션 지표)

#### 3. 금 시장 지표
- **Gold_Volume**: 금 거래량 (시장 활동성)
- **Gold_Reserves_[국가명]**: 주요 10개국 중앙은행 금 보유량

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



## 데이터 처리 및 모델링

#### 단계 1: 데이터 전처리
- **파일**: `data/gold_data.ipynb`
- **기능**: 원본 데이터 로딩, 정제, 통합
- **출력**: `data/gold_data.csv` (통합 데이터셋)

#### 단계 2: 머신러닝 모델 개발
- **파일**: `predict_gold.ipynb`
- **기능**: 
  - 특성 엔지니어링 (래그 특성, 롤링 통계, 성장률)
  - 모델 학습 및 평가
  - 예측 결과 시각화
