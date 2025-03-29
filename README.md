# SportsDrink Forecast:  
네이버 API 검색량과 날씨 데이터를 기반으로 브랜드 점유율을 예측하는 시계열 분석 프로젝트

📘 **예측 시스템 설명서 보기** → [sportsdrink-forecast.pdf](https://github.com/user-attachments/files/19523217/sportsdrink-forecast.pdf)


## 프로젝트 개요

SportsDrink Forecast는 네이버 검색량과 날씨 데이터를 기반으로 스포츠 음료 브랜드의 점유율을 예측하는 시계열 분석 프로젝트입니다.  
외부 API로 수집한 데이터를 전처리하고, LSTM 모델을 통해 브랜드별 검색량을 예측한 후, Kibana 및 Tableau를 활용해 결과를 시각화합니다.  
Slack 알림 기능을 통해 예측 변화 감지도 자동화하여 실시간 대응이 가능하도록 구성했습니다.

## 주요 목표

- 네이버 검색량과 기상 데이터를 활용해 브랜드별 검색 트렌드 분석
- LSTM 기반의 시계열 예측 모델을 통해 점유율 예측
- 예측 결과를 실시간으로 시각화하고 이상 탐지 시 Slack으로 알림 전송
- Docker 기반 분석 환경 구성

## 주요 기술 스택

| 범주         | 기술 내용 |
|--------------|-----------|
| 데이터 수집   | Naver Datalab API, 기상청 Open API |
| 데이터 처리   | Python, Pandas |
| 모델링       | Keras (LSTM) |
| 저장 및 분석  | Elasticsearch |
| 시각화       | Kibana, Tableau |
| 자동화       | Docker, Slack Webhook |

## 구현 내용

### 데이터 수집

- Naver Datalab API를 활용해 성별, 연령대, 브랜드별 검색량 데이터를 수집
- 기상청 Open API를 이용해 기온, 강수량 등 주요 날씨 데이터를 함께 수집
- 수집된 데이터를 Elasticsearch에 저장

### 데이터 전처리

- 날짜 정렬 및 결측치 처리
- 평균 기온, 강수량, 요일, 시차 변수 등 시계열 분석을 위한 feature 구성
- 브랜드, 성별, 연령대 조합 단위로 데이터 정리

### 시계열 예측 모델링 (LSTM)

- Keras 기반 LSTM 모델로 브랜드별 검색량을 예측
- 주요 설정: LSTM(128, 64), Dropout(0.3), Optimizer(Adam), Epoch(100)
- 예측 대상: 향후 7일간의 검색량 변화
- 모델 및 Scaler를 파일로 저장하여 재사용

### 예측 결과 시각화

- Kibana를 통해 실시간 Elasticsearch 데이터를 시각화
- Tableau로 시계열 트렌드, 기상 변수와의 상관관계 등을 시각화

### 알림 시스템

- 예측값 변화율이 특정 기준을 초과하면 Slack Webhook을 통해 알림 전송
- 이상 탐지를 통해 마케팅이나 운영팀이 빠르게 대응 가능

## 프로젝트 성과

- 검색량과 날씨 변수 간 상관관계를 시각적으로 분석
- LSTM 기반 예측 모델이 안정적인 성능(RMSE 기준)을 달성
- Kibana 및 Elasticsearch 기반 예측 결과 모니터링 체계 구축
- Slack을 통한 예측 결과 이상 변화 자동 감지 및 알림 전송
- Tableau를 통해 다각도 시계열 인사이트 시각화 성공

## 팀원

고정진, 백영현, 조소현, 천고은, 최덕진, 최성준
