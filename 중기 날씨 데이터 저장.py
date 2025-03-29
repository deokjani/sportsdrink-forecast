# -*- coding: utf-8 -*-
'''
기상청 중기예보 활용 → 4~10일 기온 및 강수량 예측 데이터를 CSV로 저장
'''
import os
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import unquote
from dotenv import load_dotenv

# ✅ 현재 실행 중인 파일 기준으로 .env 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # py/data_save → py → Final_project 이동
ENV_PATH = os.path.join(BASE_DIR, "docker-elk", ".env")

# ✅ 환경 변수 로드
load_dotenv(ENV_PATH)

# ✅ 기상청 API URL
BASE_URL_TA = "http://apis.data.go.kr/1360000/MidFcstInfoService/getMidTa"  # 중기기온조회
BASE_URL_LAND = "http://apis.data.go.kr/1360000/MidFcstInfoService/getMidLandFcst"  # 중기육상예보조회

# ✅ API 인증키 (환경 변수에서 로드 후 디코딩)
SERVICE_KEY = unquote(os.getenv("SERVICE_KEY"))

# ✅ 현재 날짜
now = datetime.now()
tmFc = now.strftime("%Y%m%d") + "0600"  # 예보 기준 시간 (06시 기준)

# ✅ 서울 지역 코드 (기온 & 강수 확률)
REG_ID_TA = "11B10101"  # 기온 데이터
REG_ID_LAND = "11B00000"  # 강수량 데이터

# ✅ 요청 파라미터 설정
params_ta = {
    "serviceKey": SERVICE_KEY,
    "numOfRows": 10,
    "pageNo": 1,
    "dataType": "JSON",
    "regId": REG_ID_TA,
    "tmFc": tmFc
}

params_land = {
    "serviceKey": SERVICE_KEY,
    "numOfRows": 10,
    "pageNo": 1,
    "dataType": "JSON",
    "regId": REG_ID_LAND,
    "tmFc": tmFc
}

# ✅ API 요청
response_ta = requests.get(BASE_URL_TA, params=params_ta)
response_land = requests.get(BASE_URL_LAND, params=params_land)

# ✅ 예보 데이터 저장할 딕셔너리
forecast_data = {}

# ✅ 기온 데이터 파싱
if response_ta.status_code == 200:
    try:
        data_ta = response_ta.json()
        if "response" in data_ta and "body" in data_ta["response"]:
            items = data_ta["response"]["body"]["items"]["item"]
            for item in items:
                for i in range(4, 11):  # 4~10일 데이터 추출
                    date_key = (now + timedelta(days=i)).strftime("%Y-%m-%d")
                    forecast_data.setdefault(date_key, {})
                    forecast_data[date_key]["temp_avg"] = (item[f'taMin{i}'] + item[f'taMax{i}']) / 2
    except json.JSONDecodeError:
        print(f"❌ JSON 변환 실패 (기온 데이터)\n{response_ta.text}")

# ✅ 강수량 데이터 파싱
if response_land.status_code == 200:
    try:
        data_land = response_land.json()
        if "response" in data_land and "body" in data_land["response"]:
            items = data_land["response"]["body"]["items"]["item"]
            for item in items:
                for i in range(4, 11):
                    date_key = (now + timedelta(days=i)).strftime("%Y-%m-%d")
                    am_key = f'rnSt{i}Am' if i < 8 else f'rnSt{i}'
                    pm_key = f'rnSt{i}Pm' if i < 8 else f'rnSt{i}'

                    rain_values = []
                    if am_key in item and item[am_key] is not None:
                        rain_values.append(item[am_key])
                    if pm_key in item and item[pm_key] is not None:
                        rain_values.append(item[pm_key])

                    rainfall = sum(rain_values) / len(rain_values) if rain_values else "N/A"
                    forecast_data.setdefault(date_key, {})
                    forecast_data[date_key]["rainfall"] = rainfall
    except json.JSONDecodeError:
        print(f"❌ JSON 변환 실패 (강수량 데이터)\n{response_land.text}")

# ✅ CSV로 저장
df_forecast = pd.DataFrame.from_dict(forecast_data, orient="index").reset_index()
df_forecast.rename(columns={"index": "period"}, inplace=True)
save_path = r"C:\ITWILL\Final_project\data\future_weather_forecast.csv"
df_forecast.to_csv(save_path, index=False)

print(f"✅ 중기예보 데이터 저장 완료! ({save_path})")
