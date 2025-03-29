# -*- coding: utf-8 -*-
'''
과거 예측 데이터 비교 slack 연동
'''

import pandas as pd
import requests
import os
from dotenv import load_dotenv
from datetime import datetime

# ✅ 환경 변수 로드 (Slack Webhook URL)
load_dotenv(r"C:\ITWILL\Final_project\docker-elk\.env")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# ✅ 엑셀 파일 경로
excel_path = r"C:\ITWILL\Final_project\data\future_predictions_with_past_data.csv"

# ✅ Slack 메시지 전송 함수
def send_slack_message(message):
    payload = {"text": message}
    response = requests.post(SLACK_WEBHOOK_URL, json=payload)
    if response.status_code == 200:
        print("✅ Slack 알림 전송 완료!")
    else:
        print(f"❌ Slack 전송 실패: {response.status_code}, {response.text}")

# ✅ 엑셀 데이터 불러오기 및 비교
def compare_prediction_with_past():
    df = pd.read_csv(excel_path)
    alerts = []
    
    for _, row in df.iterrows():
        predicted_share = row.get("Predicted Share (%)", None)
        past_share = row.get("Past Share (%)", None)
        brand = row.get("brand", "Unknown Brand")
        gender = row.get("gender", "Unknown Gender")
        age_group = row.get("age_group", "Unknown Age Group")
        date = row.get("date", datetime.now().strftime("%Y-%m-%d"))
        
        if predicted_share is not None and past_share is not None:
            absolute_change = abs(predicted_share - past_share)
            
            if absolute_change >= 20:  # ✅ 20% 이상 변화 감지 시 알림
                alerts.append(
                    f"🚨 [{date}] {brand} ({gender}, {age_group}) 검색량이 {absolute_change:.2f}% 변화! (과거: {past_share:.2f}%, 예측: {predicted_share:.2f}%)"
                )
    
    if alerts:
        send_slack_message("\n".join(alerts))
    else:
        print("✅ 검색량 변화 없음!")

# ✅ 실행
if __name__ == "__main__":
    print("📌 예측 데이터와 과거 데이터 비교 중...")
    compare_prediction_with_past()
    print("✅ 모든 작업 완료!")
