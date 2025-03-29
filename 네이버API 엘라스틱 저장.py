# -*- coding: utf-8 -*-
'''
네이버 검색 API 활용 → 이온음료 점유율 데이터 수집 & Elasticsearch 저장
'''
 
import urllib.request
import json
import csv
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from elasticsearch import Elasticsearch
from sklearn.preprocessing import MinMaxScaler


# Elasticsearch 클라이언트 설정 (필요시 사용)
es = Elasticsearch("http://localhost:9200")
index_name = "sports_drink_search"  # 저장할 인덱스 이름

# Elasticsearch 초기화 함수
def initialize_elasticsearch(index_name):
    if es.indices.exists(index=index_name):  # 인덱스 존재 여부 확인
        es.indices.delete(index=index_name)  # 인덱스 삭제
        print(f"기존 인덱스 '{index_name}'가 삭제되었습니다.")
    es.indices.create(index=index_name)  # 새 인덱스 생성
    print(f"새 인덱스 '{index_name}'가 생성되었습니다.")

# 인덱스 초기화 실행
# initialize_elasticsearch(index_name) # 초기화시 주석 해제

# .env 파일의 경로를 명확하게 지정
env_path = "C:\\ITWILL\\Final_project\\docker-elk\\.env"
load_dotenv(env_path)

# 환경 변수 가져오기
client_id = os.getenv("NAVER_CLIENT_ID")
client_secret = os.getenv("NAVER_CLIENT_SECRET")

if not client_id or not client_secret:
    raise ValueError("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")

# API URL (DataLab 검색 API 사용)
url = "https://openapi.naver.com/v1/datalab/search"

# 데이터 저장 경로
save_dir = "C:/ITWILL/Final_project/data"
os.makedirs(save_dir, exist_ok=True)
csv_file_path = f"{save_dir}/sports_drink_search.csv"
log_file_path = f"{save_dir}/sports_drink_search_log.txt"

# 현재 날짜 가져오기
today = datetime.now().strftime("%Y-%m-%d")

# 스포츠 음료 키워드 그룹
sports_drink = [
    {"groupName": "포카리스웨트", "keywords": ["포카리", "포카리스웨트", "포카리 스웨트", "Pocari Sweat", "POCARI SWEAT", "pocari sweat"]},
    {"groupName": "게토레이", "keywords": ["게토레이", "게토 레이", "Gatorade", "GATORADE", "gatorade"]},
    {"groupName": "파워에이드", "keywords": ["파워에이드", "파워 에이드", "Power Ade", "POWERADE", "powerade"]},
    {"groupName": "토레타", "keywords": ["토레타", "토레타!", "Toreta", "TORETA", "toreta"]},
    {"groupName": "링티", "keywords": ["링티", "Lingtea", "LINGTEA", "lingtea"]}
]

# 연령대별 매핑 (네이버 연령코드)
age_group_mapping = {
    "10대": ["2"],             # 13∼18세 (대체)
    "20대": ["3", "4"],         # 19∼24세, 25∼29세
    "30대": ["5", "6"],         # 30∼34세, 35∼39세
    "40대": ["7", "8"],         # 40∼44세, 45∼49세
    "50대": ["9", "10"],        # 50∼54세, 55∼59세
    "60대 이상": ["11"]         # 60세 이상
}

# 네이버 API에서 데이터 수집
def fetch_data(gender, ages):
    results = {}
    for batch in [sports_drink[i:i + 5] for i in range(0, len(sports_drink), 5)]:
        body = json.dumps({
            "startDate": "2024-01-01",
            "endDate": today,
            "timeUnit": "date",
            "keywordGroups": batch,
            "device": "",
            "ages": ages,
            "gender": gender
        })

        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)
        request.add_header("Content-Type", "application/json")

        try:
            response = urllib.request.urlopen(request, data=body.encode("utf-8"))
            if response.getcode() == 200:
                data = json.loads(response.read().decode('utf-8'))
                for group in data.get("results", []):
                    for entry in group.get("data", []):
                        period, ratio = entry["period"], entry["ratio"]
                        if period not in results:
                            results[period] = {item["groupName"]: 0 for item in sports_drink}
                        results[period][group["title"]] += ratio
        except Exception as e:
            print(f"❌ API 요청 오류 (Gender: {gender}, Ages: {ages}): {e}")
            continue
    return results

# 데이터 수집 및 정규화
def collect_and_normalize_data():
    aggregated = {"male": {}, "female": {}}
    for group_label, age_codes in age_group_mapping.items():
        aggregated["male"][group_label] = fetch_data("m", age_codes)
        aggregated["female"][group_label] = fetch_data("f", age_codes)

    for gender in aggregated:
        for age_group in aggregated[gender]:
            for period, group_ratios in aggregated[gender][age_group].items():
                total = sum(group_ratios.values())
                if total > 0:
                    aggregated[gender][age_group][period] = {k: round(v / total * 100, 2) for k, v in group_ratios.items()}
    return aggregated

# Elasticsearch 저장 함수 (브랜드 컬럼 추가)
def save_to_elasticsearch(index, period, gender, age_group, group_ratios):
    for brand, ratio in group_ratios.items():  # 브랜드별 데이터 저장
        doc = {
            "period": period,
            "gender": gender,
            "age_group": age_group,
            "brand": brand,
            "ratio": ratio,
            "timestamp": datetime.now().isoformat()
        }
        es.index(index=index, document=doc)

# CSV 저장 함수
def save_to_csv(aggregated_data):
    with open(csv_file_path, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["period", "gender", "age_group", "brand", "ratio"])
        for gender in aggregated_data:
            for age_group, data in aggregated_data[gender].items():
                for period, group_ratios in sorted(data.items()):
                    for brand, ratio in group_ratios.items():
                        writer.writerow([period, gender, age_group, brand, ratio])
    print(f"✅ CSV 저장 완료: {csv_file_path}")

# 로그 저장 함수
def save_to_log(aggregated_data):
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        for gender in aggregated_data:
            for age_group, data in aggregated_data[gender].items():
                for period, group_ratios in sorted(data.items()):
                    log_entry = f"{datetime.now().isoformat()} | Date: {period} | Gender: {gender} | Age Group: {age_group} | Data: {group_ratios}\n"
                    log_file.write(log_entry)
    print(f"✅ 로그 저장 완료: {log_file_path}")

# 실행 흐름
aggregated_data = collect_and_normalize_data()

for gender, age_groups in aggregated_data.items():  # 성별 단위로 반복
    for age_group, periods in age_groups.items():  # 연령대 단위로 반복
        for period, group_ratios in sorted(periods.items()):  # 날짜 단위로 반복
            save_to_elasticsearch(index_name, period, gender, age_group, group_ratios)

save_to_csv(aggregated_data)
save_to_log(aggregated_data)

print("\n✅ 모든 데이터 저장 완료!")
