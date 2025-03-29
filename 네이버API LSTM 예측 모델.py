# -*- coding: utf-8 -*-
'''
LSTM 예측 모델 → 브랜드, 성별, 연령대별로 검색 트렌드와 기상 데이터 기반 예측 모델 학습 및 저장
'''
import os
import re
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from elasticsearch import Elasticsearch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from unidecode import unidecode

# ✅ Elasticsearch 연결 설정
es = Elasticsearch("http://localhost:9200")
sport_drink_index = "sports_drink_search"
weather_index = "sports_drink_weather"

# ✅ 저장할 경로 설정
SAVE_DIR = r"C:\ITWILL\Final_project\data\trained_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 브랜드 변환 딕셔너리 (한글 → 영어)
brands_mapping = {
    "파워에이드": "powerade",
    "링티": "lingtea",
    "포카리스웨트": "pocarisweat",
    "게토레이": "gatorade",
    "토레타": "toreta"
}

# ✅ 브랜드명 변환 함수
def translate_brand_name(brand):
    return brands_mapping.get(brand, brand)  # 딕셔너리에 없으면 원래 값 반환

# ✅ Scroll API를 사용하여 25,000개 이상의 데이터를 가져오는 함수
def fetch_es_data_scroll(index, start_date, end_date, scroll_size=10000):
    query = {
        "size": scroll_size,
        "query": {
            "range": {
                "period": {
                    "gte": start_date,
                    "lte": end_date
                }
            }
        }
    }
    
    # Scroll API 초기 실행
    res = es.search(index=index, body=query, scroll="2m")
    scroll_id = res["_scroll_id"]
    data = [hit["_source"] for hit in res["hits"]["hits"]]

    while len(res["hits"]["hits"]) > 0:
        res = es.scroll(scroll_id=scroll_id, scroll="2m")
        scroll_id = res["_scroll_id"]
        data.extend([hit["_source"] for hit in res["hits"]["hits"]])
    
    # 스크롤 컨텍스트 삭제
    es.clear_scroll(scroll_id=scroll_id)

    if not data:
        print(f"⚠️ {index} 인덱스에서 데이터를 찾을 수 없습니다. 날짜 범위를 확인하세요!")
    
    return pd.DataFrame(data)

# ✅ 데이터 불러오기 (25,000개 이상도 가능)
drink_df = fetch_es_data_scroll(sport_drink_index, "2024-01-01T00:00:00.000Z", "2024-12-31T23:59:59.999Z")
weather_df = fetch_es_data_scroll(weather_index, "2024-01-01T00:00:00.000Z", "2024-12-31T23:59:59.999Z")

# ✅ 데이터 전처리
def preprocess_data(drink_df, weather_df):
    for col in ["period"]:
        if col in drink_df.columns:
            drink_df["date"] = drink_df[col].apply(lambda x: x[0] if isinstance(x, list) else x)
        if col in weather_df.columns:
            weather_df["date"] = weather_df[col].apply(lambda x: x[0] if isinstance(x, list) else x)

    drink_df["date"] = pd.to_datetime(drink_df["date"], errors='coerce')
    weather_df["date"] = pd.to_datetime(weather_df["date"], errors='coerce')

    # ✅ 브랜드명을 영어로 변환
    if "brand" in drink_df.columns:
        drink_df["brand"] = drink_df["brand"].apply(translate_brand_name)

    df = pd.merge(drink_df, weather_df, on="date", how="left")
    df.set_index("date", inplace=True)

    feature_cols = [col for col in df.columns if "ratios." in col] + ["temp_avg", "rainfall"]
    df = df[["brand", "age_group", "gender"] + feature_cols].dropna()

    return df, feature_cols

processed_df, feature_cols = preprocess_data(drink_df, weather_df)

# ✅ LSTM 모델 학습 함수 (덮어쓰기 기능 추가)
def train_lstm_model(df, feature_cols, save_path, seq_length=7):
    scaler_path = os.path.join(save_path, "scaler.pkl")
    model_path = os.path.join(save_path, "lstm_model.h5")

    # ✅ 기존 모델 및 스케일러 삭제 (덮어쓰기)
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"🔄 기존 모델 삭제: {model_path}")
    if os.path.exists(scaler_path):
        os.remove(scaler_path)
        print(f"🔄 기존 스케일러 삭제: {scaler_path}")

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[feature_cols])

    X, y = [], []
    for i in range(len(df_scaled) - seq_length):
        X.append(df_scaled[i:i+seq_length])
        y.append(df_scaled[i+seq_length])

    X, y = np.array(X), np.array(y)

    # ✅ 입력 크기 확인
    print(f"Training data shape: X={X.shape}, y={y.shape}")

    model = Sequential([
        LSTM(128, activation='relu', return_sequences=True, input_shape=(seq_length, len(feature_cols))),
        Dropout(0.3),
        LSTM(64, activation='relu'),
        Dropout(0.3),
        Dense(len(feature_cols))
    ])

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=100, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

    # ✅ 모델 및 스케일러 저장 (덮어쓰기)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    model.save(model_path)

    print(f"✅ {save_path} 모델 및 스케일러 저장 완료!")

# ✅ 폴더명 변환 (한글을 로마자로 변환)
def sanitize_folder_name(name):
    name = unidecode(name)  
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)  
    return name

# ✅ 학습 실행 (브랜드, 성별, 연령대별 저장)
def train_and_save_models(df, feature_cols, seq_length=7):
    grouped = df.groupby(["brand", "age_group", "gender"])

    for (brand, age_group, gender), group in grouped:
        brand_sanitized = sanitize_folder_name(brand)
        age_group_sanitized = sanitize_folder_name(age_group)
        gender_sanitized = sanitize_folder_name(gender)
        save_path = os.path.join(SAVE_DIR, f"{brand_sanitized}_{age_group_sanitized}_{gender_sanitized}")

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        print(f"🔹 Training model for Brand: {brand}, Age Group: {age_group}, Gender: {gender}")

        train_lstm_model(group, feature_cols, save_path, seq_length)

# ✅ 실행
train_and_save_models(processed_df, feature_cols, seq_length=7)

