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
SAVE_DIR = r"C:\ITWILL\SportsDrinkForecast\data\trained_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ Elasticsearch에서 데이터 가져오기 (최적화)
def fetch_es_data(index, start_date, end_date):
    query = {
        "size": 10000,
        "query": {
            "range": {
                "period": {
                    "gte": start_date,
                    "lte": end_date
                }
            }
        }
    }
    res = es.search(index=index, body=query)
    data = [hit["_source"] for hit in res["hits"]["hits"]]

    if not data:
        print(f"⚠️ {index} 인덱스에서 데이터를 찾을 수 없습니다. 날짜 범위를 확인하세요!")

    return pd.DataFrame(data)

# ✅ 학습용 데이터 불러오기
drink_df = fetch_es_data(sport_drink_index, "2024-01-01T00:00:00.000Z", "2024-12-31T23:59:59.999Z")
weather_df = fetch_es_data(weather_index, "2024-01-01T00:00:00.000Z", "2024-12-31T23:59:59.999Z")

# ✅ 데이터 전처리 (브랜드 포함)
def preprocess_data(drink_df, weather_df):
    for col in ["period"]:
        if col in drink_df.columns:
            drink_df["date"] = drink_df[col].apply(lambda x: x[0] if isinstance(x, list) else x)
        if col in weather_df.columns:
            weather_df["date"] = weather_df[col].apply(lambda x: x[0] if isinstance(x, list) else x)

    drink_df["date"] = pd.to_datetime(drink_df["date"], errors='coerce')
    weather_df["date"] = pd.to_datetime(weather_df["date"], errors='coerce')

    df = pd.merge(drink_df, weather_df, on="date", how="left")
    df.set_index("date", inplace=True)

    feature_cols = [col for col in df.columns if "ratios." in col] + ["temp_avg", "rainfall"]
    df = df[["brand", "age_group", "gender"] + feature_cols].dropna()

    return df, feature_cols

processed_df, feature_cols = preprocess_data(drink_df, weather_df)

# ✅ 특정 조건(Gatorade, 10대, 남성)만 필터링
filtered_df = processed_df[
    (processed_df["brand"] == "게토레이") &
    (processed_df["age_group"] == "10대") &
    (processed_df["gender"] == "male")
]

if filtered_df.empty:
    print("⚠️ 필터링된 데이터가 없습니다. 입력 조건을 다시 확인하세요.")
else:
    print(f"✅ 필터링된 데이터 개수: {len(filtered_df)}")

# ✅ LSTM 모델 학습 함수 (최적화)
def train_lstm_model(df, feature_cols, save_path):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[feature_cols])

    X, y = [], []
    seq_length = 15
    for i in range(len(df_scaled) - seq_length):
        X.append(df_scaled[i:i+seq_length])
        y.append(df_scaled[i+seq_length])

    X, y = np.array(X), np.array(y)

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

    with open(os.path.join(save_path, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    model.save(os.path.join(save_path, "lstm_model.h5"))

    print(f"✅ {save_path} 모델 및 스케일러 저장 완료!")

# ✅ 폴더명 변환 (한글을 로마자로 변환)
def sanitize_folder_name(name):
    name = unidecode(name)  # 한글을 로마자로 변환
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)  # 특수문자 제거
    return name

# ✅ 모델 학습 및 저장 (Gatorade, 10대, 남성 데이터만)
def train_and_save_model(df, feature_cols):
    if df.empty:
        print("⚠️ 학습할 데이터가 없습니다.")
        return

    brand_sanitized = sanitize_folder_name("gatorade")
    age_group_sanitized = sanitize_folder_name("10대")
    gender_sanitized = sanitize_folder_name("male")

    save_path = os.path.join(SAVE_DIR, f"{brand_sanitized}_{age_group_sanitized}_{gender_sanitized}")

    if not os.path.exists(save_path):
        print(f"📂 Creating directory: {save_path}")
        os.makedirs(save_path, exist_ok=True)

    print(f"🔹 Training model for Brand: Gatorade, Age Group: 10대, Gender: male")

    model_path = os.path.join(save_path, "lstm_model.h5")
    if os.path.exists(model_path):
        print(f"⚠️ 기존 파일 삭제: {model_path}")
        os.remove(model_path)

    train_lstm_model(df, feature_cols, save_path)

train_and_save_model(filtered_df, feature_cols)
