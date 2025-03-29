import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ✅ 미래 날씨 데이터 로드
future_weather_file = r"C:\ITWILL\SportsDrinkForecast\data\future_weather_forecast.csv"
future_weather_df = pd.read_csv(future_weather_file)

# 날짜 변환
future_weather_df["date"] = pd.to_datetime(future_weather_df["period"])
future_weather_df.drop(columns=["period"], inplace=True)

# ✅ 브랜드 변환 딕셔너리 (한글 -> 영어)
brands_mapping = {
    "파워에이드": "powerade",
    "링티": "lingtea",
    "포카리스웨트": "pocarisweat",
    "게토레이": "gatorade",
    "토레타": "toreta"
}

# ✅ 10대 남성 대상 브랜드 리스트
brands = {k: v for k, v in brands_mapping.items()}

# ✅ 모델 및 스케일러 로드 함수
def load_model_and_scaler(brand_key):
    model_path = rf"C:\ITWILL\SportsDrinkForecast\data\trained_models\{brand_key}_10dae_male\lstm_model.h5"
    scaler_path = rf"C:\ITWILL\SportsDrinkForecast\data\trained_models\{brand_key}_10dae_male\scaler.pkl"
    
    try:
        model = tf.keras.models.load_model(model_path)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        print(f"❌ 모델 또는 스케일러 로드 실패: {brand_key} - {e}")
        return None, None

# ✅ 예측 실행
predictions = []

for brand_name, brand_key in brands.items():
    model, scaler = load_model_and_scaler(brand_key)

    if model is None or scaler is None:
        continue

    # ✅ 모델 입력 데이터 변환
    scaled_features = scaler.transform(future_weather_df[["temp_avg", "rainfall"]])
    X_test = scaled_features.reshape(1, 7, 2)  # (1, 7, 2)로 변환

    # ✅ 예측 수행
    predicted_ratios = model.predict(X_test).flatten()

    # ✅ 예측 값 개수를 7개로 맞추기
    if len(predicted_ratios) < len(future_weather_df):
        predicted_ratios = np.pad(predicted_ratios, (0, len(future_weather_df) - len(predicted_ratios)), mode='edge')
    elif len(predicted_ratios) > len(future_weather_df):
        predicted_ratios = predicted_ratios[:len(future_weather_df)]

    # ✅ 결과 저장
    result_df = future_weather_df.copy()
    result_df["brand"] = brand_name
    result_df["age_group"] = "10대"
    result_df["gender"] = "male"
    result_df["predicted_ratio"] = predicted_ratios

    predictions.append(result_df)

# ✅ 최종 결과 데이터프레임 생성
predicted_df_teen_male = pd.concat(predictions, ignore_index=True)

# ✅ 날짜별 100% 맞추기
predicted_df_teen_male["total_ratio"] = predicted_df_teen_male.groupby("date")["predicted_ratio"].transform("sum")
predicted_df_teen_male["adjusted_ratio"] = predicted_df_teen_male["predicted_ratio"] / predicted_df_teen_male["total_ratio"]

# ✅ 퍼센트 변환
predicted_df_teen_male["adjusted_ratio"] = (predicted_df_teen_male["adjusted_ratio"] * 100).round(2).astype(str) + "%"

# ✅ 불필요한 컬럼 제거 후 정리
predicted_df_final = predicted_df_teen_male.drop(columns=["predicted_ratio", "total_ratio"])
predicted_df_final.rename(columns={"adjusted_ratio": "predicted_ratio"}, inplace=True)

# ✅ 날짜별, 브랜드별 정렬
predicted_df_sorted = predicted_df_final.sort_values(by=["date", "brand"])

# ✅ 날짜 형식을 YYYY-MM-DD로 변환
predicted_df_sorted["date"] = predicted_df_sorted["date"].dt.strftime('%Y-%m-%d')

# ✅ 성별을 한글로 변환
predicted_df_sorted["gender"] = predicted_df_sorted["gender"].replace({"male": "남성", "female": "여성"})

# ✅ 출력 포맷 설정
print(f"{'날짜':<12} | {'브랜드':<15} | {'나이':<6} | {'성별':<4} | {'예측 점유율':<10}")
print("-" * 60)

# ✅ 데이터 출력
for index, row in predicted_df_sorted.iterrows():
    print(f"{row['date']:<12} | {row['brand']:<15} | {row['age_group']:<6} | {row['gender']:<4} | {row['predicted_ratio']:<10}")

# ✅ 결과 저장 (필요 시 사용)
# predicted_df_sorted.to_csv(r"C:\ITWILL\SportsDrinkForecast\data\future_predictions.csv", index=False)

# ✅ 모델 입력 형태 확인
if model:
    print(f"모델이 기대하는 입력 형태: {model.input_shape}")
