# -*- coding: utf-8 -*-
'''
과거 데이터 + LSTM 모델을 활용한 미래 검색 점유율 예측
'''

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.losses import mean_squared_error
from keras.saving import register_keras_serializable

# ✅ 1. 미래 날씨 데이터 로드
future_weather_file = r"C:\ITWILL\Final_project\data\future_weather_forecast.csv"
future_weather_df = pd.read_csv(future_weather_file)

# ✅ 2. 과거 판매 데이터 로드
past_sales_file = r"C:\ITWILL\Final_project\data\sports_drink_search.csv"
past_sales_df = pd.read_csv(past_sales_file)

# ✅ 3. 날짜 변환
future_weather_df.rename(columns={"period": "date"}, inplace=True)
future_weather_df["date"] = pd.to_datetime(future_weather_df["date"])  # 미래 날짜 변환

past_sales_df.rename(columns={"period": "date", "ratio": "Past Share (%)"}, inplace=True)
past_sales_df["date"] = pd.to_datetime(past_sales_df["date"])  # 과거 날짜 변환

# ✅ 4. 과거 데이터 연도 +1 적용 (미래와 비교 가능하도록)
past_sales_df["date"] = past_sales_df["date"] + pd.DateOffset(years=1)

# ✅ 5. 미래 데이터와 동일한 날짜만 유지
valid_dates = future_weather_df["date"].unique()
past_sales_df = past_sales_df[past_sales_df["date"].isin(valid_dates)]

# ✅ 6. 연령대 및 성별 리스트 (출력용 변환)
age_groups = {
    "10dae": "10대", "20dae": "20대", "30dae": "30대",
    "40dae": "40대", "50dae": "50대", "60dae_isang": "60대 이상"
}
genders = {"male": "남성", "female": "여성"}
brands = {
    "powerade": "파워에이드",
    "lingtea": "링티",
    "pocarisweat": "포카리스웨트",
    "gatorade": "게토레이",
    "toreta": "토레타"
}

# ✅ 7. 과거 데이터에서 성별 변환 (영어 → 한글)
past_sales_df["gender"] = past_sales_df["gender"].map(genders)

# ✅ 8. LSTM 모델 및 스케일러 로드 함수
@register_keras_serializable()
def custom_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def load_model_and_scaler(brand_key, age_group_key, gender_key):
    model_path = rf"C:\ITWILL\Final_project\data\trained_models\{brand_key}_{age_group_key}_{gender_key}\lstm_model.h5"
    scaler_path = rf"C:\ITWILL\Final_project\data\trained_models\{brand_key}_{age_group_key}_{gender_key}\scaler.pkl"

    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'custom_mse': custom_mse, 'mse': custom_mse})
        print(f"✅ 모델 로드 성공: {brand_key} - {age_group_key} - {gender_key}")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        print(f"❌ 모델 또는 스케일러 로드 실패: {brand_key} - {age_group_key} - {gender_key} - {e}")
        return None, None

# ✅ 9. 예측 실행
predictions = []

for age_group_key, age_group_name in age_groups.items():
    for gender_key, gender_name in genders.items():
        for brand_key, brand_name in brands.items():
            model, scaler = load_model_and_scaler(brand_key, age_group_key, gender_key)

            if model is None or scaler is None:
                continue

            # ✅ 모델 입력 데이터 변환
            scaled_features = scaler.transform(future_weather_df[["temp_avg", "rainfall"]])
            X_test = scaled_features.reshape(1, len(future_weather_df), 2)

            # ✅ 예측 수행
            predicted_ratios = model.predict(X_test).flatten()

            # ✅ 개수 맞추기
            if len(predicted_ratios) < len(future_weather_df):
                predicted_ratios = np.pad(predicted_ratios, (0, len(future_weather_df) - len(predicted_ratios)), mode='edge')
            elif len(predicted_ratios) > len(future_weather_df):
                predicted_ratios = predicted_ratios[:len(future_weather_df)]

            # ✅ 음수 값 제거
            predicted_ratios = np.clip(predicted_ratios, 0, None)

            # ✅ 결과 저장
            result_df = future_weather_df.copy()
            result_df["brand"] = brand_name
            result_df["age_group"] = age_group_name
            result_df["gender"] = gender_name
            result_df["Predicted Share (%)"] = np.round(predicted_ratios, 2)

            predictions.append(result_df)

# ✅ 10. 예측 데이터 정리
predicted_df = pd.concat(predictions, ignore_index=True)

# ✅ 11. 날짜별 100% 맞추기
predicted_df["Predicted Share (%)"] = predicted_df.groupby(["date", "age_group", "gender"])["Predicted Share (%)"].transform(lambda x: np.round((x / x.sum()) * 100, 2))

# ✅ 12. 과거 데이터와 병합
combined_df = predicted_df.merge(
    past_sales_df,
    on=["date", "brand", "age_group", "gender"],
    how="left"
)

# ✅ 13. 정렬 (날짜별, 연령대별, 브랜드별 정렬)
combined_df = combined_df.sort_values(by=["date", "age_group", "gender", "brand"])

# ✅ 14. 최종 데이터 저장
output_file = r"C:\ITWILL\Final_project\data\future_predictions_with_past_data.csv"
combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✅ 최종 결과 저장 완료: {output_file}")

