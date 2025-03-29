# -*- coding: utf-8 -*-
'''
기상 관측 데이터 활용 → 2024년 기상 데이터 정리 & Elasticsearch 저장
'''
import pandas as pd
import logging
from elasticsearch import Elasticsearch, helpers

# ✅ Elasticsearch 연결 설정
ES_HOST = "http://localhost:9200"
weather_index = "sports_drink_weather"

# ✅ 로깅 설정
# logging.basicConfig(filename="elasticsearch_upload.log", level=logging.INFO, 
                    # format="%(asctime)s - %(levelname)s - %(message)s")

def connect_elasticsearch():
    """Elasticsearch 연결 함수"""
    try:
        es = Elasticsearch(ES_HOST, request_timeout=30)  # ✅ 타임아웃 추가
        es.info()  # ✅ 클러스터 정보 요청 (더 확실한 연결 확인)
        logging.info("✅ Elasticsearch 연결 성공!")
        print("✅ Elasticsearch 연결 성공!")
        return es
    except Exception as e:
        logging.error(f"❌ Elasticsearch 연결 오류: {str(e)}")
        print(f"❌ Elasticsearch 연결 오류: {str(e)}")
        return None


def load_weather_data(file_path):
    """ CSV 파일에서 기상 데이터를 불러와 전처리하는 함수 """
    try:
        df = pd.read_csv(file_path)

        # ✅ 컬럼명 변경
        df = df.rename(columns={
            "period": "period",
            "temp_avg": "temp_avg",
            "rainfall": "rainfall"
        })[["period", "temp_avg", "rainfall"]]  # ✅ 필요 컬럼만 남김

        # ✅ 날짜 변환 (Period → 문자열 YYYY-MM-DD)
        df["period"] = pd.to_datetime(df["period"]).dt.strftime("%Y-%m-%d")

        # ✅ 결측값 처리
        df["rainfall"].fillna(0, inplace=True)  # ✅ 강수량 NaN → 0
        df["temp_avg"] = df["temp_avg"].astype(float)
        df["rainfall"] = df["rainfall"].astype(float)

        logging.info("✅ 기상 데이터 로드 및 전처리 완료")
        print("✅ 기상 데이터 로드 및 전처리 완료")
        return df
    except Exception as e:
        logging.error(f"❌ 데이터 로드 오류: {str(e)}")
        print(f"❌ 데이터 로드 오류: {str(e)}")
        return None

def upload_to_elasticsearch(es, df, index_name):
    """ Elasticsearch에 데이터를 업로드하는 함수 """
    try:
        records = df.to_dict(orient="records")
        actions = [{"_index": index_name, "_source": record} for record in records]

        helpers.bulk(es, actions)
        logging.info(f"✅ {index_name} 인덱스에 데이터 업로드 완료!")
        print(f"✅ {index_name} 인덱스에 데이터 업로드 완료!")
    except Exception as e:
        logging.error(f"❌ 데이터 업로드 오류: {str(e)}")
        print(f"❌ 데이터 업로드 오류: {str(e)}")

if __name__ == "__main__":
    # ✅ 파일 경로
    weather_file = r"C:\ITWILL\Final_project\data\기상관측_2024.csv"

    # ✅ Elasticsearch 연결
    es = connect_elasticsearch()

    if es:
        # ✅ 기상 데이터 로드
        df_weather = load_weather_data(weather_file)

        if df_weather is not None:
            # ✅ 데이터 업로드 실행
            upload_to_elasticsearch(es, df_weather, weather_index)
