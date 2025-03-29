# Elasticsearch 초기화 코드
from elasticsearch import Elasticsearch

# Elasticsearch 클라이언트 설정
es = Elasticsearch("http://localhost:9200")  # Elasticsearch의 URL
index_name = "sports_drink_search"  # 초기화할 인덱스 이름

# Elasticsearch 초기화 함수
def initialize_elasticsearch(index_name):
    if es.indices.exists(index=index_name):  # 인덱스 존재 여부 확인
        es.indices.delete(index=index_name)  # 인덱스 삭제
        print(f"기존 인덱스 '{index_name}'가 삭제되었습니다.")

# 인덱스 초기화 실행
initialize_elasticsearch(index_name)