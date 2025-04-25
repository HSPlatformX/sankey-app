import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from google.oauth2 import service_account
from google.cloud import bigquery
import base64
import json

# 페이지 설정
st.set_page_config(layout="wide")
st.title("🧭 실시간 Sankey 다이어그램")

# URL 파라미터에서 카테고리 선택값 받기
query_params = st.experimental_get_query_params()
selected_category = query_params.get("category", ["스탠바이미"])[0]
st.markdown(f"### 🔍 선택된 카테고리: `{selected_category}`")

# 🧩 Secrets에서 base64로 인코딩된 인증 정보 가져오기
secrets = st.secrets["gcp_service_account"]

# base64로 인코딩된 private_key를 복원
private_key_json = base64.b64decode(secrets["private_key"]).decode()

# Credentials 생성
credentials_dict = json.loads(private_key_json)
credentials = service_account.Credentials.from_service_account_info(credentials_dict)

# BigQuery 클라이언트 생성
client = bigquery.Client(credentials=credentials, project=secrets["project_id"])

# BigQuery 쿼리 실행
query = """
    SELECT source, target, value
    FROM `lge-big-query-data.hsad.test_0423_1`
    WHERE category = @category
"""
job_config = bigquery.QueryJobConfig(
    query_parameters=[bigquery.ScalarQueryParameter("category", "STRING", selected_category)]
)

# DataFrame에 쿼리 결과 담기
df = client.query(query, job_config=job_config).to_dataframe()

# 노드 인덱스 맵핑
all_nodes = pd.unique(df[['source', 'target']].values.ravel())
node_map = {name: i for i, name in enumerate(all_nodes)}
df['source_id'] = df['source'].map(node_map)
df['target_id'] = df['target'].map(node_map)

# Sankey 다이어그램 그리기
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        label=list(node_map.keys()),
        line=dict(color="black", width=0.5)
    ),
    link=dict(
        source=df['source_id'],
        target=df['target_id'],
        value=df['value']
    )
)])

fig.update_layout(title_text=f"Sankey for `{selected_category}`", font_size=11)

# Streamlit에 그래프 출력
st.plotly_chart(fig, use_container_width=True)
