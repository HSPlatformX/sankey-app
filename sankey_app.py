import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from google.oauth2 import service_account
from google.cloud import bigquery
import base64

# 페이지 설정
st.set_page_config(layout="wide")
st.title("🧭 실시간 Sankey 다이어그램")

# URL 파라미터에서 카테고리 선택값 받기
query_params = st.experimental_get_query_params()
selected_category = query_params.get("category", ["스탠바이미"])[0]
st.markdown(f"### 🔍 선택된 카테고리: `{selected_category}`")

# Secrets에서 base64로 인코딩된 키 불러와 디코딩
secrets = st.secrets["gcp_service_account"]
private_key = base64.b64decode(secrets["private_key_b64"]).decode()

credentials = service_account.Credentials.from_service_account_info({
    "type": secrets["type"],
    "project_id": secrets["project_id"],
    "private_key_id": secrets["private_key_id"],
    "private_key": private_key,
    "client_email": secrets["client_email"],
    "client_id": secrets["client_id"],
    "auth_uri": secrets["auth_uri"],
    "token_uri": secrets["token_uri"],
    "auth_provider_x509_cert_url": secrets["auth_provider_x509_cert_url"],
    "client_x509_cert_url": secrets["client_x509_cert_url"]
})

client = bigquery.Client(credentials=credentials, project=secrets["project_id"])
# 쿼리 실행
query = """
    SELECT source, target, value
    FROM `lge-big-query-data.hsad.test_0423_1`
    WHERE category = @category
"""
job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter("category", "STRING", selected_category)
    ]
)
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
st.plotly_chart(fig, use_container_width=True)
