import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from google.oauth2 import service_account
from google.cloud import bigquery
import base64

# 페이지 설정
st.set_page_config(layout="wide")
st.title("🧭 Sankey Diagram")

# UI에서 카테고리 입력 받기
category_input = st.text_input('카테고리를 입력하세요:', '')  # 텍스트 입력란
category_select = st.selectbox('카테고리 선택', ['스탠바이미', '냉장고', '세탁기', 'TV'])  # 셀렉트박스

# 최종 카테고리 결정
if category_input:
    selected_category = category_input  # 텍스트 입력 값이 있으면 입력 값 사용
else:
    selected_category = category_select  # 텍스트 입력 값이 없으면 셀렉트박스 값 사용
    
st.markdown(f"### 🔍 선택된 카테고리: `{selected_category}`")


# 🌐 Streamlit secrets에서 인증 정보 가져오기
secrets = st.secrets["gcp_service_account"]

# base64로 인코딩된 private_key를 복원
private_key = base64.b64decode(secrets["private_key"]).decode()

# ✅ Credentials 생성
credentials = service_account.Credentials.from_service_account_info({
    "type": secrets["type"],
    "project_id": secrets["project_id"],
    "private_key_id": secrets["private_key_id"],
    "private_key": private_key,  # 이제 private_key를 그대로 사용
    "client_email": secrets["client_email"],
    "client_id": secrets["client_id"],
    "auth_uri": secrets["auth_uri"],
    "token_uri": secrets["token_uri"],
    "auth_provider_x509_cert_url": secrets["auth_provider_x509_cert_url"],
    "client_x509_cert_url": secrets["client_x509_cert_url"]
})

# 🚀 BigQuery 연결
client = bigquery.Client(credentials=credentials, project=secrets["project_id"])

# 쿼리 실행
query = """
       SELECT user_session_id, step, page
    FROM `lge-big-query-data.hsad.test_0423_2`
    ORDER BY user_session_id, step
"""
job_config = bigquery.QueryJobConfig(
    query_parameters=[bigquery.ScalarQueryParameter("category", "STRING", selected_category)]
)

df = client.query(query, job_config=job_config).to_dataframe()


# # 노드 인덱스 맵핑
# all_nodes = pd.unique(df[['source', 'target']].values.ravel())
# node_map = {name: i for i, name in enumerate(all_nodes)}
# df['source_id'] = df['source'].map(node_map)
# df['target_id'] = df['target'].map(node_map)

# # Sankey 다이어그램 그리기
# fig = go.Figure(data=[go.Sankey(
#     node=dict(
#         pad=15,
#         thickness=20,
#         label=list(node_map.keys()),
#         line=dict(color="black", width=0.5)
#     ),
#     link=dict(
#         source=df['source_id'],
#         target=df['target_id'],
#         value=df['value']
#     )
# )])

# fig.update_layout(title_text=f"Sankey for `{selected_category}`", font_size=11)


# 🛠️ 세션별 흐름 연결
pairs = []
for session_id, group in df.groupby('user_session_id'):
    pages = group.sort_values('step')['page'].tolist()
    for i in range(len(pages) - 1):
        pairs.append((pages[i], pages[i + 1]))

pairs_df = pd.DataFrame(pairs, columns=['source', 'target'])
pairs_agg = pairs_df.value_counts().reset_index(name='value')

# 🎯 Sankey 그리기
all_nodes = pd.unique(pairs_agg[['source', 'target']].values.ravel())
node_map = {name: i for i, name in enumerate(all_nodes)}
pairs_agg['source_id'] = pairs_agg['source'].map(node_map)
pairs_agg['target_id'] = pairs_agg['target'].map(node_map)

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        label=list(node_map.keys()),
        line=dict(color="black", width=0.5)
    ),
    link=dict(
        source=pairs_agg['source_id'],
        target=pairs_agg['target_id'],
        value=pairs_agg['value']
    )
)])
fig.update_layout(title_text=f"세션 기반 Sankey for `{selected_category}`", font_size=10)

# Streamlit에 그래프 출력
st.plotly_chart(fig, use_container_width=True)
