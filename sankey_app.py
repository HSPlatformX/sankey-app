import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from google.oauth2 import service_account
from google.cloud import bigquery
import base64

# Sankey 다이어그램 타이틀
st.set_page_config(layout="wide")
st.title("\U0001F9ED Sankey Diagram")

# UI: 카테고리 입력 및 선택
category_input = st.text_input('카테고리를 입력하세요:', '')
category_select = st.selectbox('카테고리 선택', ['스탠바이미', '냉장고', '세탁기', 'TV'])
selected_category = category_input if category_input else category_select
st.markdown(f"### \U0001F50D 선택된 카테고리: `{selected_category}`")

# 인증 처리
secrets = st.secrets["gcp_service_account"]
private_key = base64.b64decode(secrets["private_key"]).decode()
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
    SELECT user_session_id, step, page
    FROM `lge-big-query-data.hsad.test_0423_2`
    WHERE category = @category
    ORDER BY user_session_id, step
"""
job_config = bigquery.QueryJobConfig(
    query_parameters=[bigquery.ScalarQueryParameter("category", "STRING", selected_category)]
)
df = client.query(query, job_config=job_config).to_dataframe()
df = df.dropna(subset=['user_session_id', 'step', 'page'])
df['page'] = df['page'].astype(str).str.strip().str.replace(r'\s+', '', regex=True)

# step 1은 '세션 시작'으로 치환
df.loc[df['step'] == 1, 'page'] = '세션 시작'

# 세션별로 트리밍

# 경로 추출
paths = df.sort_values(['user_session_id', 'step']).groupby('user_session_id')['page'].apply(list).reset_index()
paths['path_length'] = paths['page'].apply(len)
paths = paths.sort_values('path_length')

# Sankey용 노드, 링크 생성
from collections import defaultdict, Counter

all_nodes = set()
links = Counter()

for path in paths['page']:
    for i in range(len(path) - 1):
        src = path[i]
        tgt = path[i + 1]
        links[(src, tgt)] += 1
        all_nodes.add(src)
        all_nodes.add(tgt)

node_list = list(all_nodes)
node_index = {name: i for i, name in enumerate(node_list)}

source = [node_index[src] for (src, tgt) in links.keys()]
target = [node_index[tgt] for (src, tgt) in links.keys()]
value = list(links.values())

# Sankey Chart 생성
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=node_list,
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
    ))])

fig.update_layout(title_text="Sankey Diagram", font_size=12)
st.plotly_chart(fig, use_container_width=True)
