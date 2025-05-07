import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from google.oauth2 import service_account
from google.cloud import bigquery
import base64
import re

# 페이지 설정
st.set_page_config(layout="wide")
st.title("\U0001F9ED Sankey Diagram")

# UI에서 카테고리 입력 받기
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

# 데이터 쿼리
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

# ✅ 세션 시작 노드 조건
sessions_with_step1 = df[df['step'] == 1]['user_session_id'].unique()
df = df[df['user_session_id'].isin(sessions_with_step1)]


df = df.sort_values(['user_session_id', 'step']) 
df['step'] = df.groupby('user_session_id').cumcount() + 1

last_pages = df.groupby('user_session_id').tail(1)
st.write(last_pages['page'].value_counts())

# ✅ 세션별 page 리스트로 경로 생성
session_paths = df.groupby('user_session_id')['page'].apply(list).reset_index()
session_paths['path_str'] = session_paths['page'].apply(lambda x: ' > '.join(x))
path_counts = session_paths['path_str'].value_counts().reset_index()
path_counts.columns = ['path', 'value']

# ✅ pair 생성
def path_to_pairs(path_str, value):
    steps = path_str.split(' > ')
    pairs = []
    for i in range(len(steps) - 1):
        source = f"{steps[i]} ({i+1}단계)" if i > 0 else "세션 시작"
        target = f"{steps[i+1]} ({i+2}단계)"
        pairs.append((source, target, value))
    return pairs

pairs = []
for _, row in path_counts.iterrows():
    pairs.extend(path_to_pairs(row['path'], row['value']))

pairs_df = pd.DataFrame(pairs, columns=['source', 'target', 'value'])
pairs_agg = pairs_df.groupby(['source', 'target'])['value'].sum().reset_index()

# ✅ value ≥ 5 기준 BFS 필터링
seed_edges = pairs_agg[(pairs_agg['source'] == '세션 시작') & (pairs_agg['value'] >= 5)]
if seed_edges.empty:
    st.warning("⚠️ 시작 노드가 부족합니다.")
    st.stop()
seed_targets = seed_edges['target'].unique()
valid_nodes = set(seed_targets) | {'세션 시작'}
visited_edges = set()
expanded = True
while expanded:
    current_size = len(valid_nodes)
    valid_edges = pairs_agg[
        (pairs_agg['source'].isin(valid_nodes)) &
        (pairs_agg['value'] >= 5)
    ]
    for _, row in valid_edges.iterrows():
        visited_edges.add((row['source'], row['target']))
        valid_nodes.add(row['target'])
    expanded = len(valid_nodes) > current_size
pairs_agg = pairs_agg[
    pairs_agg.apply(lambda row: (row['source'], row['target']) in visited_edges, axis=1)
]

# ✅ 노드 매핑 및 좌표 계산
all_nodes = pd.unique(pairs_agg[['source', 'target']].values.ravel())
node_map = {name: i for i, name in enumerate(all_nodes)}
pairs_agg['source_id'] = pairs_agg['source'].map(node_map)
pairs_agg['target_id'] = pairs_agg['target'].map(node_map)

def extract_step(label):
    if label == "세션 시작": return 0
    match = re.search(r"\((\d+)단계\)", label)
    return int(match.group(1)) if match else 0

valid_nodes_set = set(pairs_agg['source']).union(set(pairs_agg['target']))
depth_map = {node: extract_step(node) for node in valid_nodes_set}
max_depth = max(depth_map.values()) if depth_map else 1
node_x = [depth_map.get(name, 0) / max_depth for name in node_map.keys()]

# ✅ Sankey 시각화
fig = go.Figure(data=[go.Sankey(
    arrangement="fixed",
    node=dict(
        pad=15,
        thickness=20,
        label=list(node_map.keys()),
        line=dict(color="black", width=0.5),
        x=node_x
    ),
    link=dict(
        source=pairs_agg['source_id'],
        target=pairs_agg['target_id'],
        value=pairs_agg['value']
    )
)])
fig.update_layout(
    title_text=f"세션 기반 Sankey for `{selected_category}`",
    font_size=10,
    margin=dict(l=0, r=0, t=40, b=0)
)
st.plotly_chart(fig, use_container_width=True)
