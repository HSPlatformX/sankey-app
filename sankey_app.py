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
df['page'] = df['page'].astype(str).str.strip()

# 1. 세션 시작 조건 필터링 (step=1 포함 세션만 사용)
sessions_with_step1 = df[df['step'] == 1]['user_session_id'].unique()
df = df[df['user_session_id'].isin(sessions_with_step1)]

# 2. 정렬 및 세션 시작 플래그 지정
df = df.sort_values(['user_session_id', 'step'])
df['is_start'] = df['step'] == 1  # (여기서도 step은 원본 그대로 사용)

# 3. 경로 추출
session_paths = df.groupby('user_session_id')['page'].apply(list).reset_index()

# 4. 경로별 빈도 집계 (리스트 상태 유지)
path_counts = session_paths['page'].value_counts().reset_index()
path_counts.columns = ['path', 'value']

# ✅ pair 생성
def path_to_pairs(path, value):
    pairs = []
    for i in range(len(path) - 1):
        source = f"세션 시작" if i == 0 else f"{path[i]} ({i+1}단계)"
        target = f"{path[i+1]} ({i+2}단계)"
        pairs.append((source, target, value))
    return pairs

pairs = []
for _, row in path_counts.iterrows():
    pairs.extend(path_to_pairs(row['path'], row['value']))

pairs_df = pd.DataFrame(pairs, columns=['source', 'target', 'value'])

# ✅ pair 집계
pairs_agg = pairs_df.groupby(['source', 'target'])['value'].sum().reset_index()

# ✅ 노드 매핑 및 좌표 계산
all_nodes = pd.unique(pairs_agg[['source', 'target']].values.ravel())
node_map = {name: i for i, name in enumerate(all_nodes)}
pairs_agg['source_id'] = pairs_agg['source'].map(node_map)
pairs_agg['target_id'] = pairs_agg['target'].map(node_map)

def extract_step(label):
    if label == "세션 시작": return 0
    match = re.search(r"\((\d+)단계\)", label)
    return int(match.group(1)) if match else 0

depth_map = {node: extract_step(node) for node in all_nodes}
max_depth = max(depth_map.values()) if depth_map else 1
node_x = []
for label in node_map.keys():
    if label == "세션 시작":
        node_x.append(0.0)
    else:
        step = extract_step(label)
        node_x.append(step / max_depth if max_depth > 0 else 0.1)

# ✅ 마지막 노드만 (단계) 제거
def clean_label_for_last_node(label):
    if re.search(r'\(\d+단계\)', label) and '(1단계)' not in label:
        return re.sub(r'\s*\(\d+단계\)', '', label)
    return label

targets = set(pairs_agg['target'])
sources = set(pairs_agg['source'])
last_nodes = targets - sources

cleaned_labels = []
for label in node_map.keys():
    if label in last_nodes:
        cleaned_labels.append(clean_label_for_last_node(label))
    else:
        cleaned_labels.append(label)

# ✅ Sankey 시각화
fig = go.Figure(data=[go.Sankey(
    arrangement="fixed",
    node=dict(
        pad=20,
        thickness=30,
        label=list(cleaned_labels),
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
    font=dict(size=20),
    width=1200,
    height=1000,
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig, use_container_width=True)
