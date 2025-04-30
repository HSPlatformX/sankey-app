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
    WHERE category = @category
    ORDER BY user_session_id, step
"""
job_config = bigquery.QueryJobConfig(
    query_parameters=[bigquery.ScalarQueryParameter("category", "STRING", selected_category)]
)

df = client.query(query, job_config=job_config).to_dataframe()
df = df.dropna(subset=['user_session_id', 'step', 'page']) # ✅ 안정화: 필수 컬럼에 null 있으면 제거

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
    if 'step' in group.columns and 'page' in group.columns and len(group) >= 1:
        sorted_pages = group.sort_values('step')['page'].tolist()
        if len(sorted_pages) >= 1:
            pairs.append(("세션 시작", sorted_pages[0]))  # ✅ 세션 시작점 표시
        for i in range(len(sorted_pages) - 1):
            pairs.append((sorted_pages[i], sorted_pages[i + 1]))
        
# ✅ 빈도수 집계        
pairs_df = pd.DataFrame(pairs, columns=['source', 'target'])
pairs_agg = pairs_df.value_counts().reset_index(name='value')

# 1. ✅ 노드 매핑
all_nodes = pd.unique(pairs_agg[['source', 'target']].values.ravel())
node_map = {name: i for i, name in enumerate(all_nodes)}

# 2. ✅ source/target 인덱스 매핑
pairs_agg['source_id'] = pairs_agg['source'].map(node_map)
pairs_agg['target_id'] = pairs_agg['target'].map(node_map)

# 3. ✅ node_x 생성 (노드별 depth 기반 수평 위치 설정)
depth_map = {}

for session_id, group in df.groupby('user_session_id'):
    sorted_pages = group.sort_values('step')['page'].tolist()
    for idx, page in enumerate(sorted_pages):
        if page not in depth_map or depth_map[page] < idx:
            depth_map[page] = idx
    if sorted_pages:
        depth_map['세션 시작'] = 0  # 강제로 포함시켜줌

# 전체 depth를 0~1 범위로 정규화
max_depth = max(depth_map.values()) if depth_map else 1
node_x = [depth_map.get(name, 0) / max_depth for name in node_map.keys()]

# 🎯 Sankey 그리기
fig = go.Figure(data=[go.Sankey(
    arrangement="fixed"  # 좌표강제적용 (세션시작 고정)
    node=dict(
        pad=15,
        thickness=20,
        label=list(node_map.keys()),
        line=dict(color="black", width=0.5),
        x=node_x  # ✅ 추가된 수평 위치 적용
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

# Streamlit에 그래프 출력
st.plotly_chart(fig, use_container_width=True)
