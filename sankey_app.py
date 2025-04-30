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

# step=1이 존재하는 세션만 남긴다 (세션 시작 노드 구성 가능하도록)
sessions_with_step1 = df[df['step'] == 1]['user_session_id'].unique()
df = df[df['user_session_id'].isin(sessions_with_step1)]

#문자열 정리
df['page'] = df['page'].astype(str).str.strip()

# 주문완료 포함 세션만 유지 (이걸 자르기 전에 먼저 적용해야 함!)
sessions_with_purchase = df[df['page'] == '주문완료']['user_session_id'].unique()
df = df[df['user_session_id'].isin(sessions_with_purchase)]


# 구매완료 이후 단계는 제거하는 함수
def truncate_after_purchase(df):
    trimmed_rows = []

    for session_id, group in df.groupby('user_session_id'):
        group_sorted = group.sort_values('step')
        found = False
        for i, row in enumerate(group_sorted.itertuples()):
            if str(row.page).strip() == '주문완료':
                # ✅ 주문완료까지의 행만 남김
                trimmed = group_sorted.iloc[:i+1]
                trimmed_rows.append(trimmed)
                found = True
                break
        if not found:
            continue  # 주문완료 없는 세션은 제외

    # ✅ 모든 세션 결과 합치기
    return pd.concat(trimmed_rows).drop_duplicates().reset_index(drop=True)


# df에 적용
df = truncate_after_purchase(df)

# ✅ 주문완료 이후 단계가 남아있는 세션 확인용 디버깅 코드
after_purchase_rows = []

for session_id, group in df.groupby('user_session_id'):
    group_sorted = group.sort_values('step')
    found_purchase = False
    for row in group_sorted.itertuples():
        if found_purchase:
            after_purchase_rows.append(row)
        if str(row.page).strip() == '주문완료':  # 반드시 strip해서 비교
            found_purchase = True



# 마지막 step이 '주문완료'인 세션만 유지
last_steps = df.sort_values(['user_session_id', 'step']).groupby('user_session_id').tail(1)
valid_sessions = last_steps[last_steps['page'] == '주문완료']['user_session_id'].unique()
df = df[df['user_session_id'].isin(valid_sessions)]
df['step'] = df.groupby('user_session_id').cumcount() + 1 # 다시 step 재정의: truncate 후 step이 연속되도록 보장

# ✅ 세션 단위 흐름 리스트로 정렬
df_sorted = df.sort_values(['user_session_id', 'step'])
session_paths = df_sorted.groupby('user_session_id')['page'].apply(list).reset_index()

# ✅ 주문완료로 끝나는 세션만 유지
session_paths = session_paths[session_paths['page'].apply(lambda x: x[-1] == '주문완료')]

# ✅ 경로 문자열화 → 빈도수 집계
session_paths['path_str'] = session_paths['page'].apply(lambda x: ' > '.join(x))
path_counts = session_paths['path_str'].value_counts().reset_index()
path_counts.columns = ['path', 'value']

# 🛠️ 세션별 흐름 연결

pairs = []
for _, row in path_counts.iterrows():
    steps = row['path'].split(' > ')
    for i in range(len(steps) - 1):
        source = f"{steps[i]} ({i+1}단계)" if i > 0 else "세션 시작"
        target = f"{steps[i+1]} ({i+2}단계)"
        pairs.append((source, target, row['value']))

        
# ✅ 빈도수 집계
pairs_df = pd.DataFrame(pairs, columns=['source', 'target', 'value'])
pairs_agg = pairs_df.groupby(['source', 'target'])['value'].sum().reset_index()


# ✅ 초기 조건: 세션 시작 → X 중 value ≥ 5인 대상 노드만 seed로 사용
seed_edges = pairs_agg[
    (pairs_agg['source'] == '세션 시작') & (pairs_agg['value'] >= 5)
]

# 예외 처리: seed_edges가 비어있으면 바로 중단
if seed_edges.empty:
    st.warning("⚠️ '세션 시작'에서 시작하는 value ≥ 5 흐름이 없습니다. 데이터 조건을 확인하세요.")
    st.stop()

# ✅ Seed 노드 초기화
seed_targets = seed_edges['target'].unique()
valid_nodes = set(seed_targets) | {'세션 시작'}
visited_edges = set()

# ✅ 확장: value ≥ 5인 edge만 따라가며 valid 노드 확장
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

# ✅ 최종적으로 value ≥ 5인 유효 흐름만 필터링
pairs_agg = pairs_agg[
    pairs_agg.apply(lambda row: (row['source'], row['target']) in visited_edges, axis=1)
]

# 1. ✅ 노드 매핑
all_nodes = pd.unique(pairs_agg[['source', 'target']].values.ravel())
node_map = {name: i for i, name in enumerate(all_nodes)}

# 2. ✅ source/target 인덱스 매핑
pairs_agg['source_id'] = pairs_agg['source'].map(node_map)
pairs_agg['target_id'] = pairs_agg['target'].map(node_map)

# 3. ✅ node.x 수동 지정 (단계별로 좌표 계산)
# 단계 숫자 추출 (정규식 기반)
import re
def extract_step(label):
    if label == "세션 시작":
        return 0
    match = re.search(r"\((\d+)단계\)", label)
    return int(match.group(1)) if match else 0

# ✅ Sankey에 실제로 포함된 노드만 기반으로 depth 계산
valid_nodes_set = set(pairs_agg['source']).union(set(pairs_agg['target']))

# 🔧 실제 depth_map
depth_map = {}
for node in valid_nodes_set:
    if node == '세션 시작':
        depth_map[node] = 0
    else:
        step_num = extract_step(node)
        depth_map[node] = step_num

# 정규화
max_depth = max(depth_map.values()) if depth_map else 1
node_x = [depth_map.get(name, 0) / max_depth for name in node_map.keys()]

st.markdown("### ✅ 최종 포함된 연결 수")
st.write(len(pairs_agg))

# 🎯 Sankey 그리기
fig = go.Figure(data=[go.Sankey(
    arrangement="fixed",  # x 좌표 강제 적용
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

# Streamlit에 그래프 출력
st.plotly_chart(fig, use_container_width=True)
