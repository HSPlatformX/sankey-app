...

# ✅ 라벨에서 마지막 노드만 (단계) 제거
def clean_label_for_last_node(label):
    if re.search(r'\(\d+단계\)', label) and '(1단계)' not in label:
        return re.sub(r'\s*\(\d+단계\)', '', label)
    return label

# 마지막 노드 식별을 위해 target에만 있고 source에 없는 노드 식별
targets = set(pairs_agg['target'])
sources = set(pairs_agg['source'])
last_nodes = targets - sources

# 노드 라벨 가공
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
        label=cleaned_labels,
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
    height=1500,
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig, use_container_width=True)
