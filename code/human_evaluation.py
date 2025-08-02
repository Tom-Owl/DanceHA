import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Annotator name input at the top
annotator_name = st.text_input("Annotator Name", key="annotator_name")

# Stop app if name not entered
if not annotator_name.strip():
    st.warning("Please enter your name to start annotation.")
    st.stop()
    
# Update stored annotations with timestamp
timestamp = datetime.now().isoformat()

# ====== Configuration ======
DATA_FILE = "../data/human_eval.csv"          # Input CSV file
SAVE_FILE = "annotations.csv"        # Where annotations are saved

# ====== Load Data ======
df_all_sample = pd.read_csv(DATA_FILE)
df_all_sample["golden_label_struct"] = df_all_sample["golden_label_struct"].apply(eval)



# ====== Initialize Session State ======
if 'doc_index' not in st.session_state:
    st.session_state.doc_index = 0

if 'annotations' not in st.session_state:
    # annotations is a DataFrame with columns: doc_index, item_index, score
    if os.path.exists(SAVE_FILE):
        st.session_state.annotations = pd.read_csv(SAVE_FILE).set_index(['doc_index', 'item_index'])['score'].to_dict()
    else:
        st.session_state.annotations = {}  # { (doc_index, item_index): score }

# ====== Navigation Buttons ======
col1, col2, col3 = st.columns([1, 1, 5])
with col1:
    if st.button("Previous"):
        st.session_state.doc_index = max(0, st.session_state.doc_index - 1)
with col3:
    if st.button("Next"):
        st.session_state.doc_index = min(len(df_all_sample) - 1, st.session_state.doc_index + 1)

# ====== Current Document ======
doc_idx = st.session_state.doc_index
row = df_all_sample.iloc[doc_idx]
input_doc = row['input_doc']
items = row['golden_label_struct']

# ====== Layout for Annotation ======
left, right = st.columns(2)

with left:
    st.markdown(f"### Document {doc_idx + 1}/{len(df_all_sample)}")
    st.text_area("Input Document", value=input_doc, height=400)

with right:
    st.markdown("### Annotations")
    for item_idx, item in enumerate(items):
        key = (doc_idx, item_idx)
        # prev_score = st.session_state.annotations.get(key, {}).get("score", 0)
        val = st.session_state.annotations.get(key, 0)
        prev_score = val if isinstance(val, int) else val.get("score", 0)

        st.markdown(f"**Item {item_idx + 1}**")
        for s in item["sentences"]:
            if s.lower() not in input_doc.lower():
                st.markdown(f'<span style="color:red">[Warning] Sentence not in input_doc: "{s}"</span>', unsafe_allow_html=True)
        st.json(item, expanded=True)  # Compact by default; change to True if you want full expansion

        score = st.radio(
            "Rating (1=Disagree, 5=Agree)",
            options=[0, 1, 2, 3, 4, 5],  # 0 = unannotated
            index=prev_score,
            key=f"radio_{doc_idx}_{item_idx}",
            horizontal=True
        )

        if score != prev_score:
            st.session_state.annotations[key] = score

            # Update stored annotations with timestamp
            timestamp = datetime.now().isoformat()
            st.session_state.annotations[key] = {
                "score": score,
                "timestamp": timestamp,
                "annotator": annotator_name.strip()
            }

            # Save all annotations to file
            df_save = pd.DataFrame([
                {
                    "doc_index": k[0],
                    "item_index": k[1],
                    "score": v["score"],
                    "timestamp": v["timestamp"],
                    "annotator": v["annotator"]
                }
                for k, v in st.session_state.annotations.items()
            ])
            df_save.to_csv(SAVE_FILE, index=False)

# ====== Progress Bar ======

st.progress(doc_idx / df_all_sample.shape[0])
st.caption(f"Annotated {doc_idx} out of {df_all_sample.shape[0]} documents")
