import os
from pathlib import Path

import streamlit as st
import torch

from llm_finetune.data_prep_tools import (
    read_and_chunk_document,
    make_pretrain_data,
    make_instruct_data,
)
from llm_finetune.finetune_tool import DocumentFineTune
from unsloth import FastLanguageModel

# Page configuration and title
st.set_page_config(
    page_title="LLM Finetune Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("🤖 LLM Finetune & Chat Dashboard")
st.markdown(
    "Use the sidebar to prepare data, train models, or chat with a trained model."
)

# Sidebar controls grouped in expanders and forms
with st.sidebar.expander("1️⃣ Data Preparation", expanded=True):
    with st.form(key="prep_form", clear_on_submit=False):
        uploaded_file = st.file_uploader("Upload .txt or .pdf file", type=["txt", "pdf"])
        prep_mode = st.selectbox("Mode", ["pretrain", "instruct"]);
        chunk_size = st.number_input("Chunk size", value=1024, step=1)
        chunk_overlap = st.number_input("Chunk overlap", value=200, step=1)
        pdf_mode = st.selectbox("PDF mode", ["simple", "columns"])
        chunk_method = st.selectbox("Chunk method", ["characters", "tokens"])
        dedup = st.checkbox("Deduplicate chunks", value=True)
        encoding = st.text_input("Encoding name", value="gpt2")
        entity = st.text_input("Entity name", value="Unknown")
        doc_type = st.text_input("Doc type", value="document")
        max_q = st.number_input("Max Q&A pairs", value=3, step=1)
        delay = st.number_input("Delay (s)", value=0.5, step=0.1, format="%.2f")
        prep_submit = st.form_submit_button(label="Prepare Data")
    if prep_submit:
        if not uploaded_file:
            st.error("Please upload a file first.")
        else:
            input_path = Path(f"app_input{Path(uploaded_file.name).suffix}")
            input_path.write_bytes(uploaded_file.getvalue())
            with st.spinner("Chunking and preparing data..."):
                chunks = read_and_chunk_document(
                    input_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    mode=pdf_mode,
                    chunk_method=chunk_method,
                    dedup=dedup,
                    encoding_name=encoding,
                )
            st.success(f"Generated {len(chunks)} chunks.")
            st.write(chunks[:5])
            out_file = Path("app_data.jsonl")
            if prep_mode == "pretrain":
                make_pretrain_data(chunks, out_file, entity=entity, doc_type=doc_type)
            else:
                make_instruct_data(
                    chunks,
                    out_file,
                    max_q=max_q,
                    delay=delay,
                    entity=entity,
                    doc_type=doc_type,
                )
            st.success(f"Prepared data saved to {out_file}")
            with open(out_file, "rb") as f:
                st.download_button("📥 Download JSONL", f, file_name="app_data.jsonl")
            st.session_state["data_path"] = str(out_file)

# -- Sidebar: Training -------------------------------------
st.sidebar.header("2. Train Model")
data_path = st.sidebar.text_input(
    "Prepared data path", value=st.session_state.get("data_path", "")
)
base_model = st.sidebar.text_input(
    "Base model", value="unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
)
model_name = st.sidebar.text_input(
    "Model name", value="app_model"
)
epochs = st.sidebar.number_input("Epochs", min_value=1, value=1)

if st.sidebar.button("Train"):
    if not data_path:
        st.sidebar.error("Set the prepared data path first.")
    else:
        with st.spinner("Training model..."):
            tuner = DocumentFineTune(
                training_data_path=data_path,
                model_name=model_name,
                base_model_path=base_model,
                max_seq_length=chunk_size,
                training_mode=prep_mode,
            )
            tuner.train(num_train_epochs=epochs)
        st.success(f"Training complete. Weights at models/{model_name}")
        st.session_state["model_dir"] = f"models/{model_name}"

# -- Sidebar: Chat ------------------------------------------
st.sidebar.header("3. Chat with Model")
# List trained model directories under models/
model_dirs = [str(p) for p in Path("models").iterdir() if p.is_dir()]
if not model_dirs:
    st.sidebar.warning("No trained models found in 'models/' directory.")
selected = st.session_state.get("model_dir", "")
model_dir = st.sidebar.selectbox(
    "Model directory",
    options=model_dirs,
    index=model_dirs.index(selected) if selected in model_dirs else 0,
)
user_input = st.sidebar.text_input("Your message:")

@st.cache_resource
def load_chat_model(path, max_len):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=max_len,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

if st.sidebar.button("Send"):
    if model_dir and user_input:
        model, tokenizer = load_chat_model(model_dir, chunk_size)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_input}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        outputs = model.generate(
            inputs,
            max_new_tokens=256,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        resp = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        history = st.session_state.get("history", [])
        history.append((user_input, resp))
        st.session_state["history"] = history
    else:
        st.sidebar.error("Set model_dir and enter a message.")

st.header("Chat History")
for usr, bot in st.session_state.get("history", []):
    st.markdown(f"**You:** {usr}")
    st.markdown(f"**Bot:** {bot}")