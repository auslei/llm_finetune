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

# -- Sidebar: Data Preparation ----------------------------
st.sidebar.header("1. Data Preparation")
uploaded_file = st.sidebar.file_uploader(
    "Upload a .txt or .pdf file", type=["txt", "pdf"]
)
prep_mode = st.sidebar.selectbox("Mode", ["pretrain", "instruct"]);
chunk_size = st.sidebar.number_input("Chunk size", value=1024, step=1)
chunk_overlap = st.sidebar.number_input("Chunk overlap", value=200, step=1)
pdf_mode = st.sidebar.selectbox("PDF mode", ["simple", "columns"])
chunk_method = st.sidebar.selectbox("Chunk method", ["characters", "tokens"])
dedup = st.sidebar.checkbox("Deduplicate chunks", value=True)
encoding = st.sidebar.text_input("Encoding name", value="gpt2")
entity = st.sidebar.text_input("Entity name", value="Unknown")
doc_type = st.sidebar.text_input("Doc type", value="document")
# instruct-only params
max_q = st.sidebar.number_input("Max Q&A pairs", value=3, step=1)
delay = st.sidebar.number_input("Delay (s)", value=0.5, step=0.1, format="%.2f")

if uploaded_file:
    with open(f"app_input{Path(uploaded_file.name).suffix}", "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success(f"File saved: {uploaded_file.name}")

if st.sidebar.button("Prepare Data"):
    if not uploaded_file:
        st.sidebar.error("Please upload a file first.")
    else:
        input_path = Path(f"app_input{Path(uploaded_file.name).suffix}")
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
            st.download_button("Download JSONL", f, file_name="app_data.jsonl")
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
model_dir = st.sidebar.text_input(
    "Model directory", value=st.session_state.get("model_dir", "")
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