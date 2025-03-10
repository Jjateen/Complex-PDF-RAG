import os
from pathlib import Path
from typing import Optional

import streamlit as st
from streamlit_pandas_profiling import st_profile_report
from streamlit_pdf_viewer import pdf_viewer

from config import Config
from pdf_helper import PDFHelper, load_embedding_model

load_embedding_model(model_name=Config.EMBEDDING_MODEL_NAME)

title = "PDF Bot"
init_msg = "Hello, I'm your PDF assistant. Upload a PDF to get going."
model_name = Config.MODEL

pdfs_directory = os.path.join(str(Path.home()), 'langchain-store', 'uploads', 'pdfs')
os.makedirs(pdfs_directory, exist_ok=True)

print(f"Using model: {model_name}")
print(f"Using PDFs upload directory: {pdfs_directory}")

st.set_page_config(page_title=title, layout="wide")

def on_upload_change():
    print("File changed.")
    st.session_state.messages = [{"role": "assistant", "content": init_msg}]

def set_uploaded_file(_uploaded_file: str):
    st.session_state['uploaded_file'] = _uploaded_file

def get_uploaded_file() -> Optional[str]:
    if 'uploaded_file' in st.session_state:
        return st.session_state['uploaded_file']
    return None

with st.sidebar:
    st.title(title)
    st.write('This chatbot accepts a PDF file and lets you ask questions on it.')
    uploaded_file = st.file_uploader(
        label='Upload a PDF', type=['pdf', 'PDF'],
        accept_multiple_files=False,
        key='file-uploader',
        on_change=on_upload_change
    )

    if uploaded_file is not None:
        added = False
        my_msg = f"Great! Now, what do you want from `{uploaded_file.name}`?"
        for msg in st.session_state.messages:
            if msg["content"] == my_msg:
                added = True
        if not added:
            st.session_state.messages.append({"role": "assistant", "content": my_msg})
        bytes_data = uploaded_file.getvalue()
        target_file = os.path.join(pdfs_directory, uploaded_file.name)
        set_uploaded_file(target_file)
        with open(target_file, 'wb') as f:
            f.write(bytes_data)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": init_msg}]

col2, col3 = st.columns([3, 2])

with col2:
    st.header("View PDF")
    if uploaded_file is not None:
        pdf_viewer(os.path.join(pdfs_directory, uploaded_file.name), height=800)
    else:
        st.write("No PDF uploaded")

with col3:
    st.header("Chat")
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div style='text-align: right;'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left;'>{message['content']}</div>", unsafe_allow_html=True)

    def clear_chat_history():
        from streamlit_js_eval import streamlit_js_eval
        streamlit_js_eval(js_expressions="parent.window.location.reload()")
        st.session_state.messages = [{"role": "assistant", "content": init_msg}]

    st.button('Reset', on_click=clear_chat_history)

    if prompt := st.text_input("What do you want to know from the uploaded PDF?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div style='text-align: right;'>{prompt}</div>", unsafe_allow_html=True)

        if st.session_state.messages[-1]["role"] != "assistant":
            source_file = get_uploaded_file()
            if source_file is None:
                full_response = 'PDF file needs to be uploaded before you can ask questions on it 😟. Please upload a file.'
                st.markdown(f"<div style='text-align: left;'>{full_response}</div>", unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                pdf_helper = PDFHelper(model_name=model_name)
                response = pdf_helper.ask(pdf_file_path=source_file, question=prompt)
                full_response = response
                st.markdown(f"<div style='text-align: left;'>{full_response}</div>", unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
