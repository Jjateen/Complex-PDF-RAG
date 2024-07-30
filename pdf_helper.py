import os
import time
import uuid
from pathlib import Path

import langchain
import langchain_core
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from config import Config

def load_embedding_model(model_name, normalize_embedding=True):
    print("Loading embedding model...")
    start_time = time.time()
    hugging_face_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': Config.HUGGING_FACE_EMBEDDINGS_DEVICE_TYPE},
        encode_kwargs={'normalize_embeddings': normalize_embedding}
    )
    end_time = time.time()
    time_taken = round(end_time - start_time, 2)
    print(f"Embedding model load time: {time_taken} seconds.\n")
    return hugging_face_embeddings

def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    print("Creating embeddings...")
    e_start_time = time.time()
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    e_end_time = time.time()
    e_time_taken = round(e_end_time - e_start_time, 2)
    print(f"Embeddings creation time: {e_time_taken} seconds.\n")
    print("Writing vectorstore..")
    v_start_time = time.time()
    vectorstore.save_local(storing_path)
    v_end_time = time.time()
    v_time_taken = round(v_end_time - v_start_time, 2)
    print(f"Vectorstore write time: {v_time_taken} seconds.\n")
    return vectorstore

def load_qa_chain(retriever, llm, prompt):
    print("Loading QA chain...")
    start_time = time.time()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    end_time = time.time()
    time_taken = round(end_time - start_time, 2)
    print(f"QA chain load time: {time_taken} seconds.\n")
    return qa_chain

def get_response(query, chain) -> str:
    response = chain({'query': query})
    res = response['result']
    return res

class PDFHelper:

    def __init__(self, model_name: str = Config.MODEL,
                 embedding_model_name: str = Config.EMBEDDING_MODEL_NAME):
        self._model_name = model_name
        self._embedding_model_name = embedding_model_name

    def ask(self, pdf_file_path: str, question: str) -> str:
        vector_store_directory = os.path.join(str(Path.home()), 'langchain-store', 'vectorstore',
                                              'pdf-doc-helper-store', str(uuid.uuid4()))
        os.makedirs(vector_store_directory, exist_ok=True)
        print(f"Using vector store: {vector_store_directory}")

        llm = ChatOllama(
            temperature=0,
            model=self._model_name,
            streaming=True,
            top_k=10,
            top_p=0.3,
            num_ctx=3072,
            verbose=False
        )

        embed = load_embedding_model(model_name=self._embedding_model_name)

        # Check if the file path is a URL or a local file path
        if pdf_file_path.startswith("http"):
            loader = LLMSherpaFileLoader(
                file_path=pdf_file_path,
                new_indent_parser=True,
                apply_ocr=True,
                strategy="sections",
                llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all",
            )
        else:
            loader = PyMuPDFLoader(file_path=pdf_file_path)

        documents = loader.load()

        vectorstore = create_embeddings(chunks=documents, embedding_model=embed, storing_path=vector_store_directory)
        retriever = vectorstore.as_retriever()

        template = """
        ### System:
        You are an honest assistant.
        You will accept PDF files and you will answer the question asked by the user appropriately.
        If you don't know the answer, just say you don't know. Don't try to make up an answer.
    
        ### Context:
        {context}
    
        ### User:
        {question}
    
        ### Response:
        """

        prompt = langchain_core.prompts.PromptTemplate.from_template(template)
        chain = load_qa_chain(retriever, llm, prompt)

        start_time = time.time()
        response = get_response(question, chain)
        end_time = time.time()

        time_taken = round(end_time - start_time, 2)
        print(f"Response time: {time_taken} seconds.\n")

        return response.strip()
