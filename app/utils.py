from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import YoutubeLoader

def create_rag_chain(llm, retriever):
    # 定义系统提示
    system_prompt = (
        "你是一个精通LangChain和AI的专家,使用以下检索到的内容片段来回答问题,回答的要全面和仔细。"
        "\n\n"
        "{context}"
    )

    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 创建问答链
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def create_vectorstore(docs):
    EMBEDDING_DEVICE = "cuda"
    embeddings = HuggingFaceEmbeddings(model_name="../models/m3e-base",
                                       model_kwargs={"device": EMBEDDING_DEVICE}
                                       )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore


def load_local_files(data_dir):
    pdf_dir = os.path.join(data_dir, 'pdf')
    txt_dir = os.path.join(data_dir, 'txt')

    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    txt_files = [os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if f.endswith('.txt')]

    docs = []
    for file_path in pdf_files:
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    for file_path in txt_files:
        loader = TextLoader(file_path, encoding='utf-8')  # 指定编码
        docs.extend(loader.load())

    return docs


def load_web_urls(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:  # 指定编码
        urls = file.readlines()

    docs = []
    for url in urls:
        loader = WebBaseLoader(web_paths=(url.strip(),))
        docs.extend(loader.load())

    return docs


def load_video_urls(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:  # 指定编码
        urls = file.readlines()

    docs = []
    for url in urls:
        docs.extend(YoutubeLoader.from_youtube_url(url.strip(), add_video_info=True).load())

    return docs


def load_docs(local_data_dir, web_urls_file_path, video_urls_file_path):
    print("Loading local documents...")
    local_docs = load_local_files(local_data_dir)
    print(f"Loaded {len(local_docs)} local documents.")

    print("Loading web documents...")
    web_docs = load_web_urls(web_urls_file_path)
    print(f"Loaded {len(web_docs)} web documents.")

    print("Loading video documents...")
    video_docs = load_video_urls(video_urls_file_path)
    print(f"Loaded {len(video_docs)} video documents.")

    all_docs = local_docs + web_docs + video_docs

    # all_docs = web_docs

    return all_docs