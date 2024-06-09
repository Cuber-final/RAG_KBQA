import os
import dotenv
import sys
import io
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_zhipu import ZhipuAIEmbeddings  # 导入智谱AI嵌入式模型
from agent.tools import RapidOCRPDFLoader
from text_splitter.chinese_text_splitter import ChineseTextSplitter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
dotenv.load_dotenv()  # 加载环境变量
# 获取API_KEY
API_KEY = os.getenv("ZHIPU_API_KEY")
embeddings = ZhipuAIEmbeddings(api_key=API_KEY)
work_dir = Path(os.getcwd())
data_dir = work_dir / "datas" / "pdfs"


# def pdf_loader_simple(file_name):
#     file_path = data_dir / file_name  # 获取data_utils文件夹下的about.pdf文件路径
#     print(file_path)  # 打印文件路径
#     loader = PyMuPDFLoader(
#         str(file_path)
#     )  # 将文件路径转换为字符串格式，并调用PyMuPDFLoader加载器

#     textsplitter = ChineseTextSplitter(pdf=True)  # 创建中文文本分割器
#     docs = loader.load_and_split(textsplitter)  # 调用加载和分割方法
#     print(len(docs))  # 打印文档数量
#     return docs  # 返回文档列表


def pdf_ocr_loader(file_path):
    loader = RapidOCRPDFLoader(file_path)
    textsplitter = ChineseTextSplitter(pdf=True)  # 创建中文文本分割器
    docs = loader.load_and_split(textsplitter)
    print(len(docs))
    return docs


def init_vec_store(docs: list[Document]):
    """_summary_

    Args:
        docs (list[Document]): 文档对象列表
    """

    # 使用API_KEY初始化ZhipuAIEmbeddings

    # 使用FAISS将文档转换为向量，并将向量存储在vector_store中
    vector_store = FAISS.from_documents(docs, embedding=embeddings)
    # 将向量化的结果保存到本地
    save_faiss_path = work_dir / "tests" / "modules" / "test_faiss_index"
    vector_store.save_local(str(save_faiss_path))
    print("init_vec_store done")


def load_vec_store(docs: list[Document]):
    save_faiss_path = work_dir / "tests" / "modules" / "test_faiss_index"
    vector_store = FAISS.load_local(save_faiss_path, embeddings)
    pass


if __name__ == "__main__":
    file_name = data_dir / "llm_qa.pdf"
    docs = pdf_ocr_loader(file_name)  # 调用pdf_loader函数
    for doc in docs:
        print(doc.page_content)
    # init_vec_store(docs)
    # load_vec_store(docs)
