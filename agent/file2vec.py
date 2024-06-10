import os
import dotenv
import sys
import io
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_zhipu import ZhipuAIEmbeddings  # 导入智谱AI嵌入式模型
from agent.tools import RapidOCRPDFLoader, PDFTextLoader
from text_splitter.chinese_text_splitter import ChineseTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
dotenv.load_dotenv()  # 加载环境变量
# 获取API_KEY
API_KEY = os.getenv("ZHIPU_API_KEY")
embeddings = ZhipuAIEmbeddings(api_key=API_KEY)
work_dir = Path(os.getcwd())
data_dir = work_dir / "datas" / "pdfs"


# 不提取图片信息的分割方法
def pdf_text_loader(file_path):
    loader = PDFTextLoader(file_path)
    textsplitter = RecursiveCharacterTextSplitter()
    docs = loader.load_and_split(textsplitter)
    return docs


def pdf_ocr_loader(file_path):
    loader = RapidOCRPDFLoader(file_path)
    textsplitter = RecursiveCharacterTextSplitter()
    docs = loader.load_and_split(textsplitter)
    # 获取PDF标题
    import pymupdf

    docs = pymupdf.open(file_path)
    # print(doc.metadata)
    doc_name = docs.metadata["title"]
    # print(len(docs))
    # 处理文档内容整理
    return docs


def init_vec_store(docs: list[Document]):
    """初始化知识库向量库

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
    docs = pdf_text_loader(file_name)  # 调用pdf_loader函数
    for doc in docs:
        print(doc.page_content)
        print("---------")
    # from text_splitter import test_zh_title_enhance

    # test_zh_title_enhance(docs)
    # init_vec_store(docs)
    # load_vec_store(docs)
