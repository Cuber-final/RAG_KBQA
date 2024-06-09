import os
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, TextLoader, PyMuPDFLoader
from langchain.prompts import PromptTemplate
from langchain.vectorstores.faiss import FAISS
from text_splitter.chinese_text_splitter import ChineseTextSplitter
from langchain_zhipu import ZhipuAIEmbeddings, ChatZhipuAI
import dotenv
from pathlib import Path

dotenv.load_dotenv()

init_llm = "glm-4"
init_embedding_model = "embedding-2"
work_dir = Path(os.getcwd())


# 基于langchain调用chatglm-api 实现的agent
class KnowledgeBasedChatLLMApi:

    llm: object = None
    embeddings: object = None

    # 初始化模型配置
    def init_model_config(
        self,
        model_type: str = init_llm,
        embedding_model: str = init_embedding_model,
    ):
        API_KEY = os.getenv("ZHIPU_API_KEY")
        self.llm = ChatZhipuAI(api_key=API_KEY, model=model_type, verbose=True)
        self.embeddings = ZhipuAIEmbeddings(api_key=API_KEY, model=embedding_model)

    # 通过LLM进行向量库检索和回答
    def get_knowledge_based_answer(
        self,
        query,
        web_content,
        do_sample: bool = True,  # 是否采样
        top_k: int = 6,  # 召回的数量
        history_len: int = 3,  # 对话历史最大保留三个会话
        temperature: float = 0.01,  # (0,1) 值越大越随机
        top_p: float = 0.1,  # 采样的另一种方式
        history=[],
    ):
        self.llm.temperature = temperature
        self.llm.top_p = top_p
        self.history_len = history_len
        self.top_k = top_k

        # 利用网络检索得到的背景知识作为上下文
        if web_content:
            prompt_template = (
                f"""基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
                                已知网络检索内容：{web_content}"""
                + """
                                已知内容:
                                {context}
                                问题:
                                {question}"""
            )
        # 基于本地知识库检索得到的上下文
        else:
            prompt_template = """基于以下已知信息，请简洁并专业地回答用户的问题。
                如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。另外，答案请使用中文。

                已知内容:
                {context}

                问题:
                {question}"""

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        # self.llm.history = history[-self.history_len :] if self.history_len > 0 else []
        vector_store = self.vector_store
        retriever = vector_store.as_retriever(search_kwargs={"k": self.top_k})
        print(retriever)

        knowledge_chain = RetrievalQA.from_llm(
            llm=self.llm,
            retriever=retriever,
            prompt=prompt,
        )
        knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )

        knowledge_chain.return_source_documents = True

        result = knowledge_chain({"query": query})
        print(result)  # 会展示检索的结果
        return result

    # 根据文件类型使用不同的文本加载器
    def load_file(self, file_path):
        if file_path.lower().endswith(".md"):
            loader = UnstructuredFileLoader(file_path, mode="elements")
            docs = loader.load()
        elif file_path.lower().endswith(".pdf"):
            # 可以考虑结合其他的加载器更好用的
            loader = PyMuPDFLoader(file_path)
            textsplitter = ChineseTextSplitter(pdf=True)
            docs = loader.load_and_split(textsplitter)
        elif file_path.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf8")
            # textsplitter = ChineseTextSplitter(pdf=False)
            docs = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter())
        else:
            loader = UnstructuredFileLoader(file_path, mode="elements")
            textsplitter = ChineseTextSplitter(pdf=False)
            docs = loader.load_and_split(text_splitter=textsplitter)
        return docs

    # 初始化本地知识库
    def init_knowledge_vector_store(self, filepath):
        print("正在加载本地知识库----")
        print(filepath)
        vec_store_path = str(work_dir / "datas" / "save_faiss_index")
        self.vector_store = None
        # 检查本地知识库文件是否存在
        local_store_exists = Path(vec_store_path).exists()

        if not filepath:
            # 如果没有上传文件
            if local_store_exists and not self.vector_store:
                # 本地知识库文件存在且尚未加载，则初始化加载本地知识库
                try:
                    self.vector_store = FAISS.load_local(
                        vec_store_path, self.embeddings
                    )
                    print("本地知识库加载成功")
                except Exception as e:
                    print("本地知识库加载失败，请上传本地知识库文件")
                    print(e)
            elif not local_store_exists:
                # 本地知识库文件不存在，输出提示信息
                print("本地知识库文件缺失，请上传本地知识库文件")
        else:
            # 如果上传了文件
            docs = self.load_file(filepath)

            if not self.vector_store and local_store_exists:
                # 如果本地知识库尚未加载但文件存在，则加载本地知识库
                try:
                    self.vector_store = FAISS.load_local(
                        vec_store_path, self.embeddings
                    )
                    print("本地知识库加载成功")
                except Exception as e:
                    print("本地知识库加载失败，请检查本地知识库文件")
                    print(e)

            if self.vector_store:
                # 增量更新已有的向量库
                # 调用langchain的faiss库以及前面加载的embedding模型，对文档进行向量化
                new_vectors = self.embeddings.embed_documents(docs)
                self.vector_store.add_vectors(new_vectors)
                print("已成功增量更新本地知识库")
            else:
                # 如果没有已有的向量库，则创建新的向量库
                self.vector_store = FAISS.from_documents(docs, self.embeddings)
                print("已成功创建新的本地知识库")

            # 保存向量化的结果到本地中
            self.vector_store.save_local(vec_store_path)
            print("本地知识库保存成功")

        return self.vector_store


if __name__ == "__main__":
    pass
