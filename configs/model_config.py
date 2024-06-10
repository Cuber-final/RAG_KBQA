import torch.cuda
import torch.backends
import os
import uuid
from models import embedding_models, llm_models

embedding_model_dict = {
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "bge": embedding_models["bge"],
}

# Embedding model name
EMBEDDING_MODEL = "bge"

# Embedding running device
EMBEDDING_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


# 本地模型的加载
# llm_model_dict 处理了loader的一些预设行为，如加载位置，模型名称，模型处理器实例
LLM_MODEL_LIST = {
    "chatglm-6b": {
        "name": "chatglm-6b",
        "pretrained_model_name": "THUDM/chatglm-6b",
        "local_model_path": None,
        "provides": "ChatGLM",
    },
    # 通过 fastchat 调用的模型请参考如下格式
    "fastchat-chatglm-6b": {
        "name": "chatglm-6b",  # "name"修改为fastchat服务中的"model_name"
        "pretrained_model_name": "chatglm-6b",
        "local_model_path": None,
        "provides": "FastChatOpenAILLM",  # 使用fastchat api时，需保证"provides"为"FastChatOpenAILLM"
        "api_base_url": "http://localhost:8000/v1",  # "name"修改为fastchat服务中的"api_base_url"
    },
    "glm4-api": {
        "name": "glm4-api",
    },
}

# LLM 名称
LLM_MODEL = "glm4-api"
# LLM_MODEL = "chatglm-6b"
# 如果你需要加载本地的model，指定这个参数  ` --no-remote-model`，或者下方参数修改为 `True`
NO_REMOTE_MODEL = False
# 量化加载8bit 模型
LOAD_IN_8BIT = False
# Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.
BF16 = False
# 本地模型存放的位置
MODEL_DIR = "models/"
# 本地lora存放的位置
LORA_DIR = "loras/"

# LLM lora path，默认为空，如果有请直接指定文件夹路径
LLM_LORA_PATH = ""
USE_LORA = True if LLM_LORA_PATH else False

# LLM streaming reponse
STREAMING = True

# Use p-tuning-v2 PrefixEncoder
USE_PTUNING_V2 = False

# LLM running device
LLM_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")

UPLOAD_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "content")

# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

# 缓存知识库数量
CACHED_VS_NUM = 1

# 文本分句长度
SENTENCE_SIZE = 100

# 匹配后单段上下文长度
CHUNK_SIZE = 250

# LLM input history length
LLM_HISTORY_LEN = 3

# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 5

# 知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，经测试设置为小于500时，匹配结果更精准
VECTOR_SEARCH_SCORE_THRESHOLD = 0

NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

FLAG_USER_NAME = uuid.uuid4().hex


# 是否开启跨域，默认为False，如果需要开启，请设置为True
# is open cross domain
OPEN_CROSS_DOMAIN = False

# Bing 搜索必备变量
# 使用 Bing 搜索需要使用 Bing Subscription Key
# 具体申请方式请见 https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
BING_SUBSCRIPTION_KEY = ""


SUPPORT_AGENT_MODEL = [
    "openai-api",  # GPT4 模型
    "zhipu-api",  # 智谱AI GLM4模型
    "Qwen",  # 所有Qwen系列本地模型
]

Agent_MODEL = None
