from datetime import datetime
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from models.model_contain import model_container
from pydantic import BaseModel, Field
from langchain.chains import LLMChain

CUSTOM_PROMPT_TEMPLATE = """
根据用户提问，回答日常对话的问题
问题：{query}

回答：
"""

PROMPT = PromptTemplate(
    input_variables=["query"],
    template=CUSTOM_PROMPT_TEMPLATE,
)


class FuncWithOutInput(BaseModel):
    query: str = Field()


def get_current_time(query):
    return datetime.now()


def normal_chat(query):
    model = model_container.MODEL
    llm_math = LLMChain.from_llm(model, prompt=PROMPT, verbose=True)
    ans = llm_math.invoke(query)
    return ans
