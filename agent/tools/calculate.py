from langchain.prompts import PromptTemplate
from llm_chain import LLMMathChain
from pydantic import BaseModel, Field
from dotenv import find_dotenv, load_dotenv
import os
from langchain_zhipu import ChatZhipuAI
from models.model_contain import model_container
from pydantic import BaseModel, Field


_PROMPT_TEMPLATE = """
将数学问题翻译成可以使用Python的numexpr库执行的表达式。使用运行此代码的输出来回答问题。
问题: ${{包含数学问题的问题。}}
```text
${{解决问题的单行数学表达式}}
```
...numexpr.evaluate(query)...
```output
${{运行代码的输出}}
```
答案: ${{答案}}

这是两个例子：

问题: 37593 * 67是多少？
```text
37593 * 67
```
...numexpr.evaluate("37593 * 67")...
```output
2518731

答案: 2518731

问题: 37593的五次方根是多少？
```text
37593**(1/5)
```
...numexpr.evaluate("37593**(1/5)")...
```output
8.222831614237718

答案: 8.222831614237718


问题: 2的平方是多少？
```text
2 ** 2
```
...numexpr.evaluate("2 ** 2")...
```output
4

答案: 4


现在，这是我的问题：
问题: {question}
"""

PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
)


class CalculatorInput(BaseModel):
    query: str = Field()


def calculate_math(query: str):
    # model即通过langchain初始化的llm对象
    model = model_container.MODEL
    llm_math = LLMMathChain.from_llm(model, prompt=PROMPT, verbose=True)
    # ans = llm_math.run(query)
    ans = llm_math.invoke(query)
    return ans


if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=True)
    API_KEY = os.getenv("ZHIPUAI_API_KEY")

    llm = ChatZhipuAI(api_key=API_KEY, model="glm-4", temperature=0.8)
    # work_dir = os.getcwd()
    model_container.MODEL = llm
    result = calculate_math("2的三次方")
    print("答案:", result["answer"])
