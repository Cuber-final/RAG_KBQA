import os
from typing import List
import gradio as gr
import nltk
from agent.main_agents.chatglm_know import KnowledgeBasedChatLLMApi
from agent.tools.web_serarch import *
from configs.model_config import *

nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")] + nltk.data.path

llm_model_list = []
llm_model_dict = LLM_MODEL_LIST
init_embedding_model = EMBEDDING_MODEL
init_llm = LLM_MODEL
for model, model_description in llm_model_dict.items():
    # print(model)
    llm_model_list.append(model_description["name"])


def update_status(history, status):
    history = history + [[None, status]]
    print(status)
    return history


chat_llm = KnowledgeBasedChatLLMApi()


def init_model():
    try:
        print("开始加载模型配置")
        chat_llm.init_model_config()
        print("模型配置加载成功")
        chat_llm.llm.invoke("你好")
        return """初始模型已成功加载，可以开始对话"""
    except Exception as e:
        print(f"加载模型出错: {e}")  # 打印详细的异常信息
        return """模型未成功加载，请重新选择模型后点击"重新加载模型"按钮"""


def clear_session():
    return "", None


def reinit_model(large_language_model, embedding_model, history):
    try:
        chat_llm.init_model_config(
            large_language_model=large_language_model, embedding_model=embedding_model
        )
        model_status = """模型已成功重新加载，可以开始对话"""
    except Exception as e:
        print(e)
        model_status = """模型未成功重新加载，请点击重新加载模型"""
    return history + [[None, model_status]]


def init_vector_store(file_obj):
    # 加载文件，使用文件路径名
    print(type(file_obj))
    file_input = None
    if file_obj is not None:
        file_input = file_obj.name

    vector_store = chat_llm.init_knowledge_vector_store(file_input)

    return vector_store


def predict(input, use_web, top_k, history_len, temperature, top_p, history=None):
    if history == None:
        history = []

    if use_web == "True":
        # 网络搜索保持和初始化的大模型是同一个
        web_content = google_search(query=input, llm=chat_llm.llm)
    else:
        web_content = ""

    resp = chat_llm.get_knowledge_based_answer(
        query=input,
        web_content=web_content,
        top_k=top_k,
        history_len=history_len,
        temperature=temperature,
        top_p=top_p,
        history=history,
    )
    history.append((input, resp["result"]))
    return "", history, history


model_status = init_model()

if __name__ == "__main__":
    block = gr.Blocks()
    with block as demo:

        gr.Markdown(
            """<h1><center>LangChain-ChatLLM-Webui</center></h1>
        <center><font size=3>
        本项目基于LangChain和大型语言模型系列模型, 提供基于本地知识的自动问答应用. <br>
        目前项目提供基于<a href='https://github.com/THUDM/ChatGLM-6B' target="_blank">ChatGLM-6B </a>的LLM和包括GanymedeNil/text2vec-large-chinese、nghuyong/ernie-3.0-base-zh、nghuyong/ernie-3.0-nano-zh在内的多个Embedding模型, 支持上传 txt、docx、md、pdf等文本格式文件. <br>
        后续将提供更加多样化的LLM、Embedding和参数选项供用户尝试, 欢迎关注<a href='https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui' target="_blank">Github地址</a>.
        </center></font>
        """
        )
        model_status = gr.State(model_status)
        with gr.Row():
            with gr.Column(scale=1):
                model_choose = gr.Accordion("模型选择")
                with model_choose:
                    large_language_model = gr.Dropdown(
                        choices=llm_model_list,
                        label="large language model",
                        value=init_llm,
                    )

                    embedding_model = gr.Dropdown(
                        list(embedding_model_dict.keys()),
                        label="Embedding model",
                        value=init_embedding_model,
                    )
                    load_model_button = gr.Button("重新加载模型")
                model_argument = gr.Accordion("模型参数配置")
                with model_argument:

                    top_k = gr.Slider(
                        1,
                        10,
                        value=6,
                        step=1,
                        label="vector search top k",
                        interactive=True,
                    )

                    history_len = gr.Slider(
                        0, 5, value=3, step=1, label="history len", interactive=True
                    )

                    temperature = gr.Slider(
                        0,
                        1,
                        value=0.01,
                        step=0.01,
                        label="temperature",
                        interactive=True,
                    )
                    top_p = gr.Slider(
                        0, 1, value=0.9, step=0.1, label="top_p", interactive=True
                    )

                file = gr.File(
                    label="请上传知识库文件",
                    file_types=[".txt", ".md", ".docx", ".pdf"],
                )

                init_vs = gr.Button("知识库文件向量化")
                # 是否启用网络搜索
                use_web = gr.Radio(["True", "False"], label="Web Search", value="False")

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    [[None, model_status.value]], label="ChatLLM", height=750
                )
                message = gr.Textbox(label="请输入问题")
                state = gr.State()

                with gr.Row():
                    clear_history = gr.Button("🧹 清除历史对话")
                    send = gr.Button("🚀 发送")

            load_model_button.click(
                reinit_model,
                show_progress=True,
                inputs=[large_language_model, embedding_model, chatbot],
                outputs=chatbot,
            )
            init_vs.click(
                init_vector_store,
                show_progress=True,
                inputs=[file],
                outputs=[],
            )

            send.click(
                predict,
                inputs=[
                    message,
                    use_web,
                    top_k,
                    history_len,
                    temperature,
                    top_p,
                    state,
                ],
                outputs=[message, chatbot, state],
            )
            clear_history.click(
                fn=clear_session, inputs=[], outputs=[chatbot, state], queue=False
            )

            message.submit(
                predict,
                inputs=[
                    message,
                    use_web,
                    top_k,
                    history_len,
                    temperature,
                    top_p,
                    state,
                ],
                outputs=[message, chatbot, state],
            )
        gr.Markdown(
            """提醒：<br>
        1. 使用时请先上传自己的知识文件，并且文件中不含某些特殊字符，否则将返回error. <br>
        2. 有任何使用问题，请通过[Github Issue区](https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui/issues)进行反馈. <br>
        """
        )
    # threads to consume the request
    demo.queue(max_size=20).launch(
        server_name="127.0.0.1",  # ip for listening, 0.0.0.0 for every inbound traffic, 127.0.0.1 for local inbound
        server_port=8042,  # the port for listening
        show_api=False,  # if display the api document
        share=False,  # if register a public url
        inbrowser=False,
        max_threads=3,
    )  # if browser would be open automatically
