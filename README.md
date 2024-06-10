- 本项目整合并利用langchain，RAG，chatglm学习搭建个人知识库助手
- 兼容chatglm在langchain上的使用，可安装轮子 ``pip install langchain_zhipu``
- 运行时可能报错缺少一些组件 tiktoken,chardet，用pip自行安装即可

## 开发进度
### Level One ：基本功能实现
1. 熟悉基本的langchain 组件使用，llm主要调用chatglm api（新用户免费token）🚩
2. 实现简单的Agent 🚩
   - 可调用一些python库回答问题（数学计算，天气查询）
   - 根据问题选择知识库问答还是其他Tool
### Level Two : 知识库问答
1. 知识库相关的数据预处理
   - 语料解析
     - PDF 解析
     - HTML 解析
     - MarkDown 解析
     - 电子书 解析
   - 文本分块方法（尝试多种方式）
   - 关键词与向量库构建
   - 针对实际问题优化索引机制
2. 知识库检索
   - 向量检索
   - 关键词检索
   - 混合检索（bm25+vector）
   - RRF
   - RAG Fusion
3. 回答生成
   
---

## 参考项目
- [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)
- [LangChain-ChatGLM-Webui](https://github.com/X-D-Lab/LangChain-ChatGLM-Webui)
