# Multi-Agent

在这个LangChain项目，我基于Prompt和LangChain Agent构建了一个多功能聊天机器人，使用四个工具：RAG、Chat、Search、Calculate。

在RAG中，我实现了多数据来源：网页、本地文件（txt、pdf）、以及视频数据，在Search工具中使用DuckDuckGoSearchRun来搜索互联网以查找有关世界事件、问题、日期、年份等的信息，Chat工具负责处理日常对话，Calculate工具负责处理数学运算。

## Powered by

- LangChain
- LangSmith
- Chainlit
- LangServe
- m3e Embedding
- OpenAI、SparkLLM
- FAISS
- DuckDuckGoSearchRun

## How to Start

To install Chainlit along with any other dependencies, run:
```
pip install -r requirements.txt
```
Open a terminal in your project directory
```
cd your/file/path/app
```
run the following command:
```
chainlit run chainlit.py -w
```
对了，embedding模型文件自行下载，新建models文件夹，从huggingface下载[m3e-base](https://huggingface.co/moka-ai/m3e-base/tree/main),确保项目结构如下：

```
RAG_Practice
├─ app
├─ data
├─ models
│  └─ m3e-base
├─ README.md
└─ requirements.txt
```
