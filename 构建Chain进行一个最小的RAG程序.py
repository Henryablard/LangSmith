#测试能否通过API来接入ChatGPT来让大模型回答问题
#初步建立一个非常基础的RAG来让大模型输出我们想要的答案

import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 加载 .env 文件
load_dotenv()

# 读取 API Key
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

# LangChain 提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's request only based on the given context."),
    ("user", "Question: {question}\nContext: {context}")
])

# LLM 设置
llm = ChatOpenAI(
    model="gpt-4o-mini",      #选择要使用的模型
    openai_api_key=api_key,
    openai_api_base=base_url,
)

# 输出解析器
output_parser = StrOutputParser()

# 组装 chain
chain = prompt | llm | output_parser

# OpenAI 原生客户端
openai_client = OpenAI(api_key=api_key, base_url=base_url)

# 模拟的检索器
def retriever(query: str):
    results = ["Harrison worked at Kensho"]
    return results

# RAG 流程
def rag(question: str):
    docs = retriever(question)
    system_message = f"""仅使用以下提供的信息来回答用户的问题:

{chr(10).join(docs)}"""

    resp = openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
        model="gpt-4o-mini",
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    print("RAG 输出：", rag("Where did Harrison work?"))

