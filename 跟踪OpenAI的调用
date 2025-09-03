#跟踪OpenAI的调用
from openai import OpenAI
from langsmith.wrappers import wrap_openai

openai_client = wrap_openai(OpenAI())

# 这就是我在RAG时要使用的检索器
# 这个是模拟出来的，实际上可以是任何我们想要的

def retriever(query: str):
    results = ["Harrison worked at Kensho"]
    return results

# 这就是完整的RAG链
# 它先进行检索，然后再调用OpenAI

def rag(question):
    docs = retriever(question)
    system_message = """Answer the users question using only the provided information below:

    {docs}""".format(docs="\n".join(docs))

    return openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
        model="gpt-4o-mini",
    )
rag("where did harrison work")
