#大模型评估器
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

# 定义大模型评估器
llm_correctness = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,   # 内置的“判断答案是否正确”的提示词
    model="gpt-4o-mini",         # 评估用的模型，可以和被评估模型不同
    feedback_key="correctness",  # 在结果里显示的字段名
)
from openai import OpenAI
from langsmith import wrappers

openai_client = wrappers.wrap_openai(OpenAI())

def target(inputs: dict) -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # 被评估的模型
        messages=[
            {"role": "system", "content": "Answer the following question accurately"},
            {"role": "user", "content": inputs["question"]},
        ],
    )
    return {"answer": response.choices[0].message.content.strip()}
from langsmith import Client

client = Client()

experiment_results = client.evaluate(
    target,
    data="Sample dataset V2",   # 你在 LangSmith 创建的数据集名
    evaluators=[llm_correctness],  # 用大模型来打分
    experiment_prefix="llm-judge-eval",
    max_concurrency=2,
)
