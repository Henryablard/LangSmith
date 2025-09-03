#混合评估器
from langsmith import Client, evaluate
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

# 初始化 LangSmith 和 OpenAI 客户端
client = Client()
openai_client = wrap_openai(OpenAI())
# 定义要评估的目标函数（假设是一个简单问答应用）
def target(inputs: dict) -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer the following question accurately."},
            {"role": "user", "content": inputs["question"]},
        ],
    )
    return {"answer": response.choices[0].message.content.strip()}
# 定义评估器
# 精确匹配评估器
def exact_match(outputs: dict, reference_outputs: dict) -> dict:
    score = int(outputs["answer"].strip() == reference_outputs["answer"].strip())
    return {"exact_match": score}

# 大模型评估器
def llm_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="gpt-4o-mini",
        feedback_key="llm_correctness"
    )
    return evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs
    )
# 运行评估
experiment_results = client.evaluate(
    target,
    data="Sample dataset V2",
    evaluators=[exact_match, llm_correctness],  # 同时传两个评估器
    experiment_prefix="hybrid-eval",
    max_concurrency=2,
)

print("Evaluation done! Check results in LangSmith.")
