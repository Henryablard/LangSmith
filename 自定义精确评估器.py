#定义自定义评估器

from langsmith import evaluate

def correct(outputs: dict, reference_outputs: dict) -> bool:
    """Check if the answer exactly matches the expected answer."""
    return outputs["answer"] == reference_outputs["answer"]

def dummy_app(inputs: dict) -> dict:
    return {"answer": "hmm i'm not sure", "reasoning": "i didn't understand the question"}

results = evaluate(
    dummy_app,
    data="data_set_name",
    evaluators=[correct]
)
#这是一个最基础的精确评估器，但是这个评估器太严格了，一般不会使用精确评估器，一般是让大模型来判断答案是否符合。
