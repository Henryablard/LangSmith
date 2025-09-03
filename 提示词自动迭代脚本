# -*- coding: utf-8 -*-
"""
自动提示词迭代框架
原始 Prompt -> 评估 -> 用大模型优化 Prompt -> 存储在 LangSmith 新版本里面 -> 循环迭代
评估使用 LLM-as-a-judge（语义判断，而非精确匹配）
"""
import os
import json
from typing import List, Dict, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import convert_to_openai_messages
from requests import HTTPError

# 初始化
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
ls_api_key = os.getenv("LANGSMITH_API_KEY")
ls_endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

ls_client = Client(api_key=ls_api_key, api_url=ls_endpoint)
openai = OpenAI(api_key=api_key, base_url=base_url)

PROMPT_NAME = "iterative-prompt-python-only"

# 数据集
dataset = [
    {"inputs": {"question": "LangSmith 是做什么的？"}, "outputs": {"answer": "LangSmith 用于追踪、评估和调试 LLM 应用。"}},
    {"inputs": {"question": "Deep Research 是什么？"}, "outputs": {"answer": "Deep Research 是 LangChain 提供的自动化研究工具。"}},
]

# 工具函数
def push_prompt_safe(name: str, prompt: ChatPromptTemplate) -> None:
    """安全 push：如果提示词没变化导致 409 错误，就跳过"""
    try:
        ls_client.push_prompt(name, object=prompt)   # 不要传 version
        print(f"[LangSmith] Prompt '{name}' 已推送（会自动生成新版本）")
    except HTTPError as e:
        msg = str(e)
        if "Nothing to commit" in msg or "Conflict" in msg:
            print(f"[LangSmith] 提示词未变化，跳过推送（{msg}）")
        else:
            raise

def run_prompt(prompt: ChatPromptTemplate, question: str) -> str:
    """用给定 Prompt 调用 OpenAI，返回模型答案"""
    formatted = prompt.invoke({"question": question})
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=convert_to_openai_messages(formatted.messages),
    )
    return resp.choices[0].message.content.strip()

def llm_judge(reference: str, answer: str) -> Tuple[int, str]:
    """LLM 评审器：输出 score (0/1) + reason"""
    system = "你是一个负责评估答案的智能裁判。"
    user = f"Reference:\n{reference}\n\nCandidate:\n{answer}\n\n输出 JSON: {{\"score\": 1 或 0, \"reason\": \"...\"}}"

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
        return int(data.get("score", 0)), str(data.get("reason", ""))
    except Exception:
        return 0, f"Bad judge output: {raw}"

def evaluate_prompt(prompt: ChatPromptTemplate, dataset: List[Dict]) -> Tuple[float, List[Dict]]:
    """对数据集逐条评估，返回 accuracy 和 bad_cases"""
    results = []
    for ex in dataset:
        q = ex["inputs"]["question"]
        ref = ex["outputs"]["answer"]
        pred = run_prompt(prompt, q)
        score, why = llm_judge(ref, pred)
        results.append({"q": q, "ref": ref, "pred": pred, "score": score, "why": why})
    acc = sum(r["score"] for r in results) / max(len(results), 1)
    bad_cases = [r for r in results if r["score"] == 0]
    return acc, bad_cases

def improve_prompt(old_system_prompt: str, bad_cases: List[Dict]) -> str:
    """用大模型基于 bad cases 生成新的 system prompt 文本"""
    cases_text = "\n".join(
        f"- Q: {c['q']}\n  Ref: {c['ref']}\n  Pred: {c['pred']}\n  Judge: {c['why']}"
        for c in bad_cases[:5]
    )
    instruction = f"""
你是资深提示词工程师，请基于当前 System Prompt 与失败示例，生成一个更强的 System Prompt。
要求：
1. 保留原有能力；
2. 明确边界，避免幻觉；
3. 输出简洁、准确；
4. 与原来语言风格保持一致。

【当前 System Prompt】
{old_system_prompt}

【失败用例】
{cases_text}

请输出新的 System Prompt（仅给出提示词本身）。
"""
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a world-class prompt engineer."},
                  {"role": "user", "content": instruction}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

# 自动迭代逻辑
def iterate_prompt(max_rounds: int = 5, target_acc: float = 1.0):
    system_prompt = "你是一个非常专业的LangSmith使用专家，关于使用LangSmith没有人比你更厉害。"

    for i in range(1, max_rounds + 1):
        print(f"\n=== Round {i} ===")

        # 构造 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{question}"),
        ])

        # 推送到 LangSmith（新版本）
        push_prompt_safe(PROMPT_NAME, prompt)

        # 评估
        acc, bad_cases = evaluate_prompt(prompt, dataset)
        print("Accuracy:", acc)
        if acc >= target_acc:
            print(" 达到目标准确率，停止迭代。")
            break

        # 生成新提示词
        system_prompt = improve_prompt(system_prompt, bad_cases)

# 运行
iterate_prompt(max_rounds=5, target_acc=1.0)
