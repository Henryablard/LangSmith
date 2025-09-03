from langsmith import traceable, get_current_run_tree
from openai import OpenAI
from langsmith.wrappers import wrap_openai

# OpenAI + LangSmith 包装
client = wrap_openai(OpenAI())

# 我们的 VIP 用户名单
VIP_USERS = {"alice", "bob"}

@traceable(name="chatbot", run_type="llm", project_name="vip-chat-project")
def chatbot(question: str, user_id: str):
    rt = get_current_run_tree()

    # 非VIP → 不上传到 LangSmith
    if user_id not in VIP_USERS:
        # 不调用 rt.post()，trace 不会被提交
        return f"(非VIP用户 {user_id} 的请求未记录到 LangSmith)"

    # 对 VIP 用户，补充 metadata & tags
    rt.metadata["user_id"] = user_id
    rt.tags.append("VIP")

    # 调用 LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )

    # 提交 trace
    rt.end(outputs={"response": response.choices[0].message.content})
    rt.post()

    return response.choices[0].message.content

# 测试
print(chatbot("你好！", user_id="charlie"))  # 非VIP，不记录
print(chatbot("帮我写一首诗", user_id="alice"))  # VIP，记录到 LangSmith
