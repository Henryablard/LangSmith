#测试pycharm能否通过改变vpn端口来达到使用API来接入ChatGPT的

import os
from openai import OpenAI
import requests

# 使用 HTTP 代理
os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

#  测试代理是否生效
try:
    r = requests.get("https://api.openai.com/v1/models", proxies={
        "http": "http://127.0.0.1:7897",
        "https": "http://127.0.0.1:7897"
    }, headers={"Authorization": "Bearer sk-xxxx"})
    print("代理测试返回状态码：", r.status_code)
except Exception as e:
    print("代理测试失败：", e)

#  调用 OpenAI API
client = OpenAI(
    api_key=                   # 这里换成你的 API key
    base_url="https://api.openai.com/v1"
)

try:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "给我讲一个笑话"}]
    )
    print("返回内容：", resp.choices[0].message.content)
except Exception as e:
    print("调用出错：", e)

