#跟踪多轮对话

import openai
from langsmith import traceable
from langsmith import Client
import langsmith as ls
from langsmith.wrappers import wrap_openai

client = wrap_openai(openai.Client())
langsmith_client = Client()

#用于此示例的配置

langsmith_project = "project1"
session_id = "thread-id-1"
langsmith_extra={"project_name": langsmith_project, "metadata":{"session_id": session_id}}

#获取线程中所有LLM调用的历史，以构建对话历史

def get_thread_history(thread_id: str, project_name: str): # Filter runs by the specific thread and project
  filter_string = f'and(in(metadata_key, ["session_id","conversation_id","thread_id"]), eq(metadata_value, "{thread_id}"))' # Only grab the LLM runs
  runs = [r for r in langsmith_client.list_runs(project_name=project_name, filter=filter_string, run_type="llm")]

#按开始时间排序，以获取最新的互动

  runs = sorted(runs, key=lambda run: run.start_time, reverse=True)

#当前的对话状态

  return runs[0].inputs['messages'] + [runs[0].outputs['choices'][0]['message']]

#如果继续现有的一个对话，这个功能会查找当前运行的元数据以获取session_id，调用get_thread_history，并在调用聊天模型之前添加新的用户问题。

@traceable(name="Chat Bot")
def chat_pipeline(question: str, get_chat_history: bool = False): # Whether to continue an existing thread or start a new one
  if get_chat_history:
      run_tree = ls.get_current_run_tree()
      messages = get_thread_history(run_tree.extra["metadata"]["session_id"],run_tree.session_name) + [{"role": "user", "content": question}]
  else:
      messages = [{"role": "user", "content": question}]

 #调用模型

  chat_completion = client.chat.completions.create(
      model="gpt-4o-mini", messages=messages
  )
  return chat_completion.choices[0].message.content

#开始对话

chat_pipeline("我想给我的朋友Henry送上生日祝福，记得写上我的署名。", get_chat_history=True, langsmith_extra=langsmith_extra)



