#将跟踪发送到特定的项目
import openai
from langsmith import traceable
from langsmith.run_trees import RunTree

client = openai.Client()

messages = [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Hello!"}
]

# Use the @traceable decorator with the 'project_name' parameter to log traces to LangSmith
# Ensure that the LANGSMITH_TRACING environment variables is set for @traceable to work

@traceable(
  run_type="llm",
  name="OpenAI Call Decorator",
  project_name="My Project"
)
def call_openai(
  messages: list[dict], model: str = "gpt-4o-mini"
) -> str:
  return client.chat.completions.create(
      model=model,
      messages=messages,
  ).choices[0].message.content

# Call the decorated function
call_openai(messages)

# You can also specify the Project via the project_name parameter
# This will override the project_name specified in the @traceable decorator

call_openai(
  messages,
  langsmith_extra={"project_name": "My Overridden Project"},
)

# The wrapped OpenAI client accepts all the same langsmith_extra parameters
# as @traceable decorated functions, and logs traces to LangSmith automatically.
# Ensure that the LANGSMITH_TRACING environment variables is set for the wrapper to work.

from langsmith import wrappers
wrapped_client = wrappers.wrap_openai(client)
wrapped_client.chat.completions.create(
  model="gpt-4o-mini",
  messages=messages,
  langsmith_extra={"project_name": "My Project"},
)

# Alternatively, create a RunTree object
# You can set the project name using the project_name parameter

rt = RunTree(
  run_type="llm",
  name="OpenAI Call RunTree",
  inputs={"messages": messages},
  project_name="My Project"
)
chat_completion = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=messages,
)

# End and submit the run

rt.end(outputs=chat_completion)
rt.post()
