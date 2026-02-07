"""
사용자의 프리미엄 상태에 따라 다른 모델을 적용시켜주는 함수.
미리 Context 클래스를 만든 뒤, Custom Middleware를 만들어 Runtime 내부에 전달된 Context를 보고 모델을 다르게 적용.
"""

from dotenv import load_dotenv

load_dotenv()

from dataclasses import dataclass

@dataclass
class Context:
    user_name: str
    is_premium: bool = False
    
from langchain.agents.middleware import wrap_model_call
from langchain.chat_models import init_chat_model

@wrap_model_call
def dynamic_model_selector(request, handler):
    """
    agent 내부의 model은 모델 이름이 아니라 ChatOpenAI처럼 모델의 상태를 전달받기 때문에
    init_chat_model로 모델을 생성해서 전달.
    """
    user_name = request.runtime.context.user_name
    is_premium = request.runtime.context.is_premium
    if is_premium:
        model_name = "gpt-5"
    else:
        model_name = "gpt-5-nano"
    print(f"{user_name}님의 AI 모델은 {model_name}")
    new_model = init_chat_model(model_name)
    new_request = request.override(model=new_model)
    return handler(new_request)
    
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-5-nano",
    tools=[],
    middleware=[dynamic_model_selector],
    context_schema=Context
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "안녕하세요"}]},
    context=Context(user_name="Jung Sub", is_premium=True)
)

print(response)