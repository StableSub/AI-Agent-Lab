"""
기본적인 Tool 사용법 정리.
여러개의 툴을 사용.
System Prompt를 사용하여 Tool 사용을 강제 할 수 있음.
"""
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    """ 
    a와 b 덧셈

    Args:
        a (int): First int
        b (int): Second int
    """
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """ 
    a와 b 곱셈

    Args:
        a (int): First int
        b (int): Second int
    """
    return a * b

@tool
def divide(a: int, b: int) -> int:
    """ 
    a와 b 나눗셈

    Args:
        a (int): First int
        b (int): Second int
    """
    return a / b

tools = [add, multiply, divide]

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model = init_chat_model(
    "gpt-5-nano",
    temperature=0.9,
    api_key = api_key)

# system_prompt를 사용하지 않을 시, AI는 확률로 동작하기 때문에 tool을 쓰지 않고 답변에 대응할 수 있어서 system_prompt를 사용하여 tool 사용을 강제하는 프롬프트 추가.
agent = create_agent(
    model,
    tools,
    system_prompt="당신은 유능한 수학 선생님입니다. 사칙 연산 요청 시 모든 단계에서 Tool을 이용하시오."
)

reponse = agent.invoke(
    {"messages": [{"role": "user", "content": "42 + 3 * 23은 뭔가요?"}]}
)

print(reponse)

print(reponse["messages"][-1].content)