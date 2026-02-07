"""
기본적인 Tool 사용법 정리.
agent를 생성하여 tools 부분에 사용하였으면 하는 Tool들을 추가하여 AI가 적절히 사전에 정의된 Tool을 사용할 수 있도록 설정.
"""

from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model = init_chat_model(
    "gpt-5-nano",
    temperature=0.9,
    api_key = api_key)

from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """
    특정 지역의 날씨 정보 제공
    """
    return f"{location}의 날씨는 맑고, 영하 2도입니다." # location의 정보는 AI가 스스로 추론하여 값을 넣음 tool 호출 또한 적절히 추론하여 호출이라는 행동으로 옮긴 것

from langchain.agents import create_agent

agent = create_agent(model, tools=[get_weather])

def get_grpah():
    from IPython.display import Image, display

    png = agent.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png)

def tool_answer():
    """
    create_agent를 한 뒤, tools를 등록하면 AI의 답변은 tool 사용 시,
    바로 content를 생성하지 않고, tool_calls로 get_weather 도구를 호출 한 뒤,
    도구는 응답을 ToolMessage에 담아 응답 후 Model이 도구의 응답을 참조해 답변을 생성.
    """
    reponse = agent.invoke(
        {"messages": [{"role": "user", "content": "서울 날씨 어때요?"}]}
    )
    print(reponse["messages"][-1].content)