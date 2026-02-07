"""
이전 대화를 role 구분을 통해 같이 보내주어 맥락을 기억.
"""
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model = init_chat_model(
    "gpt-5-nano",
    temperature=0.9,
    api_key = api_key)

def message1():
    """
    메세지 구성 요소
    System Message: 개발자가 사전에 작성한 입력(프롬프트)
    Human Message: 사용자의 요청, 입력
    AI Message: Model의 답변
    """
    system_msg = SystemMessage("당신은 윤으한 로켓 전문가 입니다.")
    human_msg = HumanMessage("안녕하세요. 궁금한 것이 있어요.")

    messages = [system_msg, human_msg]

    response = model.invoke(messages)

    print(response)

def message2():
    """
    딕셔너리 포맷으로도 가능
    user - human / ai - assistant
    """
    messages = [
        {"role": "system", "content": "당신은 유능한 AI 어시스턴트입니다."},
        {"role": "human", "content": "안녕하세요. 저는 Jay입니다."},
        {"role": "assistant", "content": "안녕하세요 Jay. 어떤 도움이 필요하신가요?"},
        {"role": "human", "content": "제 이름이 뭐라고 했죠?"},
    ]

    response = model.invoke(messages)

    print(response)
    