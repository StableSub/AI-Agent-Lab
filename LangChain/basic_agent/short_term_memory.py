"""
이전 대화 맥락을 기억하기 위한 Memory를 사용.
사용 시 checkpointer를 InMemorySaver(Short Memory)로 설정 후
질문을 보낼 때 configurable의 thread_id에 동일한 문자를 등록해 주어야 함.
"""
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model = init_chat_model(
    "gpt-5-nano",
    temperature=0.9,
    api_key=api_key
)

tools = []

agent = create_agent(
    model,
    tools,
    checkpointer=InMemorySaver()
)

reponse = agent.invoke(
    {"messages": [{"role": "user", "content": "안녕하세요. 저는 StableSub의 정섭입니다."}]},
    {"configurable": {"thread_id": "1"}}
)

print(reponse)

reponse = agent.invoke(
    {"messages": [{"role": "user", "content": "제 이름이 뭐라고 했죠?"}]},
    {"configurable": {"thread_id": "1"}}
)

print(reponse)

for i, msg in enumerate(reponse["messages"]):
    print(f"--- Message {i+1} ---")
    print(msg.content)
    print()