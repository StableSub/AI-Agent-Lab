"""
프로젝트를 추적하기 위한 도구.
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model = init_chat_model(
    "gpt-5-nano",
    temperature=0.9,
    api_key = api_key)

reponse = model.invoke("랭스미스가 무엇인지 한 문장으로 설명해줘.")
print(f"답변: {reponse.content}")
