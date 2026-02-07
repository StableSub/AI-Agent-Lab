"""
알라딘 API를 통해 베스트셀러를 조회하고 N개의 베스트 셀러를 조회하는 Tool을 만들어 Agent의 tools에 등록.
"""
import os

import requests
from langchain.tools import tool
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

@tool
def get_bestseller_list() -> List[Dict[str, Any]]:
    """
    알라딘 베스트셀러 목록을 조회하고 Top N개(기본 10개)를 반환합니다.
    """
    url = "https://www.aladin.co.kr/ttb/api/ItemList.aspx"
    ttb_key = os.getenv("ALADIN_TTB_KEY")
    if not ttb_key:
        raise ValueError("ALADIN_TTB_KEY 환경 변수가 필요합니다.")
    
    # 필수 및 기본 파라미터
    params = {
        'ttbkey': ttb_key,
        'QueryType': 'Bestseller', # 베스트셀러 리스트 조회
        'MaxResults': 10,
        'start': 1,
        'output': 'js',
        'Version': 20131101,
        'SearchTarget': 'Book' # 기본값: 도서 (필요에 따라 변경 가능)
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("item", [])
    top10 = items[:10]
    return top10

tools = [get_bestseller_list]

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

api_key = os.getenv("OPENAI_API_KEY")

model = init_chat_model(
    "gpt-5-nano",
    temperature=0.9,
    api_key = api_key
)

agent = create_agent(
    model,
    tools
)

reponse = agent.invoke(
    {"messages": [{"role": "user", "content": "현재 베스트셀러에는 어떠한 것들이 있어?"}]}
)

print(reponse)

print(reponse["messages"][-1].content)
