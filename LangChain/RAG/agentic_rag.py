"""
2-Step RAG: 사용자 질문 -> 관련 문서 검색 -> 답변 생성 -> 사용자에게 답변 반환
Agentic RAG: 사용자 질문 -> 에이전트 -> 외부 정보가 필요한가(필요 시 도구 검색) -> 도구로 검색 -> 답변이 충분한가(충분하지 않으면 다시 에이전트로 돌아감) -> 최종 답변 생성 -> 사용자에게 반환
Hybrid RAG: 사용자 질문 -> 쿼리 향상 -> 문서 검색 -> 정보가 충분한가(충분하지 않으면 쿼리 수정) -> 답변 생성 -> 답변 품질이 괜찮은가(괜찮지 않을 시 다른 접근 방식 시도 yes: 쿼리 수정 no: 답변 반환) -> 최적의 답변 -> 사용자에게 반환
"""

from langchain.agents import create_agent
from langchain.tools import tool

try:
    # Package execution: python -m LangChain.RAG.agentic_rag
    from .retrieve import retriever
except ImportError:
    # Direct file execution: python LangChain/RAG/agentic_rag.py
    from retrieve import retriever

@tool
def search_protfolio(query: str):
    """
    ARKK ETF의 포트폴리오 정보를 검색할 떄 사용.
    특정기업의 보유 비중, 주식 수, 가치 등을 찾을 때 이 도구를 호출.
    """
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

agent = create_agent(
    model="gpt-5-nano",
    tools=[search_protfolio]
)

print(
    agent.invoke(
        {"messages": [{"role": "user", "content": "ARKK 펀드의 Tesla 투자 비중이 어떻개 돼?"}]}
    )['messages'][-1].content
)
