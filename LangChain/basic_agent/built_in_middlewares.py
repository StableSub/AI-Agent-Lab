"""
Middleware는 Agent 내부에서 일어나는 일을 섬세하게 컨트롤 하는 방법.
1. Agent 내부 활동 모니터링 및 컨트롤
2. 프롬프트, tool 선택, 출력 포맷 수정
3. 가드레일(개인정보 인식, 프롬프트 인젝션 방지)
사용 시 create_agent 함수 내부에 middleware 부분에 Dict 형태로 사용하고자 하는 Middleware를 등록하여야 함.
"""
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model = init_chat_model(
    "gpt-5-nano",
    temperature=0.9,
    api_key = api_key
)

@tool
def send_email_tool(to: str, subject: str, body: str) -> str:
    """
    지정한 이메일 주소로 메일을 보내는 도구입니다.
    """
    return f"✅ 이메일이 성공적으로 전송되었습니다.\n수신자: {to}\n제목: {subject}\n내용: {body[:50]}..."

@tool
def read_email_tool(limit: int = 3) -> List[Dict[str, str]]:
    """
    최근 받은 이메일 3개를 읽는 도구입니다.
    """
    return f"✅ 이메일이 성공적으로 조회되었습니다."

tools = [send_email_tool, read_email_tool]

from langchain.agents.middleware import LLMToolEmulator, TodoListMiddleware, HumanInTheLoopMiddleware, PIIMiddleware

agent = create_agent(
    model,
    tools,
    middleware=[
        LLMToolEmulator(model="gpt-5-nano"),
        TodoListMiddleware()
    ],
)

def emulator():
    """
    LLMToolEmulator: 아직 tool을 완성하지 않았을 때, LLMToolEmulator를 Middleware로 설정하면 이 Tool을 AI가 분석 후, 
    Tool 사용 시 나올 수 있는 결과를 AI가 반환하여 가상의 데이터 활용.
    Tool 호출이 불가능하거나 너무 비싼 경우, 또는 프로토타입의 경우에 활용.
    """
    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "최근 온 메일 확인하고 알아서 답장해줘."}
            ]
        }
    )

    print(response["messages"][-1].content)

def to_do_list():
    """
    TodoListMiddleware: 복잡한 요청을 처리할 때 작업 단계를 스스로 계획하고 추적하는 Middleware.
    """
    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "온 메일을 전부 확인한 뒤 나한테 요약 후 보고해. 그 다음 답장 작성하고 회손 보내줘. 마지막으로 어떻게 보냈는지 보고해."}
            ]
        }
    )
    print(response["messages"][-1].content)
    
def human_in_the_loop():
    from langgraph.checkpoint.memory import InMemorySaver
    """
    HumanInTheLoopMiddleware: 작업 전 사용자의 의사를 묻는 미들웨어로, 고도화된 업무에 적합.
    주의할 점은 사용자의 의사를 묻기 전 맥락을 기억해야 하기 떄문에 반드시 checkpointer와 함께 사용하여야 함.
    """
    checkpointer = InMemorySaver()
    
    agent = create_agent(
        model,
        tools,
        checkpointer=checkpointer,
        middleware=[
            LLMToolEmulator(model="gpt-5-nano"),
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email_tool": {
                        "allow_decisions": ["approve", "edit", "reject"]
                    },
                    "read_email_tool": False
                }
            )
        ]
    )
    
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "무슨 메일 왔는지 확인해줘"}]},
        {"configurable": {"thread_id": "1"}}
    )
    
    print(response["messages"][-1].content)
    
    print(response)
    
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "교수님한테 내일 찾아뵙겠다는 메일 작성해서 보내줘."}]},
        {"configurable": {"thread_id": "1"}}
    )
    
    print(response["messages"][-1].content)
    
    print(response)
    
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "교수님 이메일 professor_kim@edu.com, 내 이름: 섭이, 학과: 컴퓨터학부, 장소: 교수님 연구실, 내 이메일: sub@edu.com"}]},
        {"configurable": {"thread_id": "1"}}
    )
    
    print(response["messages"][-1].content)
    
    print(response)

def pii_detection():
    """
    PIIMiddleware: 민감 정보를 다루는 Agent 구축 시 활용.
    pii_type
        built-in: email, credit_card, ip, mac_address, url 등
        custom type
    블록 방식
        block: 에러 발생
        redact: 완전 마스킹
        mask: 부분 마스킹
        hash: 해싱
    적용 시점
        apply_to_input: model 호출 전
        apply_to_output: model 호출 이후
        apply_to_tool_results: tool의 응답 메세지에 대한 블록
    """
    @tool
    def save_customer_feedback(feedback: str) -> str:
        """고객 피드백을 저장하는 도구"""
        return f"고객 피드백 저장 완료: {feedback}"
    
    # 커스텀 PII Detector는 정규식으로 값을 전달하여 PII 미들웨어를 생성
    phone_number_detector_regex = r"\b(010)[-\s]?(\d{3,4})[-\s]?(\d{4})\b"
    
    phone_masking_middleware = PIIMiddleware(
        pii_type="phone_number",
        detector=phone_number_detector_regex,
        strategy="mask",
        apply_to_input=True
    )
    
    api_key_detector_regex = r"sk-proj-)[A-Za-z0-9_-]+([A-Za-z0-9]{4}"
    
    api_key_block_middleware = PIIMiddleware(
        pii_type="api_key",
        detector=api_key_detector_regex,
        strategy='block',
        apply_to_input=True
    )
    
    agent = create_agent(
        model,
        tools=[save_customer_feedback],
        middleware=[
            LLMToolEmulator(model="gpt-5-nano"),
            # 이메일 주소는 전부 마스킹
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            # 카드번호는 마지막 4자리만 남기고 나머지 마스킹
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
            # 핸드폰 번호는 마지막 4자리만 남기고 나머지 마스킹
            phone_masking_middleware,
            # API_KEY는 block
            api_key_block_middleware
        ],
    )
    
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "안녕하세요 저는 섭(이메일: sub@edu.com)입니다. 제 카드번호는 4242424242424242 입니다. 제 핸드폰 번호는 01033333333입니다."}]}
    )
    print(response)

def summarization():
    """
    Summarization: 대화가 지속될수록 AI가 기억해야할 대화가 많아지기 때문에 일정 토큰, 메세지 개수를 넘어가면 메세지를 요약해서 AI가 기억하게 해주는 Middleware.
    """

def structrued_output():
    """
    AI의 출력을 미리 정의한 형태로 구조화하여 정해진 출력을 받아오거나, 입력으로 사용할 수 있음.
    """
    from langchain.agents.structured_output import ToolStrategy
    from pydantic import BaseModel, Field
    from typing import Literal
    
    class EmailAnalysis(BaseModel):
        """이메일 내용을 분석한 결과 구조."""
        intent: Literal["complaint", "inquiry", "confirmation", "other"] = Field(
            description="이메일의 주요 의도 (예: complaint=불만, inquiry=문의, confirmation=확인, other=기타)"
        )
        sentiment: Literal["positive", "negative", "neutral"] = Field(description="이메일의 감정 상태")
        summary: str = Field(description="이메일 내용 요약")
        next_action: str = Field(description="에이전트가 수행해야 할 다음 단계 (예: 회신, 확인, 무시 등)")
    
    
    agent = create_agent(
        model,
        tools,
        response_format=ToolStrategy(EmailAnalysis),
        middleware=[
            LLMToolEmulator(model="gpt-5-nano")
        ]
    )
    
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "최근 온 메일 확인하고 고객의 의도와 감정, 요약, 그리고 어떤 행동이 필요한지 분석해."}]}
    )
    
    print(response)
    
    print(response["structured_response"])

structrued_output()