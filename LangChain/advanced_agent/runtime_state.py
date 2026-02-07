"""
RunTime: Agent 실행에 필요한 공유 정보
    Context: 사용자 ID나 데이터베이스 연결 URL처럼 Agent 실행 시 변하지 않는 정보.
    -> 보안 및 권한 제어
    Stroe: 사용자의 선호, 이름 등 장기 메모리.
    -> 개인화 제어
    Stream Writer: Agent 내부 작동 로그 등 실시간 이벤트 출력 기능.
    -> LangGraph에서 사용 
    
State: Agent가 작동하며 만든 결과물. 
-> 대화 흐름 제어
    Message: Model이 Runtime을 참조해가며 만든 결과물.
    tool_results: Tool이 만든 정보 모음.
"""
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

from dataclasses import dataclass

@dataclass
class Context:
    user_name: str

def node_style():
    """
    Node-style Hook: 특정 실행 지점에서 순차적으로 동작하는 미들웨어. 즉, Loggin 용.
    before agent, before model, after model, after agent 4가지가 존재. 입력 파라미터로 state, runtime를 사용.
    """
    from langchain.agents.middleware import before_model
    from langchain.agents import create_agent

    @before_model
    def log_before_model(state, runtime):
        print(f"State: {state}")
        print(f"Runtime: {runtime}")
        print(f"사용자 이름: {runtime.context.user_name}")
        
    agent = create_agent(
        model="gpt-5-nano",
        tools=[],
        middleware=[log_before_model],
        context_schema=Context
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "제 이름이 뭐죠?"}]},
        context=Context(user_name="Jay")
    )

    print(response)

def wrap_style():
    """
    Wrap-style Hook: 실행 흐름을 감싸서 조작하거나 흐름을 제어할 수 있는 미들웨어.
    model이나 tool이 호출될 때 실행과 제어를 가로챌 때 사용. 입력 파라미터로 request, handler를 사용.
    """
    from langchain.agents.middleware import wrap_model_call, after_model
    from langchain.messages import HumanMessage, SystemMessage
    from langchain.agents import create_agent
    
    @wrap_model_call
    def inject_user_name(request, handler):
        print(f"Request: {request}")
        user_name = request.runtime.context.user_name
        if user_name:
            sys_prompt = f"사용자의 이름은 {user_name} 입니다."
            request = request.override(system_prompt=sys_prompt)
        return handler(request)
    
    @after_model
    def log_after_model(state, runtime):
        print(f"모델 응답 완료 후 : {state['messages'][-1].content}")
        return None
    
    agent = create_agent(
        model="gpt-5-nano",
        tools=[],
        middleware=[inject_user_name, log_after_model],
        context_schema=Context
    )
    
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "제 이름이 뭐죠?"}]},
        context=Context(user_name="Jung Sub")
    )

    print(response)
wrap_style()
