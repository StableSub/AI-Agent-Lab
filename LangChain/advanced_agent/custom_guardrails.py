"""
구현방식
    Before Agent Guardrails
    After Agent Guardrails
    Combine Multiple Guardrails
"""

from dotenv import load_dotenv

load_dotenv()

def before_agent_guardrails():
    """
    AI에게 메세지를 넘기기 전에 사용자의 메세지를 본 뒤 기준이 부합하지 않으면 거부하는 가드레일.
    """
    forbidden_topcis = {
        "cheating": ["답지", "정답", "숙제"],
        "distraction": ["게임", "유튜브", "아이돌"],
        "harmful": ["담배", "술", "폭력"]
    }

    from langchain.agents.middleware import before_agent

    @before_agent(can_jump_to=["end"])
    def education_guardrail(state, runtime):
        """
        질문의 의도를 파악하여 금지어가 들어가있으면 AI에게 질문을 넘기지 않음.
        """
        if not state["messages"]:
            return None
        last_messages = state["messages"][-1]
        if last_messages.type != "human":
            return None
        user_content = last_messages.content
        for keyword in forbidden_topcis["cheating"]:
            if keyword in user_content:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "치팅 하지 마쇼"
                    }],
                    "jump_to": "end"
                }
                
        for keyword in forbidden_topcis["distraction"]:
            if keyword in user_content:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "공부 하쇼"
                    }],
                    "jump_to": "end"
                }
                
        for keyword in forbidden_topcis["harmful"]:
            if keyword in user_content:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "이상한거 보지 마쇼"
                    }],
                    "jump_to": "end"
                }
        return None
    
    from langchain.agents import create_agent
    
    agent = create_agent(
        model="gpt-5-nano",
        middleware=[education_guardrail]
    )
    
    print(agent.invoke({
        "messages": [{"role": "user", "content": "피타고라스 정리 설명 좀."}]
    })["messages"][-1].content)
    
    print(agent.invoke({
        "messages": [{"role": "user", "content": "피타고라스 정답 좀 알려줘"}]
    })["messages"][-1].content)
    
    print(agent.invoke({
        "messages": [{"role": "user", "content": "게임 유튜브 추천 좀"}]
    })["messages"][-1].content)
    
    print(agent.invoke({
        "messages": [{"role": "user", "content": "담배 종류 10가지"}]
    })["messages"][-1].content)

def after_agent_guardrails():
    """
    Agent 호출 후 답변을 확인하는 과정에서 AI를 통해 기준에 부합하는지 확인.
    만약 부합하지 않으면 AI를 통해 답변을 수정한 뒤 기존 답변에 덮어씌우기. 
    """
    from langchain.agents.middleware import after_agent
    from langchain.chat_models import init_chat_model
    from langchain.messages import AIMessage, SystemMessage, HumanMessage
    
    safety_model = init_chat_model("gpt-5-nano")
    
    @after_agent
    def answer_leakage_guardrail(state, runtime):
        """
        AI가 답변을 생성한 직후, 사용자에게 보여주기 전 내용을 검사하여 기준에 부합하지 않으면 이를 감지하고 수정.
        """
        if not state["messages"]: return None
        last_message = state["messages"][-1]
        if not isinstance (last_message, AIMessage): return None
        ai_content = last_message.content
        auditor_prompt = f"""
        당신은 엄격한 교육 감독관입니다. 다음 '튜터의 답변'을 확인하세요.
        답변이 학생을 지도하지 않고 문제의 정답이나 전체 풀이를 제공한다면 'LEAKED'로 대답하세요.
        답변이 적절한 힌트나 설명을 제공하다면 'SAFE'로 대답하세요.
        튜터의 답변: {ai_content}
        """
        
        result = safety_model.invoke([{"role": "user", "content": auditor_prompt}])
        
        if "LEAKED" in result.content:
            original_question = state["messages"][0].content
            correction_prompt = f"""
            절대 정답을 직접 말하지 말고, 학생 스스로 생갈할 수 있도록 유도하는 질문이나 핵심 개념만 설명하세요.
            사용자의 질문 {original_question}
            """
            
            corrected_response = safety_model.invoke([
                SystemMessage(content="당신은 소크라테스식 교육법을 사용하는 튜터입니다."),
                HumanMessage(content=correction_prompt)
            ])
            
            last_message.content = corrected_response.content
        return None
    
    from langchain.agents import create_agent
    
    agent = create_agent(
        model="gpt-5-nano",
        middleware=[answer_leakage_guardrail]
    )
    
    print(agent.invoke({
        "messages": [{"role": "user", "content": "직각 삼각형 두 직각변의 길이가 3, 4라면 빗변의 길이가 뭐야?"}]
    })["messages"][-1].content)