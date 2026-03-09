from langchain.messages import AnyMessage, ToolMessage, SystemMessage, AIMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gpt-5-nano")

@tool
def search_manual(query: str) -> str:
    """
    고객의 질문이 들어오면 무조건 매뉴얼을 먼저 살펴본 뒤 답변.
    고객의 질문에 답변하기 위해 참고할만한 규정이나 메뉴얼을 검색할 때 사용하는 도구.
    """
    if "비밀번호" in query:
        return "비밀번호 변경은 보안 설정에 존재"
    elif "배송" in query:
        return "3일 내 배송 예정"
    else:
        return "메뉴얼을 찾을 수 없음"
    
tools = [search_manual]

model_with_tools = model.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_step: str

def classify_node(state: AgentState):
    print("\n--- [1] 분류 단계 (LLM 판단) ---")
    last_message = state['messages'][-1]
    
    prompt = """
    당신은 고객 센터 관리자이다. 고객의 이메일을 분석해서 다음 단계를 결정해라.
    
    1. 단순 문의나 정보 요청이라면 -> 'consultant' 반환
    2. 환불 요청, 불만 제기, 화난 고객이라면 -> 'escalate' 반환
    
    답변은 오직 단어 하나만 사용.
    """
    
    response = model.invoke([SystemMessage(content=prompt), last_message])
    raw_content = response.content
    
    decision = raw_content.strip().lower()
    print(f"  -> LLM 판단 결과: {decision}")
    
    if "escalate" in decision:
        return {"next_step": "escalate"}
    else:
        return {"next_step": "consultant"}

def consultant_node(state: AgentState):
    print("\n--- [2-A] 상단 AI 답변 생성 중 ---")
    response = model_with_tools.invoke(state['messages'])
    return {'messages': [response]}

def escalate_node(state: AgentState):
    return {'messages': [AIMessage(content="해당 메일은 전문 상담원에게 이관.")]}

tools_by_name = {tool.name: tool for tool in tools}

def tool_node(state: AgentState):
    print("\n--- [3] Tool Node 진입 ---")
    result = []
    for tool_call in state['messages'][-1].tool_calls:
        tool = tools_by_name[tool_call['name']]
        tool_result = tool.invoke(tool_call['args'])
        print(f"\n--- {tool} 사용 / 결과: {tool_result} ---")
        result.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
    return {'messages': result}

agent_builder = StateGraph(AgentState)

agent_builder.add_node('classify_node', classify_node)
agent_builder.add_node('consultant_node', consultant_node)
agent_builder.add_node('escalate_node', escalate_node)
agent_builder.add_node('tool_node', tool_node)

agent_builder.add_edge(START, 'classify_node')

def route_after_classify(state: AgentState):
    if state['next_step'] == 'escalate':
        return 'escalate'
    else:
        return 'consultant'

def should_continue(state: AgentState):
    last_message = state['messages'][-1]
    
    if last_message.tool_calls:
        return "tool_node"
    return END

agent_builder.add_conditional_edges(
    'classify_node',
    route_after_classify,
    {
        'escalate': 'escalate_node',
        'consultant': 'consultant_node'
    }
)

agent_builder.add_conditional_edges(
    'consultant_node',
    should_continue,
    ['tool_node', END]
)

agent_builder.add_edge('tool_node', 'consultant_node')
agent_builder.add_edge('escalate_node', END)

agent = agent_builder.compile()

inputs = {'messages': [HumanMessage(content="비밀번호 변경은 어디서 해?")]}

response = agent.invoke(inputs)

print(response['messages'][-1].content)