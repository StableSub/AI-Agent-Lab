from langchain.messages import AnyMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gpt-5-nano")

class AgentState(TypedDict):
    email_content: str
    category: str
    next_step: str
    response: str

def read_email(state: AgentState):
    return {'email_content': state['email_content']}
    
def classify_intent(state: AgentState):
    email = state['email_content']
    if "환불" in email or "긴급" in email:
        category = "complaint"
        next_step = "escalate_to_human"
    else:
        category = "inquiry"
        next_step = "search_manual"
    return {'category': category, 'next_step': next_step}
    
def search_manual(state: AgentState):
    print("3-A 진입... 메뉴얼을 검색합니다.")
    return

def escalate_to_human(state: AgentState):
    print("3-B 진입... 상담원 이관합니다.")
    return
    
def write_reply(state: AgentState):
    email = state['email_content']
    response = model.invoke(email)
    return {'response': response}

from langgraph.graph import StateGraph, START, END

agent_builder = StateGraph(AgentState)

agent_builder.add_node('read_email', read_email)
agent_builder.add_node('classify_intent', classify_intent)
agent_builder.add_node('search_manual', search_manual)
agent_builder.add_node('escalate_to_human', escalate_to_human)
agent_builder.add_node('write_reply', write_reply)

def route_email(state: AgentState):
    return state['next_step']

agent_builder.add_edge(START, 'read_email')
agent_builder.add_edge('read_email', 'classify_intent')
agent_builder.add_conditional_edges(
    'classify_intent',
    route_email,
    ['escalate_to_human', 'search_manual']
)
agent_builder.add_edge('search_manual', 'write_reply')
agent_builder.add_edge('write_reply', END)
agent_builder.add_edge('escalate_to_human', END)

agent = agent_builder.compile()

inputs = {"email_content": "비밀번호 변경 방법"}
response = agent.invoke(inputs)
print(response)

inputs = {"email_content": "당장 환불해"}
response = agent.invoke(inputs)
print(response)