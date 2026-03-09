from langchain.messages import AnyMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gpt-5-nano")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b

tools = [multiply, add, divide]

model_with_tools = model.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int

def llm_call(state):
    response = model_with_tools.invoke(state['messages'])
    return {
        'messages': [response],
        'llm_calls': state.get('llm_calls', 0) + 1
    }

tools_by_name = {tool.name: tool for tool in tools}

def tool_node(state):
    result = []
    for tool_call in state['messages'][-1].tool_calls:
        tool = tools_by_name[tool_call['name']]
        
        tool_result = tool.invoke(tool_call['args'])
        
        result.append(ToolMessage(content=tool_result, tool_call_id=tool_call['id']))
    return {'messages': result}

from langgraph.graph import StateGraph, START, END

agent_builder = StateGraph(State)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_edge("tool_node", "llm_call")

def should_continue(state: State):
    messages = state['messages']
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "tool_node"
    else:
        return END

agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)

agent = agent_builder.compile()

from langchain.messages import HumanMessage

messages = [HumanMessage(content="3과 4를 더한다음 7을 곱해줘.")]
response = agent.invoke({'messages': messages})

print(response)