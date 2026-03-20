from PIL.Image import enum
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage

model = init_chat_model("gpt-5-nano")

class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def chat_node(state: ChatState):
    return {"messages": [model.invoke(state["messages"])]}

workflow = StateGraph(ChatState)

workflow.add_node("chat", chat_node)

workflow.add_edge(START, "chat")
workflow.add_edge("chat", END)

memory = InMemorySaver()

app = workflow.compile(checkpointer=memory)

config_1 = {"configurable": {"thread_id": "1"}}


def checkpointer():
    """
    checkponiter를 InMemorySaver로 설정 후 configuable의 thread_id를 설정 시, 해당 thread_id의 내용을 기억
    즉, 한 대화 스레드 안에서 이어지는 맥락을 유지하기 위한 장치.
    """
    input_msg1 = {"messages": [HumanMessage(content="ㅎㅇ 내 이름 정섭")]}
    response1 = app.invoke(input_msg1, config=config_1)

    input_msg2 = {"messages": [HumanMessage(content="내 이름이 뭐라고~")]}
    response2 = app.invoke(input_msg2, config=config_1)

def state_history():
    """
    단기 메모리를 사용하였다면, 그래프가 무엇을 저장하고 있는지를 불러올 수 있음.
    get_state -> 특정 스레드의 현재 상태 스냅샷을 불러옴.
    get_state_history -> 과거부터 현재까지의 스냅샷 목록을 불러옴.
    """
    input_msg1 = {"messages": [HumanMessage(content="ㅎㅇ 내 이름 정섭")]}
    app.invoke(input_msg1, config=config_1)

    input_msg2 = {"messages": [HumanMessage(content="내 이름이 뭐라고~")]}
    app.invoke(input_msg2, config=config_1)
    # get_state
    current_state = app.get_state(config_1)

    print(f"---Current State---\n{current_state}\n")
    print(f"---Current State Value---\n{current_state.values['messages'][-1].content}\n")
    print(f"---Current State Next---\n{current_state.next}\n")
    print(f"---Current State Config---\n{current_state.config}\n")
    # get_state_history
    history = list(app.get_state_history(config_1))
    for i, snapshot in enumerate(history):
        print(f"---Snapshot {i} Created At: {snapshot.created_at}---\n")

        msgs = snapshot.values.get("messages", [])
        if msgs:
            last_msg = msgs[-1]
            sender = "AI" if last_msg.type == "ai" else "User"
            print(f"---Last Message is--- \n{sender} {last_msg.content}\n")
        else:
            print("---대화 시작 전 초기 상태--- \n")
        print(f"---Next---\n{snapshot.next}\n")
        print(f"---Metadata---\n{snapshot.metadata}\n")
    