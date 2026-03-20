from validators import uuid
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.store.base import BaseStore
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool
from langchain_core.messages import ToolMessage

load_dotenv()

"""
단기 메모리는 같은 thread_id 안에서 맥락을 이어주는 장치.
하지만 실제 서비스에서는 사용자에 대한 고유 정보가 영구적으로 유지되어야 하는 경우가 많음.
이 경우 장기 메모리를 사용해서 에이전트의 영구적인 기억 장치로 활용.

Store: 장기 데이터를 영구적으로 담는 거대한 저장소 역할.
Namespace: 데이터 충돌을 막기 위해 저장 공간을 논리적으로 나누는 단위. 보통 (user_id, profile)처럼 튜플 구조를 사용하여 계층적으로 나눔.
Key: 데이터가 겹치짖 않게 uuid 등을 부여하여 고유 식별자를 부여.
Value: 딕셔너리 형태로 저장되는 실제 사용자 데이터.
"""

model = init_chat_model("gpt-5-nano")

class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def manual_save():
    """
    기억해 라는 말이 사용자의 질의문에 있으면 해당 내용을 수동으로 Store에 저장하는 방법.
    memory_id를 uuid로 중복 없이 고유 식별자를 생성.
    사용자의 마지막 메세지를 store에 저장.
    사용자의 정보를 통해 namespace 획득 후, 그 정보를 시스템 메세지에 포함하여 AI에게 전달.
    """
    def memory_agent_node(state: ChatState, config, store:BaseStore):
        user_id = config["configurable"]["user_id"]

        namespace = (user_id, "profile")

        last_message = state["messages"][-1]
        if "기억해" in last_message.content:
            print(f"\n[System] {user_id}님의 정보를 장기 기억장치에 저장.")

            memory_id = str(uuid.uuid4())
            store.put(
                namespace,
                memory_id,
                {"content": last_message.content}
            )

            memories = store.search(namespace)

            if memories:
                memory_text = "\n".join([f"- {m.value["content"]}" for m in memories])
                system_msg = f"""
                당신은 사용자의 정보를 기억하는 비서.
                [장기 기억 저장소]
                {memory_text}
                위 기억을 참고하여 답변.
                """
            else:
                system_msg = "당신은 비서. 아직 사용자에 대해서 아는 것이 없음."
            
            prompt = [SystemMessage(cotent=system_msg)] + state["messages"]
            return {"message": [model.invoke(prompt)]}

    workflow = StateGraph(ChatState)
    workflow.add_node("agent", memory_agent_node)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    checkpointer = InMemorySaver()
    store = InMemoryStore()

    app = workflow.compile(checkpointer=checkpointer, store=store)

    config_1 = {"configurable": {"thread_id": "thread-1", "user_id": "user-jay"}}

    input1 = {"messages": [HumanMessage(content="내 이름은 Jay이고, 나는 매운 음식을 싫어해. 이거 꼭 기억해.")]}
    resp1 = app.invoke(input1, config=config_1)

    print(f"응답 1: {resp1['messages'][-1].content}")

    config_2 = {"configurable": {"thread_id": "thread-2", "user_id": "user-jay"}}

    input2 = {"messages": [HumanMessage(content="나 오늘 점심 메뉴 추천해 줘.")]}
    resp2 = app.invoke(input2, config=config_2)

    print(f"응답 2: {resp2['messages'][-1].content}")

def auto_save():
    """
    tool을 통해 사용자의 정보를 자동으로 저장해주는 방법.
    save_profile이 실행되면, save_node로 이동하여 tool에서 호출한 info를 저장.
    """
    @tool
    def save_profile(info: str):
        """
        사용자에 대한 중요한 정보(이른, 취미, 특징 등)를 저장할 때 사용.
        단순한 대화나 인사는 저장 X
        """
        print(f"저장한 정보: {info}")
        return "saved"
    tools = [save_profile]
    model_with_tools = model.bind_tools(tools)

    def agent_node(state: ChatState, config, store: BaseStore):
        user_id = config["configurable"]["user_id"]
        namespace = (user_id, "profile")

        memories = store.search(namespace)
        if memories:
            info = "\n".join([f"- {m.value['data']}" for m in memories])
            system_msg = f"당신은 사용자의 기억을 담당하는 비서입니다.\n[기억된 정보]\n{info}"
        else:
            system_msg = "당신은 사용자의 기억을 담당하는 비서입니다."

        return {"messages": [model_with_tools.invoke([SystemMessage(content=system_msg)] + state["messages"])]}

    def save_node(state: ChatState, config, store: BaseStore):
        user_id = config["configurable"]["user_id"]
        namespace = (user_id, "profile")

        last_message = state["messages"][-1]
        tool_outputs = []

        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "save_profile":
                info_to_save = tool_call["args"]["info"]
                print(f"\n💾 [System] AI의 자율적 판단으로 정보를 저장합니다: '{info_to_save}'")

                memory_id = str(uuid.uuid4())
                store.put(namespace, memory_id, {"data": info_to_save})

                tool_outputs.append(
                    ToolMessage(
                        content=f"정보 저장 완료: {info_to_save}",
                        tool_call_id=tool_call["id"]
                    )
                )

        return {"messages": tool_outputs}

    workflow = StateGraph(ChatState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("save_node", save_node)

    workflow.add_edge(START, "agent")

    def should_continue(state: ChatState):
        if state["messages"][-1].tool_calls:
            return "save_node"
        return END

    workflow.add_conditional_edges("agent", should_continue, ["save_node", END])

    workflow.add_edge("save_node", "agent") 

    app = workflow.compile(checkpointer=InMemorySaver(), store=InMemoryStore())

    config = {"configurable": {"thread_id": "1", "user_id": "user-jay"}}

    input1 = {"messages": [HumanMessage(content="안녕, 나는 샌프란시스코에 사는 Jay라고 해.")]}
    resp1 = app.invoke(input1, config=config)

    print(f"응답 1: {resp1['messages'][-1].content}")

    input2 = {"messages": [HumanMessage(content="내 이름이 뭐고 어디 산다고 했지?")]}
    resp2 = app.invoke(input2, config=config)
    print(resp2["messages"][-1].content)