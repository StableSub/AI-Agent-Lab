from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool

load_dotenv()

model = init_chat_model("gpt-5-nano")

class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    context: dict

def dynamic_interrupt():
    """
    사용자가 맛집을 추천해 달라고 하고, 음식 종류를 넣지 않을 시 interrupt가 발생하여 음식 종류에 대한 사용자의 입력을 기다림.
    interrupt로 정보를 주입하고, command의 resume을 이용하여 다시 재게.
    """
    def agent_node(state: ChatState):
        print("[AI] 맛집 추천 노드 진입")
        ctx = state.get("context", {})
        food_category = ctx.get("food_category")

        if not food_category:
            print("[SYSTEM] 음시 종류 정보 부재, 사용자에게 질의 시도")

            food_category = interrupt("어떤 종류의 음식을 원하시나요?")

            print(f"[SYSTEM] 사용자 응답 수신: {food_category}")
        
        print(f"[AI] {food_category} 맛집을 생성 중...")

        system_prompt = SystemMessage(content=f"""
        당신은 서울의 맛집 전문가입니다.
        사용자가 원하는 카테고리인 '{food_category}'에 맞춰서
        실제로 유명한 맛집 1곳을 추천하고, 추천 이유를 2문장으로 설명해주세요.
        """)

        messages = [system_prompt] + state["messages"]
        response = model.invoke(messages)

        return {
            "messages": [response],
            "context": {"food_category": food_category}
        }

    workflow = StateGraph(ChatState)
    workflow.add_node("chatbot", agent_node)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)

    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)

    config_1 = {"configurable": {"thread_id": "1"}}
    input_msg1 = {
        "messages": [HumanMessage(content="강남역 근처 맛집 추천해줘.")],
        "context" : {}
    }

    response1 = app.invoke(input_msg1, config=config_1)

    snapshot = app.get_state(config_1)

    if snapshot.next:
        print(f"\n⚠️ 그래프 상태: {snapshot.next} (Interrupt 대기 중)")
        print(f"❓ 질문 내용: {snapshot.tasks[0].interrupts[0].value}")

    user_answer = str(input())

    resume_command = Command(resume=user_answer)

    response2 = app.invoke(resume_command, config=config_1)
    print(response2['messages'][-1].content)

def interrput_in_tool():
    """
    위치와 메뉴를 모를 때, tool을 호출한 뒤, tool의 인자, 즉 question 만을 빼내와서 사용자에게 interrupt로 필요한 정보를 다시 물어보는 구조.
    즉, 실제 도구가 아닌 도구에 전달되는 인자(AI가 사용자에게 원하는 추가 질문)를 가져와서 사용자에게 전달.
    """
    @tool
    def ask_human(question: str) -> str:
        """
        사용자에게 추가 정보를 물어볼 때 사용하는 도구.
        사용자의 답변을 받으려면 이 도구를 호출.
        """
        return "Human input required"
    
    tools = [ask_human]

    def agent_node(state: ChatState):
        model_with_tools = model.bind_tools(tools)
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def human_node(state: ChatState):
        """
        AI가 'ask_human' 도구를 호출했을 때 실행되는 노드.
        실제로 도구를 실행하는 대신, interrupt를 걸어 사용자 입력을 받음.
        """

        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        question_to_user = tool_call["args"]["question"]

        user_answer = interrupt(question_to_user)

        return {
            "messages": [
                ToolMessage(
                    content=user_answer,
                    tool_call_id=tool_call["id"]
                )
            ]
        }
        
    def should_continue(state: ChatState):
        last_message = state["messages"][-1]

        if last_message.tool_calls:
            if last_message.tool_calls[0]["name"] == "ask_human":
                return "human_node"
        return END

    workflow = StateGraph(ChatState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("human_node", human_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["human_node", END])
    workflow.add_edge("human_node", "agent")

    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)

    thread_config = {"configurable": {"thread_id": "2"}}

    input_data = {
        "messages": [
            SystemMessage(content="당신은 맛집 추천 비서입니다. 위치와 메뉴를 모르면 반드시 물어보세요."),
            HumanMessage(content="맛집 추천 좀 해줄래?")
        ]
    }

    app.invoke(input_data, config=thread_config)

    snapshot = app.get_state(thread_config)
    if snapshot.next:
        question = snapshot.tasks[0].interrupts[0].value
        print(f"\n⚠️ [Interrupt 발생] AI의 질문: {question}")
    
    natural_language_answer = str(input())

    result_2 = app.invoke(Command(resume=natural_language_answer), config=thread_config)
    print("\n" + result_2['messages'][-1].content)

def validate():
    """
    interrupt int tool의 humand node를 human validation node로 변경.
    미리 정한 음식 카테고리에 맞지 않으면 계속해서 interrupt를 통해 사용자에게 재질문을 요청.
    """
    @tool
    def ask_human(question: str) -> str:
        """
        사용자에게 추가 정보를 물어볼 때 사용하는 도구.
        사용자의 답변을 받으려면 이 도구를 호출.
        """
        return "Human input required"
    
    tools = [ask_human]

    def agent_node(state: ChatState):
        model_with_tools = model.bind_tools(tools)
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def human_validating_node(state: ChatState):
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        original_question = tool_call["args"]["question"]

        valid_categories = ["한식", "중식", "일식", "양식", "분식"]
        print(f"[SYSTEM] 사용자 질문 모드 집입. 허용 메뉴: {valid_categories}")

        current_prompt = original_question

        while True:
            user_input = interrupt(current_prompt)

            print(f"[CHECK] 사용자 입력값 검증 중: {user_input}")

            if any(category in user_input for category in valid_categories):
                print("[PASS] 유효한 메뉴.")

                return {
                    "messages": [
                        ToolMessage(
                            content=user_input,
                            tool_call_id=tool_call["id"]
                        )
                    ]
                }
            else:
                print("[FAIL] 메뉴에 없는 요청.")
                current_prompt= (
                    f"죄송합니다. {user_input}은 추천해드릴 수 없습니다."
                    f"가능한 메뉴는 {valid_categories} 입니다. 다시 말씀해주세요."
                )
        
    def should_continue(state: ChatState):
        last_message = state["messages"][-1]

        if last_message.tool_calls:
            if last_message.tool_calls[0]["name"] == "ask_human":
                return "human_node"
        return END

    workflow = StateGraph(ChatState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("human_node", human_validating_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["human_node", END])
    workflow.add_edge("human_node", "agent") 

    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)

    thread_config = {"configurable": {"thread_id": "3"}}

    input_data = {
        "messages": [
            SystemMessage(content="당신은 맛집 추천 비서입니다. 음식 종류가 없으면 반드시 ask_human 도구를 호출하세요."),
            HumanMessage(content="맛집 추천 좀 해줄래?")
        ]
    }

    app.invoke(input_data, config=thread_config)

    try:
        app.invoke(Command(resume="신발 튀김"), config=thread_config)
    except Exception:
        pass

    snapshot = app.get_state(thread_config)
    print(f"⚠️ [재질문] 내용: {snapshot.tasks[0].interrupts[0].value}")
    
    result = app.invoke(Command(resume="매운 한식이 땡겨"), config=thread_config)
    print("\n" + result['messages'][-1].content)