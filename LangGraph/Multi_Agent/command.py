from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage

from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gpt-5-nano")

class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    selected_menu: str

# [Node 1] 메뉴 추천
def recommender_node(state: ChatState):
    # 현재 상태(이전에 추천했던 메뉴)를 확인
    last_menu = state.get("selected_menu")

    # 만약 이전에 A코스를 추천했다면 -> 거절당하고 다시 온 것이므로 B코스 추천
    if last_menu == "A 코스 (20만원)":
        print("\n🍣 [Recommender] 그렇다면. 실속형 'B 코스 (8만원)'을 추천합니다.")
        return {"selected_menu": "B 코스 (8만원)"}

    # 기본(최초) 추천
    else:
        print("\n[Recommender] 쉐프 추천: 'A 코스 (20만원)'")
        return {"selected_menu": "A 코스 (20만원)"}


# [Node 2] 승인/거절 및 경로 변경 (Router 역할 겸임)
def human_approval_node(state: ChatState):
    menu = state["selected_menu"]

    print(f"\n👮 [Approval] '{menu}'은(는) 고가이므로 승인이 필요합니다.")

    # 1. 멈추고 물어봄 (Dynamic Interrupt)
    # 멈춘 후 Command(resume="yes" 또는 "no")로 전달된 값이 approved 변수에 들어옵니다.
    approved = interrupt(f"'{menu}' 예약을 진행하시겠습니까? (yes/no)")

    # 2. [핵심] 답변에 따라 즉시 순간이동 (Command with goto)
    if approved == "yes":
        print("▶️ [Decision] 승인됨 -> 예약 확정으로 이동")

        # update: 상태 업데이트 (예약 승인 로그 남기기)
        # goto: 다음 실행할 노드 이름 명시
        return Command(
            update={"messages": ["사용자가 예약을 승인했습니다."]},
            goto="booking",  # booking 노드로 즉시 점프!
        )

    else:
        print("▶️ [Decision] 거절됨 -> 다시 추천받으러 이동")

        return Command(
            update={"messages": ["사용자가 거절했습니다. 다른 메뉴를 찾습니다."]},
            goto="recommender",  # recommender 노드로 즉시 점프!
        )


# [Node 3] 예약 확정 (승인 시 이동할 곳)
def booking_node(state: ChatState):
    menu = state["selected_menu"]
    print(f"\n✅ [Booking] '{menu}' 예약이 확정되었습니다! (문자 발송 완료)")
    return {"messages": [f"'{menu}' 예약 완료"]}

workflow = StateGraph(ChatState)

workflow.add_node("recommender", recommender_node)
workflow.add_node("approval", human_approval_node)
workflow.add_node("booking", booking_node)

# 고정된 기본 흐름만 연결
workflow.add_edge(START, "recommender")
workflow.add_edge("recommender", "approval")

checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "goto-demo"}}

# 1. 최초 실행
app.invoke({"messages": [HumanMessage(content="예약해줘")]}, config=config)

# 2. 대기 상태 확인
snapshot = app.get_state(config)
if snapshot.next:
    print(f"⚠️ 대기 중: {snapshot.tasks[0].interrupts[0].value}")

# 3. 'no'를 입력하여 흐름 재개
app.invoke(Command(resume="no"), config=config)

# 4. 바뀐 대기 상태 확인
snapshot = app.get_state(config)
if snapshot.next:
    print(f"⚠️ 대기 중: {snapshot.tasks[0].interrupts[0].value}")

# 5. 'yes'를 입력하여 최종 승인
result = app.invoke(Command(resume="yes"), config=config)

print("\n최종 메시지 내역:", result["messages"][-1].content)