from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
from IPython.display import Image, display
from typing import Annotated, List
import operator
from typing_extensions import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage, SystemMessage
from langgraph.types import Send

llm = init_chat_model(model="gpt-5-nano")

def prompt_chaining():
    """
    프롬프트 채이닝은 각 LLM 호출이 이전 호출의 출력을 입력으로 받아서 다음 호출을 수행하는 방식.
    명확히 정의된 작업을 더 작고 검증 가능한 단계로 쪼개서 처리할 때 사용.
    """
    class State(TypedDict):
        topic: str
        joke: str
        improved_joke: str
        final_joke: str
    
    def generate_joke(state: State):
        """초기 joke를 생성하기 위한 첫번째 LLM Call"""
        msg = llm.invoke(f"write a shrot joke about {state['topic']}, ?나 !를 포함하지 마시오.")
        return {"joke": msg.content}
    
    def check_punchline(state: State):
        """joke가 펀치라인을 가지고 있는지 체크하는 함수 게이트"""
        if "?" in state['joke'] or "!" in state['joke']:
            return "Pass"
        return "Fail"
    
    def improve_joke(state: State):
        """joke를 향상시키기 위한 두번째 LLM Call"""
        msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
        return {"improved_joke": msg.content}
    
    def polish_joke(state: State):
        """촤종 다듬기를 진행할 세번째 LLM Call"""
        msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
        return {"final_joke": msg.content}
    
    # Workflow 생성
    workflow = StateGraph(State)
    
    # 노드 추가
    workflow.add_node("generate_joke", generate_joke)
    workflow.add_node("improve_joke", improve_joke)
    workflow.add_node("polish_joke", polish_joke)
    
    # 엣지를 이용하여 노드끼리 연결
    workflow.add_edge(START, "generate_joke")
    workflow.add_conditional_edges(
        "generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END}
    ) # check_punchline의 결과에 따라 분기를 선택
    workflow.add_edge("improve_joke", "polish_joke")
    workflow.add_edge("polish_joke", END)
    
    chain = workflow.compile()
    
    state = chain.invoke({"topic": "cats"})
    print("Initial joke:")
    print(state["joke"])
    print("\n--- --- ---\n")
    if "improved_joke" in state:
        print("Improved joke:")
        print(state["improved_joke"])
        print("\n--- --- ---\n")

        print("Final joke:")
        print(state["final_joke"])
    else:
        print("Final joke:")
        print(state["joke"])

def parallelization():
    """
    병렬화는 LLM이 하나의 작업을 동시에 병렬로 처리하는 방식.
    1. 서로 독립적인 여러 하위 작업을 동시에 실행하거나,
    2. 같은 작업을 여러 번 실행해 서로 다른 결과를 비교 및 검증하는 방식으로 이루어짐.
    사용 목적
    - 작업을 여러 하위 작업으로 나눠 병렬 실행함으로써 처리 속도를 높이기 위해.
    - 동일한 작업을 여러 번 실행해 결과를 비교함으로써 결과에 대한 신뢰도를 높이기 위해.
    """
    # 그래프 상태
    class State(TypedDict):
        topic: str
        joke: str
        story: str
        poem: str
        combined_output: str

    # 노드
    def call_llm_1(state: State):
        """초기의 joke를 생성"""
        msg = llm.invoke(f"Write a joke about {state['topic']}")
        return {"joke": msg.content}

    def call_llm_2(state: State):
        """이야기를 생성"""
        msg = llm.invoke(f"Write a story about {state['topic']}")
        return {"story": msg.content}

    def call_llm_3(state: State):
        """시를 생성"""
        msg = llm.invoke(f"Write a poem about {state['topic']}")
        return {"poem": msg.content}

    def aggregator(state: State):
        """joke와 이야기, 시를 결합하는 과정"""
        combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
        combined += f"STORY:\n{state['story']}\n\n"
        combined += f"JOKE:\n{state['joke']}\n\n"
        combined += f"POEM:\n{state['poem']}"
        return {"combined_output": combined}

    parallel_builder = StateGraph(State)

    parallel_builder.add_node("call_llm_1", call_llm_1)
    parallel_builder.add_node("call_llm_2", call_llm_2)
    parallel_builder.add_node("call_llm_3", call_llm_3)
    parallel_builder.add_node("aggregator", aggregator)

    parallel_builder.add_edge(START, "call_llm_1")
    parallel_builder.add_edge(START, "call_llm_2")
    parallel_builder.add_edge(START, "call_llm_3")
    parallel_builder.add_edge("call_llm_1", "aggregator")
    parallel_builder.add_edge("call_llm_2", "aggregator")
    parallel_builder.add_edge("call_llm_3", "aggregator")
    parallel_builder.add_edge("aggregator", END)
    parallel_workflow = parallel_builder.compile()

    state = parallel_workflow.invoke({"topic": "cats"})
    print(state["combined_output"])

def routing():
    """
    입력을 먼저 분석한 뒤, 그 결과에 따라 상황에 맞는 작업으로 분기시키는 방법.
    복잡한 작업에 대해 전문화된 처리 흐름을 각각 정의할 수 있음.
    """
    # 라우팅 로직 스키마
    class Route(BaseModel):
        step: Literal["poem", "story", "joke"] = Field(
            None, description="The next step in the routing process"
        )
        
    router = llm.with_structured_output(Route)

    class State(TypedDict):
        input: str
        decision: str
        output: str

    def llm_call_1(state: State):
        """이야기를 적는 LLM Call"""

        result = llm.invoke(state["input"])
        return {"output": result.content}


    def llm_call_2(state: State):
        """Joke를 적는 LLM Call"""

        result = llm.invoke(state["input"])
        return {"output": result.content}


    def llm_call_3(state: State):
        """시를 적는 LLM Call"""

        result = llm.invoke(state["input"])
        return {"output": result.content}


    def llm_call_router(state: State):
        """입력을 통해 적절한 노드로 Route."""

        decision = router.invoke(
            [
                SystemMessage(
                    content="Route the input to story, joke, or poem based on the user's request."
                ),
                HumanMessage(content=state["input"]),
            ]
        )

        return {"decision": decision.step}

    def route_decision(state: State):
        """적절한 노드로 가기 위한 분기 edge"""
        if state["decision"] == "story":
            return "llm_call_1"
        elif state["decision"] == "joke":
            return "llm_call_2"
        elif state["decision"] == "poem":
            return "llm_call_3"

    router_builder = StateGraph(State)

    router_builder.add_node("llm_call_1", llm_call_1)
    router_builder.add_node("llm_call_2", llm_call_2)
    router_builder.add_node("llm_call_3", llm_call_3)
    router_builder.add_node("llm_call_router", llm_call_router)

    router_builder.add_edge(START, "llm_call_router")
    router_builder.add_conditional_edges(
        "llm_call_router",
        route_decision,
        { 
            "llm_call_1": "llm_call_1",
            "llm_call_2": "llm_call_2",
            "llm_call_3": "llm_call_3",
        },
    )
    router_builder.add_edge("llm_call_1", END)
    router_builder.add_edge("llm_call_2", END)
    router_builder.add_edge("llm_call_3", END)

    router_workflow = router_builder.compile()

    state = router_workflow.invoke({"input": "Write me a joke about cats"})
    print(state["output"])