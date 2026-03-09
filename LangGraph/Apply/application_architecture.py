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

def evaluator_optimizer():
    """
    Evaluator-optimizer 워크플로우에서는 하나의 LLM 호출이 응답을 생성하고, 다른 하나는 그 응답을 평가.
    평가자(evaluator) 또는 human-in-the-loop(사람이 개입하는 평가 과정)가 응답이 개선이 필요하다고 판단하면, 피드백이 제공되고 응답은 다시 생성.
    이 과정은 만족할 만한 결과가 생성될 때까지 반복.

    Evaluator-optimizer 워크플로우는 작업에 대해 명확한 성공 기준이 존재하지만, 그 기준을 충족하기 위해 반복적인 개선이 필요한 경우에 주로 사용.
    예를 들어, 두 언어 간 번역을 할 때 항상 완벽하게 일치하는 표현이 바로 나오지는 않음.
    두 언어 간 의미가 동일하게 유지되도록 번역을 생성하기 위해 몇 차례의 반복이 필요할 수 있음.
    """
    class State(TypedDict):
        joke: str
        topic: str
        feedback: str
        funny_or_not: str

    class Feedback(BaseModel):
        grade: Literal["funny", "not funny"] = Field(
            description="Decide if the joke is funny or not.",
        )
        feedback: str = Field(
            description="If the joke is not funny, provide feedback on how to improve it.",
        )

    evaluator = llm.with_structured_output(Feedback)

    def llm_call_generator(state: State):
        """Joke를 생성하는 LLM"""

        if state.get("feedback"):
            msg = llm.invoke(
                f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
            )
        else:
            msg = llm.invoke(f"Write a joke about {state['topic']}")
        return {"joke": msg.content}


    def llm_call_evaluator(state: State):
        """Joke를 평가하는 LLM"""

        grade = evaluator.invoke(f"Grade the joke {state['joke']}")
        return {"funny_or_not": grade.grade, "feedback": grade.feedback}


    def route_joke(state: State):
        """평가자의 피드백에 따라 Joke 생성기로 다시 되돌리거나, 워크플로우를 종료"""

        if state["funny_or_not"] == "funny":
            return "Accepted"
        elif state["funny_or_not"] == "not funny":
            return "Rejected + Feedback"

    optimizer_builder = StateGraph(State)

    optimizer_builder.add_node("llm_call_generator", llm_call_generator)
    optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

    optimizer_builder.add_edge(START, "llm_call_generator")
    optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
    optimizer_builder.add_conditional_edges(
        "llm_call_evaluator",
        route_joke,
        { 
            "Accepted": END,
            "Rejected + Feedback": "llm_call_generator",
        },
    )

    optimizer_workflow = optimizer_builder.compile()

    state = optimizer_workflow.invoke({"topic": "Cats"})
    print(state["joke"])

def orchestrator_worker():
    """
    병렬화처럼 하위 작업을 미리 정의하기 어려운 경우, 더 큰 유연성을 제공하는 방식.
    코드를 작성하거나, 여러 파일에 거쳐 콘텐츠를 수정해야 하는 작업에서 자주 사용됨.
    """

    # planning을 위한 구조화된 스키마, 
    class Section(BaseModel):
        name: str = Field(
            description="Name for this section of the report.",
        )
        description: str = Field(
            description="Brief overview of the main topics and concepts to be covered in this section.",
        )

    class Sections(BaseModel):
        sections: List[Section] = Field(
            description="Sections of the report.",
        )

    planner = llm.with_structured_output(Sections)

    class State(TypedDict):
        topic: str # 주재
        sections: list[Section] # plan(목차)
        completed_sections: Annotated[
            list, operator.add
        ] # worker들의 결과를 이어붙인 것.
        final_report: str # 최종 결과물

    class WorkerState(TypedDict):
        section: Section
        completed_sections: Annotated[list, operator.add]

    def orchestrator(state: State):
        """레포트를 위한 계획을 작성하는 오케스트레이터"""

        # 쿼리 생성
        report_sections = planner.invoke(
            [
                SystemMessage(content="Generate a plan for the report."),
                HumanMessage(content=f"Here is the report topic: {state['topic']}"),
            ]
        )
        print(report_sections.sections)
        return {"sections": report_sections.sections}

    def llm_call(state: WorkerState):
        """Worker들은 레포트를 작성"""

        section = llm.invoke(
            [
                SystemMessage(
                    content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
                ),
                HumanMessage(
                    content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
                ),
            ]
        )

        return {"completed_sections": [section.content]}


    def synthesizer(state: State):
        """각 섹션들을 종합하여 하나의 보고서로 통합"""

        completed_sections = state["completed_sections"]
        completed_report_sections = "\n\n---\n\n".join(completed_sections)

        return {"final_report": completed_report_sections}


    # 보고서의 각 섹션을 작성하는 llm_call 워커들을 생성하기 위해, 조건부 엣지 함수를 만듬
    def assign_workers(state: State):
        """계획 각 부분들을 워커들에게 할당"""

        # Send() API를 통해 섹션 작성 작업을 병렬로 시작한다. 즉 리스트 개수만큼 노드를 생성.
        return [Send("llm_call", {"section": s}) for s in state["sections"]]

    orchestrator_worker_builder = StateGraph(State)

    orchestrator_worker_builder.add_node("orchestrator", orchestrator)
    orchestrator_worker_builder.add_node("llm_call", llm_call)
    orchestrator_worker_builder.add_node("synthesizer", synthesizer)

    orchestrator_worker_builder.add_edge(START, "orchestrator")
    orchestrator_worker_builder.add_conditional_edges(
        "orchestrator", assign_workers, ["llm_call"]
    )
    orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
    orchestrator_worker_builder.add_edge("synthesizer", END)

    orchestrator_worker = orchestrator_worker_builder.compile()

    state = orchestrator_worker.invoke({"topic": "Create a report on LLM scaling laws"})

    from IPython.display import Markdown
    Markdown(state["final_report"])