"""
Threads: checkpointer가 저장하는 각 체크포인트에 할당되는 고유 ID.
    - 하나의 run이 실행되면, 해당 assistant의 그래프 상태는 thread에 영속적으로 저장됨.
    - checkpointer와 함께 그래프를 호출할 때는, 반드시 config의 configurable 영역에 thread_id를 지정해야 함.
    
Checkpoints: 특정 시점에서의 thread 상태를 checkpoint라고 함.
    - super-step마다 저장되는 그래프 상태의 스냅샷.
    - StateSnapshot 객체로 표현.
        config: 해당 체크포인트와 연관도니 설정 정보.
        metadata: 해당 체크포인트와 연관된 메타데이터.
        values: 해당 시점의 state 채널 값들.
        next; 그래프에서 다음에 실행될 노드 이름들의 튜플.
        tasks: 다음에 실행될 작업 정보를 답은 PregelTask 객체들의 튜플.

Memory Store: 사용자가 여러 스레드에서 대화를 하더라도 그 사용자에 대한 특정 정보를 모든 대화에서 계속 유지할 때 사용.
"""
from langchain_core.runnables import RunnableConfig
from langchain.embeddings import init_embeddings
from langchain.messages import SystemMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime

from dataclasses import dataclass
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
import uuid

def checkpoints():
    """
    그래프를 실행한 뒤에는 정확히 4개의 체크포인트가 생성.
        1. 비어 있는 체크포인트
        2. 사용자 입력이 반영된 체크포인트
        3. node_a의 출력이 반영된 체크포인트
        4. node_b의 출력이 반영된 체크포인트
    START -> Input -> node_a -> node_b 순서로 체크포인트가 생성됨.
    (bar 채널 값은 add 연산으로 인해 이전 출력값도 포함)
    """

    class State(TypedDict):
        foo: str
        bar: Annotated[list[str], add]

    def node_a(state: State):
        return {"foo": "a", "bar": ["a"]}

    def node_b(state: State):
        return {"foo": "b", "bar": ["b"]}


    workflow = StateGraph(State)
    workflow.add_node(node_a)
    workflow.add_node(node_b)
    workflow.add_edge(START, "node_a")
    workflow.add_edge("node_a", "node_b")
    workflow.add_edge("node_b", END)

    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    graph.invoke({"foo": "", "bar":[]}, config)

    def getState():
        """Get state: 해당 그래프의 최신 상태 또는 특정 checkpoint 부분을 조회."""
        config = {"configurable": {"thread_id": "1"}}
        graph.get_state(config)

        config = {"configurable": {"thread_id": "1", "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"}}
        graph.get_state(config)
    
    def getHistory():
        """Get state history: 특정 thread에 대한 전체 실행 이력을 조회."""
        config = {"configurable": {"thread_id": "1"}}
        list(graph.get_state_history(config))
    
    def replay():
        """Replay: 이전에 실행된 그래프를 다시 재생하게 해줌."""
        config = {"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}
        graph.invoke(None, config=config)
        
    def updateState():
        """특정 체크포인트부터 그래프 상태를 직접 수정 가능.(as_node 지정 시 그래프의 현재 위치를 조작 가능)"""
        graph.update_state(config, {"foo": "c", "bar": ["b"]})
   
def memorty_store():
    def basicUsage():
        """
        각 메모리 항목은 Item이라는 파이썬 클래스 객체이며, dict()로 딕셔너리 형태.
            value: 메모리 값
            key: 네임스페이스 내 고유 키
            namespace: 네임스페이스
            created_at: 생성 시각
            updated_at: 갱신 시각
        """
        store = InMemoryStore()
        user_id = "1"
        namespace_for_memory = (user_id, "memories")
        memory_id = str(uuid.uuid4())
        memory = {"food_preference" : "I like pizza"}
        store.put(namespace_for_memory, memory_id, memory)
    
    def semanticSearch():
        """
        의미 기반 검색을 지원하여 store을 임베딩 모델로 설정하여 자연어 쿼리로 관련 메모리를 찾을 수 있음.
        """
        store = InMemoryStore(
            index={
                "embed": init_embeddings("openai:text-embedding-3-small"),
                "dims": 1536,
                "fields": ["food_preference", "$"]
            }
        )
        user_id = "1"
        namespace_for_memory = (user_id, "memories")
        memories = store.search(
            namespace_for_memory,
            query="What does the user like to eat?",
            limit=3
        )

        store.put(
            namespace_for_memory,
            str(uuid.uuid4()),
            {
                "food_preference": "I love Italian cuisine",
                "context": "Discussing dinner plans"
            },
            index=["food_preference"]
        )

        store.put(
            namespace_for_memory,
            str(uuid.uuid4()),
            {"system_info": "Last updated: 2024-01-01"},
            index=False
        )
    
    def usingInLangGraph():
        """
        store는 checkpointer와 함께 동작.
            - checkpointer는 thread 단위로 상태를 저장.
            - store는 thread를 넘어 접근 가능한 임의 정보를 저장.
        """
        store = InMemoryStore()
        @dataclass
        class Context:
            user_id: str

        checkpointer = InMemorySaver()

        builder = StateGraph(MessagesState, context_schema=Context)
        graph = builder.compile(checkpointer=checkpointer, store=store)
    
        config = {"configurable": {"thread_id": "1"}}

        for update in graph.stream(
            {"messages": [{"role": "user", "content": "hi"}]},
            config,
            stream_mode="updates",
            context=Context(user_id="1"),
        ):
            print(update)

        async def update_memory(state: MessagesState, runtime: Runtime[Context]):
            """
            노드 함수에서 Runtime 객체를 파라미터로 추가하면 LangGraph가 자동으로 Runtime을 주입해주어 Store에 접근할 수 있음.
            """
            user_id = runtime.context.user_id

            namespace = (user_id, "memories")

            memory_id = str(uuid.uuid4())

            await runtime.store.aput(namespace, memory_id, {"memory": "사용자는 커피를 좋아한다."})
        
        async def call_model(state: MessagesState, runtime: Runtime[Context]):
            """
            메모리는 .search를 통해 조회 가능. 
            각 메모리는 객체로 반환되며, .dict()로 확인이 가능.
            """
            user_id = runtime.context.user_id

            namespace = (user_id, "memories")

            memories = await runtime.store.asearch(
                namespace,
                query=state["messages"][-1].content,
                limit=3
            )
            info = "\n".join([d.value["memory"] for d in memories])

        # thread_id가 변경되어도 user_id가 같아 같은 메모리에 접근할 수 있음.
        config = {"configurable": {"thread_id": "2"}}

        for update in graph.stream(
            {"messages": [{"role": "user", "content": "hi, tell me about my memories"}]},
            config,
            stream_mode="updates",
            context=Context(user_id="1"),
        ):
            print(update)
    