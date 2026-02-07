from dotenv import load_dotenv

load_dotenv()

def basic():
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()

    user_id = "user_001"
    application_context = "personal_assistant"

    namespace = (user_id, application_context)

    store.put(
        namespace,
        "memory_001",
        {
            "facts": [
                "사용자는 커피보다 차를 선호함",
                "사용자는 매일 아침 6시에 일어남",
            ],
            "language": "Korean",
        }
    )

    item = store.get(namespace, "memory_001")
    items = store.search(namespace)

    store.put(
        namespace,
        "memory_002",
        {
            "facts": [
                "사용자는 그림 회화 작품을 좋아함",
                "빈센트 반 고흐의 작품을 특히 좋아함"
            ]
        }
    )

    item = store.get(namespace, "memory_002")
    items = store.search(namespace)

    from dataclasses import dataclass

    @dataclass
    class Context:
        user_id: str
        app_name: str
        
    from langchain.agents.middleware import wrap_model_call
    from langchain.messages import HumanMessage, SystemMessage

    @wrap_model_call
    def inject_memory(request, handler):
        current_user = request.runtime.context.user_id
        current_app = request.runtime.context.app_name
        memories = request.runtime.store.search((current_user, current_app))
        
        memory_content = "기록된 정보 없음"
        if memories:
            extracted_facts = []
            for item in memories:
                if "facts" in item.value:
                    extracted_facts.extend(item.value["facts"])
            memory_content = "\n- ".join(extracted_facts)
            
        system_message = f"사용자 관련 장기 메모리: {memory_content}"
        new_request = request.override(system_prompt=system_message)
        return handler(new_request)

    from langchain.agents import create_agent

    agent = create_agent(
        model="gpt-5-nano",
        store=store,
        context_schema=Context,
        middleware=[inject_memory]
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "나에 대해 알고있는 정보 알려줘"}]},
        context=Context(user_id="user_001", app_name="personal_assistant")
    )

    print(response)
    
def tool_base_memory():
    from dataclasses import dataclass
    from langgraph.store.memory import InMemoryStore
    
    @dataclass
    class Context:
        user_id: str
        app_name: str
    
    store = InMemoryStore()
    
    from typing import TypedDict
    
    class UserInfo(TypedDict):
        personal_info: str
        preference: str
    
    from langchain.tools import tool
    
    @tool
    def get_user_info(runtime) -> str:
        """
        현재 사용자의 정보 조회 (시스템 내부용 도구)
        """
        user_id = runtime.context.user_id
        app = runtime.context.app_name
        
        memories = runtime.store.search((user_id, app))
        
        if not memories:
            return "기록된 정보 없음"
        
        results = []
        for item in memories:
            data = item.value
            if "personal_information" in data:
                results.append(f"- 개인정보: {data['personal_information']}")
            if "preference" in data:
                results.append(f"- 선호도: {data['preference']}")
        return "\n".join(results) if results else "데이터 형식 불일치로 읽을 수 없음"

    import uuid
    @tool
    def save_user_info(user_info: UserInfo, runtime) -> str:
        """
        사용자의 정보를 저장하거나 업데이트
        """
        user_id = runtime.context.user_id
        app = runtime.context.app_name
        store = runtime.store
        
        memory_key = str(uuid.uuid4())
        store.put((user_id, app), memory_key, user_info)
        
        return f"정보가 안전하게 저장되었습니다. (ID: {memory_key})"
    
    from langchain.agents import create_agent
    
    agent = create_agent(
        model="gpt-5-nano",
        tools=[get_user_info, save_user_info],
        store=store,
        context_schema=Context
    )
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "내 이름은 정섭이야. 차보단 커피를 좋아해. 이 내용을 기억해."}]},
        context=Context(user_id="user_001", app_name="personal_assistant")
    )
    print(response)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "나에 대해 아는 정보를 말해."}]},
        context=Context(user_id="user_001", app_name="personal_assistant")
    )
    print(response)
tool_base_memory()
