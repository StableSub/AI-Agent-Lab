from dotenv import load_dotenv

load_dotenv()

def node_style():
    """
    사용자의 메세지에 암구호가 들어있다면 Agent에게 메세지를 전달하기 이전에 AIMessage에 보안에 주의하라는 문구를 표현.
    Agent에게 메세지를 전달하기 이전에 확인을 해야하는 작업이므로 node_style의 before_agent를 사용.
    before_agent는 AI가 호출되기 전(사용자의 메세지가 전달될 때) 한 번 실행, before_model은 Model이 호출될 때마다(Tool 등으로 인해 여러번 모델이 호출된 경우) 실행.
    """
    from langchain.messages import AIMessage
    from langchain.agents.middleware import before_agent
    @before_agent(can_jump_to=["end"])
    def validate_input(state, runtime):
        human_message = state["messages"][-1]
        if "암구호" in human_message.content:
            print("암구호 감지")
            return {
                "messages": [AIMessage(content="현재 암구호가 포함되어 있습니다. 보안에 주의하세요.")],
                "jump_to": "end"
            }
        return None

    from langchain.agents import create_agent

    agent = create_agent(
        model="gpt-5-nano",
        middleware=[validate_input],
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "오늘의 암구호는 삼각대-자동차 입니다."}]}
    )

    print(response["messages"][-1].content)

def wrap_style():
    """
    사용자의 메세지 길이에 따라 다른 모델을 선택하는 Middleware.
    Agent가 호출될 때마다 확인을 해야하는 wrap_style로 Middleware 작성.
    """
    from langchain.agents.middleware import wrap_model_call
    from langchain.chat_models import init_chat_model
    
    @wrap_model_call
    def dynamic_model_selector(request, handler):
        last_message = request.messages[-1].content if request.messages else ""
        message_len = len(last_message)
        
        if message_len < 10:
            model_name = "gpt-5-nano"
        elif message_len < 30:
            model_name = "gpt-5-mini"
        else:
            model_name = "gpt-5"
        
        print(f"메세지 길이: {message_len}, 선택된 모델: {model_name}")
        new_model = init_chat_model(model_name)
        new_request = request.override(model=new_model)
        
        return handler(new_request)

    from langchain.agents import create_agent

    agent = create_agent(
        model="gpt-5-nano",
        middleware=[dynamic_model_selector],
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "너는 어떤 모델이야?"}]}
    )

    print(response["messages"][-1].content)
wrap_style()