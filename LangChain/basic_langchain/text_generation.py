"""
기본적인 Text Generation.
invoke, stream, batch의 형태로 답변을 받을 수 있음.
구조화된 출력을 통해 원하는 형태의 출력을 얻을 수 있음.
"""
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# temperature(창의성), max_tokens(용량), timeout(시간), max_retries(안정성) 변수 조절 가능.
model = init_chat_model(
    "gpt-5-nano",
    temperature=0.9,
    api_key = api_key)

def func_invoke():
    """ 
    AI에게 질문을 하는 역할
    json 형태로 답변, 메타데이터 등을 반환
    """
    response = model.invoke("넌 누구니?")

    print(f"응답 출력: {response}\n")
    print(f"답변 출력: {response.content}\n")
    print(f"메타 데이터 출력: {response.usage_metadata}")

def func_stream():
    """ 
    stream은 AI의 답변을 청크 단위로 가져와서 연속적인 출력을 가능하게 해줌
    """
    for chunk in model.stream("AI Agent란 무엇이야?"):
        print(chunk.text, end="")

def func_batch():
    """ 
    batch는 여러가지 질문을 병렬로 AI에게 전달하는 역할
    """
    reponses = model.batch([
        "과적합이란?",
        "앵무새의 털이 화려한 이유?",
        "AI Agent 기반 서비스는 무엇이 있어?"
    ])

    for reponse in reponses:
        print(reponse)
        
def structured_output_pydantic():
    """ 
    다음 작업을 위해 답변을 특정 형태로 파싱할 때 사용
    pydantic을 사용하면 자동으로파싱 + 데이터 검증까지 지원
    """
    from pydantic import BaseModel, Field

    class Movie(BaseModel):
        """상세한 영화 정보."""
        title: str = Field(description="영화의 제목")
        year: int = Field(description="개봉 연도")
        director: str = Field(description="영화 감독 이름")
        rating: float = Field(description="영화 평점 (10점 만점)")
    model_with_structure = model.with_structured_output(Movie)
    print(model_with_structure.invoke("인셉션 영화에 대해서 설명해."))
    
    
def structured_output_json():
    """ 
    다음 작업을 위해 답변을 특정 형태로 파싱할 때 사용
    json은 pydantic과 다르게 파싱만 지원
    """
    import json
    json_schema = {
        "title": "Movie",
        "description": "A movie with details",
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "The title of the movie"},
            "year": {"type": "integer", "description": "The year the movie was released"},
            "director": {"type": "string", "description": "The director of the movie"},
            "rating": {"type": "number", "description": "The movie's rating out of 10"},
        },
        "required": ["title", "year", "director", "rating"]
    }
    model_with_json_schema = model.with_structured_output(json_schema)
    print(model_with_json_schema.invoke("인셉션 영화에 대해서 설명해."))
    
