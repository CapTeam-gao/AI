import os
import json
from dotenv import load_dotenv
load_dotenv()
import os
print(os.environ.get("UPSTAGE_API_KEY"))  
# from s_analysis_fewshot import examples
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder , FewShotChatMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal

# store = {}

# def get_session_history(session_id: str):
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]

#지금 upstage서버 요류인거같은데 내일되서 한번실행해봐야할듯
def get_llm(model = "solar-pro3"):
    return ChatUpstage(
        model = model)



def get_data():
    input_path = "/Users/kshi3430/CapTeam/data/student_analysis_data/students2.json"

    with open(input_path, "r", encoding="utf-8") as f:
        datas = json.load(f)  

    return datas



def get_prompt_chain():
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human","{input}"),
            ("ai", "{output}")
        ]
    )

    # few_shot_prompt = FewShotChatMessagePromptTemplate(
    #     examples=examples,
    #     example_prompt=example_prompt
    # )


    analysis_s_prompt = """
당신은 학생들에 코딩 실력을 분석하여 캡스톤 팀 매칭으로 이어갈수있도록 돕는 챗봇이다.
학생들의 실력을 분석할때는 아래에 학생데이터 student_data를 활용하여 답변하라.
평가 기준을 매우 엄격하게 적용하라.
없는 데이터를 만들어서 절대 사용하지 마라.


student_data : {student_data}
"""



    s_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",analysis_s_prompt),
            # few_shot_prompt,
            ("human","{input}")
        ]
    )


    return s_prompt



class StudentAnalysis(BaseModel):
    # 설명을 아무리 해줘도 제일 높은 skill_level에 조금이라도 해당 되면 지금 그냥 제일 높게 쳐주는거 같음 이분을 해결할수 있는 방법을 찾아야함.
    name : str = Field(description="student_data의 name을 그대로 여기에 작성하시오.")
    strength: str = Field(description="student_data의 stack, experience를 기반으로 학생의 핵심 강점")
    weakness: str = Field(description="student_data의 stack, experience를 기반으로 학생의 부족한 부분")
    reason: str = Field(description="strength와 weakness와 그렇게 평가한 이유")
    stack_score: str = Field(description="""
    student_data의 위에 있는 각 기술 스택별 점수를 experience를 보고 반영하여 10점 만점으로 평가.
    반드시 아래 형식으로 출력:
    스킬이름: 3점
    스킬이름: 1점
    
    점수 부여 기준 : 
    1~2점 : 기초적인 문법만.
    3~4점 : 튜토리얼 프로젝트 정도는 해봄.
    5~6점 : 간단한 프로젝트는 혼자서 구축가능함.
    7~8점 : 복잡한 프로젝트 구축가능함.
    9~10점 : 복잡한 프로젝트를 단순 구현이 아닌 아키텍처 수준에서 설계할 수 있으며, 성능 개선, 구조 개선, 기술 선택의 근거를 설명할 수 있음. 또한 문제 발생 시 원인을 분석하고 해결 전략을 스스로 수립할 수 있음.
    """)
    skill_level:  Literal['높음','보통','낮음'] = Field(description="""
    - 높음: 복잡한 팀 프로젝트를 독립적으로 설계, 구현, 문제 해결까지 수행
    - 보통: 복잡한 팀 프로젝트 구현 경험은 있으나 설계 또는 문제 해결 일부 부족 
    - 낮음: 단일 프로젝트 경험 또는 튜토리얼 기반 구현 수준, 기초 문법 및 기본 라이브러리 사용 수준
    
    다음 금지:
    - 일반적인 학습 경로를 근거로 보정하지 마라
    - 라이브러리 이름만으로 숙련도를 추정하지 마라

    다음 규칙:
    - 단일 프로젝트 또는 튜토리얼 기반 설계 경험이 있으면 '낮음'
    - 기초 문법 수준만 확인되면 '낮음'
    - 조건이 명확히 충족되지 않으면 상위 레벨 금지
    '높음'은 극소수의 학생에게만 해당되며, 대부분의 학생은 '보통' 또는 '낮음'이어야 한다.
    """)
    #skill_level이 너무 왔다갔다 함
    role: str = Field(description="student_data의 goal과 skills를 기반으로 팀에서 맡기 적합한 역할")
    suggestion: str = Field(description="이 학생과 협업하면 좋을 역할 또는 기술 스택을 가진 사람 제안")
    
    #fewshot으로 좀 추가해서 해야할듯.
    #if문으로 그냥 협업횟수 1보다 작거나 같으면 그냥 낮음 해버리는것도 괜찮을수도 

def get_analyze_stu():
    output_path = "/Users/kshi3430/CapTeam/data/student_analysis_data/analysis_output.json"
    # 이미 분석 결과 있으면 재사용
    # 테스트할때 토큰 아낄려고 사용하는건데 나중에 진짜 완성하면 있어도 그냥 덮어씌우는 형태로 갈듯
    #원래 사이즈 안보고 그냥 파일 존재 여부만보고 안에 내용이 있든지 말든지 신경안쓰고 해서 오류가 났었는데 안에 size가 있으면 기존결과대로 유지되고
    #size가 없으면 새로 분석하는그런 로직임.
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print("기존 분석 결과 불러오는 중...")

        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("분석 결과 없음. 새로 분석 시작...")

    datas = get_data()
    model = get_llm()
    s_prompt = get_prompt_chain()
    structured_llm = model.with_structured_output(StudentAnalysis)
    chain = s_prompt | structured_llm 
    results = []
    for student in datas:
        response = chain.invoke({
            "student_data": json.dumps(student, ensure_ascii=False, indent=2), #prompt에서 받아야할 student_data 넣어줌
            "input": "이 학생의 실력을 분석해줘."  #사용자 input 입력
        })

        print("="*55)
        print(f"이름: {student['name']}")
        print(response)
        results.append(response.model_dump())
        #리스트만들어서 답변나오면 그리스트 안에 들어가는 형식으로 이 만든걸 matching하는 llm에게 넘김.
        # 분석 결과 저장

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"분석 결과 저장 완료: {output_path}")

    return results

results = get_analyze_stu()

#structured output사용해서 출력 구조 조정해주기
#분석하면 잘나올때도 있고 못나올때도 있음 출력 기준이 계속 다른듯 이걸 좀 어떻게 해야할듯.