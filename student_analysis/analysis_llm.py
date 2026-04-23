import os
import json
from dotenv import load_dotenv
load_dotenv()


from langchain.chat_models import init_chat_model
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


def get_llm(model = "gpt-4"):
    return init_chat_model(model = model)



def get_data():
    input_path = "/Users/kshi3430/CapTeam/data/student_analysis_data/students2.json"

    with open(input_path, "r", encoding="utf-8") as f:
        datas = json.load(f)  

    return datas



def get_rag_chain():
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
당신은 학생들에 실력을 분석하여 팀 매칭으로 이어갈수있도록 돕는 챗봇이다.
학생들의 실력을 분석할때는 아래에 학생데이터 student_data를 활용하여 답변하라.
학생들의 실력이 우수하니 평가기준을 높여서 답변하라.
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


#   {
#     "name": "이서연",
#     "goal": "Frontend",
#     "skills": ["html","css","js"],
#     "experience": [
#       "React로 포트폴리오 웹사이트 제작 및 반응형 UI 구현",
#       "쇼핑몰 메인 페이지 클론코딩 및 상태관리(Context API) 적용",
#       "Figma로 UI 설계 후 실제 코드로 구현"
#     ],
#     "collaboration": 4
#   },

class StudentAnalysis(BaseModel): # description 좀 자세하게 해야할듯 지금 어떤거에서 무슨 작업을 하는지 모르는듯
    skill_level: str = Field(description="student_data를 보고 학생의 전체적인 기술 수준 평가") #계속 ㅈ도 아닌 실력도 상으로 자꾸 매김
    strength: str = Field(description="student_data의 skills, experience를 기반으로 학생의 핵심 강점")
    weakness: str = Field(description="student_data의 skills, experience를 기반으로 학생의 부족한 부분")
    reason: str = Field(description="strength와 weakness와 skill_level을 그렇게 평가한 이유")
    role: str = Field(description="student_data의 goal과 skills를 기반으로 팀에서 맡기 적합한 역할")
    stack_score: str = Field(description="student_data의 skills의 점수 10점 만점")
    suggestion: str = Field(description="이 학생과 협업하면 좋을 역할 또는 기술 스택을 가진 사람 제안")


def get_analyze_stu():
    datas = get_data()
    llm = get_llm()
    s_prompt = get_rag_chain()
    structured_llm = llm.with_structured_output(StudentAnalysis)
    chain = s_prompt | structured_llm 

    for student in datas:
        response = chain.invoke({
            "student_data": json.dumps(student, ensure_ascii=False, indent=2), #prompt에서 받아야할 student_data 넣어줌
            "input": "이 학생의 실력을 분석해줘."  #사용자 input 입력
        })

        print("="*55)
        print(f"이름: {student['name']}")
        print(response)

get_analyze_stu()

#structured output사용해서 출력 구조 조정해주기 3.20
#git에 ai 파일구조 올리기.