# 선호팀원 팀장선호 받아야함.
import os
import json
from dotenv import load_dotenv
load_dotenv(override=True)

from student_analysis.analysis_llm import results
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate


def load_analysis_output_json():
    #json파일 경로로 파일 가져와서 result에 넣음.
    with open("/Users/kshi3430/CapTeam/data/student_analysis_data/analysis_output.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    return results


def get_llm(model="solar-pro3"):
    return ChatUpstage(model=model)


def get_prompt_chain():
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human","{input}"),
            ("ai", "{output}")
        ]
    )
    matching_s_prompt = """
    당신은 학생들이 팀프로젝트를 할때 코팅실력을 분석한 내용을 바탕으로 팀을 매칭시켜주는 챗봇이다.
    학생들의 실력을 바탕으로 다른 팀들과 벨런스에 맞도록 매칭해라.
    학생들의 원하는 역할을 바탕으로 팀프로젝트를 원하는 분야에 맞게 팀을 매칭시켜라


student_output : {student_output}
"""

    s_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",matching_s_prompt),
            ("human","{input}")
        ]
    )


    return s_prompt

def get_response():
    results = load_analysis_output_json()
    llm = get_llm()
    s_prompt = get_prompt_chain()
    chain = s_prompt | llm
    #여기 밑으로 부터 매칭을 어떻게 나오게 할건지 짜야하고 ,선호팀원 ,팀장선호도 받아야함.
    output = []
    for student in results:
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
#python3 -m matching_student.matching_student 지금 matching_student폴더에서 실행하면 얘가 뒤로 갔다가 파일 찾아야해서 못찾음 그래서 그냥 capteam에서 실행해서 경로 지정해줘야함.