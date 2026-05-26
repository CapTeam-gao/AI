import os
import json
import re
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

# from s_analysis_fewshot import examples
from langchain_upstage import ChatUpstage
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

#apikey 다시파야할듯.
#클로드코드, codex한번 사서 써봐야할듯.
def get_llm(model = "solar-pro3"):
    llm = ChatUpstage(model = model, temperature=0.05) #모델 랜덤성이 좀 있길레 temperature=0으로 해줌
    return llm




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

평가 원칙:
- 기술 이름을 안다고 높은 점수를 주지 마라.
- "간단한", "입문", "클론코딩", "투두리스트", "폼 뼈대", "튜토리얼" 수준의 경험은 낮게 평가하라.
- 경험에 직접 적힌 내용만 근거로 사용하라. 가능성, 추정, 일반적인 학습 경로는 근거가 아니다.
- 9~10점은 아키텍처 설계, 성능 개선, 운영/장애 대응, 복잡한 문제 해결 근거가 명확할 때만 준다.
- 7~8점은 여러 기능이 결합된 프로젝트를 스스로 구축한 근거가 있을 때만 준다.
- 단순 구현, 간단한 서버, 간단한 agent, 클론코딩, 내장 데이터 실습은 6점 이하로 제한한다.

skill_level 판정:
- 높음: 복잡한 팀 프로젝트를 독립적으로 설계하고, 문제 해결/성능 개선/운영까지 수행한 근거가 명확한 경우만.
- 보통: 여러 기능이 결합된 프로젝트 구현 경험은 있으나 설계/운영/고급 문제 해결 근거가 부족한 경우.
- 낮음: 입문, 튜토리얼, 클론코딩, 단일 기능 구현, 간단한 프로젝트 위주인 경우.
- 협업 횟수 collaboration이 1 이하이면 무조건 낮음으로 평가하라.
- "간단한"이라는 표현이 핵심 경험에 포함되어 있으면 높음 금지.

출력 전 자체 검증:
- 내가 학생을 높음으로 평가했다면, student_data 안에 아키텍처/성능개선/운영/복잡한 문제해결 근거가 직접 있는지 확인하라.
- 근거가 없으면 보통 또는 낮음으로 낮춰라.

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
    
#이거 분류하는것도 llm에게 넣는것도 나쁘지 않은듯.
#학생경험에 이런단어가 많으면 낮게 보려고 만든 리스트(고평가 하지않도록 맞는 용도)
BEGINNER_WORDS = [
    "간단", "입문", "튜토리얼", "클론", "투두", "todo", "뼈대",
    "기본", "가능", "내장 데이터", "자기소개서", "로그인 폼"
]
#이거는 한게 뭐가 있는지 , 실제로 뭘했는지 흔적을 찾는 단어 
IMPLEMENTATION_WORDS = [
    "구현", "제작", "개발", "구축", "작성", "연동", "적용", "구성",
    "처리", "수집", "분류", "분석", "시각화", "개선", "향상", "배포"
]
#이거는 높음을 주기위해서 근거를 찾는 단어 리스트.
ADVANCED_WORDS = [
    "아키텍처", "설계", "성능 개선", "성능 최적화", "운영", "장애",
    "대규모", "분산", "테스트 자동화", "ci/cd", "모니터링", "리팩토링",
    "문제 해결 전략", "트러블슈팅", "최적화"
]

#이 함수는 위에서 만든 단어들 분류를 쉽게 할려고 하나에 문자열로 합쳐서 소문자로 변환하는 함수
def _student_text(student):
    experience = " ".join(student.get("experience", []))
    #experience get함수 써서 value(리스트)값을 가져와서 그 리스트를 문자열로 반환
    stack = " ".join(student.get("stack", []))
    #위와 같음.
    return f"{student.get('goal', '')} {stack} {experience}".lower()
    #goal value값 가져와서 stack 값이랑 experience랑 소문자로 같이 반환



def _count_implementation_evidence(student):
    #구현 경험 value 리스트를 가져와서
    experiences = student.get("experience", [])
    return sum(
        1 #관련 키워드가 있으면 1 카운트
        for experience in experiences #experiences 리스트 인덱스 하나씩 순회
        if any(word in experience.lower() for word in IMPLEMENTATION_WORDS)
        #IMPLEMENTATION_WORDS 단어중 하나라도 포함되면 True , True면 1을 반환(any = 조건중 하나라도 true면 true반환하는 python내장함수)
        
    )

#위에서 분류해서 합계한걸로 skill level을 분류함.
#근데 이게 좀 난이도가 있는 기술 튜토리얼 완료 한거랑 , 난이도 없는 기술 튜토리얼 완료한거랑 완전 기준이 달라서 애매함.
def decide_skill_level(student):
    text = _student_text(student)
    collaboration = int(student.get("collaboration", 0) or 0)
    experience_count = len(student.get("experience", []))
    implementation_count = _count_implementation_evidence(student)

    if collaboration <= 1:
        return "낮음"

    beginner_count = sum(1 for word in BEGINNER_WORDS if word.lower() in text)
    advanced_count = sum(1 for word in ADVANCED_WORDS if word.lower() in text)

    if advanced_count >= 2 and implementation_count >= 3 and beginner_count == 0:
        return "높음"

    if beginner_count >= 2 and implementation_count <= 2:
        return "낮음"

    if implementation_count >= 3 or (implementation_count >= 2 and experience_count >= 3):
        return "보통"

    return "낮음"


def cap_stack_score(stack_score, skill_level):
    #skill_level별로 stack_score max값을 조정
    max_score_by_level = {
        "낮음": 4,
        "보통": 7,
        "높음": 10,
    }
    #현재 skill level의 최대 점수 가져오기
    max_score = max_score_by_level[skill_level]
    #"기술명: n점" 형식 문자열 추출
    score_matches = re.findall(
        r"([A-Za-z가-힣0-9#.+/\- ]{1,30})\s*:\s*(\d{1,2})점",
        stack_score,
    )
    if score_matches:
        lines = []
        seen = set() # 중복 기술명 제거용
        for skill, score_text in score_matches:
            # 기술명 정리 및 중복/불필요 항목 제거
            skill = skill.strip(" -\n\t")
            key = skill.lower()
            if not skill or key in seen or "기준" in skill:
                continue

            seen.add(key)
            #현재 점수와 최대 점수 비교해서 더 작은 값 사용
            score = min(int(score_text), max_score)
            lines.append(f"{skill}: {score}점") #수정된 점수 리스트에 저장

        if lines: #최종 문자열 반환
            return "\n".join(lines)
    #lines가 비어있다면 
    def replace_score(match):
        score = int(match.group(1))
        return f"{min(score, max_score)}점" #maxscore과 score중에 낮은걸 반환해서 최대점수보다 낮게 만들어서 위에코드처럼 만듬

    return re.sub(r"(\d{1,2})점", replace_score, stack_score)

#여기코드 문제가 좀 있는듯
def normalize_analysis(response, student):
    # response가 pydantic 객체인지 확인
    if hasattr(response, "model_dump"):
        #pydantic객체면 dict으로 변환
        result = response.model_dump()
        #이미 딕이면 딕으로
    else:
        result = dict(response)

    #student 데이터를 기반으로 규칙 기반 skill level 재판정
    fixed_level = decide_skill_level(student)

    #LLM이 원래 판단한 skill level 저장
    original_level = result.get("skill_level")

    #LLM 결과와 상관없이 규칙 기반 skill level로 강제 덮어쓰기
    result["skill_level"] = fixed_level

    # 현재 skill level 기준으로 stack_score 최대값 제한 적용
    result["stack_score"] = cap_stack_score(
        result.get("stack_score", ""),  # stack_score 없으면 빈 문자열
        fixed_level
    )

    # 원래 skill level과 보정된 skill level이 다를 경우
    if original_level != fixed_level:

        # 최종 skill level이 "보통"일 때 보정 메시지 생성
        if fixed_level == "보통":
            correction = (
                f"최종 판정 보정: 입력 데이터 기준으로 '{original_level}'이 아니라 "
                f"'{fixed_level}'으로 판정한다. 고급 설계/운영 근거는 부족하지만, "
                "여러 기능을 결합해 구현한 경험은 확인된다."
            )

        # 최종 skill level이 "낮음" 또는 "높음"일 때 보정 메시지 생성
        else:
            correction = (
                f"최종 판정 보정: 입력 데이터 기준으로 '{original_level}'이 아니라 "
                f"'{fixed_level}'으로 판정한다. 간단한 구현/튜토리얼/단일 기능 수준의 근거는 "
                "상위 레벨의 근거로 사용하지 않는다."
            )

        # 기존 reason 뒤에 보정 사유 추가
        result["reason"] = (
            f"{result.get('reason', '')}\n\n{correction}"
        )

    # 최종 보정된 분석 결과 반환
    return result


def get_analyze_stu():
    output_path = "/Users/kshi3430/CapTeam/data/student_analysis_data/analysis_output.json"
    # 이미 분석 결과 있으면 재사용
    # 테스트할때 토큰 아낄려고 사용하는건데 나중에 진짜 완성하면 있어도 그냥 덮어씌우는 형태로 갈듯
    #원래 사이즈 안보고 그냥 파일 존재 여부만보고 안에 내용이 있든지 말든지 신경안쓰고 해서 오류가 났었는데 안에 size가 있으면 기존결과대로 유지되고
    #size가 없으면 새로 분석하는그런 로직임.
    datas = get_data()
    #이코드 알아야함.
    force_reanalyze = os.getenv("FORCE_REANALYZE", "false").lower() == "true"

    if not force_reanalyze and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print("기존 분석 결과 불러오는 중...")

        with open(output_path, "r", encoding="utf-8") as f:
            cached_results = json.load(f)

        normalized_results = [
            normalize_analysis(result, student)
            for result, student in zip(cached_results, datas)
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(normalized_results, f, ensure_ascii=False, indent=2)

        return normalized_results

    print("분석 결과 없음. 새로 분석 시작...")
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
        normalized_response = normalize_analysis(response, student)
        print(normalized_response)
        results.append(normalized_response)
        #리스트만들어서 답변나오면 그리스트 안에 들어가는 형식으로 이 만든걸 matching하는 llm에게 넘김.
        # 분석 결과 저장

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"분석 결과 저장 완료: {output_path}")

    return results

results = None

if __name__ == "__main__":
    results = get_analyze_stu()

#팀장선호와 원하는팀원을 받아야하는데 이걸 어떻게 받을지 정해야함. 원하는팀원은 몇명받을건지 팀장선호수가 넘치면 어떻게 할건지.
#스택으로 점수를 부여하는것을 좀 해야할듯 지금 허재원 react실력이 박건욱 html실력이랑 비슷하다고 해서 점수가 비슷함 이부분을 해결할수있도록 만들어야할듯