import os
import json
import re
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"), override=False)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

# from s_analysis_fewshot import examples
from langchain_upstage import ChatUpstage
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder , FewShotChatMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional

from capteam_db import fetch_analysis_results, fetch_students, save_analysis_results
from capteam_preferences import ensure_preference_profile
from capteam_traits import ensure_trait_profile

# store = {}

# def get_session_history(session_id: str):
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]

#apikey 다시파야할듯.
#클로드코드, codex한번 사서 써봐야할듯.
#이름 말고 학번으로 주 식별자.
#지금 매칭로직에서 score들이 되게 중요함 그래서 score들 점수를 잘 매겨야할듯. 상중하도 
def get_llm(model = "solar-pro3"):
    llm = ChatUpstage(
        model=model,
        temperature=0,
    ) #모델 랜덤성이 좀 있길레 temperature=0으로 해줌
    return llm




def get_data():
    return fetch_students()



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
skill_evidence는 알고리즘이 계산한 참고용 근거 점수이며 최종 판정이 아니다.
최종 skill_level과 stack_score는 student_data의 experience를 우선으로 보고 판단하라.
student_data에 성격/개발 성향 점수가 있으면 협업 방식과 추천 역할 설명에만 참고하고,
기술 숙련도 점수를 부풀리는 근거로 사용하지 마라.

평가 원칙:
- 기술 이름을 안다고 높은 점수를 주지 마라.
- "간단한", "입문", "클론코딩", "폼 뼈대", "튜토리얼" 수준의 경험은 낮게 평가하라.
- 경험에 직접 적힌 내용만 근거로 사용하라. 가능성, 추정, 일반적인 학습 경로는 근거가 아니다.
- 9~10점은 아키텍처 설계, 성능 개선, 운영/장애 대응, 복잡한 문제 해결 근거가 명확할 때만 준다.
- 7~8점은 여러 기능이 결합된 프로젝트를 스스로 구축한 근거가 있을 때만 준다.
- 단순 구현, 간단한 서버, 간단한 agent, 클론코딩, 내장 데이터 실습은 6점 이하로 제한한다.
- beginner_evidence_count가 있어도 상태 관리, API 연동, 배포, 모델 개선, 최적화처럼 구체적인 구현 근거가 있으면 낮음으로 단정하지 마라.
- collaboration이 1 이하이면 협업 경험 부족으로 skill_level은 낮음으로 제한한다.
- advanced_evidence_count가 부족하면 skill_level 높음은 금지한다.

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

skill_evidence : {skill_evidence}
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
    student_data의 stack에 있는 각 기술 스택별 숙련도를 10점 만점으로 평가한다.
    점수는 기술 이름의 난이도가 아니라, experience에 드러난 실제 사용 깊이와 구현 난이도를 기준으로 한다.
    stack 목록에 있다는 이유만으로 높은 점수를 주지 말고, 반드시 experience에 직접 적힌 근거만 사용한다.

    평가할 때 아래 요소를 함께 본다:
    - 해당 기술로 무엇을 만들었는가
    - 단순 문법/태그 사용인지, 실제 기능 구현인지
    - 단일 기능인지, 여러 기능이 연결된 프로젝트인지
    - 상태 관리, 데이터 처리, API 연동, 인증, 배포, 최적화, 모델 개선, 충돌 처리 등 구체적인 구현 근거가 있는지
    - 설계, 구조화, 성능 개선, 문제 해결, 운영 경험이 있는지
    - "간단한", "입문", "튜토리얼", "클론코딩", "뼈대", "기본 사용 가능" 수준인지

    점수 기준:
    1점: 기술 이름만 있거나 매우 기초적인 문법/태그/명령어 사용 수준
    2점: 간단한 화면, 폼, 기본 문법, 기본 설정처럼 입문 수준의 사용
    3점: 튜토리얼, 클론코딩, 단일 기능처럼 제한된 구현 경험
    4점: 해당 기술로 명확한 기능을 직접 구현한 경험이 있음
    5점: 여러 기능을 연결한 작은 프로젝트를 혼자 구현할 수 있음
    6점: 데이터 처리, API 연동, 인증, 상태 관리, 배포 등 실전 요소 일부를 포함한 프로젝트 경험
    7점: 여러 기능이 결합된 프로젝트를 구조적으로 구현하고 주요 문제를 해결한 경험
    8점: 복잡한 프로젝트에서 설계, 구조화, 최적화, 협업 개발 경험이 뚜렷함
    9점: 아키텍처 설계, 성능 개선, 운영/장애 대응, 테스트 자동화 등 고급 경험이 명확함
    10점: 해당 기술을 깊게 이해하고 기술 선택, 구조 설계, 문제 해결 전략까지 주도할 수 있음

    중요 규칙:
    - 같은 skill_level이 "낮음"이어도 입문 수준과 의미 있는 기능 구현 경험은 점수 차이를 둔다.
    - 기술 수가 많다고 평균적으로 높게 주지 않는다. 각 기술마다 experience 근거가 있는 만큼만 점수를 준다.
    - 구체적인 경험이 없는 기술은 1~2점으로 제한한다.
    - "가능", "사용 가능", "경험 있음"처럼 구체적인 산출물이 없는 표현은 낮게 평가한다.
    - 고급 기술 이름이 있어도 간단한 실습이면 낮은 점수를 준다.
    - 기초 기술이어도 여러 기능 구현, 구조화, 배포, 문제 해결 근거가 있으면 그만큼 점수를 올릴 수 있다.

    반드시 아래 형식으로 출력:
    스킬이름: 3점
    스킬이름: 1점
    """)
    #skill_level 중상 , 중하 이런식으로 더 만들어서 실력 분포를 더 다양하게 만들도록 하면 괜찮을거 같기도
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
    role: str = Field(description="student_data의 goal과 skills를 기반으로 팀에서 맡기 적합한 역할") #이렇게 하니까 역할좀 이상하게 나오는거 같기도 하고 그냥 사용자가 원하는거 그대로 계속 받는것도 괜찮을거 같은데
    suggestion: str = Field(description="이 학생과 협업하면 좋을 역할 또는 기술 스택을 가진 사람 제안")
    

#이거 이런식으로 분류하는거 ㅈㄴ 별로인듯 어떻게 작성하다 보니 높음 키워드가 많이 들어가있으면 높게 나와버리니까
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
#LLM에게 넘길 중간 난이도 구현 근거. 이 단어만으로 판정하지 않고 참고 점수로만 사용한다.
INTERMEDIATE_WORDS = [
    "상태 관리", "상태관리", "usestate", "usecallback", "usememo",
    "api 연결", "api 연동", "연동", "인증", "jwt", "crud", "curd",
    "배포", "docker", "ec2", "컨테이너", "데이터 처리", "모델 개선",
    "시각화", "충돌 처리", "점수 시스템", "로컬 저장", "ui 설계",
    "반응형", "최적화", "실제 코드"
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


def _count_words(text, words):
    return sum(1 for word in words if word.lower() in text)


def make_skill_evidence(student):
    # 규칙 기반으로 최종 skill_level을 정하지 않고, LLM이 참고할 근거 점수만 만든다.
    text = _student_text(student)
    collaboration = int(student.get("collaboration", 0) or 0)
    experience_count = len(student.get("experience", []))
    implementation_count = _count_implementation_evidence(student)
    beginner_count = _count_words(text, BEGINNER_WORDS)
    intermediate_count = _count_words(text, INTERMEDIATE_WORDS)
    advanced_count = _count_words(text, ADVANCED_WORDS)

    return {
        "collaboration": collaboration,
        "experience_count": experience_count,
        "beginner_evidence_count": beginner_count,
        "implementation_evidence_count": implementation_count,
        "intermediate_evidence_count": intermediate_count,
        "advanced_evidence_count": advanced_count,
        "notes": (
            "이 값들은 최종 판정이 아니라 참고용이다. "
            "experience의 실제 구현 깊이를 우선해서 skill_level과 stack_score를 판단한다."
        ),
    }


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


def normalize_analysis(response, student):
    # response가 pydantic 객체인지 확인
    if hasattr(response, "model_dump"):
        #pydantic객체면 dict으로 변환
        result = response.model_dump()
        #이미 딕이면 딕으로
    else:
        result = dict(response)

    skill_evidence = make_skill_evidence(student)
    original_level = result.get("skill_level")
    fixed_level = original_level

    # LLM 판단을 기본으로 두되, 강한 안전장치만 적용한다.
    if skill_evidence["collaboration"] <= 1: #낮음이 나왔는데 프로젝트 경험이 1이거나 1보다 작으면 낮음
        fixed_level = "낮음"
    elif original_level == "높음" and skill_evidence["advanced_evidence_count"] < 1: #높음이 나왔는데 1보다 작으면 보통
        fixed_level = "보통"

    result["skill_level"] = fixed_level

    # 현재 skill level 기준으로 stack_score 최대값 제한 적용
    result["stack_score"] = cap_stack_score(
        result.get("stack_score", ""),  # stack_score 없으면 빈 문자열
        fixed_level
    )
    if student.get("role") or student.get("goal"):
        result["role"] = student.get("role") or student.get("goal")
    #
    if original_level != fixed_level:
        if skill_evidence["collaboration"] <= 1:
            correction = (
                f"최종 판정 보정: 입력 데이터 기준으로 '{original_level}'이 아니라 "
                f"'{fixed_level}'으로 제한한다. collaboration이 1 이하이면 협업 경험 부족으로 "
                "상위 레벨을 부여하지 않는다."
            )
        elif fixed_level == "보통":
            correction = (
                f"최종 판정 보정: 입력 데이터 기준으로 '{original_level}'이 아니라 "
                f"'{fixed_level}'으로 제한한다. 높음 판정에는 아키텍처, 운영, 성능 개선, "
                "복잡한 문제 해결 같은 고급 근거가 명확해야 한다."
            )
        else:
            correction = (
                f"최종 판정 보정: 입력 데이터 기준으로 '{original_level}'이 아니라 "
                f"'{fixed_level}'으로 제한한다."
            )

        # 기존 reason 뒤에 보정 사유 추가
        result["reason"] = (
            f"{result.get('reason', '')}\n\n{correction}"
        )

    # 설문 성향 점수는 LLM 응답과 무관하게 원본 입력 기준으로 정규화해서 항상 포함한다.
    trait_result = ensure_trait_profile(result, source=student)
    return ensure_preference_profile({**student, **trait_result})


def _get_cached_analysis_for_students(
    cached_results: Optional[List[Dict[str, Any]]],
    students: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    if not cached_results:
        return None

    cached_by_name = {
        result.get("name"): result
        for result in cached_results
        if result.get("name")
    }
    student_names = [student.get("name") for student in students if student.get("name")]

    if not student_names or any(name not in cached_by_name for name in student_names):
        return None

    return [
        normalize_analysis(cached_by_name[student.get("name")], student)
        for student in students
    ]


def get_analyze_stu(students: Optional[List[Dict[str, Any]]] = None):
    # 이미 분석 결과 있으면 재사용
    # 테스트할때 토큰 아낄려고 사용하는건데 나중에 진짜 완성하면 있어도 그냥 덮어씌우는 형태로 갈듯
    #원래 사이즈 안보고 그냥 파일 존재 여부만보고 안에 내용이 있든지 말든지 신경안쓰고 해서 오류가 났었는데 안에 size가 있으면 기존결과대로 유지되고
    #size가 없으면 새로 분석하는그런 로직임.
    datas = students if students is not None else get_data()
    force_reanalyze = os.getenv("FORCE_REANALYZE", "false").lower() == "true"

    if not datas:
        raise RuntimeError("분석할 학생 데이터가 없습니다.")

    if not force_reanalyze:
        cached_results = fetch_analysis_results()
        normalized_results = _get_cached_analysis_for_students(cached_results, datas)

        if normalized_results:
            print("기존 분석 결과를 MySQL에서 불러오는 중...")
            save_analysis_results(normalized_results)
            return normalized_results

    print("분석 결과 없음. 새로 분석 시작...")
    model = get_llm()
    s_prompt = get_prompt_chain()
    structured_llm = model.with_structured_output(StudentAnalysis)
    chain = s_prompt | structured_llm 
    results = []
    for student in datas:
        print("="*55, flush=True)
        print(f"분석 시작: {student['name']}", flush=True)

        response = chain.invoke({
            "student_data": json.dumps(student, ensure_ascii=False, indent=2), #prompt에서 받아야할 student_data 넣어줌
            "skill_evidence": json.dumps(make_skill_evidence(student), ensure_ascii=False, indent=2),
            "input": "이 학생의 실력을 분석해줘."  #사용자 input 입력
        })

        print(f"이름: {student['name']}")
        normalized_response = normalize_analysis(response, student)
        print(normalized_response)
        results.append(normalized_response)
        #리스트만들어서 답변나오면 그리스트 안에 들어가는 형식으로 이 만든걸 matching하는 llm에게 넘김.
        # 분석 결과 저장

    save_analysis_results(results)

    print("분석 결과 MySQL 저장 완료")

    return results

results = None

if __name__ == "__main__":
    results = get_analyze_stu()

#팀장선호와 원하는팀원을 받아야하는데 이걸 어떻게 받을지 정해야함. 원하는팀원은 몇명받을건지 팀장선호수가 넘치면 어떻게 할건지.
#스택으로 점수를 부여하는것을 좀 해야할듯 지금 허재원 react실력이 박건욱 html실력이랑 비슷하다고 해서 점수가 비슷함 이부분을 해결할수있도록 만들어야할듯
