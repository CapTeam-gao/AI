import os
import json
import re
import time
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"), override=False)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

# from s_analysis_fewshot import examples
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder , FewShotChatMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional

from capteam_db import fetch_analysis_results_for_students, fetch_students, save_analysis_results
from capteam_preferences import ensure_preference_profile, ensure_preference_profiles
from capteam_traits import ensure_trait_profile

# store = {}

# def get_session_history(session_id: str):
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]

#apikey 다시파야할듯.
#클로드코드, codex한번 사서 써봐야할듯.
#이름 말고 학번으로 주 식별자.
# 매칭에서 사용하는 기술 점수가 안정적으로 이어지도록 5단계 등급과 스택 점수를 함께 관리한다.
# 객체를 한 번만 생성해 학생별 분석 호출에서 재사용한다.
# GPT-5.4가 학생 경험의 경계 사례를 충분히 검토하도록 기본 추론 강도는 medium으로 둔다.
llm = ChatOpenAI(
    model=os.getenv("OPENAI_ANALYSIS_MODEL", "gpt-5.4"),
    reasoning_effort=os.getenv("OPENAI_ANALYSIS_REASONING_EFFORT", "medium"),
    timeout=int(os.getenv("OPENAI_TIMEOUT", "120")),
    max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
    temperature = 0
)


def get_llm():
    return llm


def is_rate_limit_error(error: Exception) -> bool:
    error_text = str(error).lower()
    status_code = getattr(error, "status_code", None)
    return (
        status_code == 429
        or "ratelimiterror" in type(error).__name__.lower()
        or "too_many_requests" in error_text
        or "error code: 429" in error_text
    )


def invoke_analysis_with_retry(chain, payload: Dict[str, Any], student_name: str):
    max_retries = int(os.getenv("ANALYSIS_LLM_MAX_RETRIES", "3"))
    base_delay = float(os.getenv("ANALYSIS_LLM_RETRY_DELAY", "5"))

    for attempt in range(max_retries + 1):
        try:
            return chain.invoke(payload)
        except Exception as error:
            if not is_rate_limit_error(error):
                raise

            if attempt >= max_retries:
                print(
                    f"{student_name} 분석 RateLimit 재시도 초과: {type(error).__name__}: {error}",
                    flush=True,
                )
                raise

            delay = base_delay * (2 ** attempt)
            print(
                f"{student_name} 분석 RateLimit 발생. {delay:.1f}초 후 재시도 "
                f"({attempt + 1}/{max_retries})",
                flush=True,
            )
            time.sleep(delay)


def build_failed_analysis(student: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    role = student.get("role") or student.get("goal") or ""
    failed_result = {
        **student,
        "name": student.get("name", ""),
        "strength": "",
        "weakness": "",
        "reason": "학생 분석 중 API 요청 제한으로 분석을 완료하지 못했습니다.",
        "stack_score": "",
        "skill_level": "하",
        "role": role,
        "suggestion": "",
        "analysis_status": "FAILED",
        "analysis_error": f"{type(error).__name__}: {error}",
    }
    return ensure_preference_profile(ensure_trait_profile(failed_result, source=student))




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
최종 skill_level과 stack_score는 student_data의 experience를 우선으로 보고 판단하라.
student_data에 성격/개발 성향 점수가 있으면 협업 방식과 추천 역할 설명에만 참고하고,
기술 숙련도와 skill_level을 올리거나 내리는 근거로 사용하지 마라.

평가 원칙:
- 기술 이름을 안다고 높은 점수를 주지 마라.
- 특정 단어의 포함 여부나 반복 횟수로 평가하지 말고 문장 전체에서 실제 수행한 범위와 깊이를 해석하라.
- 구현 범위, 기능 간 연결, 주도성, 구조화, 문제 해결, 배포와 운영 경험을 종합해서 평가하라.
- 경험에 직접 적힌 내용만 근거로 사용하라. 가능성, 추정, 일반적인 학습 경로는 근거가 아니다.
- 9~10점은 아키텍처 설계, 성능 개선, 운영/장애 대응, 복잡한 문제 해결 근거가 명확할 때만 준다.
- 7~8점은 여러 기능이 결합된 프로젝트를 스스로 구축한 근거가 있을 때만 준다.
- 단순 구현, 간단한 서버, 간단한 agent, 클론코딩, 내장 데이터 실습은 6점 이하로 제한한다.

skill_level 판정:
- 상: 복잡한 프로젝트에서 설계, 주도적 구현, 복잡한 문제 해결과 최적화 또는 운영 경험이 명확한 경우.
- 중상: 여러 기능을 연결한 실제 프로젝트를 구현했고 구조화 또는 의미 있는 문제 해결 경험이 있는 경우.
- 중: 구체적인 프로젝트 구현 경험은 있으나 설계, 운영, 고급 문제 해결 근거가 부족한 경우.
- 중하: 단일 기능이나 작은 범위의 구현 경험은 확인되지만 범위, 깊이, 주도성 근거가 제한적인 경우.
- 하: 기초 사용이나 실습 수준이거나 실제로 구현한 결과를 확인할 구체적인 근거가 부족한 경우.

출력 전 자체 검증:
- skill_level의 기준을 만족하는 직접적인 경험 근거가 student_data에 있는지 확인하라.
- 기술 이름이나 표현의 인상만으로 상위 등급을 선택했다면 실제 수행 범위에 맞게 낮춰라.

관리자용 reason 작성 규칙:
- reason은 학생의 역량을 확인하는 교사가 읽는 설명이다. AI의 채점 과정을 보여주지 마라.
- 3~4문장으로 작성하고, 실제로 구현한 내용, 팀에서 맡을 수 있는 업무, 아직 확인되지 않은 경험이나 보완점 순서로 설명하라.
- skill_level, stack_score 같은 필드명을 reason에 쓰지 마라.
- '상', '중상', '중', '중하', '하' 등급을 reason에 쓰거나 등급끼리 비교하지 마라.
- 기술별 점수, 점수 차이, 점수를 정한 과정은 reason에 쓰지 마라.
- '엄격하게 보면', '과대평가하면 안 된다', '상위 숙련 근거' 같은 채점자 관점의 표현을 쓰지 마라.
- 전문 용어가 필요하면 학생이 실제로 한 행동과 결과를 함께 적어 쉽게 이해할 수 있게 하라.

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

#stack_score를 뺄지말지 생각좀 해봐야할듯 구현경험으로는 stack_score를 명확하게 판단할 수 없는거 같음

class StudentAnalysis(BaseModel):
    # 설명을 아무리 해줘도 제일 높은 skill_level에 조금이라도 해당 되면 지금 그냥 제일 높게 쳐주는거 같음 이분을 해결할수 있는 방법을 찾아야함.
    name : str = Field(description="student_data의 name을 그대로 여기에 작성하시오.")
    strength: str = Field(description="student_data의 stack, experience를 기반으로 학생의 핵심 강점")
    weakness: str = Field(description="student_data의 stack, experience를 기반으로 학생의 부족한 부분")
    reason: str = Field(description="""
    관리자인 교사가 학생의 현재 역량을 쉽게 이해할 수 있는 3~4문장의 설명.
    첫째, experience에서 확인된 실제 구현 내용을 설명한다.
    둘째, 팀 프로젝트에서 맡을 수 있는 구체적인 업무를 설명한다.
    셋째, 입력에서 아직 확인되지 않은 경험이나 보완할 부분을 설명한다.
    내부 필드명, 5단계 등급 이름, 기술 점수, 등급과 점수의 산정 과정은 절대 적지 마라.
    """)
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
    - 같은 skill_level이 "하"여도 입문 수준과 의미 있는 기능 구현 경험은 점수 차이를 둔다.
    - 기술 수가 많다고 평균적으로 높게 주지 않는다. 각 기술마다 experience 근거가 있는 만큼만 점수를 준다.
    - 구체적인 경험이 없는 기술은 1~2점으로 제한한다.
    - "가능", "사용 가능", "경험 있음"처럼 구체적인 산출물이 없는 표현은 낮게 평가한다.
    - 고급 기술 이름이 있어도 간단한 실습이면 낮은 점수를 준다.
    - 기초 기술이어도 여러 기능 구현, 구조화, 배포, 문제 해결 근거가 있으면 그만큼 점수를 올릴 수 있다.

    반드시 아래 형식으로 출력:
    스킬이름: 3점
    스킬이름: 1점
    """)
    skill_level: Literal['상', '중상', '중', '중하', '하'] = Field(description="""
    - 상: 복잡한 프로젝트에서 설계, 주도적 구현, 문제 해결과 최적화 또는 운영 경험이 명확함
    - 중상: 여러 기능을 연결한 프로젝트와 구조화 또는 의미 있는 문제 해결 경험이 있음
    - 중: 구체적인 프로젝트 구현 경험은 있으나 설계, 운영, 고급 문제 해결 근거가 부족함
    - 중하: 단일 기능이나 작은 범위의 구현 경험으로 범위, 깊이, 주도성 근거가 제한됨
    - 하: 기초 사용이나 실습 수준이거나 구체적인 구현 근거가 부족함
    
    다음 금지:
    - 일반적인 학습 경로를 근거로 보정하지 마라
    - 라이브러리 이름만으로 숙련도를 추정하지 마라
    - 특정 키워드가 있다는 이유만으로 등급을 결정하지 마라
    - 협업 점수를 기술 숙련도 등급의 근거로 사용하지 마라
    """)
    #skill_level이 너무 왔다갔다 함
    role: str = Field(description="student_data의 goal과 skills를 기반으로 팀에서 맡기 적합한 역할") #이렇게 하니까 역할좀 이상하게 나오는거 같기도 하고 그냥 사용자가 원하는거 그대로 계속 받는것도 괜찮을거 같은데
    suggestion: str = Field(description="이 학생과 협업하면 좋을 역할 또는 기술 스택을 가진 사람 제안")
    

def cap_stack_score(stack_score, skill_level):
    #skill_level별로 stack_score max값을 조정
    max_score_by_level = {
        "하": 4,
        "중하": 5,
        "중": 6,
        "중상": 7,
        "상": 9,
        "낮음": 4,
        "보통": 7,
        "높음": 10,
        "LOWER": 4,
        "LOWER_MIDDLE": 5,
        "MIDDLE_LOWER": 5,
        "MIDDLE": 6,
        "UPPER_MIDDLE": 7,
        "MIDDLE_UPPER": 7,
        "UPPER": 9,
    }
    #현재 skill level의 최대 점수 가져오기
    max_score = max_score_by_level.get(skill_level, 7)
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

    skill_level = result.get("skill_level")

    # 현재 skill level 기준으로 stack_score 최대값 제한 적용
    result["stack_score"] = cap_stack_score(
        result.get("stack_score", ""),  # stack_score 없으면 빈 문자열
        skill_level
    )
    if student.get("role") or student.get("goal"):
        result["role"] = student.get("role") or student.get("goal")
    # 설문 성향 점수는 LLM 응답과 무관하게 원본 입력 기준으로 정규화해서 항상 포함한다.
    trait_result = ensure_trait_profile(result, source=student)
    student_user_id = (
        student.get("user_id")
        or student.get("userId")
        or student.get("student_id")
        or student.get("studentId")
    )
    merged_result = {
        **student,
        **trait_result,
        "user_id": student_user_id,
        "preferred_members": (
            student.get("preferred_members")
            or student.get("preferredMembers")
            or trait_result.get("preferred_members", [])
        ),
        "wants_leader": (
            student.get("wants_leader")
            if "wants_leader" in student
            else student.get("wantsLeader", trait_result.get("wants_leader", False))
        ),
    }
    return ensure_preference_profile(merged_result)


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

    return ensure_preference_profiles([
        normalize_analysis(cached_by_name[student.get("name")], student)
        for student in students
    ])


BACKEND_LEVEL_TO_SKILL_LEVEL = {
    "UPPER": "상",
    "UPPER_MIDDLE": "중상",
    "MIDDLE_UPPER": "중상",
    "MIDDLE": "중",
    "LOWER_MIDDLE": "중하",
    "MIDDLE_LOWER": "중하",
    "LOWER": "하",
}


def _student_identifier(student: Dict[str, Any]) -> Optional[str]:
    return (
        student.get("user_id")
        or student.get("userId")
        or student.get("student_id")
        or student.get("studentId")
    )


def _embedded_analysis(student: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    analysis_result = student.get("analysis_result") or student.get("analysisResult")
    student_level = student.get("skill_level") or student.get("student_level") or student.get("studentLevel")
    if not analysis_result or not student_level:
        return None

    skill_level = BACKEND_LEVEL_TO_SKILL_LEVEL.get(str(student_level).upper(), student_level)
    return {
        "name": student.get("name"),
        "strength": student.get("strength") or analysis_result,
        "weakness": student.get("weakness") or "추가 프로젝트 경험을 통해 역량을 구체적으로 확인할 필요가 있습니다.",
        "reason": analysis_result,
        "stack_score": student.get("stack_score") or student.get("stackScore") or "",
        "skill_level": skill_level,
        "role": student.get("role") or student.get("goal") or "",
        "suggestion": student.get("suggestion") or "다른 역할군의 팀원과 협업하는 구성이 적합합니다.",
        "analysis_status": "SUCCESS",
    }


def _resolve_reusable_analyses(
    students: List[Dict[str, Any]],
    cached_results: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    cached_by_id = {
        identifier: result
        for result in cached_results
        if (identifier := _student_identifier(result))
    }
    cached_by_name = {
        result.get("name"): result
        for result in cached_results
        if result.get("name")
    }

    reusable = {}
    for index, student in enumerate(students):
        cached = cached_by_id.get(_student_identifier(student)) or cached_by_name.get(student.get("name"))
        source = cached or _embedded_analysis(student)
        if source:
            normalized = normalize_analysis(source, student)
            normalized["analysis_status"] = source.get("analysis_status", "SUCCESS")
            reusable[index] = normalized
    return reusable


def get_analyze_stu(students: Optional[List[Dict[str, Any]]] = None):
    # 이미 분석 결과 있으면 재사용
    # 테스트할때 토큰 아낄려고 사용하는건데 나중에 진짜 완성하면 있어도 그냥 덮어씌우는 형태로 갈듯
    #원래 사이즈 안보고 그냥 파일 존재 여부만보고 안에 내용이 있든지 말든지 신경안쓰고 해서 오류가 났었는데 안에 size가 있으면 기존결과대로 유지되고
    #size가 없으면 새로 분석하는그런 로직임.
    datas = students if students is not None else get_data()
    force_reanalyze = os.getenv("FORCE_REANALYZE", "false").lower() == "true"

    if not datas:
        raise RuntimeError("분석할 학생 데이터가 없습니다.")

    results_by_index: Dict[int, Dict[str, Any]] = {}
    if not force_reanalyze:
        cached_results = fetch_analysis_results_for_students(datas)
        results_by_index = _resolve_reusable_analyses(datas, cached_results)

    pending_indices = [index for index in range(len(datas)) if index not in results_by_index]
    if not pending_indices:
        results = ensure_preference_profiles([results_by_index[index] for index in range(len(datas))])
        print(f"기존 학생 분석 {len(results)}명을 재사용합니다.")
        save_analysis_results(results)
        return results

    print(
        f"기존 학생 분석 {len(results_by_index)}명을 재사용하고 "
        f"미분석 학생 {len(pending_indices)}명만 새로 분석합니다."
    )
    model = get_llm()
    s_prompt = get_prompt_chain()
    structured_llm = model.with_structured_output(StudentAnalysis)
    chain = s_prompt | structured_llm 
    for index in pending_indices:
        student = datas[index]
        print("="*55, flush=True)
        print(f"분석 시작: {student['name']}", flush=True)

        request_payload = {
            "student_data": json.dumps(student, ensure_ascii=False, indent=2), #prompt에서 받아야할 student_data 넣어줌
            "input": "이 학생의 실력을 분석해줘."  #사용자 input 입력
        }

        try:
            response = invoke_analysis_with_retry(chain, request_payload, student["name"])
        except Exception as error:
            if not is_rate_limit_error(error):
                raise

            failed_response = build_failed_analysis(student, error)
            print(f"분석 실패 상태로 저장: {student['name']}")
            print(failed_response)
            results_by_index[index] = failed_response
            continue

        print(f"이름: {student['name']}")
        normalized_response = normalize_analysis(response, student)
        normalized_response["analysis_status"] = "SUCCESS"
        print(normalized_response)
        results_by_index[index] = normalized_response
        #리스트만들어서 답변나오면 그리스트 안에 들어가는 형식으로 이 만든걸 matching하는 llm에게 넘김.
        # 분석 결과 저장

    results = ensure_preference_profiles([results_by_index[index] for index in range(len(datas))])
    save_analysis_results(results)

    print("분석 결과 MySQL 저장 완료")

    return results

results = None

if __name__ == "__main__":
    results = get_analyze_stu()

#팀장선호와 원하는팀원을 받아야하는데 이걸 어떻게 받을지 정해야함. 원하는팀원은 몇명받을건지 팀장선호수가 넘치면 어떻게 할건지.
#스택으로 점수를 부여하는것을 좀 해야할듯 지금 허재원 react실력이 박건욱 html실력이랑 비슷하다고 해서 점수가 비슷함 이부분을 해결할수있도록 만들어야할듯
