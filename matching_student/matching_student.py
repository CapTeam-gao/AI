# 선호팀원 팀장선호 받아야함.
import json
import math
import re
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate

ANALYSIS_OUTPUT_PATH = "/Users/kshi3430/CapTeam/data/student_analysis_data/analysis_output.json"
MATCHING_OUTPUT_PATH = "/Users/kshi3430/CapTeam/data/student_analysis_data/matching_output.json"

# skill_level을 숫자로 바꿔서 팀별 실력 합계를 계산하기 위한 기준.
# 높음/보통/낮음만으로는 정렬과 합산을 하기 어려워서 점수화한다.
SKILL_LEVEL_SCORE = {
    "높음": 3,
    "보통": 2,
    "낮음": 1,
}


def load_analysis_output_json():
    #json파일 경로로 파일 가져와서 result에 넣음.
    with open(ANALYSIS_OUTPUT_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    return results


def get_llm(model="solar-pro3"):
    return ChatUpstage(model=model)


def get_prompt_chain():
    # 규칙 기반으로 만든 1차 팀 배정안을 LLM에게 넘기고,
    # LLM은 역할 구성과 설명을 더 자연스럽게 정리하는 역할을 한다.
    matching_s_prompt = """
당신은 학생들의 코딩 실력 분석 결과를 바탕으로 캡스톤 프로젝트 팀을 매칭하는 챗봇이다.
아래 학생 분석 결과와 1차 팀 배정안을 참고해서 균형 잡힌 최종 팀을 만들어라.

매칭 기준:
- 각 팀의 전체 실력 합계가 너무 차이 나지 않게 한다.
- 가능하면 프론트엔드, 백엔드, AI/데이터, 기타 역할이 한 팀에 몰리지 않게 한다.
- 낮음 학생은 보통 또는 높음 학생과 함께 배치한다.
- 같은 역할만 모인 팀은 피한다.
- 아직 선호 팀원, 팀장 선호도 데이터는 없으므로 실력/역할 균형을 우선한다.

이름 사용 규칙:
- 아래 allowed_student_names에 있는 이름만 사용한다.
- 학생 이름은 한 글자도 바꾸지 말고 그대로 복사한다.
- allowed_student_names에 없는 이름을 만들거나 추가하지 않는다.
- 모든 학생은 정확히 한 팀에만 들어가야 한다.

출력 형식:
팀 1:
- 팀원:
- 역할 구성:
- 실력 균형 설명:
- 추천 팀장:
- 매칭 이유:

팀 2:
...

allowed_student_names:
{allowed_student_names}

student_analysis:
{student_analysis}

initial_teams:
{initial_teams}
"""

    s_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",matching_s_prompt),
            ("human","{input}"),
        ]
    )

    return s_prompt


def parse_stack_score(stack_score):
    # stack_score는 "python: 7점\nfastapi: 7점" 같은 문자열이라
    # 정규식으로 숫자만 뽑아서 평균 점수를 계산한다.
    scores = [int(score) for score in re.findall(r"(\d{1,2})점", stack_score or "")]
    if not scores:
        return 0
    return sum(scores) / len(scores)


def get_student_score(student):
    # 학생의 전체 실력 점수.
    # skill_level을 큰 기준으로 보고, stack_score 평균을 보조 점수로 더한다.
    # 예: 보통(2 * 10) + 스택 평균 5점 = 25점
    level_score = SKILL_LEVEL_SCORE.get(student.get("skill_level"), 1)
    stack_score = parse_stack_score(student.get("stack_score", ""))
    return level_score * 10 + stack_score


def get_role_group(role):
    # role 문장은 학생마다 표현이 다를 수 있어서,
    # 키워드 기준으로 큰 역할군(frontend/backend/ai_data/game/etc)으로 묶는다.
    # 이 값은 같은 역할이 한 팀에 몰리는 것을 줄이는 데 사용한다.
    role_text = (role or "").lower()

    if any(keyword in role_text for keyword in ["frontend", "front", "프론트"]):
        return "frontend"
    if any(keyword in role_text for keyword in ["backend", "back", "서버", "server", "백엔드"]):
        return "backend"
    if any(keyword in role_text for keyword in ["ai", "데이터", "머신러닝", "ml"]):
        return "ai_data"
    if any(keyword in role_text for keyword in ["unity", "게임"]):
        return "game"
    return "etc"


def make_student_summary(student):
    # LLM에게 원본 분석 결과 전체를 그대로 주면 너무 길어질 수 있으므로,
    # 1차 팀 배정에 필요한 핵심 정보만 추린 학생 요약 데이터를 만든다.
    return {
        "name": student.get("name"),
        "skill_level": student.get("skill_level"),
        "score": round(get_student_score(student), 2),
        "role": student.get("role"),
        "role_group": get_role_group(student.get("role")),
        "strength": student.get("strength"),
        "weakness": student.get("weakness"),
    }


def get_student_names(students):
    # LLM이 이름을 바꾸는 문제를 막기 위해 원본 이름 목록을 따로 만든다.
    # 프롬프트에 이 목록을 넘기고, 응답 후에도 누락된 이름이 있는지 검사한다.
    return [student.get("name") for student in students if student.get("name")]


def make_empty_teams(team_count):
    # 팀 배정을 시작하기 전 빈 팀 틀을 만든다.
    # members에는 학생이 들어가고, total_score는 팀 실력 합계,
    # role_groups는 팀 안에 어떤 역할군이 몇 명 있는지 기록한다.
    return [
        {
            "team_name": f"팀 {index + 1}",
            "members": [],
            "total_score": 0,
            "role_groups": {},
        }
        for index in range(team_count)
    ]


def choose_team_for_student(teams, student, max_team_size):
    # 현재 학생을 어느 팀에 넣을지 고르는 함수.
    # 기준 1: 팀 실력 합계가 낮은 팀을 우선한다.
    # 기준 2: 같은 역할군이 적은 팀을 우선한다.
    # 기준 3: 인원수가 적은 팀을 우선한다.
    role_group = student["role_group"]
    available_teams = [
        team for team in teams
        if len(team["members"]) < max_team_size
    ]

    return min(
        available_teams,
        key=lambda team: (
            team["total_score"],
            team["role_groups"].get(role_group, 0),
            len(team["members"]),
        ),
    )


def create_initial_teams(students, team_size=4, team_count=None):
    # 분석 결과를 바탕으로 LLM에게 넘길 1차 팀 배정안을 만든다.
    # 여기서는 LLM을 호출하지 않고, 점수와 역할군만 보고 빠르게 균형을 맞춘다.
    if not students:
        return []

    # team_count가 없으면 team_size 기준으로 필요한 팀 수를 자동 계산한다.
    # 예: 학생 8명, team_size 4명 => 2팀
    if team_count is None:
        team_count = math.ceil(len(students) / team_size)

    # 학생 수가 딱 나누어떨어지지 않을 수 있으므로 팀당 최대 인원을 계산한다.
    # 예: 학생 10명, 3팀 => 최대 4명
    max_team_size = math.ceil(len(students) / team_count)
    teams = make_empty_teams(team_count)
    student_summaries = [make_student_summary(student) for student in students]

    # 실력이 높은 학생부터 배치해야 팀별 실력 합계를 맞추기 쉽다.
    sorted_students = sorted(
        student_summaries,
        key=lambda student: student["score"],
        reverse=True,
    )

    for student in sorted_students:
        # 현재 학생에게 가장 적합한 팀을 고르고, 해당 팀의 상태를 업데이트한다.
        team = choose_team_for_student(teams, student, max_team_size)
        team["members"].append(student)
        team["total_score"] += student["score"]
        team["role_groups"][student["role_group"]] = (
            team["role_groups"].get(student["role_group"], 0) + 1
        )

    for team in teams:
        # JSON으로 저장했을 때 보기 좋게 소수 둘째 자리까지만 남긴다.
        team["total_score"] = round(team["total_score"], 2)

    return teams


def save_matching_result(result):
    # 최종 결과를 matching_output.json으로 저장한다.
    # use_ai=False이면 1차 배정안만 저장되고, use_ai=True이면 AI 최종 설명까지 저장된다.
    with open(MATCHING_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def validate_ai_matching_names(ai_matching, allowed_student_names):
    # AI 응답에 원본 학생 이름이 모두 들어있는지 확인한다.
    # 이름을 다르게 쓰거나 빼먹으면 해당 이름이 missing_names에 잡힌다.
    missing_names = [
        name for name in allowed_student_names
        if name not in ai_matching
    ]

    return {
        "is_valid": not missing_names,
        "missing_names": missing_names,
    }


def request_ai_matching(chain, results, initial_teams, allowed_student_names, input_message):
    # LLM 호출에 필요한 값을 한 곳에서 구성한다.
    # allowed_student_names를 별도로 넘기는 이유는 이름 변형을 최대한 막기 위해서다.
    return chain.invoke({
        "allowed_student_names": json.dumps(allowed_student_names, ensure_ascii=False),
        "student_analysis": json.dumps(results, ensure_ascii=False, indent=2),
        "initial_teams": json.dumps(initial_teams, ensure_ascii=False, indent=2),
        "input": input_message,
    })


def get_response(team_size=4, team_count=None, use_ai=True, retry_on_invalid_name=True):
    # 전체 매칭 실행 함수.
    # 1. 학생 분석 결과를 읽는다.
    # 2. 규칙 기반 1차 팀 배정안을 만든다.
    # 3. use_ai=True이면 LLM에게 최종 매칭 설명을 요청한다.
    results = load_analysis_output_json()
    allowed_student_names = get_student_names(results)
    initial_teams = create_initial_teams(
        results,
        team_size=team_size,
        team_count=team_count,
    )

    if not use_ai:
        # 테스트하거나 LLM 비용을 쓰고 싶지 않을 때 쓰는 경로.
        # 규칙 기반으로 만든 팀만 저장하고 바로 반환한다.
        output = {
            "allowed_student_names": allowed_student_names,
            "initial_teams": initial_teams,
            "ai_matching": None,
            "name_validation": None,
        }
        save_matching_result(output)
        return output

    llm = get_llm()
    s_prompt = get_prompt_chain()
    chain = s_prompt | llm

    # 학생 분석 원본, 원본 이름 목록, 1차 팀 배정안을 함께 넘긴다.
    # LLM은 이 정보를 보고 최종 팀 구성, 추천 팀장, 매칭 이유를 작성한다.
    response = request_ai_matching(
        chain,
        results,
        initial_teams,
        allowed_student_names,
        "1차 팀 배정안을 바탕으로 균형 잡힌 최종 프로젝트 팀을 만들어줘.",
    )
    ai_matching = response.content
    name_validation = validate_ai_matching_names(ai_matching, allowed_student_names)

    if retry_on_invalid_name and not name_validation["is_valid"]:
        # 이름이 누락되면 같은 초안을 유지한 채 이름 규칙을 더 강하게 알려주고 한 번 재요청한다.
        # 무한 반복을 피하기 위해 재시도는 1회만 한다.
        retry_response = request_ai_matching(
            chain,
            results,
            initial_teams,
            allowed_student_names,
            (
                "이전 응답에서 학생 이름이 누락되거나 다르게 작성되었다. "
                f"누락된 이름: {', '.join(name_validation['missing_names'])}. "
                "allowed_student_names의 이름만 정확히 사용해서 다시 작성해줘."
            ),
        )
        ai_matching = retry_response.content
        name_validation = validate_ai_matching_names(ai_matching, allowed_student_names)

    output = {
        "allowed_student_names": allowed_student_names,
        # initial_teams를 같이 저장해두면 AI가 어떤 초안을 보고 답했는지 추적할 수 있다.
        "initial_teams": initial_teams,
        "ai_matching": ai_matching,
        "name_validation": name_validation,
    }
    save_matching_result(output)

    print(ai_matching)
    if not name_validation["is_valid"]:
        print(f"이름 검증 실패: {name_validation['missing_names']}")
    print(f"매칭 결과 저장 완료: {MATCHING_OUTPUT_PATH}")

    return output


if __name__ == "__main__":
    get_response()
#python3 -m matching_student.matching_student 지금 matching_student폴더에서 실행하면 얘가 뒤로 갔다가 파일 찾아야해서 못찾음 그래서 그냥 capteam에서 실행해서 경로 지정해줘야함.
#팀 매칭할때 이름같은게 다르게 나올때가 있음 그런것들을 고쳐줘야할듯 데이터늘려서 한번 해봐야할듯

# 팀 매칭 기준
# 한 팀 당 인원은 4명에서 5명으로 팀을 생성할게요

# 웹앱팀은 기술 스택이 골고루 분포되게 팀을 생성할게요

# 게임팀은 설문조사 답변을 반영하여 선호팀원과 팀을 생성할게요

# 기술적으로 이끌어 갈 사람과 참여할 의욕이 있는 학생을 포함되게 팀을 생성할게요