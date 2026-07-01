from typing import Any,List,TypedDict,Dict,Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import os
import sys
import json
import math
import re
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"), override=False)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=False)

from capteam_db import fetch_analysis_results, fetch_matching_result, fetch_students, save_matching_result

from capteam_preferences import (
    breaks_preference_constraints,
    build_preference_rejections,
    choose_preference_aware_leader,
    ensure_preference_profile,
    ensure_preference_profiles,
    get_preferred_members,
    preference_bonus,
    team_preference_notes,
)
from capteam_traits import (
    CORE_RISK_TRAITS,
    TRAIT_LABELS,
    build_leader_reason,
    build_team_trait_risks,
    calculate_trait_averages,
    ensure_trait_profile,
    get_high_trait_names,
    get_leader_score,
    get_low_trait_names,
    get_response_reliability,
    get_trait_score,
)
#역할을 다양하게 균형 잡힌 팀 1순위 , 선호팀원 2순위 
# OpenAI Chat 모델 객체를 생성해서 반환한다.
# 기본값은 학생 분석 모델과 동일한 gpt-5.4를 쓰고,
# OPENAI_MATCHING_MODEL 환경변수로 쉽게 교체할 수 있게 둔다.
def get_llm(model=None):
    model = model or os.getenv("OPENAI_MATCHING_MODEL", "gpt-5.4")
    kwargs = {
        "model": model,
        "timeout": int(os.getenv("OPENAI_TIMEOUT", "120")),
        "max_retries": int(os.getenv("OPENAI_MAX_RETRIES", "2")),
    }
    temperature = os.getenv("OPENAI_MATCHING_TEMPERATURE")
    if temperature not in (None, ""):
        kwargs["temperature"] = float(temperature)
    return ChatOpenAI(**kwargs)


#state
# LangGraph 워크플로우에서 노드들이 공유하는 상태 구조다.
# 학생 분석, 팀 초안, 검증 결과, 조정 기록, 최종 결과를 저장한다.
class MatchingState(TypedDict):
    analyzed_students : List[Dict[str,Any]] #학생 분석결과 들고와서 매칭할때 사용
    teams : List[Dict[str,Any]] #현재 팀 매칭상태 저장
    balance_result : Dict[str,Any] #현재 전체적인 팀 매칭 평가결과 상태 저장. 상태가 안좋을때 재조정하고 좋을때 멈추기위해 사용
    team_evaluations: List[Dict[str, Any]] #팀별 팀매칭 평가결과 상태
    adjustment_history:List[Dict[str,Any]] #팀이 어떻게 수정 됬는지 기록. 계속해서 같은 결과가 나오지 않도록 하기위해 사용
    final_result:Dict[str,Any] #최종 결과 저장
    llm_result:Dict[str,Any] #llm이 생성한 매칭상태 저장
    iteration_count: int #팀 생성 수정한 횟수 저장하여 무한반복을 막음
#team으로 알고리즘으로 team상태 저장하고
#llm_result로 llm이 제안한 팀 상태 저장하고
#검증할때 실패하면 다시 알고리즘 보고 할수있도록 알고리즘은 그대로 두고 llm_result만 계속 덮어 씌어지면서 수정

# skill_level을 팀 생성용 숫자 점수로 바꾸기 위한 기준.
# 팀 간 실력 균형을 맞추려면 5단계 문자열보다 숫자가 다루기 편함.
SKILL_LEVEL_SCORE = {
    "상": 5,
    "중상": 4,
    "중": 3,
    "중하": 2,
    "하": 1,
    # 기존 3단계 분석 캐시 호환
    "높음": 5,
    "보통": 3,
    "낮음": 1,
}

ROLE_GROUPS_TO_KEEP_TOGETHER = {"game"}







#node
# 최신 설문 원본의 선호 팀원/팀장 희망을 분석 결과에 덮어쓴다.
# 분석 캐시가 있어도 새 설문 선호가 매칭에 빠지지 않게 보정한다.
def merge_latest_preferences(analyzed_students: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        source_students = fetch_students()
    except Exception:
        source_students = []

    source_by_name = {
        student.get("name"): student
        for student in source_students
        if student.get("name")
    }

    merged_students = []
    for student in analyzed_students:
        source = source_by_name.get(student.get("name"))
        if source:
            merged = {
                **student,
                "preferred_members": source.get("preferred_members", student.get("preferred_members", [])),
                "wants_leader": source.get("wants_leader", student.get("wants_leader", False)),
            }
        else:
            merged = dict(student)
        merged_students.append(merged)

    return ensure_preference_profiles([ensure_trait_profile(student) for student in merged_students])


# MySQL, analysis_output.json, matching_output.json 순서로 학생 분석 결과를 읽는다.
# 읽은 학생 데이터에는 성향/선호 프로필을 보강해서 매칭 입력으로 반환한다.
def load_analysis_output_json():
    results = fetch_analysis_results()
    if results:
        return merge_latest_preferences(results)

    input_path = Path(__file__).resolve().parents[1] / "data/student_analysis_data/analysis_output.json"
    if input_path.exists() and input_path.stat().st_size > 0:
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                return merge_latest_preferences(json.load(f))
        except json.JSONDecodeError:
            pass

    matching_path = Path(__file__).resolve().parents[1] / "data/student_analysis_data/matching_output.json"
    if matching_path.exists() and matching_path.stat().st_size > 0:
        with open(matching_path, "r", encoding="utf-8") as f:
            matching_output = json.load(f)
        analyzed_students = matching_output.get("analyzed_students", [])
        if analyzed_students:
            return merge_latest_preferences(analyzed_students)

    raise RuntimeError("MySQL, analysis_output.json, matching_output.json에 학생 분석 결과가 없습니다. student_analysis.analysis_llm을 먼저 실행하세요.")

#팀생성 노드
#일단 파일불러와서 프롬포트 작성과 알고리즘으로 팀 생성 team에 저장

#정규표현식으로 학생분석데이터에서 숫자만 뽑아서 평균을 계산함
# "기술명: n점" 형태의 stack_score에서 숫자 점수만 뽑아 평균을 낸다.
# 점수가 없으면 기술 스택 보조 점수 0을 반환한다.
def parse_stack_score(stack_score):
    # stack_score는 "python: 7점\nfastapi: 7점" 같은 문자열 형태라서
    # 정규식으로 숫자만 뽑고 평균을 계산함.
    scores = [int(score) for score in re.findall(r"(\d{1,2})점", stack_score or "")]
    if not scores:
        return 0
    return sum(scores) / len(scores)

#여기 수정해야함
#여기 점수 계산 로직 이거 수정해야할듯 지금 구조가 너무 점수에 의존하는 구조인데 지금 그냥 평균내거나 곱해서 그냥 점수 계산하니까 성능이 좀 많이 안좋아짐.
# 학생 분석 결과를 받아 매칭용 기술 점수를 계산한다.
# skill_level 기본점수에 stack_score 평균을 더해 기술 역량 기준값을 만든다.
def get_technical_score(student):
    # 학생 한 명의 기술 점수 계산.
    # skill_level을 큰 기준으로 보고, stack_score 평균을 보조 점수로 더함.
    level_score = SKILL_LEVEL_SCORE.get(student.get('skill_level'),1)
    # 신규 5단계와 기존 3단계 모두 위 점수표로 변환한다.
    stack_score = parse_stack_score(student.get('stack_score',""))
    #stack_score가져와서 함수써서 숫자만 추출함 없으면 빈 문자열.
    return level_score * 10 + stack_score


# 학생 분석 결과를 받아 최종 배치용 학생 점수를 계산한다.
# 기술 점수 70%와 성향 점수 30%를 합쳐 팀 균형 계산에 사용한다.
def get_student_score(student):
    # 최종 매칭 점수는 기술 70%, 성향 30%로 계산한다.
    technical_score = get_technical_score(student)
    trait_score = get_trait_score(student)
    return technical_score * 0.7 + trait_score * 0.3



#학생의 희망 역할을 큰 카테고리로 바꾸어줌
#같은 분야에 사람이 한팀에 많이 들어오지 않게 하기 위해서
# 자유 입력 역할 문자열을 매칭용 role_group으로 분류한다.
# 프론트/백엔드/AI/앱/디자인/게임/풀스택/DevOps/보안/기타 중 하나를 반환한다.
#이거 더 채워야할듯 우리학교 분야가 너무 많아서 
def get_role_group(role):
    # role은 "Frontend Developer", "백엔드 개발자", "AI 엔지니어"처럼 표현이 제각각이라
    # 큰 역할군으로 묶어서 한 팀에 같은 역할이 몰리지 않게 할 때 사용함.
    role_text = (role or "").lower()
    if any(keyword in role_text for keyword in ["unity", "unreal", "game", "게임", "유니티", "언리얼"]):
        return "game"
    if any(keyword in role_text for keyword in ["fullstack", "full-stack", "풀스택"]):
        return "fullstack"
    if any(keyword in role_text for keyword in ["devops", "dev ops", "인프라", "배포", "ci/cd", "cicd"]):
        return "devops"
    if any(keyword in role_text for keyword in ["security", "secure", "보안", "owasp"]):
        return "security"
    #근데 테스트 해보니까 role에서 ai하고 backend가 둘다 들어가 있을때가 있음(ai/backend엔지니어 이런느낌으로) 그거 방지도 해야할듯
    if any(keyword in role_text for keyword in ["frontend", "front", "프론트"]):
        return "frontend"
    if any(keyword in role_text for keyword in ["backend", "back", "서버", "server", "백엔드"]):
        return "backend"
    if any(keyword in role_text for keyword in ["ai", "데이터", "머신러닝", "ml"]):
        return "ai_data"
    if any(keyword in role_text for keyword in ["app", "android", "ios", "mobile", "앱", "모바일"]):
        return "app"
    if any(keyword in role_text for keyword in ["design", "designer", "figma", "ui/ux", "ui", "ux", "디자인", "디자이너"]):
        return "design"
    return "etc"


# 학생 원본 분석 dict를 팀 배치에 필요한 요약 dict로 변환한다.
# 기술점수, 성향점수, 역할군, 선호팀원, 리더점수 등 핵심 필드를 채운다.
def make_student_summary(student):
    # 팀 생성에 필요한 정보만 추린 학생 요약 데이터.
    # 원본 분석 결과 전체를 teams에 넣으면 너무 길어져서 핵심 필드만 저장함.
    student = ensure_preference_profile(ensure_trait_profile(student))
    role = student.get("role") or student.get("goal")
    technical_score = round(get_technical_score(student), 2)
    trait_score = round(get_trait_score(student), 2)
    matching_score = round(get_student_score(student), 2)
    return {
        "user_id": student.get("user_id") or student.get("userId") or student.get("student_id"),
        "name": student.get("name"),
        "skill_level": student.get("skill_level"),
        "score": matching_score,
        "technical_score": technical_score,
        "trait_score": trait_score,
        "response_reliability": get_response_reliability(student),
        "role": role, #학생이 원하는 역할 
        "role_group": get_role_group(role), #팀에서 할 역할
        "stack_score": student.get("stack_score", ""),
        "experience": student.get("experience", []),
        "strength": student.get("strength"),
        "weakness": student.get("weakness"),
        "suggestion": student.get("suggestion"),
        "personality_scores": student.get("personality_scores", {}),
        "development_scores": student.get("development_scores", {}),
        "personality_summary": student.get("personality_summary", {}),
        "development_summary": student.get("development_summary", {}),
        "matching_traits": student.get("matching_traits", {}),
        "leader_score": get_leader_score(student),
        "low_traits": student.get("matching_traits", {}).get("low_traits", []),
        "high_traits": student.get("matching_traits", {}).get("high_traits", []),
        "preferred_members": student.get("preferred_members", []),
        "wants_leader": student.get("wants_leader", False),
        "matching_preferences": student.get("matching_preferences", {}),
    }


# 팀에 학생을 실제로 추가하고 점수/역할 분포를 갱신한다.
# 선호 기반 채우기와 기존 순차 배치가 같은 추가 로직을 쓰도록 분리했다.
def add_student_to_team(team, student):
    team["members"].append(student)
    team["total_score"] += student["score"]
    team["role_groups"][student["role_group"]] = (
        team["role_groups"].get(student["role_group"], 0) + 1
    )


# 전체 학생 수와 목표 팀 크기를 받아 팀별 목표 인원 목록을 만든다.
# 팀 인원 차이가 최대 1명만 나도록 capacity 리스트를 반환한다.
def build_team_capacities(total_students, target_team_size=5):
    if total_students <= 0:
        return []

    team_count = math.ceil(total_students / target_team_size)
    base_size = total_students // team_count
    larger_team_count = total_students % team_count

    return [
        base_size + 1 if index < larger_team_count else base_size
        for index in range(team_count)
    ]


# 전체 학생 수 기준으로 3명 팀이 구조적으로 피할 수 없는지 확인한다.
# 팀 capacity 계산 결과에 3이 있으면 True를 반환한다.
def is_three_person_team_unavoidable(total_students):
    return any(capacity == 3 for capacity in build_team_capacities(total_students))


# 팀 capacity 목록을 받아 빈 팀 dict 목록을 만든다.
# 각 팀에는 이름, 멤버 목록, 총점, 역할 분포, 목표 인원이 초기화된다.
def make_empty_teams(team_capacities):
    # 실제 학생을 넣기 전에 빈 팀 틀을 먼저 만든다.
    # total_score는 팀 실력 합계, role_groups는 역할군 분포 기록용.
    if isinstance(team_capacities, int):
        team_capacities = [5 for _ in range(team_capacities)]

    return [
        {
            "team_name": f"팀 {index + 1}",
            "members": [],
            "total_score": 0,
            "role_groups": {},
            "capacity": capacity,
        }
        for index, capacity in enumerate(team_capacities)
    ]
    #team_count만큼 팀 생성

# 학생 한 명을 현재 팀들 중 어느 팀에 넣을지 선택한다.
# 선호팀원, 역할 다양성, 인원수, 성향 보완, 팀 총점을 기준으로 가장 적합한 팀을 반환한다.
def choose_team_for_student(teams, student):
    # 현재 학생을 어느 팀에 넣을지 고르는 함수.
    # 우선순위:
    # 1. total_score가 낮은 팀
    # 2. 같은 role_group이 적은 팀
    # 3. 현재 인원이 적은 팀
    role_group = student["role_group"] #팀에서 어떤 역할을 할지 role_group에 저장
    
    available_teams = [
        team for team in teams #팀에다가 teams 반복돌려서 조건에 맞으면 리스트 형태로 저장
        if len(team["members"]) < team.get("capacity", 5) #팀에 members수가 팀 목표 인원보다 적으면
    ]

    # 후보 팀에 학생을 넣었을 때 성향 약점이 반복되는 정도를 계산한다.
    # 낮은 성향이 높은 성향으로 보완되면 penalty를 낮춰 팀 선택에 유리하게 만든다.
    def trait_penalty(team):
        members = team.get("members", [])
        low_traits = get_low_trait_names(student)
        high_traits = get_high_trait_names(student)
        member_low_traits = set().union(*(get_low_trait_names(member) for member in members)) if members else set() #여러 타입을 허용하여 유연한 데이터 처리와 가독성을 동시에 제공
        member_high_traits = set().union(*(get_high_trait_names(member) for member in members)) if members else set()
        repeated_low_count = len(low_traits & member_low_traits)
        complemented_count = len((low_traits & member_high_traits) | (high_traits & member_low_traits))
        return repeated_low_count - complemented_count

    # 선호 팀원 보너스를 적용하되 인원 균형이나 제약을 깨면 0으로 처리한다.
    # 현재 팀 멤버와의 선호 관계가 안전하게 유지될 때만 preference_bonus를 반환한다.
    def safe_preference_bonus(team):
        members = team.get("members", [])
        bonus = preference_bonus(student, members)
        if not bonus:
            return 0
        projected_members = members + [student]
        if breaks_preference_constraints(projected_members):
            return 0

        projected_sizes = []
        for candidate_team in teams:
            candidate_size = len(candidate_team["members"])
            if candidate_team is team:
                candidate_size += 1
            projected_sizes.append(candidate_size)

        if projected_sizes and min(projected_sizes) > 0 and max(projected_sizes) - min(projected_sizes) > 1:
            return 0

        return bonus

    # 역할군 배치 우선순위를 계산한다.
    # 일반 역할은 다양성을 우선하고, game 역할군은 같은 팀에 모이도록 반대로 계산한다.
    def role_group_priority(team):
        current_count = team["role_groups"].get(role_group, 0)
        if role_group in ROLE_GROUPS_TO_KEEP_TOGETHER:
            return -current_count
        return current_count

    return min(
        available_teams,
        key=lambda team: ( #tuple이여서 위에서 아래순으로 우선순위를 따짐
            -safe_preference_bonus(team),
            role_group_priority(team), #기본은 역할 다양성, game은 같은 역할군이 있는 팀 우선.
            len(team["members"]), #팀 인원 ,팀 인원이 적은 팀 우선.
            trait_penalty(team),
            team["total_score"], #팀 총합 스코어는 마지막 보조 기준으로 사용.
        ),
    )


# 선호 정보가 있는 데이터에서 한 팀에 넣을 다음 학생을 고른다.
# 현재 팀원과 서로 선호 관계가 있으면 최우선으로 보고, 그다음 역할 다양성과 점수를 본다.
def choose_preference_candidate_for_team(team, candidates):
    members = team.get("members", [])
    role_groups = team.get("role_groups", {})

    def candidate_key(candidate):
        role_count = role_groups.get(candidate["role_group"], 0)
        return (
            -preference_bonus(candidate, members),
            role_count,
            -candidate["score"],
        )

    return min(candidates, key=candidate_key)


# 선호 팀원 입력이 있는 경우 팀 내부를 선호 연결 중심으로 먼저 채운다.
# 한 팀을 채울 때 현재 팀원과 연결된 학생을 우선 선택해 선호 반영률을 높인다.
def create_preference_seeded_teams(student_summaries, team_capacities):
    teams = make_empty_teams(team_capacities)
    unassigned = list(student_summaries)

    for team in teams:
        while unassigned and len(team["members"]) < team.get("capacity", 5):
            if not team["members"]:
                candidate = unassigned.pop(0)
            else:
                candidate = choose_preference_candidate_for_team(team, unassigned)
                unassigned.remove(candidate)

            add_student_to_team(team, candidate)

    return teams


# 팀 목록에서 같은 팀에 들어간 선호 관계 수를 계산한다.
# 값이 높을수록 학생들이 선택한 preferred_members가 더 많이 반영된 팀 구성이다.
def count_preference_hits(teams):
    team_by_student = {}
    for index, team in enumerate(teams):
        for member in team.get("members", []):
            if member.get("name"):
                team_by_student[member["name"]] = index

    return sum(
        1
        for team in teams
        for member in team.get("members", [])
        for preferred_name in get_preferred_members(member)
        if team_by_student.get(member.get("name")) == team_by_student.get(preferred_name)
    )


# 팀별 역할 다양성 합계를 계산한다.
# 선호 반영 수가 같을 때 한 역할군으로 쏠리는 배치를 피하기 위한 보조 기준이다.
def count_role_diversity(teams):
    return sum(
        len({
            member.get("role_group")
            for member in team.get("members", [])
            if member.get("role_group")
        })
        for team in teams
    )


# 팀 점수 격차를 계산한다.
# 선호 반영 수와 역할 다양성이 같을 때 점수 격차가 작은 배치를 고르는 보조 기준이다.
def get_team_score_gap(teams):
    scores = [
        sum(member.get("score", 0) for member in team.get("members", []))
        for team in teams
    ]
    if not scores:
        return 0
    return round(max(scores) - min(scores), 2)


# 팀 배치를 비교하기 위한 점수 tuple을 만든다.
# 역할 다양성과 점수 격차를 지키면서 선호 반영 수를 올리는 용도로 사용한다.
def get_preference_layout_score(teams):
    return (
        count_preference_hits(teams),
        count_role_diversity(teams),
        -get_team_score_gap(teams),
    )


# 선호 최적화 swap이 팀 균형을 과하게 깨는지 확인한다.
# 역할 다양성이 낮아지거나 점수 격차가 크게 벌어지는 교환은 적용하지 않는다.
def keeps_team_balance(candidate_teams, baseline_score_gap, min_role_diversity):
    score_gap = get_team_score_gap(candidate_teams)
    role_diversity = count_role_diversity(candidate_teams)
    allowed_score_gap = max(baseline_score_gap + 5, baseline_score_gap * 1.15)

    if role_diversity < min_role_diversity:
        return False
    if score_gap > allowed_score_gap:
        return False
    return True


# 학생 교환을 반복해 선호 반영 수가 늘어나는 팀 구성을 찾는다.
# 팀 크기는 유지하고, 역할/점수 균형을 해치지 않는 swap만 적용한다.
def optimize_teams_for_preferences(teams, max_passes=20):
    if not teams or not any(
        get_preferred_members(member)
        for team in teams
        for member in team.get("members", [])
    ):
        return teams

    optimized_teams = copy.deepcopy(teams)
    current_score = get_preference_layout_score(optimized_teams)
    baseline_score_gap = get_team_score_gap(optimized_teams)
    min_role_diversity = count_role_diversity(optimized_teams)

    for _ in range(max_passes):
        best_score = current_score
        best_swap = None

        for first_team_index in range(len(optimized_teams)):
            for second_team_index in range(first_team_index + 1, len(optimized_teams)):
                first_members = optimized_teams[first_team_index].get("members", [])
                second_members = optimized_teams[second_team_index].get("members", [])

                for first_member_index in range(len(first_members)):
                    for second_member_index in range(len(second_members)):
                        candidate_teams = copy.deepcopy(optimized_teams)
                        candidate_teams[first_team_index]["members"][first_member_index], candidate_teams[second_team_index]["members"][second_member_index] = (
                            candidate_teams[second_team_index]["members"][second_member_index],
                            candidate_teams[first_team_index]["members"][first_member_index],
                        )
                        if not keeps_team_balance(candidate_teams, baseline_score_gap, min_role_diversity):
                            continue
                        candidate_score = get_preference_layout_score(candidate_teams)
                        if candidate_score > best_score:
                            best_score = candidate_score
                            best_swap = (
                                first_team_index,
                                second_team_index,
                                first_member_index,
                                second_member_index,
                            )

        if best_swap is None:
            break

        first_team_index, second_team_index, first_member_index, second_member_index = best_swap
        optimized_teams[first_team_index]["members"][first_member_index], optimized_teams[second_team_index]["members"][second_member_index] = (
            optimized_teams[second_team_index]["members"][second_member_index],
            optimized_teams[first_team_index]["members"][first_member_index],
        )
        current_score = best_score

    return optimized_teams


# 팀 멤버 변경 후 total_score와 role_groups를 다시 계산한다.
# 선호 최적화 swap 이후 저장되는 팀 메타데이터가 실제 멤버와 맞게 한다.
def refresh_team_stats(team):
    role_groups = {}
    total_score = 0
    for member in team.get("members", []):
        total_score += member.get("score", 0)
        role_group = member.get("role_group")
        if role_group:
            role_groups[role_group] = role_groups.get(role_group, 0) + 1

    team["total_score"] = round(total_score, 2)
    team["role_groups"] = role_groups
    return team


# 분석된 학생 목록을 받아 LLM 전 1차 규칙 기반 팀 초안을 만든다.
# 점수, 역할군, 성향 보완, 선호팀원을 고려해 팀 capacity에 맞게 배정한다.
def create_initial_teams(analyzed_students, team_size=5, team_count=None):
    # 분석된 학생 리스트를 받아서 1차 팀 배정안을 만든다.
    # 이 단계는 LLM 없이 규칙 기반으로 빠르게 팀을 나누는 기본 틀이다.
    if not analyzed_students:
        return []

    if team_count is None:
        team_capacities = build_team_capacities(len(analyzed_students), team_size)
    else:
        base_size = len(analyzed_students) // team_count
        larger_team_count = len(analyzed_students) % team_count
        team_capacities = [
            base_size + 1 if index < larger_team_count else base_size
            for index in range(team_count)
        ]

    teams = make_empty_teams(team_capacities) #make_empty함수에 팀별 목표 인원을 넣어 팀 생성
    student_summaries = [ #키워드 꺼낸거 리스트에 저장
        make_student_summary(student) #student 넣어서 분석데이터에서 키워드만 꺼냄.
        for student in analyzed_students #analyzed_student for문 돌려서 student에 넣음
    ]

    has_preferences = any(student.get("preferred_members") for student in student_summaries)
    if has_preferences:
        teams = create_preference_seeded_teams(student_summaries, team_capacities)
    else:
        # 점수가 높은 학생부터 배치하되, 함께 배치해야 하는 역할군은 먼저 배치해 자리를 확보한다.
        sorted_students = sorted(
            student_summaries,#정렬할 데이터
            key=lambda student: (
                student["role_group"] in ROLE_GROUPS_TO_KEEP_TOGETHER,
                student["score"],
            ),#key = 정렬기준, 우선 묶음 역할군과 score를 기준으로 내림차순
            reverse=True, #내림차순
        )

        for student in sorted_students:
            team = choose_team_for_student(teams, student) #어떤 팀에 넣을지 정하는 함수에서 변수 넣어줌
            add_student_to_team(team, student)

    teams = optimize_teams_for_preferences(teams)

    for team in teams: #모든 팀 순회해서
        refresh_team_stats(team)
        leader = choose_preference_aware_leader(team["members"])
        team["leader"] = leader.get("name", "")
        team["leader_score"] = get_leader_score(leader) if leader else 0
        team["leader_reason"] = build_leader_reason(leader)
        team["personality_averages"] = calculate_trait_averages(team["members"], "personality_scores")
        team["development_averages"] = calculate_trait_averages(team["members"], "development_scores")
        team["trait_risks"] = build_team_trait_risks(team["members"])
        team["preference_notes"] = team_preference_notes(team["members"])
        team["matching_evidence"] = build_team_matching_evidence(team)
    #팀 점수를 소수점 둘째 자리까지 반올림.
    return teams

#llm
#LLM = 자연어 분석 기반 팀 조합 보정
#검증 코드 = LLM 실수 방지
# 학생 분석 목록에서 유효한 이름만 뽑아 리스트로 반환한다.
# LLM이 없는 이름을 만들지 못하게 allowed_student_names 입력으로 사용한다.
def get_student_names(students):
    # LLM이 이름을 새로 만들거나 오타를 내는 것을 줄이기 위해 원본 이름만 따로 넘긴다.
    return [student.get("name") for student in students if student.get("name")] #리스트 컴프리헨션


REASON_ROLE_LABELS = {
    "frontend": "프론트엔드",
    "backend": "백엔드",
    "ai_data": "AI",
    "app": "앱",
    "design": "디자인",
    "game": "게임",
    "fullstack": "풀스택",
    "devops": "DevOps",
    "security": "보안",
    "etc": "기타",
}


# role_group 값을 배정 이유 문장에 사용할 한국어 라벨로 바꾼다.
# 알 수 없는 값은 원래 값이나 기타로 반환한다.
def get_reason_role_label(role_group: str) -> str:
    return REASON_ROLE_LABELS.get(role_group or "", role_group or "기타")


# stack_score 문자열에서 점수 높은 기술명만 추출한다.
# 배정 이유 context에 넣을 대표 기술 목록을 limit개 반환한다.
def parse_reason_stack_names(stack_score: str, limit: int = 2) -> List[str]:
    scored_stacks = []

    for stack, score in re.findall(r"([^:\n]+):\s*(\d{1,2})점", stack_score or ""):
        scored_stacks.append((stack.strip(), int(score)))

    scored_stacks.sort(key=lambda item: item[1], reverse=True)
    return [stack for stack, _ in scored_stacks[:limit]]


# 학생 한 명의 높은/낮은 성향 목록을 이유 생성용으로 요약한다.
# matching_traits에서 high_traits, low_traits를 최대 4개씩 반환한다.
def summarize_reason_traits(member: Dict[str, Any]) -> Dict[str, Any]:
    matching_traits = member.get("matching_traits", {})
    return {
        "high_traits": matching_traits.get("high_traits", [])[:4],
        "low_traits": matching_traits.get("low_traits", [])[:4],
    }


# 팀원들 사이에서 낮은 성향을 높은 성향이 보완하는 관계를 찾는다.
# 이유 생성 LLM에 넘길 trait, low_member, supporters 목록을 만든다.
def build_trait_complements(members: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    complements = []

    for weak_member in members:
        weak_name = weak_member.get("name")
        for low_trait in weak_member.get("matching_traits", {}).get("low_traits", []):
            trait = low_trait.get("trait")
            supporters = []

            for supporter in members:
                if supporter.get("name") == weak_name:
                    continue
                for high_trait in supporter.get("matching_traits", {}).get("high_traits", []):
                    if high_trait.get("trait") == trait:
                        supporters.append({
                            "name": supporter.get("name"),
                            "score": high_trait.get("score"),
                        })

            if supporters:
                complements.append({
                    "trait": trait,
                    "label": low_trait.get("label") or TRAIT_LABELS.get(trait, trait),
                    "low_member": {
                        "name": weak_name,
                        "score": low_trait.get("score"),
                    },
                    "supporters": supporters[:3],
                })

    return complements[:6]


PERSONALITY_REASON_TRAITS = {
    "communication",
    "responsibility",
    "collaboration",
    "flexibility",
    "emotionalStability",
}


# 배정 이유에 사용할 신뢰 가능한 성격 성향 근거를 이름과 함께 정리한다.
# LOW 신뢰도 응답은 제외하고 점수 대신 성향 라벨과 보완 관계만 전달한다.
def build_personality_reason_evidence(members: List[Dict[str, Any]], limit: int = 3) -> Dict[str, Any]:
    reliable_members = [
        member
        for member in members
        if get_response_reliability(member) != "LOW"
    ]
    complements = []
    for complement in build_trait_complements(reliable_members):
        if complement.get("trait") not in PERSONALITY_REASON_TRAITS:
            continue
        complements.append({
            "trait": complement.get("label"),
            "member_to_support": (complement.get("low_member") or {}).get("name"),
            "supporters": [
                supporter.get("name")
                for supporter in complement.get("supporters", [])
                if supporter.get("name")
            ],
        })
        if len(complements) >= limit:
            break

    strengths = []
    for member in reliable_members:
        personality_scores = member.get("personality_scores", {})
        high_traits = [
            {
                "trait": TRAIT_LABELS.get(trait, trait),
                "score": score,
            }
            for trait, score in personality_scores.items()
            if trait in PERSONALITY_REASON_TRAITS and score >= 4
        ]
        high_traits.sort(key=lambda item: item["score"], reverse=True)
        if high_traits:
            strengths.append({
                "name": member.get("name"),
                "traits": [item["trait"] for item in high_traits[:2]],
            })

    return {
        "complements": complements,
        "strengths": strengths[: max(2, limit * 2)],
    }


# 후보 팀 결과와 학생 분석을 받아 이유 생성/보정용 팀 context를 만든다.
# 팀별 멤버, 역할 분포, 대표 기술, 성향 보완 관계, 성향 평균을 담아 반환한다.
def build_reason_context(candidate_result, analyzed_students):
    student_lookup = build_student_lookup(analyzed_students)
    contexts = []

    for team in get_candidate_teams(candidate_result):
        member_names = get_member_names(team)
        members = [student_lookup[name] for name in member_names if name in student_lookup]
        leader = team.get("leader") or choose_preference_aware_leader(members).get("name", "")
        role_counts = {}

        for member in members:
            role_group = member.get("role_group") or get_role_group(member.get("role"))
            role_counts[role_group] = role_counts.get(role_group, 0) + 1
        matching_evidence = team.get("matching_evidence") or build_team_matching_evidence({
            **team,
            "members": members,
        })

        contexts.append({
            "team_name": team.get("team_name"),
            "leader": leader,
            "member_names": member_names,
            "role_distribution": [
                {"role": get_reason_role_label(role_group), "count": count}
                for role_group, count in role_counts.items()
            ],
            "members": [
                {
                    "name": member.get("name"),
                    "role": get_reason_role_label(member.get("role_group") or get_role_group(member.get("role"))),
                    "skill_level": member.get("skill_level"),
                    "top_skills": parse_reason_stack_names(member.get("stack_score", ""), limit=2),
                    "traits": summarize_reason_traits(member),
                }
                for member in members
            ],
            "trait_complements": build_trait_complements(members),
            "personality_averages": calculate_trait_averages(members, "personality_scores"),
            "development_averages": calculate_trait_averages(members, "development_scores"),
        })

    return contexts

#팀 구성 TeamMatchingResult에 final_teams에 list로 들어감.
# LLM structured output에서 역할군별 인원 수를 표현하는 schema다.
# role_group 이름과 count를 받아 FinalTeam.role_groups에 들어간다.
class RoleGroupCount(BaseModel):
    role_group: str = Field(description="역할군 이름. 예: frontend, backend, ai_data, app, design, game, fullstack, devops, security, etc")
    count: int = Field(description="해당 역할군 인원 수")


# 화면에 보여줄 배정 이유 카드 schema다.
# title과 description을 받아 final team reason_cards에 저장한다.
class ReasonCard(BaseModel):
    title: str = Field(description="화면에 보여줄 배정 이유 카드 제목")
    description: str = Field(description="카드 제목에 맞는 2~3문장의 구체적인 팀 배정 이유 설명")


# LLM이 반환해야 하는 최종 팀 한 개의 schema다.
# 팀 이름, 멤버 이름, 총점, 역할 분포, 팀장, 이유 필드를 검증한다.
class FinalTeam(BaseModel):
    team_name: str = Field(description="팀 이름")
    members: List[str] = Field(description="팀원 이름 목록")
    total_score: float = Field(description="팀원 score 합계")
    role_groups: List[RoleGroupCount] = Field(description="역할군별 인원 수")
    leader: str = Field(description="추천 팀장 이름")
    reason_cards: List[ReasonCard] = Field(default_factory=list, description="최종 확정 후 별도 생성되는 배정 이유 카드")
    reason: str = Field(default="", description="최종 확정 후 별도 생성되는 호환용 요약 설명")

#팀 매칭할때 변경되었는거 알려주는
# 팀 매칭 LLM의 전체 structured output schema다.
# 최종 팀 목록과 변경 여부, 변경 요약, 검증 메모를 담는다.
class TeamMatchingResult(BaseModel):
    final_teams: List[FinalTeam] = Field(description="최종 팀 매칭 결과")
    changed: bool = Field(description="initial_teams에서 팀원이 바뀌었으면 true")
    change_summary: str = Field(description="사용자 재생성 요청을 어떻게 반영했는지 포함한 변경 내용 요약")
    validation_notes: str = Field(description="중복/누락/팀 수/인원 차이 검증 결과와 요청을 반영하지 못한 이유")



# 1차 팀 초안을 LLM이 최소 보정하도록 하는 프롬프트 chain을 만든다.
# 학생 분석, allowed names, initial teams, reason context를 입력받는 템플릿을 반환한다.
def get_matching_prompt_chain():
    system_prompt = """
당신은 캡스톤 프로젝트 팀 매칭을 보정하는 전문가다.
입력으로 학생 분석 데이터와 알고리즘이 만든 1차 팀 초안을 받는다.

역할:
- 알고리즘 초안을 기본 정답으로 보고, 자연어 분석상 명확히 더 좋은 조합이 있을 때만 최소한으로 보정한다.
- 보정이 필요하지 않으면 initial_teams를 그대로 유지하고 이유만 설명한다.
- 팀별 총점, 역할 다양성, 하위 등급 학생의 지원 가능성을 함께 본다.
- 성격 성향과 개발 성향 점수는 팀 보완 관계를 판단할 때 사용하되, reason에는 숫자 점수를 직접 쓰지 않는다.
- preferred_members는 강하게 고려하되, 점수/역할군/성향 균형을 깨면 선호를 분리할 수 있다.

매칭 기준:
- 한 학생은 정확히 한 팀에만 배정한다.
- 팀 수는 initial_teams의 팀 수와 동일하게 유지한다.
- 각 팀 인원 차이는 1명 이하를 유지한다.
- 팀 총점 차이를 크게 악화시키는 재배정은 하지 않는다.
- 하 또는 낮음 학생은 가능하면 중 이상의 학생과 함께 둔다.
- 같은 role_group만으로 구성된 팀은 가능하면 피하되, game 역할군은 프로젝트 특성상 가능한 같은 팀에 유지한다.
- suggestion, strength, weakness는 내부 판단 근거로만 사용하고 reason에 학생별 분석문을 옮겨 쓰지 않는다.
- 성향/개발 점수는 전체 점수표처럼 나열하지 말고, 낮은 성향을 높은 성향의 팀원이 보완하는 관계를 설명할 때만 사용한다.
- reason에는 "2점", "4.5점", "점수가 3"처럼 숫자 점수 표현을 쓰지 말고 "소통이 낮은 편", "책임감이 높은 팀원"처럼 자연어로 표현한다.
- 팀장 추천은 leadership, planning, problemSolving 조합인 leader_score를 우선한다.
- 팀 안에 wants_leader=true인 학생이 있으면 그 학생들 중 leader_score와 technical_score가 높은 학생을 팀장으로 추천한다.
- preferred_members를 떨어뜨린 경우에는 균형을 위해 분리했다는 이유를 reason에 자연스럽게 포함한다.

계산 규칙:
- 점수는 student_analysis 또는 initial_teams에 있는 score 값만 사용한다.
- 새로운 점수를 만들거나 skill_level을 바꾸지 않는다.
- total_score는 최종 팀원의 score 합으로 계산한다.
- 계산 과정을 장황하게 쓰지 말고 최종 JSON에 결과만 담는다.

이름 사용 규칙:
- allowed_student_names에 있는 이름만 사용한다.
- 학생 이름은 한 글자도 바꾸지 말고 그대로 복사한다.
- allowed_student_names에 없는 학생을 만들지 않는다.

출력 규칙:
- 반드시 지정된 structured output schema에 맞춰 출력한다.
- 최상위 키는 final_teams, changed, change_summary, validation_notes만 사용한다.
	- final_teams의 각 팀은 team_name, members, total_score, role_groups, leader를 포함한다.
		- role_groups는 [{{"role_group": "backend", "count": 1}}] 같은 배열 형식으로 작성한다. 가능한 role_group은 frontend, backend, ai_data, app, design, game, fullstack, devops, security, etc이다.
		- members는 학생 이름 문자열 배열로만 작성한다.
		- changed는 initial_teams에서 팀원이 바뀌었으면 true, 그대로면 false다.
        - reason_cards와 reason은 작성하지 않거나 빈 값으로 둔다. 배정 이유는 최종 팀 확정 후 별도 노드에서 생성한다.
"""

    user_prompt = """
아래 데이터를 바탕으로 최종 팀 매칭안을 만들어라.

allowed_student_names:
{allowed_student_names}

student_analysis:
{student_analysis}

initial_teams:
{initial_teams}

reason_context:
{reason_context}

요청:
1차 팀 초안을 기준으로 유지할지, 최소한으로 보정할지 판단해라.
이 단계에서는 팀원 배정, 역할 분포, 팀장만 결정해라.
배정 이유는 작성하지 마라.
최종 결과는 지정된 JSON 형식으로만 출력해라.
"""

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )
    #프롬포트에 딱히 문제될건 없는거 같음.

# LangGraph에서 1차 알고리즘 팀을 LLM에게 보내 최소 보정을 받는 노드다.
# state/analyzed_students를 입력받아 teams와 repair된 llm_result를 반환한다.
def llm_analyzed(state: MatchingState, analyzed_students=None):
    # 1차 알고리즘 팀 생성 이후 LLM에게 자연어 분석 기반 보정을 요청한다.
    if analyzed_students is None: #ananlyzed_student변수가 안들어오면 
        analyzed_students = state.get("analyzed_students", []) #state에서 불러와서 가져옴

    initial_teams = state.get("teams") or create_initial_teams(analyzed_students) #학생분석한거 state에 teams에 있으면 그거쓰고 없으면 다시 알고리즘으로 팀 짜서 가져옴
    allowed_student_names = get_student_names(analyzed_students) #이름 다르게 안나오게 하는 함수로 검증
    reason_context = build_reason_context(initial_teams, analyzed_students)

    llm = get_llm() #llm불러옴
    structured_llm = llm.with_structured_output(TeamMatchingResult) #TeamMatchingResult 형식으로 llm이 답변 생성하도록함
    chain = get_matching_prompt_chain() | structured_llm #chatprompttemplate함수에서 반환해서 structured_llm이랑 연결

    response = chain.invoke({
        #json.dump = 데이터를 json으로 저장하는 함수
        #ensure_ascil = 한글 같은 유니코드를 ASCII 형태로 바꿀지 결정하는 옵션. indent = 보기좋게 들여쓰기 할때 사용
        "allowed_student_names": json.dumps(allowed_student_names, ensure_ascii=False),
        "student_analysis": json.dumps(analyzed_students, ensure_ascii=False, indent=0),
        "initial_teams": json.dumps(initial_teams, ensure_ascii=False, indent=0),
        "reason_context": json.dumps(reason_context, ensure_ascii=False, indent=0),
    })
    #삼항연산자 (A if 조건 else B)

    ai_matching = response.model_dump() if hasattr(response, "model_dump") else response
    ai_matching = repair_team_matching_result(ai_matching, analyzed_students, initial_teams)
    #hasattr = 객체에 특정 속성(attribute)이나 함수(method)가 있는지 확인하는 함수.
    #model_dump = basemodel객체를 dict로 반환 
    #basemodel 객체를 상속하면 model_dump함수가 있기 때문에 있으면
    #response를 받아서 model_dump메서드 적용해서 basemodel객체를 dict로 반환 
    #아니면 그냥 response반환

    return {
        "teams": initial_teams,
        "llm_result": ai_matching, 
    }
#팀장하고 선호팀원 받아야하는데 선호팀원이 좀 빡셀듯. 선호팀원을 너무 잘하는사람끼리 모여있으면 선호팀원 안되게 할지 이거 내일 물어볼거. 아니면 그냥일단 다 배치 할지.
#그리고 몇몇 팀은 괜찮고 몇몇 팀은 별로일때 별로인 팀만 바꾸는 걸 만들어도 괜찮을듯

# LangGraph에서 1차 규칙 기반 팀 생성을 담당하는 노드다.
# state의 analyzed_students를 입력받아 생성된 teams만 업데이트 값으로 반환한다.
def create_team_node(state: MatchingState) -> Dict[str, Any]:
    # LangGraph에서 사용할 팀 생성 노드.
    # 입력 state에서 analyzed_students를 꺼내 1차 팀을 만들고,
    # 바뀐 값만 {"teams": teams} 형태로 반환하면 LangGraph가 state에 합쳐준다.
    analyzed_students = state.get("analyzed_students", []) #state에서 analyzed_students 값 받아서 
    teams = create_initial_teams(analyzed_students) 

    return {
        "teams": teams,
    } #팀을 teams state에 저장








#지금 여기 검증로직에서 그냥 점수만 대충 맞으면 llm도 알았다하고 알고리즘도 통과되서 검증뚫기가 너무 쉬움 여기부분 최적화 해야할듯


#밸런스 검사 노드
#llm_result 또는 teams를 가져와서 전체 평가 결과는 balance_result에,
#팀별 평가 결과는 team_evaluations에 저장한다.
#이 알고리즘이 위에 llm이 팀 생성한 것을 다시 알고리즘으로 잘했는지 검증함
# LLM 결과 dict나 알고리즘 팀 list에서 실제 팀 목록만 꺼낸다.
# candidate_result 형태가 달라도 이후 검증 함수가 같은 리스트로 처리하게 만든다.
def get_candidate_teams(candidate_result):
    # structured output 결과는 {"final_teams": [...]} 형태이고,
    # 알고리즘 초안은 바로 팀 리스트 형태라서 둘 다 받을 수 있게 정리한다.
    if isinstance(candidate_result, dict): #candidate_result가 dict인지 확인
        return candidate_result.get("final_teams", []) #candidate_result 딕셔너리에서 "final_teams" 키의 value 값을 가져오고,없으면 [] 반환
    if isinstance(candidate_result, list):
        return candidate_result #list면 그냥 반환.
    return [] #둘다 아니면 빈 리스트



# 팀 dict의 members에서 학생 이름만 추출한다.
# members가 dict 목록이든 문자열 목록이든 이름 문자열 리스트로 변환한다.
def get_member_names(team):
    # 알고리즘 초안의 members는 dict 리스트, LLM 결과의 members는 이름 문자열 리스트다.
    # 검증에서는 이름만 필요하므로 같은 형태로 맞춘다.
    member_names = [] #팀원 즉 사람이름을 넣어서 검증할 리스트
    for member in team.get("members", []): #member에 value가져와서 member로 순회
        if isinstance(member, dict): #member가 dict이면  isinstance는 객체가 어떤 타입인지 확인하는 함수.
            member_names.append(member.get("name")) #member에 name key에 value받아서 member_name 리스트에 추가.
        else:
            member_names.append(member) #만약 dict형태로 저장 되있지 않으면 그냥 member를 리스트에 추가 
    return [name for name in member_names if name] #member_name을 순회해서 name에 넣은 다음 name에 빈값없으면추가


# 분석된 학생 목록을 이름 기준 lookup dict로 만든다.
# 값은 make_student_summary 결과라 점수/역할/성향 정보를 빠르게 조회할 수 있다.
def build_student_lookup(analyzed_students):
    # 학생 이름으로 score, skill_level, role_group을 빠르게 찾기 위한 dict.
    return {
        student.get("name"): make_student_summary(student) #key : value로 저장
        for student in analyzed_students #학생들을 하나씩 순회.
        if student.get("name") #이름 없는 학생들 제외
    }


# 팀 하나의 현재 상태를 학생 lookup 기준으로 다시 계산한다.
# 멤버 수, 총점, 역할 분포, 실력 분포, 성향 평균, 리스크를 반환한다.
def calculate_team_status(team, student_lookup):
    # 팀원 이름 목록을 기준으로 실제 총점과 역할 분포를 다시 계산한다.
    member_names = get_member_names(team) #이름 가져와서 member_names에 저장
    total_score = 0 #팀 총점
    role_groups = {} #역할군 분포
    skill_levels = {} #실력 레벨 분포
    unknown_names = [] #분석 데이터에 없는이름
    members = []

    #팀원이름 검사
    for name in member_names: 
        student = student_lookup.get(name)
        if student is None:
            unknown_names.append(name)
            continue

        members.append(student)
        total_score += student["score"] #for문 돌면서 점수 더함.
        role_group = student["role_group"] #역할 넣어주기
        skill_level = student["skill_level"] 
        role_groups[role_group] = role_groups.get(role_group, 0) + 1 #총 팀 역할 더해주기
        skill_levels[skill_level] = skill_levels.get(skill_level, 0) + 1 

    return {
        "team_name": team.get("team_name"),
        "members": member_names,
        "member_count": len(member_names),
        "total_score": round(total_score, 2),
        "role_groups": role_groups,
        "skill_levels": skill_levels,
        "unknown_names": unknown_names,
        "leader_score": round(max([get_leader_score(member) for member in members] or [0]), 2),
        "personality_averages": calculate_trait_averages(members, "personality_scores"),
        "development_averages": calculate_trait_averages(members, "development_scores"),
        "trait_risks": build_team_trait_risks(members),
    }


# 후보 팀에서 학생별 선호 팀원이 같은 팀에 배치됐는지 확인한다.
# 미반영 선호 목록은 검증 단계에서 조정 노드로 보내기 위한 근거로 사용한다.
def find_unmet_preference_pairs(candidate_teams, analyzed_students):
    team_by_student = {}
    for team in get_candidate_teams(candidate_teams):
        for member_name in get_member_names(team):
            team_by_student[member_name] = team.get("team_name")

    unmet_pairs = []
    for student in analyzed_students:
        student_name = student.get("name")
        student_team = team_by_student.get(student_name)
        if not student_name or not student_team:
            continue

        for preferred_name in get_preferred_members(student):
            preferred_team = team_by_student.get(preferred_name)
            if preferred_team and preferred_team != student_team:
                unmet_pairs.append({
                    "student": student_name,
                    "preferred_member": preferred_name,
                    "student_team": student_team,
                    "preferred_team": preferred_team,
                })

    return unmet_pairs


# 함께 유지해야 하는 role_group이 여러 팀으로 쪼개졌는지 찾는다.
# team_evaluations를 입력받아 분리된 role_group 목록을 반환한다.
def find_split_keep_together_roles(team_evaluations):
    split_roles = []
    max_team_size = max(
        [evaluation["member_count"] for evaluation in team_evaluations] or [0]
    )

    for role_group in ROLE_GROUPS_TO_KEEP_TOGETHER:
        counts = [
            evaluation["role_groups"].get(role_group, 0)
            for evaluation in team_evaluations
        ]
        total_count = sum(counts)
        containing_team_count = sum(1 for count in counts if count > 0)

        if total_count > 1 and containing_team_count > 1 and total_count <= max_team_size:
            split_roles.append(role_group)

    return split_roles


# 후보 팀 결과를 코드로 검증해 balance_result와 team_evaluations를 만든다.
# 이름 중복/누락, 팀 수, 인원, 점수, 역할, 성향 리스크를 검사한다.
def validation_balance_team(candidate_result, analyzed_students, base_teams=None):
    # 알고리즘으로 팀 검증하여 수정할 필요가 있는지 없는지 판단한다.
    # 여기서는 LLM을 쓰지 않고, 이름/중복/누락/점수/인원/역할 분포를 코드로 검사한다.
    candidate_teams = get_candidate_teams(candidate_result) #candidate_result를 리스트로 저장
    student_lookup = build_student_lookup(analyzed_students) #이름 없으면 빼주기
    allowed_names = set(student_lookup.keys()) #이름 가져온것을 set으로이거 중복 제거
    base_team_count = len(base_teams or []) #팀 개수
    errors = [] #밑에서 추가할 에러 리스트
    warnings = [] #팀 오류 리스트
    team_evaluations = [] #밑에서 검증하면서 추가했던거 저장
    total_student_count = len(allowed_names)
    three_person_team_allowed = is_three_person_team_unavoidable(total_student_count)

    if not candidate_teams:
        errors.append("검증할 팀 결과가 없습니다.")

    if base_team_count and len(candidate_teams) != base_team_count:
        errors.append(
            f"팀 수가 맞지 않습니다. expected={base_team_count}, actual={len(candidate_teams)}"
        )

    all_member_names = []
    for team in candidate_teams:
        team_status = calculate_team_status(team, student_lookup)
        team_errors = []
        team_warnings = []

        if team_status["unknown_names"]:
            team_errors.append(f"없는 학생 이름이 포함되었습니다: {team_status['unknown_names']}")

        if team_status["member_count"] == 0:
            team_errors.append("팀원이 없습니다.")
        elif team_status["member_count"] < 3:
            team_warnings.append(f"팀 인원이 적습니다. member_count={team_status['member_count']}")
        elif team_status["member_count"] == 3 and not three_person_team_allowed:
            team_warnings.append("3명 팀이 포함되어 있습니다.")
        elif team_status["member_count"] > 5:
            team_errors.append(f"팀 인원이 5명을 초과했습니다. member_count={team_status['member_count']}")

        lower_level_count = (
            team_status["skill_levels"].get("하", 0)
            + team_status["skill_levels"].get("낮음", 0)
        )
        if lower_level_count == team_status["member_count"]:
            team_errors.append("하위 등급 학생만으로 구성된 팀입니다.")

        only_role_group = next(iter(team_status["role_groups"]), None)
        if (
            len(team_status["role_groups"]) == 1
            and team_status["member_count"] > 1
            and only_role_group not in ROLE_GROUPS_TO_KEEP_TOGETHER
        ):
            team_warnings.append("한 가지 역할군으로만 구성된 팀입니다.")

        for trait in CORE_RISK_TRAITS:
            trait_average = (
                team_status["personality_averages"].get(trait)
                or team_status["development_averages"].get(trait)
            )
            if trait_average is not None and trait_average < 3:
                team_warnings.append(f"{TRAIT_LABELS[trait]} 평균이 3 미만입니다.")

        if team_status["leader_score"] < 4.0:
            team_warnings.append("leader_score 4.0 이상 팀장 후보가 없습니다.")

        team_warnings.extend(team_status["trait_risks"])

        reported_score = team.get("total_score")
        if reported_score is not None:
            try:
                if abs(float(reported_score) - team_status["total_score"]) > 0.1:
                    team_warnings.append(
                        f"총점이 맞지 않습니다. reported={reported_score}, calculated={team_status['total_score']}"
                    )
            except (TypeError, ValueError):
                team_warnings.append(f"total_score가 숫자가 아닙니다: {reported_score}")

        team_evaluations.append({
            "team_name": team_status["team_name"],
            "is_valid": not team_errors,
            "errors": team_errors,
            "warnings": team_warnings,
            "member_count": team_status["member_count"],
            "total_score": team_status["total_score"],
            "role_groups": team_status["role_groups"],
            "skill_levels": team_status["skill_levels"],
            "leader_score": team_status["leader_score"],
            "personality_averages": team_status["personality_averages"],
            "development_averages": team_status["development_averages"],
            "trait_risks": team_status["trait_risks"],
        })

        errors.extend([f"{team_status['team_name']}: {error}" for error in team_errors])
        warnings.extend([f"{team_status['team_name']}: {warning}" for warning in team_warnings])
        all_member_names.extend(team_status["members"])

    duplicate_names = sorted({
        name for name in all_member_names
        if all_member_names.count(name) > 1
    })
    missing_names = sorted(allowed_names - set(all_member_names))
    unknown_names = sorted(set(all_member_names) - allowed_names)

    if duplicate_names:
        errors.append(f"중복 배정된 학생이 있습니다: {duplicate_names}")
    if missing_names:
        errors.append(f"누락된 학생이 있습니다: {missing_names}")
    if unknown_names:
        errors.append(f"분석 데이터에 없는 학생이 있습니다: {unknown_names}")

    member_counts = [
        evaluation["member_count"]
        for evaluation in team_evaluations
    ]
    if member_counts and max(member_counts) - min(member_counts) > 1:
        warnings.append(f"팀 인원 차이가 1명을 초과합니다: {member_counts}")

    split_keep_together_roles = find_split_keep_together_roles(team_evaluations)
    if split_keep_together_roles:
        errors.append(
            "같은 팀에 배치 가능한 역할군이 여러 팀으로 분리되었습니다: "
            f"{split_keep_together_roles}"
        )

    unmet_preference_pairs = find_unmet_preference_pairs(candidate_teams, analyzed_students)
    if unmet_preference_pairs:
        warnings.append(
            "선호 팀원이 같은 팀에 배치되지 않았습니다. "
            f"역할/점수/성향 균형을 우선해 미반영 선호를 기록합니다: {unmet_preference_pairs}"
        )

    team_scores = [
        evaluation["total_score"]
        for evaluation in team_evaluations
    ]
    score_gap = round(max(team_scores) - min(team_scores), 2) if team_scores else 0
    average_score = round(sum(team_scores) / len(team_scores), 2) if team_scores else 0
    soft_score_gap = max(15, average_score * 0.25) if average_score else 0
    hard_score_gap = max(30, average_score * 0.45) if average_score else 0

    if team_scores and score_gap > hard_score_gap:
        errors.append(
            f"팀 점수 차이가 허용 범위를 크게 초과합니다. gap={score_gap}, hard_max={round(hard_score_gap, 2)}"
        )
    elif team_scores and score_gap > soft_score_gap:
        warnings.append(
            f"팀 점수 차이가 큽니다. gap={score_gap}, recommended_max={round(soft_score_gap, 2)}"
        )

    balance_result = {
        "is_balanced": not errors,
        "need_adjustment": bool(errors),
        "errors": errors,
        "warnings": warnings,
        "missing_names": missing_names,
        "duplicate_names": duplicate_names,
        "unknown_names": unknown_names,
        "unmet_preference_pairs": unmet_preference_pairs,
        "team_count": len(candidate_teams),
        "member_counts": member_counts,
        "score_gap": score_gap,
        "average_score": average_score,
        "soft_score_gap": round(soft_score_gap, 2),
        "hard_score_gap": round(hard_score_gap, 2),
    }

    return balance_result, team_evaluations


# 보정된 팀 멤버 목록에서 role_groups 리스트를 다시 계산한다.
# 멤버 dict 목록을 받아 [{"role_group": ..., "count": ...}] 형태로 반환한다.
def build_repaired_role_groups(members: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    role_counts: Dict[str, int] = {}
    for member in members:
        role_group = member.get("role_group") or get_role_group(member.get("role"))
        role_counts[role_group] = role_counts.get(role_group, 0) + 1

    return [
        {"role_group": role_group, "count": count}
        for role_group, count in role_counts.items()
    ]


# 알고리즘 초안에서 학생 이름별 원래 팀 index를 기록한다.
# 누락 학생을 복구할 때 가능하면 원래 팀으로 되돌리기 위해 사용한다.
def build_base_team_index(base_teams: List[Dict[str, Any]]) -> Dict[str, int]:
    base_team_index = {}
    for index, team in enumerate(get_candidate_teams(base_teams)):
        for member_name in get_member_names(team):
            if member_name not in base_team_index:
                base_team_index[member_name] = index
    return base_team_index


# 누락 학생을 복구할 대상 팀 index를 선택한다.
# 원래 팀이 가능하면 우선하고, 아니면 인원/역할중복/총점이 낮은 팀을 고른다.
def choose_repair_team_index(
    repaired_teams: List[Dict[str, Any]],
    student: Dict[str, Any],
    preferred_index: Optional[int] = None,
) -> int:
    role_group = student.get("role_group")
    if (
        preferred_index is not None
        and 0 <= preferred_index < len(repaired_teams)
        and len(repaired_teams[preferred_index]["members"]) < 5
    ):
        return preferred_index

    available_indexes = [
        index
        for index, team in enumerate(repaired_teams)
        if len(team["members"]) < 5
    ] or list(range(len(repaired_teams)))

    # 보정 후보 팀의 우선순위를 계산한다.
    # 인원이 적고 같은 역할군이 적으며 총점이 낮은 팀일수록 먼저 선택된다.
    def sort_key(index: int):
        team = repaired_teams[index]
        role_count = sum(
            1
            for member_name in team["members"]
            if team["student_lookup"].get(member_name, {}).get("role_group") == role_group
        )
        return (
            len(team["members"]),
            role_count,
            team["total_score"],
        )

    return min(available_indexes, key=sort_key)


# 보정된 멤버 이름 목록으로 최종 팀 dict를 다시 만든다.
# 총점, 역할 분포, 팀장, 빈 reason 필드를 코드 기준으로 재계산한다.
def rebuild_repaired_team(team_name: str, member_names: List[str], student_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    members = [
        student_lookup[name]
        for name in member_names
        if name in student_lookup
    ]
    leader = choose_preference_aware_leader(members)
    return {
        "team_name": team_name,
        "members": [member["name"] for member in members],
        "total_score": round(sum(member.get("score", 0) for member in members), 2),
        "role_groups": build_repaired_role_groups(members),
        "leader": leader.get("name", ""),
        "reason_cards": [],
        "reason": "",
    }


# LLM 매칭 결과의 중복/누락/없는 이름을 코드로 보정한다.
# matching_result, 학생 분석, 기준 팀을 입력받아 안전한 final_teams와 repair_report를 반환한다.
def repair_team_matching_result(
    matching_result: Dict[str, Any],
    analyzed_students: List[Dict[str, Any]],
    base_teams: List[Dict[str, Any]],
) -> Dict[str, Any]:
    student_lookup = build_student_lookup(analyzed_students)
    allowed_names = get_student_names(analyzed_students)
    base_candidate_teams = get_candidate_teams(base_teams)
    base_team_index = build_base_team_index(base_candidate_teams)
    candidate_teams = get_candidate_teams(matching_result) or base_candidate_teams
    team_count = len(base_candidate_teams) or len(candidate_teams)
    if team_count == 0:
        return matching_result

    repaired_teams = []
    for index in range(team_count):
        source_team = candidate_teams[index] if index < len(candidate_teams) else {}
        base_team = base_candidate_teams[index] if index < len(base_candidate_teams) else {}
        repaired_teams.append({
            "team_name": source_team.get("team_name") or base_team.get("team_name") or f"팀 {index + 1}",
            "members": [],
            "total_score": 0,
            "student_lookup": student_lookup,
        })

    assigned_names = set()
    repair_report = {
        "ignored_unknown_names": [],
        "ignored_duplicate_names": [],
        "restored_missing_names": [],
        "restored_to_base_team": [],
        "restored_by_balance": [],
    }
    for index, source_team in enumerate(candidate_teams[:team_count]):
        for member_name in get_member_names(source_team):
            if member_name not in student_lookup:
                repair_report["ignored_unknown_names"].append(member_name)
                continue

            if member_name in assigned_names:
                repair_report["ignored_duplicate_names"].append(member_name)
                continue

            repaired_teams[index]["members"].append(member_name)
            repaired_teams[index]["total_score"] += student_lookup[member_name].get("score", 0)
            assigned_names.add(member_name)

    for missing_name in [name for name in allowed_names if name not in assigned_names]:
        student = student_lookup[missing_name]
        preferred_index = base_team_index.get(missing_name)
        target_index = choose_repair_team_index(repaired_teams, student, preferred_index)
        repaired_teams[target_index]["members"].append(missing_name)
        repaired_teams[target_index]["total_score"] += student.get("score", 0)
        assigned_names.add(missing_name)
        repair_report["restored_missing_names"].append(missing_name)
        report_key = (
            "restored_to_base_team"
            if preferred_index is not None and target_index == preferred_index
            else "restored_by_balance"
        )
        repair_report[report_key].append({
            "name": missing_name,
            "team_name": repaired_teams[target_index]["team_name"],
        })

    repaired_final_teams = [
        rebuild_repaired_team(team["team_name"], team["members"], student_lookup)
        for team in repaired_teams
    ]

    return {
        **matching_result,
        "final_teams": repaired_final_teams,
        "changed": matching_result.get("changed", True),
        "change_summary": matching_result.get("change_summary", ""),
        "validation_notes": (
            (matching_result.get("validation_notes") or "").strip()
            + " 구조 검증을 위해 중복/누락/점수/역할 분포를 코드로 보정했습니다."
        ).strip(),
        "repair_report": repair_report,
    }


# LangGraph에서 후보 팀의 균형 검증을 담당하는 노드다.
# 코드 검증을 우선하고, 구조 오류가 있을 때만 LLM 검증 결과를 참고로 합친다.
def evaluate_balance_node(state: MatchingState) -> Dict[str, Any]:
    # llm_result가 있으면 LLM 제안안을 검증하고, 없으면 알고리즘 teams를 검증한다.
    # 1. 알고리즘 검증을 먼저 돌린다.
    # 2. 코드 검증이 통과하면 LLM 검증을 생략한다.
    # 3. 구조 오류가 있을 때만 LLM 검증을 참고 경고로 합친다.
    # final_result는 여기서 저장하지 않고 finalize_node에서 따로 저장하는 구조가 좋다.
    analyzed_students = state.get("analyzed_students", [])
    base_teams = state.get("teams", [])
    candidate_result = state.get("llm_result") or base_teams

    algorithm_result, team_evaluations = validation_balance_team( #알고리즘으로 검증
        candidate_result=candidate_result,
        analyzed_students=analyzed_students,
        base_teams=base_teams,
    )
    if algorithm_result.get("errors"):
        llm_result = llm_validation_balance_team( #llm으로 검증
            candidate_result=candidate_result,
            analyzed_students=analyzed_students,
            algorithm_result=algorithm_result,
        )
    else:
        llm_result = {
            "is_balanced": True,
            "need_adjustment": False,
            "overall_reason": "코드 검증 통과로 LLM 검증을 생략했습니다.",
            "adjustment_request": "",
            "team_evaluations": [],
        }
    balance_result = merge_balance_results( #llm으로 검증한 결과와 알고리즘으로 검증한 결과 병합해서 최종적으로 어떻게 할건지 반환
        algorithm_result=algorithm_result,
        llm_result=llm_result,
    )

    return {
        "balance_result": balance_result,
        "team_evaluations": team_evaluations,
    }

#llm 검증로직
#팀 하나에 대한 평가
# LLM이 팀 하나의 정성 평가를 반환할 때 쓰는 schema다.
# 팀별 균형 여부, 이유, 강점, 리스크, 조정 제안을 검증한다.
class LLMBalanceTeamEvaluation(BaseModel):
    team_name: str = Field(description = "평가 대상 팀 이름") #팀이름
    is_balanced: bool = Field(description = "해당 팀이 역할, 실력, 협업 리스크 측면에서 균형적인지 여부") #팀 균형
    need_adjustment: bool = Field(description = "해당 팀에 팀원 조정이 필요한지 여부") #조정 필요한지 bool
    matching_reason : str = Field (description = (
        "화면에 보여줄 자연스러운 팀 배정 이유를 작성한다. "
        "역할 분담, 기술/경험 보완, 협업 시너지를 중심으로 왜 이 조합이 좋은지 설명한다. "
        "내부 필드명, true/false, 점수 나열, 학생 분석 전문은 포함하지 않는다."
    ))
    strengths: str = Field(description = "해당 팀의 특별한 강점을 구체적으로 설명") #팀 강점 
    risks: str = Field(description = "해당 팀의 특별한 리스크 또는 약점을 구체적으로 설명") #팀 약점
    adjustment_suggestion: str = Field(description = "조정이 필요할 경우 구체적인 수정 제안") #조정 제안

#전체적인 팀 매칭 결과에 대한 평가
# LLM이 전체 팀 매칭 검증 결과를 반환할 때 쓰는 schema다.
# 전체 균형 여부, 조정 필요성, 요청사항, 팀별 평가 목록을 담는다.
class LLMBalanceResult(BaseModel):
    is_balanced: bool = Field(description = "전체 팀 매칭 결과가 균형적인지 여부") #전체적인 벨런스
    need_adjustment: bool = Field(description = "전체 팀 매칭 결과에 수정이 필요한지 여부") #수정이 필요한지 bool
    overall_reason: str = Field(description = "전체 균형 평가 판단 이유") #이렇게 생각한 이유 
    adjustment_request: str = Field(description = "수정이 필요할 경우 adjust_team_node에 전달할 구체적인 수정 요청") #수정 요청 수정이 필요하면 어떻게 수정할지
    team_evaluations: List[LLMBalanceTeamEvaluation] = Field(description = "팀별 균형 평가 결과") #각 팀 평가
#이거 strutured output으로 쓰는데 description이 안적혀있음
# 후보 팀 결과를 LLM이 정성 검증하도록 하는 프롬프트 chain을 만든다.
# candidate_result, student_analysis, algorithm_result를 입력 변수로 사용한다.
def get_llm_balance_prompt_chain():
    system_prompt = """
    당신은 캡스톤 프로젝트 팀 매칭 결과를 검증하는 평가자다.
    숫자 계산, 이름 중복, 누락 검증은 algorithm_result를 우선한다.
    너는 협업 시너지, 역할 적합성, 팀별 리스크를 평가한다.
    """

    user_prompt = """
    candidate_result:
    {candidate_result}

    student_analysis:
    {student_analysis}

    algorithm_result:
    {algorithm_result}
    """

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt),
    ])


#llm으로 검증
# 후보 팀 결과를 LLM으로 정성 평가한다.
# 후보 팀, 학생 분석, 코드 검증 결과를 입력받아 LLMBalanceResult dict를 반환한다.
def llm_validation_balance_team(candidate_result, analyzed_students, algorithm_result):
    llm = get_llm()
    structured_llm = llm.with_structured_output(LLMBalanceResult) #스키마 넣어줘서 structured_llm 만들어줌
    chain = get_llm_balance_prompt_chain() | structured_llm

    response = chain.invoke({
        "candidate_result": json.dumps(candidate_result, ensure_ascii=False, indent=0),
        "student_analysis": json.dumps(analyzed_students, ensure_ascii=False, indent=0),
        "algorithm_result": json.dumps(algorithm_result, ensure_ascii=False, indent=0),
    })

    return response.model_dump()
#model_dump : basemodel 객에 dict으로 반환


# 코드 검증 결과와 LLM 검증 결과를 하나의 balance_result로 합친다.
# 실제 조정 여부는 코드 검증의 구조 오류를 기준으로 결정하고 LLM 의견은 warning으로 남긴다.
def merge_balance_results(algorithm_result, llm_result):
    # 알고리즘 검증 + LLM 검증 결과를 하나의 balance_result로 합친다.
    # 코드 검증은 중복/누락/없는 이름 같은 구조 오류만 하드 게이트로 사용한다.
    # LLM 검증은 정성 평가이므로 수정 루프를 강제하지 않고 경고/참고 의견으로만 남긴다.
    algorithm_is_balanced = algorithm_result.get("is_balanced", False) #균형 맞는지 is_balanced 반환하고 없으면 False반환
    algorithm_need_adjustment = algorithm_result.get("need_adjustment", True) #수정 필요한지 need_adjustment반환하고 없으면 True반환

    is_balanced = algorithm_is_balanced
    need_adjustment = algorithm_need_adjustment

    warnings = list(algorithm_result.get("warnings", []))
    if llm_result.get("need_adjustment"):
        llm_warning = llm_result.get("adjustment_request") or llm_result.get("overall_reason")
        if llm_warning:
            warnings.append(f"LLM 검증 참고: {llm_warning}")

    return {
        "is_balanced": is_balanced,
        "need_adjustment": need_adjustment,
        "next_node": "finalize_node" if is_balanced and not need_adjustment else "adjust_team_node", #다음 노드 뭘로 할지
        "algorithm_result": algorithm_result,
        "llm_result": llm_result,
        "errors": algorithm_result.get("errors", []),
        "warnings": warnings,
        "missing_names": algorithm_result.get("missing_names", []),
        "duplicate_names": algorithm_result.get("duplicate_names", []),
        "unknown_names": algorithm_result.get("unknown_names", []),
        "unmet_preference_pairs": algorithm_result.get("unmet_preference_pairs", []),
        "adjustment_request": llm_result.get("adjustment_request", ""),
    }


MAX_ITERATION = 3

# 최종 fallback이 필요한 치명적 배정 오류인지 확인한다.
# 선호팀원 미반영은 제외하고 누락/중복/없는 이름만 True로 본다.
def has_structural_assignment_error(balance_result: Dict[str, Any]) -> bool:
    algorithm_result = balance_result.get("algorithm_result", {})
    return bool(
        balance_result.get("missing_names")
        or balance_result.get("duplicate_names")
        or balance_result.get("unknown_names")
        or algorithm_result.get("missing_names")
        or algorithm_result.get("duplicate_names")
        or algorithm_result.get("unknown_names")
    )

# LangGraph 조건 분기에서 다음 노드를 결정한다.
# balance_result와 iteration_count를 보고 finalize_node 또는 adjust_team_node 문자열을 반환한다.
def should_adjust(state: MatchingState):
    # LangGraph 조건 분기 함수.
    # 코드 검증에서 치명 오류가 없으면 finalize_node, 있으면 adjust_team_node로 보낸다.
    #balance_result로 finalize_node adjust_team_node 구별
    balance_result = state.get("balance_result", {}) 
    if balance_result.get("is_balanced") and not balance_result.get("need_adjustment"):
        return "finalize_node"
    if state.get('iteration_count',0) >= MAX_ITERATION:
        return "finalize_node"
    return "adjust_team_node"











#팀 수정 노드
#team으로 알고리즘으로 team상태 저장하고
#llm_result로 llm이 제안한 팀 상태 저장하고
#검증할때 실패하면 다시 알고리즘 보고 할수있도록 알고리즘은 그대로 두고 llm_result만 계속 덮어 씌어지면서 수정
#수정 노드 작성할때 수정한 횟수 만큼 iteration_count올려야함 
# TeamMatchingResult, FinalTeam, get_llm()은 위에서 정의한 것을 재사용한다.
# 검증 실패한 팀 후보를 LLM이 다시 조정하도록 하는 프롬프트 chain을 만든다.
# 현재 후보, 알고리즘 초안, 검증 결과, 조정 이력을 입력 변수로 사용한다.
def get_adjust_team_prompt_chain():
    system_prompt = """
    당신은 캡스톤 프로젝트 팀 매칭 결과를 수정하는 담당자다.
    현재 팀 후보는 검증을 통과하지 못했거나 수정이 필요하다는 평가를 받았다.
    balance_result의 알고리즘 검증 결과와 LLM 검증 결과를 모두 반영해서 팀을 다시 제안하라.

    수정 원칙:
    - current_candidate를 기본으로 유지하고, 문제가 있는 부분만 최소한으로 수정한다.
    - algorithm_teams는 원래 알고리즘 초안이므로 참고 기준으로 사용한다.
    - 모든 학생은 정확히 한 팀에만 배정한다.
    - 팀 수는 algorithm_teams와 동일하게 유지한다.
    - 팀별 인원 차이는 1명 이하로 유지한다.
    - 팀 총점 차이를 크게 악화시키지 않는다.
    - 하 또는 낮음 학생은 가능하면 중 이상의 학생과 함께 둔다.
    - 같은 role_group만으로 구성된 팀은 가능하면 피하되, game 역할군은 프로젝트 특성상 가능한 같은 팀에 유지한다.
    - 팀 인원, 점수, 역할군, 성향 균형과 algorithm_result.errors 해결을 preferred_members보다 우선한다.
    - 서로를 선택한 상호 선호 페어는 균형이 비슷한 대안 중에서 우선 유지한다.
    - 역할군 이동이나 학생 교환 시 상호 선호 페어를 함께 이동해도 균형이 악화되지 않는지 먼저 검토한다.
    - 선호를 유지하면 필수 오류가 남거나 팀 균형이 뚜렷하게 나빠지는 경우에는 분리할 수 있으며, 이유를 validation_notes에 쓴다.
    - 단방향 preferred_members도 팀 균형을 해치지 않는 범위에서 고려한다.
    - 팀 안에 wants_leader=true인 학생이 있으면 그 학생들 중 leader_score와 technical_score가 높은 학생을 팀장으로 추천한다.
    - adjustment_history와 같은 수정 패턴을 반복하지 않는다.

    반영해야 할 정보:
    - balance_result.algorithm_result.errors는 반드시 해결한다.
    - balance_result.algorithm_result.warnings는 가능하면 완화한다.
    - balance_result.llm_result.adjustment_request는 사용자의 재생성 요청이다. 중복/누락/없는 이름/팀 수/팀 인원 차이 규칙을 깨지 않는 범위에서 적극적으로 반영한다.
    - 사용자 요청이 특정 역할군 분산, 팀원 이동, 팀장 변경처럼 실행 가능한 조건이면 current_candidate에서 최소 이동으로 반영한다.
    - current_candidate가 이미 사용자 요청을 만족하거나 검증 규칙 때문에 반영할 수 없는 경우가 아니라면 changed=false로 두지 않는다.
    - 사용자 요청을 전부 반영할 수 없으면 가능한 부분만 반영하고, 불가능한 이유를 validation_notes에 명확히 쓴다.
    - student_analysis의 strength, weakness, suggestion은 내부 판단 근거로만 사용하고 reason에 학생별 분석문을 옮겨 쓰지 않는다.
    - 성격성향/개발성향 점수는 팀 보완 관계를 판단할 때 사용하되, reason에는 숫자 점수를 직접 쓰지 않는다.

    계산 규칙:
    - 역할군 응집 오류를 고칠 때 해당 역할군의 전체 학생 수, 현재 팀별 인원, 이동할 학생 수를 먼저 계산한다.
    - 특정 역할군 전체를 한 팀에 모을 수 있으면 팀 균형과 선호 관계를 함께 비교해 오류가 남지 않는 배치를 선택한다.
    - 점수는 student_analysis 또는 algorithm_teams에 있는 score 값만 사용한다.
    - 새로운 점수나 skill_level을 만들지 않는다.
    - total_score는 최종 팀원의 score 합으로 작성한다.

    이름 사용 규칙:
    - allowed_student_names에 있는 이름만 사용한다.
    - 학생 이름은 한 글자도 바꾸지 말고 그대로 복사한다.
    - allowed_student_names에 없는 학생을 만들지 않는다.

    출력 규칙:
    - 반드시 지정된 structured output schema에 맞춰 출력한다.
    - final_teams의 각 팀은 team_name, members, total_score, role_groups, leader를 포함한다.
    - role_groups는 [{{"role_group": "backend", "count": 1}}] 같은 배열 형식으로 작성한다. 가능한 role_group은 frontend, backend, ai_data, app, design, game, fullstack, devops, security, etc이다.
    - members는 학생 이름 문자열 배열로만 작성한다.
    - changed는 current_candidate에서 팀원이 바뀌었으면 true, 그대로면 false다.
    - reason_cards와 reason은 작성하지 않거나 빈 값으로 둔다. 배정 이유는 최종 팀 확정 후 별도 노드에서 생성한다.
    - change_summary에는 사용자 요청을 어떻게 반영했는지와 어떤 검증 실패를 어떻게 고쳤는지 함께 작성한다.
    - validation_notes에는 중복/누락/없는 이름/팀 수/인원 차이 검증 결과와, 사용자 요청을 반영하지 못한 부분이 있으면 그 이유를 작성한다.
    """

    user_prompt = """
    아래 검증 실패 정보를 바탕으로 팀 후보를 수정해라.
    algorithm_result.errors의 중복 배정, 누락 학생, 없는 이름, 팀 수 오류를 최우선으로 해결해라.
    역할군 응집 오류는 전체 대상 인원과 팀 정원을 계산해 해결하되, 균형이 비슷한 대안이라면 상호 선호 페어를 유지해라.
    balance_result.llm_result.adjustment_request는 사용자 재생성 요청이다. 중복/누락/없는 이름/팀 수/인원 차이 규칙을 깨지 않는 범위에서 적극적으로 반영해라.
    사용자 요청을 완전히 반영할 수 없으면 가능한 대안을 적용하고 validation_notes에 반영하지 못한 이유를 써라.
    이 단계에서는 팀원 배정, 역할 분포, 팀장만 결정하고 배정 이유는 작성하지 마라.

    allowed_student_names:
    {allowed_student_names}

    student_analysis:
    {student_analysis}

    algorithm_teams:
    {algorithm_teams}

    current_candidate:
    {current_candidate}

    reason_context:
    {reason_context}

    balance_result:
    {balance_result}

    adjustment_history:
    {adjustment_history}
    """

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt),
    ])
# class MatchingState
#iteration_count 3으로할지 4로할지.
# LangGraph에서 검증 실패한 팀 후보를 LLM으로 재조정하는 노드다.
# state를 입력받아 새 llm_result, 증가한 iteration_count, adjustment_history를 반환한다.
def adjust_team_node(state: MatchingState) -> Dict[str, Any]:
    # balance_result를 보고 LLM이 팀 후보를 다시 수정한다.
    # 수정된 결과는 final_result가 아니라 llm_result에 덮어쓴다.
    # 이후 evaluate_balance_node에서 다시 검증하는 흐름이다.
    analyzed_students = state.get("analyzed_students", []) #state에서 get해서 analyzed_student value값을 가져옴.
    algorithm_teams = state.get("teams", []) #알고리즘으로 팀 저장한거 가져옴
    current_candidate = state.get("llm_result") or algorithm_teams #llm이 팀 저장한거 가져오는데 없으면 알고리즘 팀 사용.
    balance_result = state.get("balance_result", {}) #전체적인 팀균형 맞는지 가져옴.
    adjustment_history = state.get("adjustment_history", []) #같은 결과 나오지않게 어떻게 수정했는지 기록.
    iteration_count = state.get("iteration_count", 0) #무한 반복이 되지 않도록 초기값 0으로하고 state에서 가져옴.
    allowed_student_names = get_student_names(analyzed_students) #학생 이름 중복되지 않도록 검증.
    reason_context = build_reason_context(current_candidate, analyzed_students)

    llm = get_llm()
    structured_llm = llm.with_structured_output(TeamMatchingResult)
    chain = get_adjust_team_prompt_chain() | structured_llm

    response = chain.invoke({
        "allowed_student_names": json.dumps(allowed_student_names, ensure_ascii=False),
        "student_analysis": json.dumps(analyzed_students, ensure_ascii=False, indent=0),
        "algorithm_teams": json.dumps(algorithm_teams, ensure_ascii=False, indent=0),
        "current_candidate": json.dumps(current_candidate, ensure_ascii=False, indent=0),
        "reason_context": json.dumps(reason_context, ensure_ascii=False, indent=0),
        "balance_result": json.dumps(balance_result, ensure_ascii=False, indent=0),
        "adjustment_history": json.dumps(adjustment_history, ensure_ascii=False, indent=0),
    })

    adjusted_result = response.model_dump() if hasattr(response, "model_dump") else response
    raw_adjusted_result = copy.deepcopy(adjusted_result)
    adjusted_result = repair_team_matching_result(adjusted_result, analyzed_students, algorithm_teams)
    #hasattr = 객체에 특정 속성(attribute)이나 함수(method)가 있는지 확인하는 함수.
    #model_dump = basemodel객체를 dict로 반환 
    #basemodel 객체를 상속하면 model_dump함수가 있기 때문에 있으면
    #response를 받아서 model_dump메서드 적용해서 basemodel객체를 dict로 반환 
    #아니면 그냥 response반환
    next_iteration_count = iteration_count + 1
    adjustment_record = {
        "iteration": next_iteration_count, #몇번째 조정인지 
        "reason": balance_result.get("adjustment_request", ""), #조정이 왜 필요 했는지
        "algorithm_errors": balance_result.get("errors", []), #알고리즘에서 에러목록
        "algorithm_warnings": balance_result.get("warnings", []),#팀매칭에서 경고
        "raw_result": raw_adjusted_result, #LLM이 조정해서 만든 원본 결과
        "repair_report": adjusted_result.get("repair_report", {}), #코드 보정에서 제거/복구한 내용
        "result": adjusted_result, #코드 보정까지 끝난 최종 조정 결과
    }

    return {
        "llm_result": adjusted_result, #이번에 조정된 팀 매칭 결과
        "iteration_count": next_iteration_count, #반복수
        "adjustment_history": adjustment_history + [adjustment_record], #기존 조정기록에 요번에 추가한 리스트
    }


# 배정 이유에서 실제 팀장과 다른 학생을 팀장/리더로 언급한 문장을 제거한다.
# reason, 확정 leader_name, 팀원 이름 목록을 받아 정리된 reason 문자열을 반환한다.
def remove_conflicting_leader_sentences(reason: str, leader_name: str, member_names: List[str]) -> str:
    if not reason:
        return ""

    conflicting_names = [
        name
        for name in member_names
        if name and name != leader_name
    ]
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。])\s+", reason)
        if sentence.strip()
    ]
    filtered_sentences = []

    for sentence in sentences:
        mentions_wrong_leader = (
            any(name in sentence for name in conflicting_names)
            and any(keyword in sentence for keyword in ["팀장", "리더"])
        )
        if not mentions_wrong_leader:
            filtered_sentences.append(sentence)

    return " ".join(filtered_sentences)


# 이유 문자열에서 [강점], [약점], [리스크] 같은 후속 섹션을 잘라낸다.
# 최종 화면에 보여줄 핵심 reason 문장만 반환한다.
def clean_reason_sections(reason: str) -> str:
    if not reason:
        return ""

    return re.split(r"\s*\[(?:강점|보완점|약점|리스크)\]", reason, maxsplit=1)[0].strip()


# 최종 팀 이유 문구가 확정 팀장과 충돌하지 않게 정리한다.
# 팀 dict, 팀장 이름, 멤버 분석 목록을 입력받아 reason이 정리된 팀 dict를 반환한다.
def align_reason_with_leader(team: Dict[str, Any], leader_name: str, members: List[Dict[str, Any]]) -> Dict[str, Any]:
    member_names = [member.get("name") for member in members if member.get("name")]
    original_reason = team.get("reason", "")
    reason = clean_reason_sections(
        remove_conflicting_leader_sentences(original_reason, leader_name, member_names)
    )

    if not reason:
        reason = clean_reason_sections(original_reason)

    return {
        **team,
        "reason": reason,
    }


# 최종 팀 목록에 팀장 근거, 성향 평균, 리스크, 선호 메모를 보강한다.
# final_teams와 분석 학생 목록을 받아 화면/저장용 enriched team 목록 또는 dict를 반환한다.
def enrich_final_teams(final_teams, analyzed_students):
    student_lookup = build_student_lookup(analyzed_students)
    enriched_teams = []

    for team in get_candidate_teams(final_teams):
        member_names = get_member_names(team)
        members = [
            student_lookup[name]
            for name in member_names
            if name in student_lookup
        ]
        leader = choose_preference_aware_leader(members)
        leader_name = leader.get("name", team.get("leader", ""))
        enriched_team = align_reason_with_leader(dict(team), leader_name, members)
        enriched_team["leader"] = leader.get("name", enriched_team.get("leader", ""))
        enriched_team["leader_reason"] = build_leader_reason(leader)
        enriched_team["personality_averages"] = calculate_trait_averages(members, "personality_scores")
        enriched_team["development_averages"] = calculate_trait_averages(members, "development_scores")
        enriched_team["trait_risks"] = build_team_trait_risks(members)
        enriched_team["preference_notes"] = team_preference_notes(members)
        enriched_team["matching_evidence"] = build_team_matching_evidence({
            **enriched_team,
            "members": members,
        })
        enriched_teams.append(enriched_team)

    if isinstance(final_teams, dict):
        result = dict(final_teams)
        result["final_teams"] = enriched_teams
        return result

    return enriched_teams


# 팀원들의 stack_score에서 중복 없는 대표 기술명을 모은다.
# 이유 생성 context에 넣을 기술 스택을 limit개까지 반환한다.
def collect_representative_stacks_for_reason(members: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    stacks = []
    seen = set()

    for member in members:
        for stack in parse_reason_stack_names(member.get("stack_score", ""), limit=2):
            if stack not in seen:
                seen.add(stack)
                stacks.append(stack)
            if len(stacks) >= limit:
                return stacks

    return stacks


# 이유 생성 context에 넣을 학생 분석 텍스트를 자르지 않고 한 줄로 정리한다.
# 리스트는 문자열로 합치고 None은 빈 문자열로 반환한다.
def normalize_reason_text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value if item)
    if value is None:
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


# 팀 내 역할 분포가 어떤 배정 근거가 되는지 짧게 정리한다.
# 최종 reason LLM이 기술 나열 대신 실제 팀 구성 기준을 설명하도록 전달한다.
def build_role_balance_evidence(role_counts: Dict[str, int]) -> List[str]:
    evidence = []
    required_roles = ["backend", "frontend", "ai_data", "design", "app"]
    present_roles = [role for role in required_roles if role_counts.get(role)]
    missing_roles = [role for role in required_roles if not role_counts.get(role)]

    if len(present_roles) >= 4:
        evidence.append("핵심 구현 역할이 한쪽에 몰리지 않도록 역할군을 분산했습니다.")
    if missing_roles:
        evidence.append(
            "부족한 역할군은 "
            + ", ".join(get_reason_role_label(role) for role in missing_roles)
            + "입니다."
        )

    repeated_roles = [
        get_reason_role_label(role_group)
        for role_group, count in role_counts.items()
        if count > 1
    ]
    if repeated_roles:
        evidence.append(
            ", ".join(repeated_roles)
            + " 역할이 겹치므로 작업 범위를 초반에 나누는 것이 좋습니다."
        )

    return evidence


# 팀 안에서 실제로 반영된 선호 팀원 관계와 미반영 선호를 분리해 반환한다.
# reason 생성 시 "왜 같이 배치했는지"와 "왜 못 맞췄는지"를 근거로 사용한다.
def build_preference_evidence(members: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    member_names = {member.get("name") for member in members}
    matched = []
    unmatched = []

    for member in members:
        member_name = member.get("name")
        for preferred_name in get_preferred_members(member):
            if preferred_name in member_names:
                matched.append(f"선호 반영: {member_name} -> {preferred_name}")
            else:
                unmatched.append(f"선호 미반영: {member_name} -> {preferred_name}")

    return {
        "matched": matched[:6],
        "unmatched": unmatched[:4],
    }


# 팀장 선정이 팀장 희망을 반영했는지 정리한다.
# wants_leader 학생이 있으면 그 사실을 reason LLM에 명시적으로 넘긴다.
def build_leader_selection_evidence(leader_name: str, members: List[Dict[str, Any]]) -> Dict[str, Any]:
    leader_candidates = [
        member.get("name")
        for member in members
        if member.get("wants_leader")
    ]
    if leader_name and leader_name in leader_candidates:
        return {
            "leader": leader_name,
            "reason": "wants_leader",
            "candidate_names": leader_candidates,
        }
    if leader_candidates:
        return {
            "leader": leader_name,
            "reason": "balanced_leader_choice",
            "candidate_names": leader_candidates,
        }
    if leader_name:
        return {
            "leader": leader_name,
            "reason": "leader_score",
            "candidate_names": [],
        }
    return {}


# 팀 내 대표 학생 2~3명의 핵심 배치 근거를 만든다.
# 선호 반영, 팀장 희망, 역할 보완 순서로 중요한 근거만 골라 reason LLM에 전달한다.
def build_key_placements(
    members: List[Dict[str, Any]],
    role_counts: Dict[str, int],
    preference_evidence: Dict[str, List[str]],
    leader_evidence: Dict[str, Any],
) -> List[Dict[str, str]]:
    member_names = {member.get("name") for member in members}
    placements = []
    seen = set()

    def add_placement(student_name: str, reason_type: str, reason: str):
        if not student_name or student_name in seen or student_name not in member_names:
            return
        seen.add(student_name)
        placements.append({
            "student": student_name,
            "reason_type": reason_type,
            "reason": reason,
        })

    for matched in preference_evidence.get("matched", []):
        match = re.match(r"선호 반영:\s*(.+?)\s*->\s*(.+)", matched)
        if not match:
            continue
        student_name, preferred_name = match.group(1), match.group(2)
        add_placement(
            student_name,
            "preference",
            f"{preferred_name} 선호가 반영되어 초반 역할 조율 비용을 줄일 수 있습니다.",
        )
        if len(placements) >= 3:
            return placements

    if leader_evidence.get("reason") == "wants_leader":
        leader_name = leader_evidence.get("leader")
        add_placement(
            leader_name,
            "leader_preference",
            "팀장 희망이 반영되어 일정 정리와 의사결정 중심 역할을 맡기 좋습니다.",
        )

    single_role_members = [
        member
        for member in members
        if role_counts.get(member.get("role_group"), 0) == 1
    ]
    for member in single_role_members:
        add_placement(
            member.get("name"),
            "role_balance",
            f"{get_reason_role_label(member.get('role_group'))} 역할을 맡아 팀의 구현 범위를 보완합니다.",
        )
        if len(placements) >= 3:
            break

    return placements[:3]


# 팀원 역할 조합에서 실제 구현 흐름으로 이어질 수 있는 연결 근거를 만든다.
# 기술명 나열 대신 역할 간 산출물 흐름을 설명할 수 있게 reason LLM에 전달한다.
def build_implementation_connections(members: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    role_members: Dict[str, List[Dict[str, Any]]] = {}
    for member in members:
        role_group = member.get("role_group") or get_role_group(member.get("role"))
        role_members.setdefault(role_group, []).append(member)

    connection_specs = [
        ("backend", "frontend", "API와 화면 연동 흐름을 함께 맞출 수 있습니다."),
        ("backend", "app", "서버 기능을 앱 화면과 안정적으로 연결할 수 있습니다."),
        ("ai_data", "backend", "분석 결과를 API와 저장 구조로 연결하기 좋습니다."),
        ("ai_data", "frontend", "분석 결과를 사용자가 이해할 수 있는 화면으로 풀어낼 수 있습니다."),
        ("design", "frontend", "사용자 흐름과 웹 화면 구현을 함께 다듬기 좋습니다."),
        ("design", "app", "모바일 사용 흐름과 앱 화면 구현을 함께 맞추기 좋습니다."),
        ("fullstack", "frontend", "풀스택 경험을 화면 구현과 API 연동 사이의 조율에 활용할 수 있습니다."),
        ("fullstack", "backend", "풀스택 경험을 서버 구조와 화면 요구사항 사이의 조율에 활용할 수 있습니다."),
        ("devops", "backend", "배포와 서버 구현 흐름을 초반부터 함께 맞출 수 있습니다."),
        ("security", "backend", "인증과 권한 처리 리스크를 서버 구현 단계에서 함께 점검할 수 있습니다."),
    ]
    connections = []

    for source_role, target_role, reason in connection_specs:
        source = role_members.get(source_role, [None])[0]
        target = role_members.get(target_role, [None])[0]
        if not source or not target:
            continue
        connections.append({
            "source": source.get("name"),
            "source_role": get_reason_role_label(source_role),
            "target": target.get("name"),
            "target_role": get_reason_role_label(target_role),
            "reason": reason,
        })
        if len(connections) >= 4:
            break

    return connections


# 최종 팀 멤버 기준으로 reason 생성에 사용할 실제 매칭 근거를 구조화한다.
# 선호, 팀장, 구현 연결, 리스크처럼 사용자가 납득할 수 있는 관찰 근거만 저장한다.
def build_team_matching_evidence(team: Dict[str, Any]) -> Dict[str, Any]:
    members = [
        member
        for member in team.get("members", [])
        if isinstance(member, dict)
    ]
    role_counts: Dict[str, int] = {}
    for member in members:
        role_group = member.get("role_group") or get_role_group(member.get("role"))
        role_counts[role_group] = role_counts.get(role_group, 0) + 1

    leader_name = team.get("leader") or choose_preference_aware_leader(members).get("name", "")
    preference_evidence = build_preference_evidence(members)
    leader_evidence = build_leader_selection_evidence(leader_name, members)
    key_placements = build_key_placements(
        members,
        role_counts,
        preference_evidence,
        leader_evidence,
    )

    return {
        "role_distribution": [
            {"role": get_reason_role_label(role_group), "count": count}
            for role_group, count in role_counts.items()
        ],
        "preference": {
            **preference_evidence,
            "matched_count": len(preference_evidence.get("matched", [])),
            "unmatched_count": len(preference_evidence.get("unmatched", [])),
        },
        "leader_selection": leader_evidence,
        "key_placements": key_placements,
        "implementation_connections": build_implementation_connections(members),
        "risks": team.get("trait_risks", [])[:4] or build_team_trait_risks(members)[:4],
        "team_notes": team.get("preference_notes", [])[:6] or team_preference_notes(members)[:6],
        "leader_reason": team.get("leader_reason", ""),
    }


# reason LLM에 넘길 matching_evidence에서 내부 계산/알고리즘 산출물을 제외한다.
# 최종 문장은 선호, 팀장 희망, 구현 연결, 위험 요소 같은 설명 가능한 근거만 보게 한다.
def build_public_matching_evidence(matching_evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in (matching_evidence or {}).items()
        if key not in {"score_summary", "role_balance"}
    }


def dumps_llm_context(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_final_team_context(final_teams, analyzed_students, trait_complement_limit: int = 3):
    student_lookup = build_student_lookup(analyzed_students)
    contexts = []

    for team in get_candidate_teams(final_teams):
        member_names = get_member_names(team)
        members = [
            student_lookup[name]
            for name in member_names
            if name in student_lookup
        ]
        role_counts: Dict[str, int] = {}
        for member in members:
            role_group = member.get("role_group") or get_role_group(member.get("role"))
            role_counts[role_group] = role_counts.get(role_group, 0) + 1
        matching_evidence = team.get("matching_evidence") or build_team_matching_evidence({
            **team,
            "members": members,
        })
        public_matching_evidence = build_public_matching_evidence(matching_evidence)

        contexts.append({
            "team_name": team.get("team_name"),
            "member_count": len(member_names),
            "members": member_names,
            "leader": team.get("leader") or choose_preference_aware_leader(members).get("name", ""),
            "member_profiles": [
                {
                    "name": member.get("name"),
                    "role": get_reason_role_label(member.get("role_group") or get_role_group(member.get("role"))),
                    "top_skills": parse_reason_stack_names(member.get("stack_score", ""), limit=2),
                    "response_reliability": member.get("response_reliability", "HIGH"),
                    "strength": normalize_reason_text(member.get("strength")),
                    "suggestion": normalize_reason_text(member.get("suggestion")),
                    "experience": normalize_reason_text(member.get("experience", [])),
                }
                for member in members
            ],
            "matching_evidence": public_matching_evidence,
            "role_distribution": [
                {"role": get_reason_role_label(role_group), "count": count}
                for role_group, count in role_counts.items()
            ],
            "trait_complements": [
                {"trait": complement.get("label")}
                for complement in build_trait_complements(members)[:trait_complement_limit]
                if complement.get("label")
            ],
        })

    return contexts


# 최종 이유 카드 LLM에 넣을 팀별 context를 만든다.
# 팀별 멤버, 리더, 학생 분석 근거, 공개 가능한 매칭 근거를 반환한다.
def build_reason_card_context(final_teams, analyzed_students):
    contexts = build_final_team_context(
        final_teams,
        analyzed_students,
        trait_complement_limit=3,
    )
    student_lookup = build_student_lookup(analyzed_students)

    for context in contexts:
        members = [
            student_lookup[name]
            for name in context.get("members", [])
            if name in student_lookup
        ]
        context.pop("trait_complements", None)
        context["personality_evidence"] = build_personality_reason_evidence(members)

    return contexts


def build_team_analysis_context(final_teams, analyzed_students):
    return build_final_team_context(
        final_teams,
        analyzed_students,
        trait_complement_limit=2,
    )


# 최종 이유 카드 LLM이 팀 하나에 대해 반환해야 하는 schema다.
# 팀 이름, reason_cards, 호환용 reason 문자열을 검증한다.
class FinalTeamReasonCards(BaseModel):
    team_name: str = Field(description="reason_cards를 생성할 팀 이름")
    reason_cards: List[ReasonCard] = Field(
        min_length=2,
        max_length=4,
        description="팀에서 가장 설득력 있는 배정 이유 카드 최소 2개, 최대 4개",
    )
    reason: str = Field(description="reason_cards의 description들을 공백으로 이어 붙인 호환용 요약 설명")


# 최종 이유 카드 LLM의 전체 응답 schema다.
# 팀별 FinalTeamReasonCards 목록을 teams 필드에 담는다.
class FinalReasonCardsResult(BaseModel):
    teams: List[FinalTeamReasonCards] = Field(description="최종 확정 팀별 배정 이유 카드 목록")


# 최종 팀 강점/약점 LLM이 팀 하나에 대해 반환해야 하는 schema다.
# 팀 이름과 관리자 화면에 보여줄 strengths, weaknesses 문장을 검증한다.
class FinalTeamAnalysis(BaseModel):
    team_name: str = Field(description="강점/약점을 생성할 팀 이름")
    strengths: str = Field(description="팀원 조합으로 만들 수 있는 시너지를 설명하는 강점 문장")
    weaknesses: str = Field(description="팀 구성상 생길 수 있는 리스크와 보완 방식을 설명하는 약점 문장")


# 최종 팀 강점/약점 LLM의 전체 응답 schema다.
# 팀별 FinalTeamAnalysis 목록을 teams 필드에 담는다.
class FinalTeamAnalysisResult(BaseModel):
    teams: List[FinalTeamAnalysis] = Field(description="최종 확정 팀별 강점/약점 목록")


# 최종 확정 팀의 강점/약점을 예시 톤으로 생성하는 프롬프트 chain을 만든다.
# 팀원 변경 없이 member_profiles 근거를 사용해 시너지와 보완점을 쓰게 한다.
def get_final_team_analysis_prompt_chain():
    system_prompt = """
    당신은 캡스톤 팀 추천 결과를 관리자에게 설명하는 담당자다.
    팀원 배정은 이미 끝났으므로 팀원, 팀 수, 팀 이름, 팀장, 역할 분포를 절대 바꾸지 않는다.
    목표는 최종 팀만 보고 그럴듯한 문장을 만드는 것이 아니라, matching_evidence에 있는 실제 배치 근거를 바탕으로 설명하는 것이다.
    matching_evidence에는 알고리즘 초안, 점수 합계, 검증 결과가 들어 있지 않으므로 임의로 추측하지 않는다.

    출력 규칙:
    - 반드시 지정된 structured output schema에 맞춰 출력한다.
    - teams의 각 항목은 team_name, strengths, weaknesses만 포함한다.
    - strengths는 2문장으로 작성한다.
    - weaknesses는 2문장으로 작성한다.
    - 모든 문장은 관리자 화면에 그대로 노출된다. 반드시 존댓말로 작성하고, 문장 끝은 "-습니다", "-입니다", "-됩니다", "-합니다" 중 하나로 끝낸다.
    - 숫자 점수는 쓰지 말고 "구현 경험이 충분한 편", "소통이 안정적인 편"처럼 자연어로 표현한다.
    - strengths 첫 문장은 matching_evidence의 preference, leader_selection, key_placements 중 하나를 반드시 반영한다.
    - strengths 둘째 문장은 "A의 어떤 능력과 B의 어떤 능력이 만나 어떤 결과를 만들 수 있는지"를 구체적으로 쓴다.
    - weaknesses는 부족한 역할이나 병목 가능성을 말한 뒤, 어떤 방식으로 보완하면 좋은지까지 쓴다.
    - 학생 이름은 현재 팀원 중 필요한 1~2명만 언급한다. 모든 학생을 나열하지 않는다.
    - 기술 스택은 꼭 필요할 때만 1~2개 사용한다.
    - "역할이 고르게 배치되어", "협업 성향이 안정적인 팀", "시너지 극대화", "동시에 만족", "알고리즘", "점수 기준" 같은 일반적이거나 기계적인 표현은 쓰지 않는다.
    - matching_evidence에 없는 소통/협업 안정성은 새로 만들어 쓰지 않는다.
    - member_profiles의 response_reliability가 LOW인 학생은 성향 점수를 강한 근거로 쓰지 말고, 기술 스택, 구현 경험, 희망 역할을 중심으로 설명한다.
    - 현재 팀원이 아닌 학생 이름은 절대 언급하지 않는다.

    """

    user_prompt = """

    reason_context:
    {reason_context}

    요청:
    위 최종 팀 구성은 확정된 결과다. 팀원을 바꾸지 말고 각 팀의 strengths와 weaknesses만 작성해라.
    """

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt),
    ])


# LLM이 만든 강점/약점 문장을 화면 노출용으로 정리한다.
# 값이 비어 있으면 알고리즘 문장으로 대체하지 않고 빈 문자열을 반환한다.
def sanitize_team_analysis_text(value: Any) -> str:
    text = normalize_reason_text(value)
    text = re.sub(r"^\s*\[(?:강점|보완점|약점|리스크)\]\s*", "", text).strip()
    return text


def chunk_team_batches(teams: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [
        teams[index:index + batch_size]
        for index in range(0, len(teams), batch_size)
    ]


def run_parallel_team_batches(
    final_teams,
    analyzed_students,
    worker_fn,
    worker_env_name: str,
    batch_env_name: str,
    default_workers: int = 3,
    default_batch_size: int = 6,
    error_fields: Optional[Dict[str, Any]] = None,
    error_key: str = "generation_error",
    task_label: str = "팀 설명",
):
    teams = get_candidate_teams(final_teams)
    if not teams:
        return []

    max_workers = max(1, int(os.getenv(worker_env_name, str(default_workers))))
    batch_size = max(1, int(os.getenv(batch_env_name, str(default_batch_size))))
    batches = chunk_team_batches(teams, batch_size)
    results = [None] * len(batches)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker_fn, batch, analyzed_students): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            index = futures[future]
            batch = batches[index]
            try:
                results[index] = future.result()
            except Exception as error:
                batch_names = ", ".join(team.get("team_name", "") for team in batch)
                print(f"{batch_names} {task_label} batch 처리 실패: {type(error).__name__}: {error}")
                results[index] = [
                    {
                        **team,
                        **(error_fields or {}),
                        error_key: f"{type(error).__name__}: {error}",
                    }
                    for team in batch
                ]

    fixed_teams = []
    for batch_result in results:
        if batch_result:
            fixed_teams.extend(batch_result)
    return fixed_teams


def parallelization_strength_weakness_batch(
    teams: List[Dict[str, Any]],
    analyzed_students: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    try:
        analysis_context = build_team_analysis_context(teams, analyzed_students)
        llm = get_llm()
        structured_llm = llm.with_structured_output(FinalTeamAnalysisResult)
        chain = get_final_team_analysis_prompt_chain() | structured_llm
        response = chain.invoke({
            "reason_context": dumps_llm_context(analysis_context),
        })
        result = response.model_dump() if hasattr(response, "model_dump") else response
    except Exception as error:
        batch_names = ", ".join(team.get("team_name", "") for team in teams)
        print(f"{batch_names} 강점/약점 생성 실패: {type(error).__name__}: {error}")
        return [
            {
                **team,
                "strengths": "",
                "weaknesses": "",
                "analysis_generation_error": f"{type(error).__name__}: {error}",
            }
            for team in teams
        ]

    analysis_by_team = {
        team.get("team_name"): team
        for team in result.get("teams", [])
        if isinstance(team, dict)
    }
    fixed_teams = []
    for team in teams:
        fixed_team = dict(team)
        generated = analysis_by_team.get(fixed_team.get("team_name"), {})
        fixed_team["strengths"] = sanitize_team_analysis_text(generated.get("strengths"))
        fixed_team["weaknesses"] = sanitize_team_analysis_text(generated.get("weaknesses"))
        fixed_teams.append(fixed_team)

    return fixed_teams


def run_parallel_strength_weakness(final_teams, analyzed_students):
    return run_parallel_team_batches(
        final_teams=final_teams,
        analyzed_students=analyzed_students,
        worker_fn=parallelization_strength_weakness_batch,
        worker_env_name="FINAL_ANALYSIS_WORKERS",
        batch_env_name="FINAL_ANALYSIS_BATCH_SIZE",
        default_workers=3,
        default_batch_size=6,
        error_fields={
            "strengths": "",
            "weaknesses": "",
        },
        error_key="analysis_generation_error",
        task_label="강점/약점",
    )


# 최종 팀 목록에 대해 LLM으로 강점/약점 문장을 생성한다.
# 실패하거나 누락된 팀은 알고리즘 fallback 없이 strengths/weaknesses를 비워 둔다.
def generate_final_team_analysis(final_teams, analyzed_students):
    try:
        analysis_context = build_team_analysis_context(final_teams, analyzed_students)
        llm = get_llm()
        structured_llm = llm.with_structured_output(FinalTeamAnalysisResult)
        chain = get_final_team_analysis_prompt_chain() | structured_llm
        response = chain.invoke({
            "reason_context": dumps_llm_context(analysis_context),
        })
        result = response.model_dump() if hasattr(response, "model_dump") else response
    except Exception as error:
        print(f"최종 팀 강점/약점 생성 실패: {type(error).__name__}: {error}")
        return [
            {
                **team,
                "strengths": "",
                "weaknesses": "",
            }
            for team in get_candidate_teams(final_teams)
        ]

    analysis_by_team = {
        team.get("team_name"): team
        for team in result.get("teams", [])
        if isinstance(team, dict)
    }
    fixed_teams = []
    for team in get_candidate_teams(final_teams):
        fixed_team = dict(team)
        generated = analysis_by_team.get(fixed_team.get("team_name"), {})
        fixed_team["strengths"] = sanitize_team_analysis_text(generated.get("strengths"))
        fixed_team["weaknesses"] = sanitize_team_analysis_text(generated.get("weaknesses"))
        fixed_teams.append(fixed_team)

    return fixed_teams


# 최종 확정 팀의 관리자 노출용 reason_cards 생성 프롬프트를 만든다.
# final_teams와 reason_context를 입력받아 팀원 변경 없이 이유 카드만 생성하게 한다.
def get_final_reason_cards_prompt_chain():
    system_prompt = """
    당신은 최종 확정된 캡스톤 팀 구성에 대해 관리자 화면에 보여줄 배정 이유 카드를 작성한다.
    팀원 배정은 이미 끝났으므로 팀원, 팀 수, 팀 이름, 팀장, 역할 분포를 절대 바꾸지 않는다.
    이 작업은 규칙 기반 fallback 문구를 대체하기 위한 최종 사용자 노출 문구 작성이다.
    절대 알고리즘 설명처럼 쓰지 말고, 실제 관리자가 납득할 수 있는 자연스러운 존댓말 문장으로 작성한다.
    핵심은 matching_evidence, member_profiles, personality_evidence를 비교해 이 팀에서 가장 설득력 있는 배정 이유를 기본 3개 고르는 것이다.
    서로 다른 강한 근거가 충분하면 4개까지 작성하고, 근거가 부족하면 억지로 늘리지 말고 2개만 작성한다.
    matching_evidence에는 알고리즘 초안, 점수 합계, 검증 결과가 없으므로 그런 값을 근거로 쓰지 않는다.
    근거가 약한 리더십/역할 균형/기술 조합 카드를 억지로 만들지 않는다.

    출력 규칙:
    - 반드시 지정된 structured output schema에 맞춰 출력한다.
    - teams의 각 항목은 team_name, reason_cards, reason만 포함한다.
    - 각 팀의 reason_cards는 기본 3개 작성한다.
    - 서로 다른 강한 근거가 충분하면 4개까지 작성할 수 있다.
    - 설득력 있는 근거가 부족하면 반복하거나 추측하지 말고 2개만 작성한다.
    - reason_cards의 title은 팀의 가장 강한 배정 근거를 구체적으로 드러낸다.
    - "리더십 중심의 팀 운영 가능", "팀장 희망을 반영한 운영 중심 팀"은 팀장 희망 반영이 이 팀의 가장 중요한 이유일 때만 사용한다.
    - 각 description은 제목을 반복하지 말고 130~220자 정도의 2~3문장으로 작성한다.
    - 모든 description 문장은 관리자 화면에 그대로 노출된다. 반드시 존댓말로 작성하고, 모든 문장 끝은 "-습니다", "-입니다", "-됩니다", "-합니다" 중 하나로 끝낸다.
    - 절대 쓰면 안 되는 종결: "한다", "된다", "높인다", "해소한다", "유지한다", "기대된다", "가능하다", "충족시킨다".
    - 절대 쓰면 안 되는 표현: "알고리즘", "규칙 기반", "fallback", "점수 기준", "균형 계산", "시너지 극대화", "동시에 만족", "품질을 높인다".
    - 각 reason_card는 서로 다른 배정 근거를 담는다. 같은 말을 제목만 바꿔 반복하지 않는다.
    - personality_evidence에 두 명 이상의 신뢰 가능한 서로 다른 성향 정보가 있으면 성향 보완 또는 성향 강점 조합을 설명하는 카드를 반드시 1개 작성한다.
    - 성향 카드는 구체적인 팀원 이름과 소통, 책임감, 협업, 유연성 같은 실제 라벨을 사용해 누가 어떤 부분을 보완하는지 설명한다.
    - 성향이 모두 좋다는 식의 칭찬이나 성향만으로 성과를 단정하는 문장은 쓰지 않는다.
    - 카드 중 최소 1개는 matching_evidence.key_placements 또는 preference.matched를 반영한다.
    - 최소 1개는 matching_evidence.implementation_connections를 반영해 역할 간 구현 흐름을 설명한다.
    - preference.matched가 있으면 선호 관계를 우선 검토하되, 역할 균형이나 구현 연결이 약하면 억지로 쓰지 않는다.
    - leader_selection은 보조 근거다. 팀장 희망이 실제 팀 운영상 핵심 장점일 때만 한 문장 이내로 언급한다.
    - risks는 약점/주의점 근거로만 사용하고, 장점처럼 포장하지 않는다.
    - suggestion은 참고 근거로만 사용하고 원문을 요약하거나 복사하지 않는다.
    - 각 카드 description에는 현재 팀원 중 필요한 1~2명의 이름을 언급하고, 왜 같은 팀에 둔 판단인지 설명한다.
    - "A가 선호한 B와 같은 팀에 배치해 초반 소통 비용을 줄이고, C가 부족한 역할을 보완하도록 구성했습니다"처럼 배정 의도를 드러낸다.
    - 학생별 경험 목록, 스택 목록, 구현 기능 목록을 나열하지 않는다. 한 학생당 대표 능력 하나만 고른다.
    - 기술 스택은 꼭 필요할 때만 카드당 1~2개 사용한다.
    - 현재 팀원이 아닌 학생 이름은 절대 언급하지 않는다.
    - "각 구성원의 소통, 책임감, 협업, 유연성 등 성격 성향이 고르게 분포되어", "협업 성향이 안정적인 팀" 같은 일반 템플릿 문장을 쓰지 않는다.
    - matching_evidence에 없는 협업 안정성, 소통 안정성, 책임감 안정성은 새로 만들어 쓰지 않는다.
    - member_profiles의 response_reliability가 LOW인 학생은 성향 점수를 강한 배정 근거로 쓰지 않는다. 필요하면 "설문 응답 성향 정보는 참고 수준으로 활용하고 구현 경험과 희망 직군을 중심으로 배치했습니다."처럼 완곡하게 설명한다.
    - 숫자 점수는 되도록 쓰지 말고 "소통이 낮은 편", "책임감이 높은 편", "구현 경험이 풍부한 편"처럼 자연어로 표현한다.
    - reason은 reason_cards의 모든 description을 공백으로 이어 붙여 작성한다.

    나쁜 문장 예시:
    - "백엔드, 프론트엔드, 디자인, 앱, AI 역할이 고르게 배치되어 각 담당자가 맡은 범위에 집중하기 쉬운 구성입니다."
    - "Spring Boot와 Java를 활용한 백엔드 구현, Vue를 활용한 프론트엔드 화면 구성, Flutter와 Dart를 활용한 앱 개발이 자연스럽게 이어집니다."
    - "이 팀은 역할이 균형 있게 구성되어 안정적인 협업이 가능합니다."
    - "김아린의 Redux Toolkit 역량과 신우진의 Pandas 역량이 함께하면 협업 성향이 안정적인 팀이 됩니다."
    - "권서준이 리더십/정리 역량 기준 팀장 추천으로 리더십을 발휘하고, 민하준과 김아린이 각각 앱과 프론트엔드 핵심 역량을 제공해 역할 균형을 맞췄습니다."
    - "김도현은 Spring Boot와 Spring Security를 활용한 인증/인가 시스템 구현, JPA 연관관계 매핑, Docker Compose 개발 환경 구축 경험이 있습니다."
    """

    user_prompt = """

    reason_context:
    {reason_context}

    요청:
    위 최종 팀 구성은 확정된 결과다. 팀원을 바꾸지 말고 각 팀에서 가장 괜찮은 매칭 이유 reason_cards를 기본 3개 작성해라.
    서로 다른 강한 근거가 충분하면 4개까지 작성하고, 근거가 부족하면 2개만 작성해라.
    """

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt),
    ])


def parallelization_reason_cards_batch(
    teams: List[Dict[str, Any]],
    analyzed_students: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    try:
        reason_context = build_reason_card_context(teams, analyzed_students)
        llm = get_llm()
        structured_llm = llm.with_structured_output(FinalReasonCardsResult)
        chain = get_final_reason_cards_prompt_chain() | structured_llm
        response = chain.invoke({
            "reason_context": dumps_llm_context(reason_context),
        })
        result = response.model_dump() if hasattr(response, "model_dump") else response
    except Exception as error:
        batch_names = ", ".join(team.get("team_name", "") for team in teams)
        print(f"{batch_names} 배정 이유 생성 실패: {type(error).__name__}: {error}")
        return [
            {
                **team,
                "reason_cards": [],
                "reason": "",
                "reason_generation_error": f"{type(error).__name__}: {error}",
            }
            for team in teams
        ]

    cards_by_team = {
        team.get("team_name"): team
        for team in result.get("teams", [])
        if isinstance(team, dict)
    }
    fixed_teams = []
    for team in teams:
        fixed_team = dict(team)
        generated = cards_by_team.get(fixed_team.get("team_name"), {})
        fixed_team["reason_cards"] = generated.get("reason_cards") or []
        fixed_team["reason"] = generated.get("reason", "")
        fixed_teams.append(fixed_team)

    return fixed_teams


def run_parallel_reason_cards(final_teams, analyzed_students):
    return run_parallel_team_batches(
        final_teams=final_teams,
        analyzed_students=analyzed_students,
        worker_fn=parallelization_reason_cards_batch,
        worker_env_name="FINAL_REASON_WORKERS",
        batch_env_name="FINAL_REASON_BATCH_SIZE",
        default_workers=3,
        default_batch_size=6,
        error_fields={
            "reason_cards": [],
            "reason": "",
        },
        error_key="reason_generation_error",
        task_label="배정 이유",
    )


# 최종 팀 목록에 대해 LLM으로 배정 이유 카드를 생성한다.
# LLM이 준 reason_cards/reason을 별도 보정 없이 팀별로 병합한다.
def generate_final_reason_cards(final_teams, analyzed_students):
    try:
        reason_context = build_reason_card_context(final_teams, analyzed_students)
        llm = get_llm()
        structured_llm = llm.with_structured_output(FinalReasonCardsResult)
        chain = get_final_reason_cards_prompt_chain() | structured_llm
        response = chain.invoke({
            "reason_context": dumps_llm_context(reason_context),
        })
        result = response.model_dump() if hasattr(response, "model_dump") else response
    except Exception as error:
        print(f"최종 팀 배정 이유 생성 실패: {type(error).__name__}: {error}")
        return get_candidate_teams(final_teams)

    cards_by_team = {
        team.get("team_name"): team
        for team in result.get("teams", [])
        if isinstance(team, dict)
    }
    fixed_teams = []
    for team in get_candidate_teams(final_teams):
        fixed_team = dict(team)
        generated = cards_by_team.get(fixed_team.get("team_name"), {})
        fixed_team["reason_cards"] = generated.get("reason_cards") or []
        fixed_team["reason"] = generated.get("reason", "")
        fixed_teams.append(fixed_team)

    return fixed_teams











#최종 설명노드
#검증된 매칭된 팀 final_result출력
# LangGraph 마지막 단계에서 최종 매칭 결과를 확정한다.
# state를 입력받아 final_teams, 검증 결과, 조정 기록, 선호 거절 사유를 final_result에 담는다.
def finalize_node(state: MatchingState) -> Dict[str, Any]:
    balance_result = state.get("balance_result", {})
    iteration_count = state.get("iteration_count", 0)
    analyzed_students = state.get("analyzed_students", [])

    if balance_result.get("is_balanced") and not balance_result.get("need_adjustment"):
        finalized_by = "validation_passed"
    elif iteration_count >= MAX_ITERATION:
        finalized_by = "max_iteration"
    else:
        finalized_by = "manual_finalize"

    candidate_teams = state.get("llm_result") or state.get("teams", [])
    has_assignment_error = has_structural_assignment_error(balance_result)

    if finalized_by == "max_iteration" and has_assignment_error:
        # LLM이 반복 수정 후에도 누락/중복을 남기면 화면에는 검증 가능한 규칙 기반 팀을 보낸다.
        candidate_teams = state.get("teams", [])
        algorithm_balance, algorithm_evaluations = validation_balance_team(
            candidate_teams,
            analyzed_students,
            base_teams=state.get("teams", []),
        )
        balance_result = {
            **algorithm_balance,
            "next_node": "finalize_node",
            "algorithm_result": algorithm_balance,
            "llm_result": balance_result.get("llm_result", {}),
            "warnings": algorithm_balance.get("warnings", [])
            + ["LLM 조정 결과에 누락/중복이 있어 규칙 기반 팀으로 확정했습니다."],
            "adjustment_request": "",
        }
        team_evaluations = algorithm_evaluations
        finalized_by = "algorithm_after_validation_failure"
    else:
        team_evaluations = state.get("team_evaluations", [])

    enriched_teams = enrich_final_teams(candidate_teams, analyzed_students)
    analyzed_teams = run_parallel_strength_weakness(enriched_teams, analyzed_students)
    final_teams = run_parallel_reason_cards(analyzed_teams, analyzed_students)
    llm_result = state.get("llm_result", {})

    final_result = {
        "final_teams": final_teams,
        "changed": llm_result.get("changed", False),
        "change_summary": llm_result.get("change_summary", ""),
        "validation_notes": llm_result.get("validation_notes", ""),
        "adjustment_history": state.get("adjustment_history", []),
        "preference_rejections": build_preference_rejections(
            get_candidate_teams(final_teams),
            analyzed_students,
        ),
        "iteration_count": iteration_count,
        "finalized_by": finalized_by,
    }

    return {
        "final_result": final_result
    }

#langgrph로 workflow형식으로 구축
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(MatchingState)
# START
# -> create_team_node
# -> llm_analyzed
# -> evaluate_balance_node
# -> should_adjust
#    -> finalize_node
#    -> adjust_team_node
# -> adjust_team_node
# -> evaluate_balance_node
# -> END
#노드 추가.
workflow.add_node('create_team_node',create_team_node) #알고리즘으로 팀 생성
workflow.add_node('llm_analyzed',llm_analyzed) #llm으로 생성
workflow.add_node('evaluate_balance_node',evaluate_balance_node) #팀 검증 노드
workflow.add_node('adjust_team_node',adjust_team_node) #팀 수정 노드
workflow.add_node("finalize_node", finalize_node) #최종결과 노드

workflow.add_edge(START,'create_team_node')
workflow.add_edge('create_team_node','llm_analyzed')
workflow.add_edge('llm_analyzed','evaluate_balance_node')

workflow.add_conditional_edges(
    'evaluate_balance_node',
    should_adjust,#조건 분기 함수 하나라도 수정필요면 adjust_team_node로 보내서 수정시킴.
    {
        'finalize_node' : 'finalize_node',
        'adjust_team_node' : 'adjust_team_node',
    }
)   

workflow.add_edge('adjust_team_node','evaluate_balance_node')
workflow.add_edge('finalize_node',END)

app = workflow.compile()


# 워크플로우 전체 결과를 MySQL과 matching_output.json에 저장한다.
# result dict에서 공개 결과만 추려 DB 저장 후 로컬 파일에도 JSON으로 기록한다.
def build_public_workflow_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "analyzed_students": result.get("analyzed_students", []),
        "final_result": result.get("final_result", {}),
    }


def save_workflow_result(result):
    public_result = build_public_workflow_result(result)
    save_matching_result(public_result)
    output_path = Path(__file__).resolve().parents[1] / "data/student_analysis_data/matching_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(public_result, f, ensure_ascii=False, indent=0)


# force_rematch 설정을 보고 기존 매칭 결과 캐시를 읽을지 결정한다.
# 재매칭이 강제되지 않으면 MySQL의 최신 매칭 결과를 반환한다.
def load_cached_matching_result(force_rematch=False):
    # 이미 MySQL에 매칭 결과가 있으면 기존 결과를 재사용한다.
    # FORCE_REMATCH=true 환경변수를 주면 기존 결과가 있어도 새로 매칭한다.
    if force_rematch:
        return None
    if os.getenv("FORCE_REMATCH", "false").lower() == "true":
        return None

    return fetch_matching_result()

#프롬포트 기반 재생성
# 재생성 요청의 멤버 값에서 학생 이름 문자열만 뽑는다.
# member가 dict면 여러 이름 key를 순서대로 확인하고, 아니면 문자열로 변환한다.
def normalize_current_team_member(member: Any) -> str:
    if isinstance(member, dict):
        return (
            member.get("name")
            or member.get("studentName")
            or member.get("userName")
            or member.get("user_id")
            or member.get("userId")
            or ""
        )
    return str(member) if member is not None else ""


# 재생성 요청으로 들어온 팀 하나를 내부 표준 팀 구조로 변환한다.
# 팀 이름, 멤버 이름, 총점, 역할 분포, 팀장만 유지하고 이전 이유 캐시는 버린다.
def normalize_current_team(team: Dict[str, Any], index: int) -> Dict[str, Any]:
    members = [
        member_name
        for member_name in (
            normalize_current_team_member(member)
            for member in team.get("members", [])
        )
        if member_name
    ]

    return {
        "team_name": team.get("team_name") or team.get("teamName") or team.get("name") or f"팀 {index + 1}",
        "members": members,
        "total_score": float(team.get("total_score") or team.get("totalScore") or 0),
        "role_groups": normalize_role_groups_for_regenerate(team.get("role_groups") or team.get("roleCounts")),
        "leader": team.get("leader") or team.get("leaderName") or "",
        "reason": "",
        "reason_cards": [],
    }


# 재생성 요청의 role_groups 값을 표준 리스트 형태로 맞춘다.
# list/dict 입력을 모두 [{"role_group": ..., "count": ...}] 구조로 변환한다.
def normalize_role_groups_for_regenerate(role_groups: Any) -> List[Dict[str, Any]]:
    if isinstance(role_groups, list):
        return [
            {
                "role_group": item.get("role_group") or item.get("roleGroup"),
                "count": int(item.get("count", 0) or 0),
            }
            for item in role_groups
            if isinstance(item, dict) and (item.get("role_group") or item.get("roleGroup"))
        ]
    if isinstance(role_groups, dict):
        return [
            {"role_group": role_group, "count": int(count or 0)}
            for role_group, count in role_groups.items()
            if role_group
        ]
    return []


# 재생성 요청의 current_teams 전체를 표준 팀 리스트로 변환한다.
# dict 래퍼나 잘못된 타입을 처리하고 유효한 팀 dict만 반환한다.
def normalize_current_teams(current_teams: Any) -> List[Dict[str, Any]]:
    if not current_teams:
        return []
    if isinstance(current_teams, dict):
        current_teams = current_teams.get("teams") or current_teams.get("final_teams") or []
    if not isinstance(current_teams, list):
        return []
    return [
        normalize_current_team(team, index)
        for index, team in enumerate(current_teams)
        if isinstance(team, dict)
    ]


# 사용자 재생성 프롬프트를 반영할 초기 LangGraph state를 만든다.
# 분석 학생, prompt, 현재 팀을 입력받아 adjust_team_node부터 시작 가능한 상태를 반환한다.
def build_regenerate_state(
    analyzed_students: List[Dict[str, Any]],
    prompt: str,
    current_teams: Optional[List[Dict[str, Any]]] = None,
) -> MatchingState:
    algorithm_teams = create_initial_teams(analyzed_students)
    current_candidate = normalize_current_teams(current_teams)
    current_candidate_source = "request_current_teams" if current_candidate else ""
    if not current_candidate:
        cached_result = load_cached_matching_result(force_rematch=False) or {}
        current_candidate = normalize_current_teams(
            get_candidate_teams((cached_result.get("final_result") or cached_result))
        )
        current_candidate_source = "cached_matching_result" if current_candidate else ""
    if not current_candidate:
        current_candidate = algorithm_teams
        current_candidate_source = "algorithm_initial_teams"

    algorithm_result, team_evaluations = validation_balance_team(
        candidate_result=current_candidate,
        analyzed_students=analyzed_students,
        base_teams=algorithm_teams,
    )
    balance_result = {
        "is_balanced": False,
        "need_adjustment": True,
        "next_node": "adjust_team_node",
        "algorithm_result": algorithm_result,
        "llm_result": {
            "is_balanced": False,
            "need_adjustment": True,
            "overall_reason": "사용자 프롬프트 기반 재생성이 요청되었습니다.",
            "adjustment_request": prompt,
            "team_evaluations": [],
        },
        "errors": algorithm_result.get("errors", []),
        "warnings": algorithm_result.get("warnings", []),
        "adjustment_request": prompt,
    }

    return {
        "analyzed_students": analyzed_students,
        "teams": algorithm_teams,
        "balance_result": balance_result,
        "team_evaluations": team_evaluations,
        "adjustment_history": [],
        "final_result": {},
        "llm_result": {
            "final_teams": current_candidate,
            "changed": False,
            "change_summary": "사용자 프롬프트 재생성 전 현재 팀 구성입니다.",
            "validation_notes": (
                "사용자 프롬프트를 반영해 조정합니다. "
                f"기준 팀 출처: {current_candidate_source}."
            ),
        },
        "iteration_count": 0,
    }


# 프롬프트 기반 팀 재생성 워크플로우를 실행한다.
# prompt와 현재 팀/분석 학생을 받아 조정-검증 루프 후 최종 상태를 저장하고 반환한다.
def run_regenerate_workflow(
    prompt: str,
    current_teams: Optional[List[Dict[str, Any]]] = None,
    analyzed_students: Optional[List[Dict[str, Any]]] = None,
):
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("재생성 프롬프트가 비어 있습니다.")

    state = build_regenerate_state(
        analyzed_students=analyzed_students or load_analysis_output_json(),
        prompt=prompt,
        current_teams=current_teams,
    )

    while state.get("iteration_count", 0) < MAX_ITERATION:
        state = {
            **state,
            **adjust_team_node(state),
        }
        evaluation_update = evaluate_balance_node(state)
        state = {
            **state,
            **evaluation_update,
        }
        balance_result = state.get("balance_result", {})
        if balance_result.get("is_balanced") and not balance_result.get("need_adjustment"):
            break

    result = finalize_node(state)
    final_state = {
        **state,
        **result,
    }
    save_workflow_result(final_state)
    return final_state


# 일반 팀 매칭 워크플로우의 최초 state를 만든다.
# 학생 분석 결과를 로드하고 나머지 상태 필드는 빈 값으로 초기화한다.
def build_initial_state(analyzed_students: Optional[List[Dict[str, Any]]] = None) -> MatchingState:
    return {
        "analyzed_students": analyzed_students or load_analysis_output_json(), #여기서 불러온거 학생분석 analyzed state에 넣어줌
        "teams": [],
        "balance_result": {},
        "team_evaluations": [],
        "adjustment_history": [],
        "final_result": {},
        "llm_result": {},
        "iteration_count": 0,
    }


# LLM 호출 실패 시 규칙 기반 팀 초안만으로 최종 결과를 만든다.
# 기존 state와 예외를 입력받아 algorithm_fallback 상태를 반환한다.
def build_algorithm_only_result(state: MatchingState, error: Exception) -> MatchingState:
    teams = state.get("teams") or create_initial_teams(state.get("analyzed_students", []))
    final_teams = []

    for team in teams:
        members = team.get("members", [])
        member_names = [member.get("name") for member in members if member.get("name")]
        leader = choose_preference_aware_leader(members)
        role_groups = [
            {"role_group": role_group, "count": count}
            for role_group, count in team.get("role_groups", {}).items()
        ]
        final_teams.append({
            "team_name": team.get("team_name"),
            "members": member_names,
            "total_score": team.get("total_score", 0),
            "role_groups": role_groups,
            "leader": leader.get("name", ""),
            "leader_reason": build_leader_reason(leader),
            "personality_averages": team.get("personality_averages", {}),
            "development_averages": team.get("development_averages", {}),
            "trait_risks": team.get("trait_risks", []),
            "preference_notes": team.get("preference_notes", []),
            "reason": "",
        })

    return {
        **state,
        "teams": teams,
        "final_result": {
            "final_teams": final_teams,
            "adjustment_history": [],
            "preference_rejections": build_preference_rejections(
                final_teams,
                state.get("analyzed_students", []),
            ),
            "iteration_count": state.get("iteration_count", 0),
            "finalized_by": "algorithm_fallback",
        },
        "llm_result": {
            "final_teams": final_teams,
            "changed": False,
            "change_summary": "LLM 호출 실패로 규칙 기반 초안을 사용했습니다.",
            "validation_notes": f"LLM fallback used: {type(error).__name__}",
        },
    }


# 일반 팀 매칭 워크플로우의 공개 진입점이다.
# 캐시 사용 여부를 확인하고, LangGraph 실행 실패 시 규칙 기반 fallback 결과를 저장/반환한다.
def run_workflow(force_rematch=False, analyzed_students: Optional[List[Dict[str, Any]]] = None):
    cached_result = load_cached_matching_result(force_rematch=force_rematch)
    if cached_result is not None:
        print("기존 매칭 결과를 MySQL에서 불러오는 중")
        return cached_result

    initial_state = build_initial_state(analyzed_students=analyzed_students)
    try:
        result = app.invoke(initial_state)
    except Exception as error:
        print(f"LLM 매칭 실패. 규칙 기반 팀 배정으로 fallback합니다: {error}")
        result = build_algorithm_only_result(initial_state, error)

    save_workflow_result(result)
    return result


if __name__ == "__main__":
    result = run_workflow()
    print(json.dumps(result.get("final_result", result), ensure_ascii=False, indent=0))
