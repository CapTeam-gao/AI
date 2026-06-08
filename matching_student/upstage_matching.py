#현재 팀원선호를 받아야하는데 
#보안 팀은 보안끼리 붙이기, 게임도.
#그리고 지금 로직이 매칭하면 분석하고 매칭되는 로직이라 ㅈㄴ 오래걸리는데 이거를 설문 받으면 분석해서 분석해논걸 가지고 매칭해서 시간 단축하기.
from langchain_upstage import ChatUpstage
from typing import Any,List,TypedDict,Dict
import os
import json
import math
import re
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

from capteam_db import fetch_analysis_results, fetch_matching_result, save_matching_result
from capteam_preferences import (
    breaks_preference_constraints,
    build_preference_rejections,
    choose_preference_aware_leader,
    ensure_preference_profile,
    ensure_preference_profiles,
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
    get_trait_score,
)
#역할을 다양하게 균형 잡힌 팀 1순위 , 선호팀원 2순위 
def get_llm(model = "solar-pro3"):
    return ChatUpstage(
        model=model,
        base_url=os.getenv("UPSTAGE_API_BASE", "https://api.upstage.ai/v1"),
        temperature=0.05,
    )


#state
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
# 팀 간 실력 균형을 맞추려면 "보통", "낮음" 같은 문자열보다 숫자가 다루기 편함.
SKILL_LEVEL_SCORE = {
    "높음": 3,
    "보통": 2,
    "낮음": 1,
}







#node
def load_analysis_output_json():
    results = fetch_analysis_results()
    if results:
        return ensure_preference_profiles([ensure_trait_profile(student) for student in results])

    input_path = Path(__file__).resolve().parents[1] / "data/student_analysis_data/analysis_output.json"
    if input_path.exists() and input_path.stat().st_size > 0:
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                return ensure_preference_profiles([ensure_trait_profile(student) for student in json.load(f)])
        except json.JSONDecodeError:
            pass

    matching_path = Path(__file__).resolve().parents[1] / "data/student_analysis_data/matching_output.json"
    if matching_path.exists() and matching_path.stat().st_size > 0:
        with open(matching_path, "r", encoding="utf-8") as f:
            matching_output = json.load(f)
        analyzed_students = matching_output.get("analyzed_students", [])
        if analyzed_students:
            return ensure_preference_profiles([ensure_trait_profile(student) for student in analyzed_students])

    raise RuntimeError("MySQL, analysis_output.json, matching_output.json에 학생 분석 결과가 없습니다. student_analysis.analysis_llm을 먼저 실행하세요.")

#팀생성 노드
#일단 파일불러와서 프롬포트 작성과 알고리즘으로 팀 생성 team에 저장

#정규표현식으로 학생분석데이터에서 숫자만 뽑아서 평균을 계산함
def parse_stack_score(stack_score):
    # stack_score는 "python: 7점\nfastapi: 7점" 같은 문자열 형태라서
    # 정규식으로 숫자만 뽑고 평균을 계산함.
    scores = [int(score) for score in re.findall(r"(\d{1,2})점", stack_score or "")]
    if not scores:
        return 0
    return sum(scores) / len(scores)

#여기 수정해야함
#여기 점수 계산 로직 이거 수정해야할듯 지금 구조가 너무 점수에 의존하는 구조인데 지금 그냥 평균내거나 곱해서 그냥 점수 계산하니까 성능이 좀 많이 안좋아짐.
def get_technical_score(student):
    # 학생 한 명의 기술 점수 계산.
    # skill_level을 큰 기준으로 보고, stack_score 평균을 보조 점수로 더함.
    level_score = SKILL_LEVEL_SCORE.get(student.get('skill_level'),1)
    #ex skill_level이 보통이면 skill_level_score에서 보통에 value가 2여서 2를 저장
    stack_score = parse_stack_score(student.get('stack_score',""))
    #stack_score가져와서 함수써서 숫자만 추출함 없으면 빈 문자열.
    return level_score * 10 + stack_score


def get_student_score(student):
    # 최종 매칭 점수는 기술 70%, 성향 30%로 계산한다.
    technical_score = get_technical_score(student)
    trait_score = get_trait_score(student)
    return technical_score * 0.7 + trait_score * 0.3



#학생의 희망 역할을 큰 카테고리로 바꾸어줌
#같은 분야에 사람이 한팀에 많이 들어오지 않게 하기 위해서
def get_role_group(role):
    # role은 "Frontend Developer", "백엔드 개발자", "AI 엔지니어"처럼 표현이 제각각이라
    # 큰 역할군으로 묶어서 한 팀에 같은 역할이 몰리지 않게 할 때 사용함.
    role_text = (role or "").lower()
    #근데 테스트 해보니까 role에서 ai하고 backend가 둘다 들어가 있을때가 있음(ai/backend엔지니어 이런느낌으로) 그거 방지도 해야할듯
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
    # 팀 생성에 필요한 정보만 추린 학생 요약 데이터.
    # 원본 분석 결과 전체를 teams에 넣으면 너무 길어져서 핵심 필드만 저장함.
    student = ensure_preference_profile(ensure_trait_profile(student))
    technical_score = round(get_technical_score(student), 2)
    trait_score = round(get_trait_score(student), 2)
    matching_score = round(get_student_score(student), 2)
    return {
        "name": student.get("name"),
        "skill_level": student.get("skill_level"),
        "score": matching_score,
        "technical_score": technical_score,
        "trait_score": trait_score,
        "role": student.get("role"), #학생이 원하는 역할 
        "role_group": get_role_group(student.get("role")), #팀에서 할 역할
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


def make_empty_teams(team_count):
    # 실제 학생을 넣기 전에 빈 팀 틀을 먼저 만든다.
    # total_score는 팀 실력 합계, role_groups는 역할군 분포 기록용.
    return [
        {
            "team_name": f"팀 {index + 1}",
            "members": [],
            "total_score": 0,
            "role_groups": {},
        }
        for index in range(team_count)
    ]
    #team_count만큼 팀 생성

def choose_team_for_student(teams, student, max_team_size):
    # 현재 학생을 어느 팀에 넣을지 고르는 함수.
    # 우선순위:
    # 1. total_score가 낮은 팀
    # 2. 같은 role_group이 적은 팀
    # 3. 현재 인원이 적은 팀
    role_group = student["role_group"] #팀에서 어떤 역할을 할지 role_group에 저장
    
    available_teams = [
        team for team in teams #팀에다가 teams 반복돌려서 조건에 맞으면 리스트 형태로 저장
        if len(team["members"]) < max_team_size #팀에 members수가 최대 팀원 양보다 적으면
    ]

    def trait_penalty(team):
        members = team.get("members", [])
        low_traits = get_low_trait_names(student)
        high_traits = get_high_trait_names(student)
        member_low_traits = set().union(*(get_low_trait_names(member) for member in members)) if members else set()
        member_high_traits = set().union(*(get_high_trait_names(member) for member in members)) if members else set()
        repeated_low_count = len(low_traits & member_low_traits)
        complemented_count = len((low_traits & member_high_traits) | (high_traits & member_low_traits))
        return repeated_low_count - complemented_count

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

    return min(
        available_teams,
        key=lambda team: ( #tuple이여서 위에서 아래순으로 우선순위를 따짐
            -safe_preference_bonus(team),
            team["role_groups"].get(role_group, 0),#팀에서 역할, 해당 역할 그룹 인원이 적은 팀 우선.
            len(team["members"]), #팀 인원 ,팀 인원이 적은 팀 우선.
            trait_penalty(team),
            team["total_score"], #팀 총합 스코어는 마지막 보조 기준으로 사용.
        ),
    )


def create_initial_teams(analyzed_students, team_size=4, team_count=None):
    # 분석된 학생 리스트를 받아서 1차 팀 배정안을 만든다.
    # 이 단계는 LLM 없이 규칙 기반으로 빠르게 팀을 나누는 기본 틀이다.
    if not analyzed_students:
        return []

    # team_count를 직접 안 주면 team_size 기준으로 필요한 팀 수를 계산한다.
    if team_count is None:
        #ceil 올림함수
        team_count = math.ceil(len(analyzed_students) / team_size) #위에서 정한 팀 사이즈로 총학생수와 나누어 팀수를 구함

    max_team_size = math.ceil(len(analyzed_students) / team_count) #그다음 팀수와 총 학생수를 나누어 최대 팀 인원을 구함
    teams = make_empty_teams(team_count) #make_empty함수에 팀 수를 넣어 팀수 만큼 팀생성
    student_summaries = [ #키워드 꺼낸거 리스트에 저장
        make_student_summary(student) #student 넣어서 분석데이터에서 키워드만 꺼냄.
        for student in analyzed_students #analyzed_student for문 돌려서 student에 넣음
    ]

    # 점수가 높은 학생부터 배치하면 팀별 총점 차이를 줄이기 쉽다.
    sorted_students = sorted(
        student_summaries,#정렬할 데이터
        key=lambda student: student["score"],#key = 정렬기준, student에 score를 기준으로 내림차순
        reverse=True, #내림차순
    )

    for student in sorted_students:
        team = choose_team_for_student(teams, student, max_team_size) #어떤 팀에 넣을지 정하는 함수에서 변수 넣어줌
        team["members"].append(student) #선택된 맴버 리스트에 학생추가
        team["total_score"] += student["score"] #현재 학생score를 팀 스코어에 더함
        team["role_groups"][student["role_group"]] = ( #학생역할의 value값을 팀 총팀인원 역할에 더해줌
            team["role_groups"].get(student["role_group"], 0) + 1
        )

    for team in teams: #모든 팀 순회해서
        team["total_score"] = round(team["total_score"], 2)
        leader = choose_preference_aware_leader(team["members"])
        team["leader"] = leader.get("name", "")
        team["leader_score"] = get_leader_score(leader) if leader else 0
        team["leader_reason"] = build_leader_reason(leader)
        team["personality_averages"] = calculate_trait_averages(team["members"], "personality_scores")
        team["development_averages"] = calculate_trait_averages(team["members"], "development_scores")
        team["trait_risks"] = build_team_trait_risks(team["members"])
        team["preference_notes"] = team_preference_notes(team["members"])
    #팀 점수를 소수점 둘째 자리까지 반올림.
    return teams

#llm
#LLM = 자연어 분석 기반 팀 조합 보정
#검증 코드 = LLM 실수 방지
def get_student_names(students):
    # LLM이 이름을 새로 만들거나 오타를 내는 것을 줄이기 위해 원본 이름만 따로 넘긴다.
    return [student.get("name") for student in students if student.get("name")] #리스트 컴프리헨션

#팀 구성 TeamMatchingResult에 final_teams에 list로 들어감.
class RoleGroupCount(BaseModel):
    role_group: str = Field(description="역할군 이름. 예: frontend, backend, ai_data, game, etc")
    count: int = Field(description="해당 역할군 인원 수")


class FinalTeam(BaseModel):
    team_name: str = Field(description="팀 이름")
    members: List[str] = Field(description="팀원 이름 목록")
    total_score: float = Field(description="팀원 score 합계")
    role_groups: List[RoleGroupCount] = Field(description="역할군별 인원 수")
    leader: str = Field(description="추천 팀장 이름")
    reason: str = Field(description=(
        "해당 팀원들이 함께 배정된 구체적인 이유. "
        "학생별 역할, 실력 수준, 구현 경험, 강점과 약점의 보완 관계를 근거로 작성한다. "
        "'초안 유지', '총점 변화 없음', '점수 차이가 작음', '구성이 안정적임', '균형이 유지됨' 같은 "
        "추상적인 검증 문구만으로 답변하지 않는다."
    ))

#팀 매칭할때 변경되었는거 알려주는
class TeamMatchingResult(BaseModel):
    final_teams: List[FinalTeam] = Field(description="최종 팀 매칭 결과")
    changed: bool = Field(description="initial_teams에서 팀원이 바뀌었으면 true")
    change_summary: str = Field(description="변경한 내용 요약")
    validation_notes: str = Field(description="매칭 기준 검증 메모")



def get_matching_prompt_chain():
    system_prompt = """
당신은 캡스톤 프로젝트 팀 매칭을 보정하는 전문가다.
입력으로 학생 분석 데이터와 알고리즘이 만든 1차 팀 초안을 받는다.

역할:
- 알고리즘 초안을 기본 정답으로 보고, 자연어 분석상 명확히 더 좋은 조합이 있을 때만 최소한으로 보정한다.
- 보정이 필요하지 않으면 initial_teams를 그대로 유지하고 이유만 설명한다.
- 팀별 총점, 역할 다양성, 낮음 학생의 지원 가능성을 함께 본다.
- 성격 성향과 개발 성향 점수는 팀별 평균 균형과 상호 보완 관계를 판단하는 데 사용한다.
- preferred_members는 강하게 고려하되, 점수/역할군/성향 균형을 깨면 선호를 분리할 수 있다.

매칭 기준:
- 한 학생은 정확히 한 팀에만 배정한다.
- 팀 수는 initial_teams의 팀 수와 동일하게 유지한다.
- 각 팀 인원 차이는 1명 이하를 유지한다.
- 팀 총점 차이를 크게 악화시키는 재배정은 하지 않는다.
- 낮음 학생은 가능하면 보통 또는 높음 학생과 함께 둔다.
- 같은 role_group만으로 구성된 팀은 가능하면 피한다.
- suggestion, strength, weakness를 활용해 협업 시너지를 판단한다.
- communication, responsibility, implementation, problemSolving이 낮은 학생은 해당 점수가 높은 학생과 함께 두는 것을 선호한다.
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
	- final_teams의 각 팀은 team_name, members, total_score, role_groups, leader, reason을 포함한다.
		- role_groups는 [{{"role_group": "backend", "count": 1}}] 같은 배열 형식으로 작성한다.
		- members는 학생 이름 문자열 배열로만 작성한다.
		- changed는 initial_teams에서 팀원이 바뀌었으면 true, 그대로면 false다.
        - reason에는 팀원이 함께 배정된 구체적인 이유를 작성한다.
        - reason에서 '초안 유지', '총점 변화 없음', '점수 차이가 작음', '구성이 안정적임', '균형이 유지됨' 같은 추상적 검증 문구만 쓰지 않는다.
        - reason은 학생의 역할, 구현 경험, 강점과 약점 보완 관계, 성향 보완 관계를 직접 언급해야 한다.
"""

    user_prompt = """
아래 데이터를 바탕으로 최종 팀 매칭안을 만들어라.

allowed_student_names:
{allowed_student_names}

student_analysis:
{student_analysis}

initial_teams:
{initial_teams}

요청:
1차 팀 초안을 기준으로 유지할지, 최소한으로 보정할지 판단해라.
최종 결과는 지정된 JSON 형식으로만 출력해라.
"""

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )
    #프롬포트에 딱히 문제될건 없는거 같음.

def llm_analyzed(state: MatchingState, analyzed_students=None):
    # 1차 알고리즘 팀 생성 이후 LLM에게 자연어 분석 기반 보정을 요청한다.
    if analyzed_students is None: #ananlyzed_student변수가 안들어오면 
        analyzed_students = state.get("analyzed_students", []) #state에서 불러와서 가져옴

    initial_teams = state.get("teams") or create_initial_teams(analyzed_students) #학생분석한거 state에 teams에 있으면 그거쓰고 없으면 다시 알고리즘으로 팀 짜서 가져옴
    allowed_student_names = get_student_names(analyzed_students) #이름 다르게 안나오게 하는 함수로 검증

    llm = get_llm() #llm불러옴
    structured_llm = llm.with_structured_output(TeamMatchingResult) #TeamMatchingResult 형식으로 llm이 답변 생성하도록함
    chain = get_matching_prompt_chain() | structured_llm #chatprompttemplate함수에서 반환해서 structured_llm이랑 연결

    response = chain.invoke({
        #json.dump = 데이터를 json으로 저장하는 함수
        #ensure_ascil = 한글 같은 유니코드를 ASCII 형태로 바꿀지 결정하는 옵션. indent = 보기좋게 들여쓰기 할때 사용
        "allowed_student_names": json.dumps(allowed_student_names, ensure_ascii=False),
        "student_analysis": json.dumps(analyzed_students, ensure_ascii=False, indent=2),
        "initial_teams": json.dumps(initial_teams, ensure_ascii=False, indent=2),
    })
    #삼항연산자 (A if 조건 else B)

    ai_matching = response.model_dump() if hasattr(response, "model_dump") else response
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
def get_candidate_teams(candidate_result):
    # structured output 결과는 {"final_teams": [...]} 형태이고,
    # 알고리즘 초안은 바로 팀 리스트 형태라서 둘 다 받을 수 있게 정리한다.
    if isinstance(candidate_result, dict): #candidate_result가 dict인지 확인
        return candidate_result.get("final_teams", []) #candidate_result 딕셔너리에서 "final_teams" 키의 value 값을 가져오고,없으면 [] 반환
    if isinstance(candidate_result, list):
        return candidate_result #list면 그냥 반환.
    return [] #둘다 아니면 빈 리스트



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


def build_student_lookup(analyzed_students):
    # 학생 이름으로 score, skill_level, role_group을 빠르게 찾기 위한 dict.
    return {
        student.get("name"): make_student_summary(student) #key : value로 저장
        for student in analyzed_students #학생들을 하나씩 순회.
        if student.get("name") #이름 없는 학생들 제외
    }


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

        if team_status["skill_levels"].get("낮음", 0) == team_status["member_count"]:
            team_warnings.append("낮음 학생만으로 구성된 팀입니다.")

        if len(team_status["role_groups"]) == 1 and team_status["member_count"] > 1:
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
                    team_errors.append(
                        f"총점이 맞지 않습니다. reported={reported_score}, calculated={team_status['total_score']}"
                    )
            except (TypeError, ValueError):
                team_errors.append(f"total_score가 숫자가 아닙니다: {reported_score}")

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
        errors.append(f"팀 인원 차이가 1명을 초과합니다: {member_counts}")

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
        "team_count": len(candidate_teams),
        "member_counts": member_counts,
        "score_gap": score_gap,
        "average_score": average_score,
        "soft_score_gap": round(soft_score_gap, 2),
        "hard_score_gap": round(hard_score_gap, 2),
    }

    return balance_result, team_evaluations


def evaluate_balance_node(state: MatchingState) -> Dict[str, Any]:
    # llm_result가 있으면 LLM 제안안을 검증하고, 없으면 알고리즘 teams를 검증한다.
    # 1. 알고리즘 검증을 먼저 돌린다.
    # 2. 알고리즘 검증 결과를 LLM prompt에 넣어서 LLM 검증을 돌린다.
    # 3. 둘 다 통과해야 finalize_node로 갈 수 있도록 balance_result를 합친다.
    # final_result는 여기서 저장하지 않고 finalize_node에서 따로 저장하는 구조가 좋다.
    analyzed_students = state.get("analyzed_students", [])
    base_teams = state.get("teams", [])
    candidate_result = state.get("llm_result") or base_teams

    algorithm_result, team_evaluations = validation_balance_team( #알고리즘으로 검증
        candidate_result=candidate_result,
        analyzed_students=analyzed_students,
        base_teams=base_teams,
    )
    llm_result = llm_validation_balance_team( #llm으로 검증
        candidate_result=candidate_result,
        analyzed_students=analyzed_students,
        algorithm_result=algorithm_result,
    )
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
class LLMBalanceTeamEvaluation(BaseModel):
    team_name: str = Field(description = "평가 대상 팀 이름") #팀이름
    is_balanced: bool = Field(description = "해당 팀이 역할, 실력, 협업 리스크 측면에서 균형적인지 여부") #팀 균형
    need_adjustment: bool = Field(description = "해당 팀에 팀원 조정이 필요한지 여부") #조정 필요한지 bool
    matching_reason : str = Field (description = (
        "해당 팀원들이 함께 배정된 구체적인 이유를 작성한다. "
        "반드시 학생별 역할, 실력 수준, 구현 경험, 강점과 약점의 보완 관계를 근거로 설명한다. "
        "특정 역할 인원이 적더라도 해당 역할 담당자의 실력이나 경험으로 감당 가능하다면 그 근거를 명확히 쓴다. "
        "팀당 2~3개의 핵심 이유를 자연스러운 문장으로 작성한다. "
        "'초안 유지', '총점 변화 없음', '점수 차이가 작음', '구성이 안정적임', '균형이 유지됨', '역할 다양성이 좋음' 같은 일반적인 평가 문구만으로 답변하면 안 된다. "
        "점수나 균형 상태를 요약하지 말고, 왜 이 학생들이 같은 팀이어야 하는지 설명한다."
    ))
    strengths: str = Field(description = "해당 팀의 특별한 강점을 구체적으로 설명") #팀 강점 
    risks: str = Field(description = "해당 팀의 특별한 리스크 또는 약점을 구체적으로 설명") #팀 약점
    adjustment_suggestion: str = Field(description = "조정이 필요할 경우 구체적인 수정 제안") #조정 제안

#전체적인 팀 매칭 결과에 대한 평가
class LLMBalanceResult(BaseModel):
    is_balanced: bool = Field(description = "전체 팀 매칭 결과가 균형적인지 여부") #전체적인 벨런스
    need_adjustment: bool = Field(description = "전체 팀 매칭 결과에 수정이 필요한지 여부") #수정이 필요한지 bool
    overall_reason: str = Field(description = "전체 균형 평가 판단 이유") #이렇게 생각한 이유 
    adjustment_request: str = Field(description = "수정이 필요할 경우 adjust_team_node에 전달할 구체적인 수정 요청") #수정 요청 수정이 필요하면 어떻게 수정할지
    team_evaluations: List[LLMBalanceTeamEvaluation] = Field(description = "팀별 균형 평가 결과") #각 팀 평가
#이거 strutured output으로 쓰는데 description이 안적혀있음
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
def llm_validation_balance_team(candidate_result, analyzed_students, algorithm_result):
    llm = get_llm()
    structured_llm = llm.with_structured_output(LLMBalanceResult) #스키마 넣어줘서 structured_llm 만들어줌
    chain = get_llm_balance_prompt_chain() | structured_llm

    response = chain.invoke({
        "candidate_result": json.dumps(candidate_result, ensure_ascii=False, indent=2),
        "student_analysis": json.dumps(analyzed_students, ensure_ascii=False, indent=2),
        "algorithm_result": json.dumps(algorithm_result, ensure_ascii=False, indent=2),
    })

    return response.model_dump()
#model_dump : basemodel 객에 dict으로 반환


def merge_balance_results(algorithm_result, llm_result):
    # 알고리즘 검증 + LLM 검증 결과를 하나의 balance_result로 합친다.
    # 둘 다 통과해야 최종 통과이고, 하나라도 수정 필요하면 수정 노드로 보내기 위한 형태다.
    algorithm_is_balanced = algorithm_result.get("is_balanced", False) #균형 맞는지 is_balanced 반환하고 없으면 False반환
    llm_is_balanced = llm_result.get("is_balanced", False) #위랑 마찬가지
    algorithm_need_adjustment = algorithm_result.get("need_adjustment", True) #수정 필요한지 need_adjustment반환하고 없으면 True반환
    llm_need_adjustment = llm_result.get("need_adjustment", True) #위랑 마찬가지

    is_balanced = algorithm_is_balanced and llm_is_balanced #and로 둘다 True아니면 수정
    need_adjustment = algorithm_need_adjustment or llm_need_adjustment #True면 수정해야 되기 때문에 or로 둘다 false아니면 수정

    return {
        "is_balanced": is_balanced,
        "need_adjustment": need_adjustment,
        "next_node": "finalize_node" if is_balanced and not need_adjustment else "adjust_team_node", #다음 노드 뭘로 할지
        "algorithm_result": algorithm_result,
        "llm_result": llm_result,
        "errors": algorithm_result.get("errors", []),
        "warnings": algorithm_result.get("warnings", []),
        "adjustment_request": llm_result.get("adjustment_request", ""),
    }


MAX_ITERATION = 3

def should_adjust(state: MatchingState):
    # LangGraph 조건 분기 함수.
    # 둘 다 통과하면 finalize_node, 하나라도 실패/수정 필요이면 adjust_team_node로 보낸다.
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
    - 낮음 학생은 가능하면 보통 또는 높음 학생과 함께 둔다.
    - 같은 role_group만으로 구성된 팀은 가능하면 피한다.
    - preferred_members는 강하게 고려하되, 점수/역할군/성향 균형을 깨면 선호를 분리할 수 있다.
    - 팀 안에 wants_leader=true인 학생이 있으면 그 학생들 중 leader_score와 technical_score가 높은 학생을 팀장으로 추천한다.
    - adjustment_history와 같은 수정 패턴을 반복하지 않는다.

    반영해야 할 정보:
    - balance_result.algorithm_result.errors는 반드시 해결한다.
    - balance_result.algorithm_result.warnings는 가능하면 완화한다.
    - balance_result.llm_result.adjustment_request를 우선적으로 반영한다.
    - student_analysis의 strength, weakness, suggestion을 활용해 협업 시너지를 높인다.

    계산 규칙:
    - 점수는 student_analysis 또는 algorithm_teams에 있는 score 값만 사용한다.
    - 새로운 점수나 skill_level을 만들지 않는다.
    - total_score는 최종 팀원의 score 합으로 작성한다.

    이름 사용 규칙:
    - allowed_student_names에 있는 이름만 사용한다.
    - 학생 이름은 한 글자도 바꾸지 말고 그대로 복사한다.
    - allowed_student_names에 없는 학생을 만들지 않는다.

    출력 규칙:
    - 반드시 지정된 structured output schema에 맞춰 출력한다.
    - final_teams의 각 팀은 team_name, members, total_score, role_groups, leader, reason을 포함한다.
    - role_groups는 [{{"role_group": "backend", "count": 1}}] 같은 배열 형식으로 작성한다.
    - members는 학생 이름 문자열 배열로만 작성한다.
    - changed는 current_candidate에서 팀원이 바뀌었으면 true, 그대로면 false다.
    - reason에는 팀원이 함께 배정된 구체적인 이유를 작성한다.
    - reason에서 '초안 유지', '총점 변화 없음', '점수 차이가 작음', '구성이 안정적임', '균형이 유지됨' 같은 추상적 검증 문구만 쓰지 않는다.
    - reason은 학생의 역할, 구현 경험, 강점과 약점 보완 관계를 직접 언급해야 한다.
    - change_summary에는 어떤 검증 실패를 어떻게 고쳤는지 작성한다.
    - validation_notes에는 다시 검증할 때 확인해야 할 내용을 작성한다.
    """

    user_prompt = """
    아래 검증 실패 정보를 바탕으로 팀 후보를 수정해라.

    allowed_student_names:
    {allowed_student_names}

    student_analysis:
    {student_analysis}

    algorithm_teams:
    {algorithm_teams}

    current_candidate:
    {current_candidate}

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

    llm = get_llm()
    structured_llm = llm.with_structured_output(TeamMatchingResult)
    chain = get_adjust_team_prompt_chain() | structured_llm

    response = chain.invoke({
        "allowed_student_names": json.dumps(allowed_student_names, ensure_ascii=False),
        "student_analysis": json.dumps(analyzed_students, ensure_ascii=False, indent=2),
        "algorithm_teams": json.dumps(algorithm_teams, ensure_ascii=False, indent=2),
        "current_candidate": json.dumps(current_candidate, ensure_ascii=False, indent=2),
        "balance_result": json.dumps(balance_result, ensure_ascii=False, indent=2),
        "adjustment_history": json.dumps(adjustment_history, ensure_ascii=False, indent=2),
    })

    adjusted_result = response.model_dump() if hasattr(response, "model_dump") else response
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
        "result": adjusted_result, #llm이 조정해서 만든결과
    }

    return {
        "llm_result": adjusted_result, #이번에 조정된 팀 매칭 결과
        "iteration_count": next_iteration_count, #반복수
        "adjustment_history": adjustment_history + [adjustment_record], #기존 조정기록에 요번에 추가한 리스트
    }


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


def align_reason_with_leader(team: Dict[str, Any], leader_name: str, member_names: List[str]) -> Dict[str, Any]:
    reason = remove_conflicting_leader_sentences(
        team.get("reason", ""),
        leader_name,
        member_names,
    )
    leader_prefix = f"{leader_name}을(를) 추천 팀장으로 지정했습니다."

    if reason:
        reason = f"{leader_prefix} {reason}"
    else:
        reason = leader_prefix

    return {
        **team,
        "reason": reason,
    }


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
        enriched_team = align_reason_with_leader(dict(team), leader_name, member_names)
        enriched_team["leader"] = leader.get("name", enriched_team.get("leader", ""))
        enriched_team["leader_reason"] = build_leader_reason(leader)
        enriched_team["personality_averages"] = calculate_trait_averages(members, "personality_scores")
        enriched_team["development_averages"] = calculate_trait_averages(members, "development_scores")
        enriched_team["trait_risks"] = build_team_trait_risks(members)
        enriched_team["preference_notes"] = team_preference_notes(members)
        enriched_teams.append(enriched_team)

    if isinstance(final_teams, dict):
        result = dict(final_teams)
        result["final_teams"] = enriched_teams
        return result

    return enriched_teams











#최종 설명노드
#검증된 매칭된 팀 final_result출력
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
    algorithm_errors = balance_result.get("algorithm_result", {}).get("errors", [])
    has_assignment_error = bool(
        algorithm_errors
        or balance_result.get("missing_names")
        or balance_result.get("duplicate_names")
        or balance_result.get("unknown_names")
    )

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

    final_teams = enrich_final_teams(candidate_teams, analyzed_students)

    final_result = {
        "final_teams": final_teams,
        "algorithm_teams": state.get("teams", []),
        "balance_result": balance_result,
        "team_evaluations": team_evaluations,
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


def save_workflow_result(result):
    save_matching_result(result)
    output_path = Path(__file__).resolve().parents[1] / "data/student_analysis_data/matching_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def load_cached_matching_result(force_rematch=False):
    # 이미 MySQL에 매칭 결과가 있으면 기존 결과를 재사용한다.
    # FORCE_REMATCH=true 환경변수를 주면 기존 결과가 있어도 새로 매칭한다.
    if force_rematch:
        return None
    if os.getenv("FORCE_REMATCH", "false").lower() == "true":
        return None

    return fetch_matching_result()


def build_initial_state() -> MatchingState:
    return {
        "analyzed_students": load_analysis_output_json(), #여기서 불러온거 학생분석 analyzed state에 넣어줌
        "teams": [],
        "balance_result": {},
        "team_evaluations": [],
        "adjustment_history": [],
        "final_result": {},
        "llm_result": {},
        "iteration_count": 0,
    }


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
            "reason": "LLM 호출이 실패하여 규칙 기반 점수, 역할군, 성향 균형으로 생성한 팀입니다.",
        })

    return {
        **state,
        "teams": teams,
        "final_result": {
            "final_teams": final_teams,
            "algorithm_teams": teams,
            "balance_result": {
                "is_balanced": True,
                "need_adjustment": False,
                "next_node": "finalize_node",
                "algorithm_result": {},
                "llm_result": {
                    "team_evaluations": [],
                },
                "errors": [],
                "warnings": [f"LLM fallback used: {type(error).__name__}"],
                "adjustment_request": "",
            },
            "team_evaluations": [],
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


def run_workflow(force_rematch=False):
    cached_result = load_cached_matching_result(force_rematch=force_rematch)
    if cached_result is not None:
        print("기존 매칭 결과를 MySQL에서 불러오는 중")
        return cached_result

    initial_state = build_initial_state()
    try:
        result = app.invoke(initial_state)
    except Exception as error:
        print(f"LLM 매칭 실패. 규칙 기반 팀 배정으로 fallback합니다: {error}")
        result = build_algorithm_only_result(initial_state, error)

    save_workflow_result(result)
    return result


if __name__ == "__main__":
    result = run_workflow()
    print(json.dumps(result.get("final_result", result), ensure_ascii=False, indent=2))
#balance_result와 team_evalution결과가 좋으면 final_result에 바로 저장하고 안좋으면 수정하여 adjustment_history에 기록하여 같은결과과 나오지 않도록함
#그리고 또 별로라면 다시 수정 밸런스검사 좋을때까지 반복.
#   - adjust: 팀 상태가 별로라서 다시 고치는 단계
#   - finalize: 팀 상태가 좋아서 최종 확정하는 단계
#이런식으로 저장

# load_analysis_node

#   MySQL의 분석 결과를 읽어서 analyzed_students에 저장.

#   create_team_node

#   분석 결과를 보고 1차 팀 생성. 결과를 teams에 저장.

#   evaluate_balance_node

#   현재 teams를 평가해서:

#   - 전체 평가: balance_result
#   - 팀별 평가: team_evaluations

#   에 저장.

#   should_adjust`

#   노드라기보다는 조건 분기 함수.
#   balance_result["is_balanced"] 같은 값을 보고:

#   좋음 -> finalize_node
#   나쁨 -> adjust_team_node

#   로 보냄.

#   adjust_team_node

#   팀을 수정하고 adjustment_history에 기록.
#   수정된 팀은 다시 teams에 저장.

#   finalize_node

#   최종 팀 결과와 설명을 final_result에 저장.

#근데 이거 코드 ㅈㄴ 길어서 토큰값 레전드로 많이 나올듯 이걸 계속 검증하는거니까
#팀 매칭시 학생의 성격과 개발 성향을 고려하여 팀매칭하는로직

#현재 팀 매칭 로직은 LangGraph 기반 workflow 구조로 설계했습니다. 
#각 단계는 노드로 분리되어 있고, 팀 생성, LLM 보정, 알고리즘 검증, LLM 검증, 수정, 최종 확정 노드가 state를 주고받으며 동작합니다.
#검증 결과에 따라 최종 확정 또는 수정 노드로 분기하고, 수정된 결과는 다시 검증 단계로 돌아가는 반복 구조입니다.
