from langchain.chat_models import init_chat_model
from typing import Any,List,TypedDict,Dict
import json
import math
import re
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

def get_llm(model = 'gpt-5-nano'):
    return init_chat_model(model = model)


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

ANALYSIS_OUTPUT_PATH = "/Users/kshi3430/CapTeam/data/student_analysis_data/analysis_output.json"
MATCHING_OUTPUT_PATH = "/Users/kshi3430/CapTeam/data/student_analysis_data/matching_output.json"

# skill_level을 팀 생성용 숫자 점수로 바꾸기 위한 기준.
# 팀 간 실력 균형을 맞추려면 "보통", "낮음" 같은 문자열보다 숫자가 다루기 편함.
SKILL_LEVEL_SCORE = {
    "높음": 3,
    "보통": 2,
    "낮음": 1,
}





#node
def load_analysis_output_json():
    with open(ANALYSIS_OUTPUT_PATH,'r',encoding='utf-8') as f: #r은 읽기모드로 열겠다는 뜻이고 ,as f는 f라는 변수로 받겠다는거임. 
    #with은 작업끝나면 알아서 닫아주는 걸 해주는 거라고 생각하면 됨.
        results = json.load(f) #f가 가리키는 json으로 읽고 dict이나 list로 변환해서 results에 저장
    return results

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



def get_student_score(student):
    # 학생 한 명의 매칭용 점수 계산.
    # skill_level을 큰 기준으로 보고, stack_score 평균을 보조 점수로 더함.
    level_score = SKILL_LEVEL_SCORE.get(student.get('skill_level'),1)
    #ex skill_level이 보통이면 skill_level_score에서 보통에 value가 2여서 2를 저장
    stack_score = parse_stack_score(student.get('stack_score',""))
    #stack_score가져와서 함수써서 숫자만 추출함 없으면 빈 문자열.
    return level_score * 10 + stack_score



#학생의 희망 역할을 큰 카테고리로 바꾸어줌
#같은 분야에 사람이 한팀에 많이 들어오지 않게 하기 위해서
def get_role_group(role):
    # role은 "Frontend Developer", "백엔드 개발자", "AI 엔지니어"처럼 표현이 제각각이라
    # 큰 역할군으로 묶어서 한 팀에 같은 역할이 몰리지 않게 할 때 사용함.
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
    # 팀 생성에 필요한 정보만 추린 학생 요약 데이터.
    # 원본 분석 결과 전체를 teams에 넣으면 너무 길어져서 핵심 필드만 저장함.
    return {
        "name": student.get("name"),
        "skill_level": student.get("skill_level"),
        "score": round(get_student_score(student), 2), #함수에서 skill_level과 stack_level합친 점수를 score에 저장 round = 반올림함수
        "role": student.get("role"), #학생이 원하는 역할 
        "role_group": get_role_group(student.get("role")), #팀에서 할 역할
        "strength": student.get("strength"),
        "weakness": student.get("weakness"),
        "suggestion": student.get("suggestion"), 
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

    return min(
        available_teams,
        key=lambda team: ( #tuple이여서 위에서 아래순으로 우선순위를 따짐
            team["total_score"], #팀 총합 스코어, 팀 총점이 가장 낮은 팀 우선.
            team["role_groups"].get(role_group, 0),#팀에서 역할, 해당 역할 그룹 인원이 적은 팀 우선.
            len(team["members"]), #팀 인원 ,팀 인원이 적은 팀 우선.
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
    #팀 점수를 소수점 둘째 자리까지 반올림.
    return teams

#llm
#LLM = 자연어 분석 기반 팀 조합 보정
#검증 코드 = LLM 실수 방지
def get_student_names(students):
    # LLM이 이름을 새로 만들거나 오타를 내는 것을 줄이기 위해 원본 이름만 따로 넘긴다.
    return [student.get("name") for student in students if student.get("name")] #리스트 컴프리헨션

#팀 구성 TeamMatchingResult에 final_teams에 list로 들어감.
class FinalTeam(BaseModel):
    team_name: str = Field(description="팀 이름")
    members: List[str] = Field(description="팀원 이름 목록")
    total_score: float = Field(description="팀원 score 합계")
    role_groups: Dict[str, int] = Field(description="역할군별 인원 수")
    leader: str = Field(description="추천 팀장 이름")
    reason: str = Field(description="팀 매칭 이유")

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

매칭 기준:
- 한 학생은 정확히 한 팀에만 배정한다.
- 팀 수는 initial_teams의 팀 수와 동일하게 유지한다.
- 각 팀 인원 차이는 1명 이하를 유지한다.
- 팀 총점 차이를 크게 악화시키는 재배정은 하지 않는다.
- 낮음 학생은 가능하면 보통 또는 높음 학생과 함께 둔다.
- 같은 role_group만으로 구성된 팀은 가능하면 피한다.
- suggestion, strength, weakness를 활용해 협업 시너지를 판단한다.

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
- members는 학생 이름 문자열 배열로만 작성한다.
- changed는 initial_teams에서 팀원이 바뀌었으면 true, 그대로면 false다.
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









#밸런스 검사 노드
#llm_result 또는 teams를 가져와서 전체 평가 결과는 balance_result에,
#팀별 평가 결과는 team_evaluations에 저장한다.

def get_candidate_teams(candidate_result):
    # structured output 결과는 {"final_teams": [...]} 형태이고,
    # 알고리즘 초안은 바로 팀 리스트 형태라서 둘 다 받을 수 있게 정리한다.
    if isinstance(candidate_result, dict):
        return candidate_result.get("final_teams", [])
    if isinstance(candidate_result, list):
        return candidate_result
    return []


def get_member_names(team):
    # 알고리즘 초안의 members는 dict 리스트, LLM 결과의 members는 이름 문자열 리스트다.
    # 검증에서는 이름만 필요하므로 같은 형태로 맞춘다.
    member_names = []
    for member in team.get("members", []):
        if isinstance(member, dict):
            member_names.append(member.get("name"))
        else:
            member_names.append(member)
    return [name for name in member_names if name]


def build_student_lookup(analyzed_students):
    # 학생 이름으로 score, skill_level, role_group을 빠르게 찾기 위한 dict.
    return {
        student.get("name"): make_student_summary(student)
        for student in analyzed_students
        if student.get("name")
    }


def calculate_team_status(team, student_lookup):
    # 팀원 이름 목록을 기준으로 실제 총점과 역할 분포를 다시 계산한다.
    member_names = get_member_names(team)
    total_score = 0
    role_groups = {}
    skill_levels = {}
    unknown_names = []

    for name in member_names:
        student = student_lookup.get(name)
        if student is None:
            unknown_names.append(name)
            continue

        total_score += student["score"]
        role_group = student["role_group"]
        skill_level = student["skill_level"]
        role_groups[role_group] = role_groups.get(role_group, 0) + 1
        skill_levels[skill_level] = skill_levels.get(skill_level, 0) + 1

    return {
        "team_name": team.get("team_name"),
        "members": member_names,
        "member_count": len(member_names),
        "total_score": round(total_score, 2),
        "role_groups": role_groups,
        "skill_levels": skill_levels,
        "unknown_names": unknown_names,
    }


def validation_balance_team(candidate_result, analyzed_students, base_teams=None):
    # 알고리즘으로 팀 검증하여 수정할 필요가 있는지 없는지 판단한다.
    # 여기서는 LLM을 쓰지 않고, 이름/중복/누락/점수/인원/역할 분포를 코드로 검사한다.
    candidate_teams = get_candidate_teams(candidate_result)
    student_lookup = build_student_lookup(analyzed_students)
    allowed_names = set(student_lookup.keys())
    base_team_count = len(base_teams or [])
    errors = []
    warnings = []
    team_evaluations = []

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
    max_score_gap = max(10, average_score * 0.2) if average_score else 0

    if team_scores and score_gap > max_score_gap:
        warnings.append(
            f"팀 점수 차이가 큽니다. gap={score_gap}, recommended_max={round(max_score_gap, 2)}"
        )

    balance_result = {
        "is_balanced": not errors,
        "need_adjustment": bool(errors or warnings),
        "errors": errors,
        "warnings": warnings,
        "missing_names": missing_names,
        "duplicate_names": duplicate_names,
        "unknown_names": unknown_names,
        "team_count": len(candidate_teams),
        "member_counts": member_counts,
        "score_gap": score_gap,
        "average_score": average_score,
    }

    return balance_result, team_evaluations


def evaluate_balance_node(state: MatchingState) -> Dict[str, Any]:
    # llm_result가 있으면 LLM 제안안을 검증하고, 없으면 알고리즘 teams를 검증한다.
    # 검증 결과만 state에 저장하고 final_result는 finalize_node에서 따로 저장하는 구조가 좋다.
    analyzed_students = state.get("analyzed_students", [])
    base_teams = state.get("teams", [])
    candidate_result = state.get("llm_result") or base_teams

    balance_result, team_evaluations = validation_balance_team(
        candidate_result=candidate_result,
        analyzed_students=analyzed_students,
        base_teams=base_teams,
    )

    return {
        "balance_result": balance_result,
        "team_evaluations": team_evaluations,
    }
    # validation_balance_team에서 수정할 필요 없다고 해도 여기서 한번 챗봇으로 검증하는 로직은
    # 나중에 별도 llm_evaluate_balance_node로 분리하는 편이 안전하다.



#team으로 알고리즘으로 team상태 저장하고
#llm_result로 llm이 제안한 팀 상태 저장하고
#검증할때 실패하면 다시 알고리즘 보고 할수있도록 알고리즘은 그대로 두고 llm_result만 계속 덮어 씌어지면서 수정


#팀 수정 노드
#balance_result와 team_evalution결과가 좋으면 final_result에 바로 저장하고 안좋으면 수정하여 adjustment_history에 기록하여 같은결과과 나오지 않도록함
#그리고 또 별로라면 다시 수정 밸런스검사 좋을때까지 반복.
#   - adjust: 팀 상태가 별로라서 다시 고치는 단계
#   - finalize: 팀 상태가 좋아서 최종 확정하는 단계
#이런식으로 저장

#최종 설명노드
#검증된 매칭된 팀 final_result출력


# load_analysis_node

#   analysis_output.json을 읽어서 analyzed_students에 저장.

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
