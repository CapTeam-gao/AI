"""Hackathon-oriented student team matching engine.

This module is deliberately independent from the capstone matcher.  It builds a
deterministic, skill-balanced draft first and only accepts LLM rearrangements
that preserve that development-skill balance.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


load_dotenv(override=False)

TEAM_SIZE = 5
HIGH_SCORE = 4
LOW_SCORE = 2
MAX_ITERATIONS = 3
EPSILON = 1e-9

SKILL_LEVEL_SCORE = {
    "상": 5,
    "중상": 4,
    "중": 3,
    "중하": 2,
    "하": 1,
    "높음": 5,
    "보통": 3,
    "낮음": 1,
    "HIGH": 5,
    "UPPER_MIDDLE": 4,
    "MIDDLE": 3,
    "LOWER_MIDDLE": 2,
    "LOW": 1,
}

PERSONALITY_KEYS = (
    "ideaPlanning",
    "communication",
    "roleFlexibility",
    "timePressure",
    "staminaFocus",
)
DEVELOPMENT_KEYS = (
    "implementation",
    "problemSolving",
    "completionQuality",
    "presentation",
    "leadership",
)
ALL_TRAIT_KEYS = PERSONALITY_KEYS + DEVELOPMENT_KEYS
TRAIT_LABELS = {
    "ideaPlanning": "아이디어/기획",
    "communication": "협업/소통",
    "roleFlexibility": "역할 유연성",
    "timePressure": "시간 압박 대응",
    "staminaFocus": "체력/집중 유지",
    "implementation": "개발 실행력",
    "problemSolving": "문제 해결력",
    "completionQuality": "완성도 추구",
    "presentation": "발표/설명",
    "leadership": "리더십/정리",
}


class MatchingState(TypedDict, total=False):
    analyzed_students: List[Dict[str, Any]]
    team_size: int
    teams: List[Dict[str, Any]]
    candidate_teams: List[Dict[str, Any]]
    best_teams: List[Dict[str, Any]]
    baseline_metrics: Dict[str, Any]
    balance_result: Dict[str, Any]
    best_balance_result: Dict[str, Any]
    best_score: List[float]
    iteration_count: int
    adjustment_history: List[Dict[str, Any]]
    llm_available: bool
    final_result: Dict[str, Any]


class LLMTeam(BaseModel):
    team_name: str = Field(description="기존 팀 이름")
    members: List[str] = Field(description="이 팀에 배정할 학생 이름")


class LLMMatchingResult(BaseModel):
    teams: List[LLMTeam]
    change_summary: str = ""


class ReasonCard(BaseModel):
    title: str
    description: str


class TeamStrengthWeakness(BaseModel):
    team_name: str
    strengths: str
    weaknesses: str


class StrengthWeaknessResult(BaseModel):
    teams: List[TeamStrengthWeakness]


class TeamReasonCards(BaseModel):
    team_name: str
    reason_cards: List[ReasonCard] = Field(min_length=2, max_length=4)
    reason: str


class ReasonCardsResult(BaseModel):
    teams: List[TeamReasonCards]


def _first_dict(student: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    for key in keys:
        value = student.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _raw_trait(student: Dict[str, Any], key: str) -> Any:
    personality = _first_dict(
        student,
        "hackathon_personality_scores",
        "personality_scores",
        "personalityScores",
    )
    development = _first_dict(
        student,
        "hackathon_development_scores",
        "development_scores",
        "developmentScores",
    )
    if key in personality:
        return personality[key]
    if key in development:
        return development[key]
    return student.get(key)


def validate_and_normalize_students(students: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate all ten hackathon scores and return a normalized copy.

    Missing or out-of-range values are rejected instead of silently becoming a
    neutral score, because that would make the team result look valid when the
    new survey payload was not connected correctly.
    """
    if not isinstance(students, list) or not students:
        raise ValueError("해커톤 팀 생성에는 최소 1명의 학생 데이터가 필요합니다.")

    errors: List[str] = []
    normalized: List[Dict[str, Any]] = []
    seen_names = set()
    for index, original in enumerate(students):
        if not isinstance(original, dict):
            errors.append(f"{index + 1}번째 학생 데이터가 객체가 아닙니다.")
            continue
        student = copy.deepcopy(original)
        name = str(student.get("name") or "").strip()
        label = name or f"{index + 1}번째 학생"
        if not name:
            errors.append(f"{label}: name이 없습니다.")
        elif name in seen_names:
            errors.append(f"{label}: 같은 이름이 중복되었습니다.")
        seen_names.add(name)

        personality: Dict[str, float] = {}
        development: Dict[str, float] = {}
        for key in ALL_TRAIT_KEYS:
            value = _raw_trait(student, key)
            if isinstance(value, bool):
                errors.append(f"{label}: {key}는 1~5 숫자여야 합니다.")
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                errors.append(f"{label}: {key} 점수가 누락되었습니다.")
                continue
            if not 1 <= numeric <= 5:
                errors.append(f"{label}: {key}={value!r}, 허용 범위는 숫자 1~5입니다.")
                continue
            target = personality if key in PERSONALITY_KEYS else development
            target[key] = int(numeric) if numeric.is_integer() else numeric

        student["name"] = name
        student["personality_scores"] = personality
        student["development_scores"] = development
        normalized.append(student)

    if errors:
        raise ValueError("해커톤 성향 점수 검증 실패:\n- " + "\n- ".join(errors))
    return normalized


def parse_stack_score(stack_score: Any) -> float:
    if isinstance(stack_score, dict):
        values = [float(value) for value in stack_score.values() if isinstance(value, (int, float))]
    elif isinstance(stack_score, list):
        values = [float(value) for value in stack_score if isinstance(value, (int, float))]
    else:
        values = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*점", str(stack_score or ""))]
    return mean(values) if values else 0.0


def get_technical_score(student: Dict[str, Any]) -> float:
    raw_level = student.get("skill_level") or student.get("student_level") or student.get("level")
    level_key = str(raw_level or "").strip()
    level_score = SKILL_LEVEL_SCORE.get(level_key, SKILL_LEVEL_SCORE.get(level_key.upper()))
    if level_score is None:
        level_score = 1
    return round(level_score * 10 + parse_stack_score(student.get("stack_score")), 2)


def get_role_group(role: Any) -> str:
    text = str(role or "").lower()
    groups = (
        ("game", ("unity", "unreal", "game", "게임", "유니티", "언리얼")),
        ("fullstack", ("fullstack", "full-stack", "풀스택")),
        ("devops", ("devops", "인프라", "배포", "ci/cd")),
        ("security", ("security", "보안", "owasp")),
        ("frontend", ("frontend", "front", "프론트")),
        ("backend", ("backend", "back", "server", "서버", "백엔드")),
        ("ai_data", ("ai", "머신러닝", "ml", "데이터")),
        ("app", ("app", "android", "ios", "mobile", "모바일", "앱")),
        ("design", ("design", "figma", "ui/ux", "디자인")),
    )
    for group, keywords in groups:
        if any(keyword in text for keyword in keywords):
            return group
    return "etc"


def make_student_summary(student: Dict[str, Any]) -> Dict[str, Any]:
    personality = student["personality_scores"]
    development = student["development_scores"]
    technical_score = get_technical_score(student)
    execution_score = round(mean((development["implementation"], development["problemSolving"])), 2)
    role = student.get("role") or student.get("goal") or student.get("desired_role") or ""
    return {
        "user_id": student.get("user_id") or student.get("userId") or student.get("student_id"),
        "name": student["name"],
        "skill_level": student.get("skill_level") or student.get("student_level"),
        "stack_score": student.get("stack_score", ""),
        "technical_score": technical_score,
        "execution_score": execution_score,
        "role": role,
        "role_group": get_role_group(role),
        "personality_scores": personality,
        "development_scores": development,
        "preferred_members": student.get("preferred_members") or student.get("preferredMembers") or [],
        "wants_leader": bool(student.get("wants_leader") or student.get("wantsLeader")),
    }


def build_team_capacities(total: int, team_size: int = TEAM_SIZE) -> List[int]:
    if team_size <= 0:
        raise ValueError("team_size는 1 이상이어야 합니다.")
    count = max(1, math.ceil(total / team_size))
    base, remainder = divmod(total, count)
    return [base + (1 if index < remainder else 0) for index in range(count)]


def _trait(member: Dict[str, Any], key: str) -> float:
    group = "personality_scores" if key in PERSONALITY_KEYS else "development_scores"
    return float(member[group][key])


def _average(members: List[Dict[str, Any]], field: str) -> float:
    return round(mean(float(member[field]) for member in members), 3) if members else 0.0


def _trait_average(members: List[Dict[str, Any]], key: str) -> float:
    return round(mean(_trait(member, key) for member in members), 3) if members else 0.0


def _role_counts(members: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for member in members:
        role = member["role_group"]
        counts[role] = counts.get(role, 0) + 1
    return counts


def _preferred_names(member: Dict[str, Any]) -> set[str]:
    raw = member.get("preferred_members", [])
    if isinstance(raw, str):
        raw = re.split(r"[,\n/]", raw)
    return {str(value).strip() for value in raw if str(value).strip()}


def _team_averages(teams: List[Dict[str, Any]], field: str) -> List[float]:
    return [_average(team["members"], field) for team in teams if team.get("members")]


def _spread(values: List[float]) -> float:
    return round(max(values) - min(values), 3) if values else 0.0


def _hackathon_penalty(teams: List[Dict[str, Any]], expected_names: Optional[set[str]] = None) -> float:
    penalty = 0.0
    for team in teams:
        members = team.get("members", [])
        roles = _role_counts(members)
        penalty += sum(max(0, count - 1) ** 2 for role, count in roles.items() if role != "etc") * 2
        penalty += max(0, sum(_trait(member, "timePressure") <= LOW_SCORE for member in members) - 1) * 3
        penalty += max(0, sum(_trait(member, "staminaFocus") <= LOW_SCORE for member in members) - 1) * 3
        penalty += 5 if members and not any(_trait(member, "presentation") >= HIGH_SCORE for member in members) else 0
        penalty += 5 if members and not any(_trait(member, "leadership") >= HIGH_SCORE for member in members) else 0
        member_names = {member["name"] for member in members}
        penalty -= sum(len(_preferred_names(member) & member_names) for member in members) * 0.1
    return round(penalty, 3)


def _candidate_score(teams: List[Dict[str, Any]], expected_names: set[str]) -> List[float]:
    names = [member["name"] for team in teams for member in team.get("members", [])]
    structural = len(names) - len(set(names)) + len(expected_names - set(names)) + len(set(names) - expected_names)
    return [
        float(structural),
        _spread(_team_averages(teams, "technical_score")),
        _spread(_team_averages(teams, "execution_score")),
        _hackathon_penalty(teams, expected_names),
    ]


def _placement_key(teams: List[Dict[str, Any]], team_index: int, student: Dict[str, Any], target_technical: float, target_execution: float) -> tuple:
    team = teams[team_index]
    projected = team["members"] + [student]
    role_count = _role_counts(team["members"]).get(student["role_group"], 0)
    low_pressure = sum(_trait(member, "timePressure") <= LOW_SCORE for member in projected)
    low_stamina = sum(_trait(member, "staminaFocus") <= LOW_SCORE for member in projected)
    has_presentation = any(_trait(member, "presentation") >= HIGH_SCORE for member in team["members"])
    has_leader = any(_trait(member, "leadership") >= HIGH_SCORE for member in team["members"])
    preference_hits = len(_preferred_names(student) & {member["name"] for member in team["members"]})
    return (
        abs(_average(projected, "technical_score") - target_technical),
        abs(_average(projected, "execution_score") - target_execution),
        role_count,
        max(0, low_pressure - 1) + max(0, low_stamina - 1),
        0 if _trait(student, "presentation") >= HIGH_SCORE and not has_presentation else 1,
        0 if _trait(student, "leadership") >= HIGH_SCORE and not has_leader else 1,
        -preference_hits,
        len(team["members"]),
        team_index,
    )


def _optimize_swaps(teams: List[Dict[str, Any]], expected_names: set[str], max_passes: int = 12) -> List[Dict[str, Any]]:
    optimized = copy.deepcopy(teams)
    current = _candidate_score(optimized, expected_names)
    for _ in range(max_passes):
        best_score = current
        best_swap = None
        for left in range(len(optimized)):
            for right in range(left + 1, len(optimized)):
                for left_member in range(len(optimized[left]["members"])):
                    for right_member in range(len(optimized[right]["members"])):
                        candidate = copy.deepcopy(optimized)
                        candidate[left]["members"][left_member], candidate[right]["members"][right_member] = (
                            candidate[right]["members"][right_member], candidate[left]["members"][left_member]
                        )
                        score = _candidate_score(candidate, expected_names)
                        if score < best_score:
                            best_score = score
                            best_swap = (left, right, left_member, right_member)
        if best_swap is None:
            break
        left, right, left_member, right_member = best_swap
        optimized[left]["members"][left_member], optimized[right]["members"][right_member] = (
            optimized[right]["members"][right_member], optimized[left]["members"][left_member]
        )
        current = best_score
    return optimized


def create_initial_teams(students: List[Dict[str, Any]], team_size: int = TEAM_SIZE) -> List[Dict[str, Any]]:
    summaries = [make_student_summary(student) for student in students]
    capacities = build_team_capacities(len(summaries), team_size)
    teams = [
        {"team_name": f"팀 {index + 1}", "capacity": capacity, "members": []}
        for index, capacity in enumerate(capacities)
    ]
    target_technical = mean(member["technical_score"] for member in summaries)
    target_execution = mean(member["execution_score"] for member in summaries)

    # Serpentine-like greedy placement: strongest developers are considered first,
    # while the projected team averages remain the first two selection criteria.
    ordered = sorted(
        summaries,
        key=lambda member: (
            member["technical_score"],
            member["execution_score"],
            _trait(member, "presentation"),
            _trait(member, "leadership"),
            member["name"],
        ),
        reverse=True,
    )
    for student in ordered:
        available = [index for index, team in enumerate(teams) if len(team["members"]) < team["capacity"]]
        selected = min(
            available,
            key=lambda index: _placement_key(teams, index, student, target_technical, target_execution),
        )
        teams[selected]["members"].append(student)

    return _optimize_swaps(teams, {student["name"] for student in summaries})


def _team_warnings(team: Dict[str, Any]) -> List[str]:
    members = team.get("members", [])
    warnings = []
    if members and not any(_trait(member, "presentation") >= HIGH_SCORE for member in members):
        warnings.append("presentation 4점 이상 발표 후보가 없습니다.")
    if members and not any(_trait(member, "leadership") >= HIGH_SCORE for member in members):
        warnings.append("leadership 4점 이상 리더 후보가 없습니다.")
    for key in ("timePressure", "staminaFocus"):
        low_count = sum(_trait(member, key) <= LOW_SCORE for member in members)
        if low_count >= 2:
            warnings.append(f"{TRAIT_LABELS[key]} 2점 이하 학생이 {low_count}명입니다.")
    return warnings


def validate_teams(teams: List[Dict[str, Any]], students: List[Dict[str, Any]], baseline: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    expected = {student["name"] for student in students}
    names = [member.get("name") for team in teams for member in team.get("members", [])]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    missing_names = sorted(expected - set(names))
    unknown_names = sorted(set(names) - expected)
    technical_spread = _spread(_team_averages(teams, "technical_score"))
    execution_spread = _spread(_team_averages(teams, "execution_score"))
    capacities_valid = all(len(team.get("members", [])) == team.get("capacity") for team in teams)
    structural_valid = not duplicate_names and not missing_names and not unknown_names and capacities_valid
    technical_preserved = baseline is None or technical_spread <= baseline["technical_spread"] + EPSILON
    execution_preserved = baseline is None or execution_spread <= baseline["execution_spread"] + EPSILON
    evaluations = []
    warnings = []
    for team in teams:
        team_warnings = _team_warnings(team)
        warnings.extend(f"{team['team_name']}: {warning}" for warning in team_warnings)
        evaluations.append({
            "team_name": team["team_name"],
            "technical_average": _average(team["members"], "technical_score"),
            "execution_average": _average(team["members"], "execution_score"),
            "role_groups": _role_counts(team["members"]),
            "warnings": team_warnings,
        })
    if not technical_preserved:
        warnings.append("LLM 수정안이 규칙 기반 초안보다 기술 점수 편차를 키웠습니다.")
    if not execution_preserved:
        warnings.append("LLM 수정안이 규칙 기반 초안보다 개발 실행 역량 편차를 키웠습니다.")
    score = _candidate_score(teams, expected)
    return {
        "is_valid": structural_valid and technical_preserved and execution_preserved,
        "needs_adjustment": bool(warnings) and structural_valid,
        "structural_valid": structural_valid,
        "technical_preserved": technical_preserved,
        "execution_preserved": execution_preserved,
        "technical_spread": technical_spread,
        "execution_spread": execution_spread,
        "duplicate_students": duplicate_names,
        "missing_students": missing_names,
        "unknown_students": unknown_names,
        "capacities_valid": capacities_valid,
        "warnings": warnings,
        "team_evaluations": evaluations,
        "score": score,
    }


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_HACKATHON_MATCHING_MODEL", os.getenv("OPENAI_MATCHING_MODEL", "gpt-5.4")),
        timeout=int(os.getenv("OPENAI_TIMEOUT", "120")),
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
    )


def is_llm_enabled() -> bool:
    """Return whether the optional LLM correction stage may make API calls."""
    flag = os.getenv("HACKATHON_MATCHING_LLM_ENABLED", "true").strip().lower()
    return flag not in {"0", "false", "no", "off"} and bool(os.getenv("OPENAI_API_KEY"))


def _compact_teams(teams: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "team_name": team["team_name"],
            "capacity": team["capacity"],
            "members": [
                {
                    "name": member["name"],
                    "role_group": member["role_group"],
                    "technical_score": member["technical_score"],
                    "execution_score": member["execution_score"],
                    **member["personality_scores"],
                    **member["development_scores"],
                }
                for member in team["members"]
            ],
        }
        for team in teams
    ]


def _request_llm_adjustment(teams: List[Dict[str, Any]], warnings: List[str]) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 해커톤 팀 매칭 보정자다. 학생 이름, 팀 수, 팀별 정원은 절대 바꾸지 않는다.
최우선 조건은 팀별 technical_score와 execution_score 균형이다. 이 균형을 악화시키면서 성향 조건을 맞추지 않는다.
그 범위에서 역할 중복 최소화, implementation/problemSolving 고득점자 분산, presentation/leadership 4점 이상 분산,
timePressure/staminaFocus 2점 이하 집중 방지, ideaPlanning 기획 활용, roleFlexibility 부족 역할 보완 순으로 교환한다.
경고가 개선되지 않으면 원안을 그대로 반환한다."""),
        ("human", "현재 팀:\n{teams}\n\n개선 대상 경고:\n{warnings}"),
    ])
    chain = prompt | _llm().with_structured_output(LLMMatchingResult)
    response = chain.invoke({
        "teams": json.dumps(_compact_teams(teams), ensure_ascii=False),
        "warnings": json.dumps(warnings, ensure_ascii=False),
    })
    return response.model_dump() if hasattr(response, "model_dump") else dict(response)


def _rebuild_llm_teams(raw: Dict[str, Any], base_teams: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    member_map = {member["name"]: member for team in base_teams for member in team["members"]}
    capacity_map = {team["team_name"]: team["capacity"] for team in base_teams}
    rebuilt = []
    for raw_team in raw.get("teams", []):
        team_name = raw_team.get("team_name")
        rebuilt.append({
            "team_name": team_name,
            "capacity": capacity_map.get(team_name, -1),
            "members": [member_map[name] for name in raw_team.get("members", []) if name in member_map],
        })
    return rebuilt


def create_team_node(state: MatchingState) -> Dict[str, Any]:
    teams = create_initial_teams(state["analyzed_students"], state.get("team_size", TEAM_SIZE))
    baseline = validate_teams(teams, state["analyzed_students"])
    metrics = {
        "technical_spread": baseline["technical_spread"],
        "execution_spread": baseline["execution_spread"],
    }
    return {
        "teams": teams,
        "candidate_teams": teams,
        "best_teams": teams,
        "baseline_metrics": metrics,
        "best_balance_result": baseline,
        "best_score": baseline["score"],
        "balance_result": baseline,
    }


def llm_analyzed(state: MatchingState) -> Dict[str, Any]:
    if not state.get("llm_available"):
        return {"candidate_teams": state["teams"]}
    try:
        raw = _request_llm_adjustment(state["teams"], state["balance_result"].get("warnings", []))
        return {"candidate_teams": _rebuild_llm_teams(raw, state["teams"])}
    except Exception as error:
        history = list(state.get("adjustment_history", []))
        history.append({"iteration": 0, "accepted": False, "reason": f"LLM 보정 실패: {error}"})
        return {"candidate_teams": state["teams"], "llm_available": False, "adjustment_history": history}


def evaluate_balance_node(state: MatchingState) -> Dict[str, Any]:
    candidate = state.get("candidate_teams") or state["teams"]
    result = validate_teams(candidate, state["analyzed_students"], state["baseline_metrics"])
    best_score = state.get("best_score", [math.inf] * 4)
    accepted = result["is_valid"] and result["score"] < best_score
    history = list(state.get("adjustment_history", []))
    if candidate is not state.get("teams") or state.get("iteration_count", 0):
        history.append({
            "iteration": state.get("iteration_count", 0),
            "accepted": accepted,
            "score": result["score"],
            "warnings": result["warnings"],
        })
    update: Dict[str, Any] = {"balance_result": result, "adjustment_history": history}
    if accepted:
        update.update({"best_teams": candidate, "best_balance_result": result, "best_score": result["score"]})
    return update


def adjust_team_node(state: MatchingState) -> Dict[str, Any]:
    iteration = state.get("iteration_count", 0) + 1
    if not state.get("llm_available"):
        return {"iteration_count": iteration, "candidate_teams": state["best_teams"]}
    try:
        raw = _request_llm_adjustment(state["best_teams"], state["balance_result"].get("warnings", []))
        candidate = _rebuild_llm_teams(raw, state["best_teams"])
        return {"iteration_count": iteration, "candidate_teams": candidate}
    except Exception as error:
        history = list(state.get("adjustment_history", []))
        history.append({"iteration": iteration, "accepted": False, "reason": f"LLM 재보정 실패: {error}"})
        return {
            "iteration_count": iteration,
            "candidate_teams": state["best_teams"],
            "llm_available": False,
            "adjustment_history": history,
        }


def _choose_member(members: List[Dict[str, Any]], key: str, extra: str = "technical_score") -> Dict[str, Any]:
    return max(members, key=lambda member: (_trait(member, key), member.get(extra, 0), member["name"]))


def _enrich_team(team: Dict[str, Any]) -> Dict[str, Any]:
    members = team["members"]
    leader = max(
        members,
        key=lambda member: (
            _trait(member, "leadership"),
            _trait(member, "communication"),
            member["execution_score"],
            member["name"],
        ),
    )
    presenter = _choose_member(members, "presentation")
    planner = _choose_member(members, "ideaPlanning")
    flexible = _choose_member(members, "roleFlexibility")
    personality_averages = {key: _trait_average(members, key) for key in PERSONALITY_KEYS}
    development_averages = {key: _trait_average(members, key) for key in DEVELOPMENT_KEYS}
    reasons = [
        f"기술 점수 평균 {_average(members, 'technical_score'):.2f}, 개발 실행 역량 평균 {_average(members, 'execution_score'):.2f}로 실력 균형을 맞췄습니다.",
        f"{presenter['name']} 학생은 presentation {_trait(presenter, 'presentation')}점으로 발표 후보입니다.",
        f"{planner['name']} 학생은 ideaPlanning {_trait(planner, 'ideaPlanning')}점으로 기획 후보입니다.",
        f"{flexible['name']} 학생은 roleFlexibility {_trait(flexible, 'roleFlexibility')}점으로 부족 역할 보완 후보입니다.",
    ]
    enriched = {
        "team_name": team["team_name"],
        "members": members,
        "capacity": team["capacity"],
        "leader": leader["name"],
        "presentation_candidate": presenter["name"],
        "planning_candidate": planner["name"],
        "flexible_supporter": flexible["name"],
        "role_groups": _role_counts(members),
        "technical_average": _average(members, "technical_score"),
        "execution_average": _average(members, "execution_score"),
        "personality_averages": personality_averages,
        "development_averages": development_averages,
        "assignment_reasons": reasons,
        "warnings": _team_warnings(team),
    }
    return _apply_fallback_explanation(enriched)


def _apply_fallback_explanation(team: Dict[str, Any]) -> Dict[str, Any]:
    """Add evidence-based service text even when the explanation LLM is unavailable."""
    members = team["members"]
    strongest_implementation = _choose_member(members, "implementation")
    strongest_problem_solver = _choose_member(members, "problemSolving")
    presenter_name = team["presentation_candidate"]
    planner_name = team["planning_candidate"]
    role_names = [role for role, count in team["role_groups"].items() if count]
    role_text = ", ".join(role_names) if role_names else "기타 역할"
    strengths = (
        f"{strongest_implementation['name']}의 개발 실행력과 {strongest_problem_solver['name']}의 문제 해결력을 중심으로 "
        f"해커톤 구현을 진행할 수 있습니다. {presenter_name}이 발표를 맡고 {planner_name}이 아이디어를 정리해 "
        "구현부터 발표까지 역할을 연결할 수 있습니다."
    )
    if team["warnings"]:
        weaknesses = " ".join(team["warnings"]) + " 역할을 초기에 명확히 정하고 중간 점검으로 보완해야 합니다."
    else:
        lowest_key = min(
            ALL_TRAIT_KEYS,
            key=lambda key: (
                team["personality_averages"].get(key)
                if key in PERSONALITY_KEYS
                else team["development_averages"].get(key)
            ),
        )
        weaknesses = (
            f"상대적으로 {TRAIT_LABELS[lowest_key]} 평균이 낮을 수 있으므로 체크포인트를 짧게 나누고 "
            "담당자 간 진행 상황을 자주 공유하는 방식으로 보완해야 합니다."
        )
    cards = [
        {
            "title": "개발 실력 균형",
            "description": (
                f"팀 기술 점수 평균은 {team['technical_average']:.2f}, 개발 실행 역량 평균은 "
                f"{team['execution_average']:.2f}로 구성했습니다."
            ),
        },
        {
            "title": "핵심 구현과 문제 해결",
            "description": (
                f"{strongest_implementation['name']}의 implementation {_trait(strongest_implementation, 'implementation')}점과 "
                f"{strongest_problem_solver['name']}의 problemSolving {_trait(strongest_problem_solver, 'problemSolving')}점을 "
                "핵심 개발 역량으로 활용합니다."
            ),
        },
        {
            "title": "기획·발표 역할 연결",
            "description": f"{planner_name}이 아이디어를 정리하고 {presenter_name}이 결과 발표를 담당하도록 배치했습니다.",
        },
        {
            "title": "역할 분배",
            "description": f"{role_text} 역할을 바탕으로 구현 업무를 나누고 유연 역할 후보가 빈 자리를 보완합니다.",
        },
    ]
    team["strengths"] = strengths
    team["weaknesses"] = weaknesses
    team["reason_cards"] = cards
    team["reason"] = " ".join(card["description"] for card in cards)
    return team


def _explanation_context(teams: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "team_name": team["team_name"],
            "members": [
                {
                    "name": member["name"],
                    "role_group": member["role_group"],
                    "technical_score": member["technical_score"],
                    "execution_score": member["execution_score"],
                    "personality_scores": member["personality_scores"],
                    "development_scores": member["development_scores"],
                }
                for member in team["members"]
            ],
            "leader": team["leader"],
            "presentation_candidate": team["presentation_candidate"],
            "planning_candidate": team["planning_candidate"],
            "flexible_supporter": team["flexible_supporter"],
            "role_groups": team["role_groups"],
            "technical_average": team["technical_average"],
            "execution_average": team["execution_average"],
            "warnings": team["warnings"],
        }
        for team in teams
    ]


def _chunk_team_batches(teams: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [teams[index:index + batch_size] for index in range(0, len(teams), batch_size)]


def _run_parallel_team_batches(
    teams: List[Dict[str, Any]],
    worker_fn,
    worker_env_name: str,
    batch_env_name: str,
    task_label: str,
) -> List[Dict[str, Any]]:
    """Use the same ordered batch/future merge pattern as the capstone workflow."""
    if not teams:
        return []
    max_workers = max(1, int(os.getenv(worker_env_name, "3")))
    batch_size = max(1, int(os.getenv(batch_env_name, "6")))
    batches = _chunk_team_batches(teams, batch_size)
    results: List[Optional[List[Dict[str, Any]]]] = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker_fn, batch): index
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
                        f"{task_label}_generation_error": f"{type(error).__name__}: {error}",
                    }
                    for team in batch
                ]
    merged: List[Dict[str, Any]] = []
    for batch_result in results:
        if batch_result:
            merged.extend(batch_result)
    return merged


def _strength_weakness_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", """당신은 최종 확정된 해커톤 팀의 강점과 약점을 관리자에게 설명한다.
팀원, 팀 이름, 팀장과 역할 후보를 절대 변경하지 않는다. 입력에 있는 점수와 경고만 근거로 사용한다.
strengths는 implementation·problemSolving·ideaPlanning·presentation의 실제 조합이 해커톤에서 어떻게 이어지는지 2문장으로 작성한다.
weaknesses는 timePressure·staminaFocus 저점, 역할 중복, 발표·리더 후보 부족처럼 입력에서 확인되는 리스크와 실행 가능한 보완책을 2문장으로 작성한다.
없는 경험이나 성격을 추측하지 않고 숫자 나열보다 팀원 이름과 역할 연결을 자연스러운 존댓말로 설명한다."""),
        ("human", "확정된 해커톤 팀 근거:\n{context}\n\n팀 구성은 바꾸지 말고 strengths와 weaknesses만 작성해 주세요."),
    ])


def _reason_cards_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", """당신은 최종 확정된 해커톤 팀의 배정 이유 카드를 관리자 화면용으로 작성한다.
팀원, 팀 수, 팀 이름, 팀장과 역할 후보를 절대 변경하지 않는다. 입력에 있는 점수와 역할만 근거로 사용한다.
reason_cards는 서로 다른 근거로 2~4개 작성하며 개발 실력 균형 카드를 반드시 포함한다.
나머지 카드는 역할 분배, implementation·problemSolving 고득점자 분산, 발표·리더 후보, 기획·유연 역할 활용 중 실제 근거가 강한 항목을 고른다.
각 description은 구체적인 팀원 이름을 사용하고 모든 문장을 자연스러운 존댓말로 작성한다. 알고리즘, 규칙 기반, fallback 같은 내부 표현은 쓰지 않는다.
reason은 reason_cards의 description을 순서대로 자연스럽게 이어 붙인다."""),
        ("human", "확정된 해커톤 팀 근거:\n{context}\n\n팀 구성은 바꾸지 말고 reason_cards와 reason만 작성해 주세요."),
    ])


def _parallel_strength_weakness_batch(teams: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        chain = _strength_weakness_prompt() | _llm().with_structured_output(StrengthWeaknessResult)
        response = chain.invoke({"context": json.dumps(_explanation_context(teams), ensure_ascii=False)})
        result = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    except Exception as error:
        batch_names = ", ".join(team.get("team_name", "") for team in teams)
        print(f"{batch_names} 강점/약점 생성 실패: {type(error).__name__}: {error}")
        return [
            {**team, "analysis_generation_error": f"{type(error).__name__}: {error}"}
            for team in teams
        ]
    generated_by_name = {
        item.get("team_name"): item for item in result.get("teams", []) if isinstance(item, dict)
    }
    fixed = []
    for team in teams:
        enriched = dict(team)
        generated = generated_by_name.get(team["team_name"], {})
        enriched["strengths"] = str(generated.get("strengths") or team["strengths"]).strip()
        enriched["weaknesses"] = str(generated.get("weaknesses") or team["weaknesses"]).strip()
        fixed.append(enriched)
    return fixed


def _parallel_reason_cards_batch(teams: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        chain = _reason_cards_prompt() | _llm().with_structured_output(ReasonCardsResult)
        response = chain.invoke({"context": json.dumps(_explanation_context(teams), ensure_ascii=False)})
        result = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    except Exception as error:
        batch_names = ", ".join(team.get("team_name", "") for team in teams)
        print(f"{batch_names} 배정 이유 생성 실패: {type(error).__name__}: {error}")
        return [
            {**team, "reason_generation_error": f"{type(error).__name__}: {error}"}
            for team in teams
        ]
    generated_by_name = {
        item.get("team_name"): item for item in result.get("teams", []) if isinstance(item, dict)
    }
    fixed = []
    for team in teams:
        enriched = dict(team)
        generated = generated_by_name.get(team["team_name"], {})
        cards = generated.get("reason_cards") or []
        if 2 <= len(cards) <= 4:
            enriched["reason_cards"] = cards
            enriched["reason"] = str(
                generated.get("reason")
                or " ".join(card.get("description", "") for card in cards)
            ).strip()
        fixed.append(enriched)
    return fixed


def run_parallel_strength_weakness(teams: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _run_parallel_team_batches(
        teams,
        _parallel_strength_weakness_batch,
        "FINAL_ANALYSIS_WORKERS",
        "FINAL_ANALYSIS_BATCH_SIZE",
        "강점/약점",
    )


def run_parallel_reason_cards(teams: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _run_parallel_team_batches(
        teams,
        _parallel_reason_cards_batch,
        "FINAL_REASON_WORKERS",
        "FINAL_REASON_BATCH_SIZE",
        "배정 이유",
    )


def generate_service_explanations(teams: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run capstone-style parallel final analysis with hackathon-specific prompts."""
    if not is_llm_enabled():
        return teams
    analyzed_teams = run_parallel_strength_weakness(teams)
    return run_parallel_reason_cards(analyzed_teams)


def finalize_node(state: MatchingState) -> Dict[str, Any]:
    best_teams = state["best_teams"]
    final_teams = [_enrich_team(team) for team in best_teams]
    final_teams = generate_service_explanations(final_teams)
    balance = validate_teams(best_teams, state["analyzed_students"], state["baseline_metrics"])
    if not state.get("llm_available"):
        balance["warnings"] = balance["warnings"] + ["LLM 보정이 비활성화되어 규칙 기반 결과로 확정했습니다."]
    return {
        "final_result": {
            "final_teams": final_teams,
            "balance_result": balance,
            "adjustment_history": state.get("adjustment_history", []),
            "iteration_count": state.get("iteration_count", 0),
            "finalized_by": "best_validated_candidate",
        }
    }


def should_adjust(state: MatchingState) -> str:
    if not state.get("llm_available"):
        return "finalize"
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "finalize"
    result = state.get("balance_result", {})
    if not result.get("needs_adjustment"):
        return "finalize"
    return "adjust"


workflow = StateGraph(MatchingState)
workflow.add_node("create_team_node", create_team_node)
workflow.add_node("llm_analyzed", llm_analyzed)
workflow.add_node("evaluate_balance_node", evaluate_balance_node)
workflow.add_node("adjust_team_node", adjust_team_node)
workflow.add_node("finalize_node", finalize_node)
workflow.add_edge(START, "create_team_node")
workflow.add_edge("create_team_node", "llm_analyzed")
workflow.add_edge("llm_analyzed", "evaluate_balance_node")
workflow.add_conditional_edges(
    "evaluate_balance_node",
    should_adjust,
    {"adjust": "adjust_team_node", "finalize": "finalize_node"},
)
workflow.add_edge("adjust_team_node", "evaluate_balance_node")
workflow.add_edge("finalize_node", END)
app = workflow.compile()


def run_workflow(analyzed_students: List[Dict[str, Any]], team_size: int = TEAM_SIZE) -> Dict[str, Any]:
    """Create hackathon teams without writing to the capstone DB or output file."""
    normalized = validate_and_normalize_students(analyzed_students)
    initial_state: MatchingState = {
        "analyzed_students": normalized,
        "team_size": team_size,
        "iteration_count": 0,
        "adjustment_history": [],
        "llm_available": is_llm_enabled(),
    }
    result = app.invoke(initial_state)
    return {
        "analyzed_students": normalized,
        "final_result": result["final_result"],
    }


def _member_name(member: Any) -> str:
    if isinstance(member, dict):
        return str(member.get("name") or "").strip()
    return str(member or "").strip()


def normalize_current_teams(
    current_teams: Any,
    analyzed_students: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rebuild API/DB team payloads with trusted analyzed student summaries."""
    if isinstance(current_teams, dict):
        current_teams = current_teams.get("teams") or current_teams.get("final_teams") or []
    if not isinstance(current_teams, list) or not current_teams:
        raise ValueError("재생성할 현재 해커톤 팀이 없습니다.")

    summaries = {student["name"]: make_student_summary(student) for student in analyzed_students}
    normalized_teams = []
    for index, team in enumerate(current_teams):
        if not isinstance(team, dict):
            raise ValueError(f"{index + 1}번째 현재 팀 형식이 올바르지 않습니다.")
        names = [_member_name(member) for member in team.get("members", [])]
        names = [name for name in names if name]
        normalized_teams.append({
            "team_name": team.get("team_name") or team.get("teamName") or f"팀 {index + 1}",
            "capacity": len(names),
            "members": [summaries[name] for name in names if name in summaries],
        })
    validation = validate_teams(normalized_teams, analyzed_students)
    if not validation["structural_valid"]:
        raise ValueError(
            "현재 팀 검증 실패: "
            f"누락={validation['missing_students']}, 중복={validation['duplicate_students']}, "
            f"알 수 없는 학생={validation['unknown_students']}"
        )
    return normalized_teams


def _request_user_regeneration(
    teams: List[Dict[str, Any]],
    prompt_text: str,
) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 확정된 해커톤 팀의 재생성 담당자다. 사용자의 요청을 가능한 범위에서 반영하되,
학생 추가·누락·중복, 팀 수 변경, 팀별 인원 변경은 금지한다. 팀별 technical_score와 execution_score 편차를
현재 결과보다 악화시키지 않는다. presentation/leadership 후보 분산, 역할 균형, 저압박·저집중 학생 분산도
가능한 한 유지한다. 요청이 이 제약과 충돌하면 팀을 바꾸지 않는다."""),
        ("human", "현재 팀:\n{teams}\n\n사용자 재생성 요청:\n{request}"),
    ])
    chain = prompt | _llm().with_structured_output(LLMMatchingResult)
    response = chain.invoke({
        "teams": json.dumps(_compact_teams(teams), ensure_ascii=False),
        "request": prompt_text,
    })
    return response.model_dump() if hasattr(response, "model_dump") else dict(response)


def run_regenerate_workflow(
    prompt: str,
    current_teams: Any,
    analyzed_students: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Regenerate saved/current hackathon teams while preserving skill balance."""
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("재생성 프롬프트가 비어 있습니다.")
    if not is_llm_enabled():
        raise ValueError("해커톤 팀 재생성에는 활성화된 OpenAI API 설정이 필요합니다.")

    normalized_students = validate_and_normalize_students(analyzed_students)
    base_teams = normalize_current_teams(current_teams, normalized_students)
    baseline = validate_teams(base_teams, normalized_students)
    baseline_metrics = {
        "technical_spread": baseline["technical_spread"],
        "execution_spread": baseline["execution_spread"],
    }
    try:
        raw = _request_user_regeneration(base_teams, prompt)
    except Exception as error:
        raise RuntimeError(f"해커톤 팀 재생성 LLM 호출 실패: {type(error).__name__}: {error}") from error

    candidate = _rebuild_llm_teams(raw, base_teams)
    candidate_validation = validate_teams(candidate, normalized_students, baseline_metrics)
    accepted = candidate_validation["is_valid"]
    selected = candidate if accepted else base_teams
    final_validation = candidate_validation if accepted else baseline
    final_teams = generate_service_explanations([_enrich_team(team) for team in selected])
    change_summary = str(raw.get("change_summary") or "").strip() if accepted else ""
    if not accepted:
        change_summary = "재생성안이 학생 배정 또는 개발 실력 균형 검증을 통과하지 못해 기존 팀을 유지했습니다."
    return {
        "analyzed_students": normalized_students,
        "final_result": {
            "final_teams": final_teams,
            "balance_result": final_validation,
            "changed": accepted and [
                [member["name"] for member in team["members"]] for team in candidate
            ] != [
                [member["name"] for member in team["members"]] for team in base_teams
            ],
            "change_summary": change_summary,
            "regeneration_prompt": prompt,
            "iteration_count": 1,
            "finalized_by": "validated_regeneration" if accepted else "regeneration_rejected",
        },
    }


__all__ = [
    "app",
    "create_initial_teams",
    "get_technical_score",
    "run_workflow",
    "run_regenerate_workflow",
    "validate_and_normalize_students",
    "validate_teams",
]
