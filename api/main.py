#총인원, 팀이름, 직군별 사람수, 팀장, 학생당 스택점수 제일 높은거 2개, 팀 배정 이유,팀마다 강점약점, 학생마다 skill_level : 상중하
#팀 재생성 프롬포트 넣어서 팀 재생성 누르면 가능하도록 최종 팀에서 재생성 프롬포트넣어서 llm이 수정하도록 하기.
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from fastapi import Body, FastAPI, HTTPException

from capteam_db import fetch_matching_result
from capteam_traits import (
    build_leader_reason,
    build_team_trait_risks,
    calculate_trait_averages,
    choose_team_leader,
    ensure_trait_profile,
    get_leader_score,
)


BASE_DIR = Path(__file__).resolve().parents[1]
MATCHING_OUTPUT_PATH = BASE_DIR / "data/student_analysis_data/matching_output.json"

app = FastAPI(title="CapTeam Matching API")


def _first_present(data: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data.get(key) is not None:
            return data.get(key)
    return None


def _copy_trait_scores(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    personality = source.get("personality_scores") or source.get("personalityScores") or {}
    development = source.get("development_scores") or source.get("developmentScores") or {}

    target.update({
        "communication": _first_present(source, "communication") or personality.get("communication"),
        "responsibility": _first_present(source, "responsibility") or personality.get("responsibility"),
        "collaboration": _first_present(source, "collaboration") or personality.get("collaboration"),
        "flexibility": _first_present(source, "flexibility") or personality.get("flexibility"),
        "emotionalStability": (
            _first_present(source, "emotionalStability", "emotional_stability")
            or personality.get("emotionalStability")
            or personality.get("emotional_stability")
        ),
        "leadership": _first_present(source, "leadership") or development.get("leadership"),
        "problemSolving": (
            _first_present(source, "problemSolving", "problem_solving")
            or development.get("problemSolving")
            or development.get("problem_solving")
        ),
        "implementation": _first_present(source, "implementation") or development.get("implementation"),
        "learningAbility": (
            _first_present(source, "learningAbility", "learning_ability")
            or development.get("learningAbility")
            or development.get("learning_ability")
        ),
        "planning": _first_present(source, "planning") or development.get("planning"),
    })


def normalize_request_students(students: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if students is None:
        return None

    normalized_students = []
    for student in students:
        normalized = dict(student)
        normalized["name"] = _first_present(student, "name", "studentName")
        normalized["role"] = _first_present(student, "role", "studentRole")
        normalized["stack"] = _first_present(student, "stack", "skill", "skills") or []
        normalized["experience"] = _first_present(student, "experience", "experiences") or []
        normalized["wants_leader"] = _first_present(student, "wants_leader", "wantsLeader")
        normalized["preferred_members"] = (
            _first_present(student, "preferred_members", "preferredMembers", "preferredTeammates")
            or []
        )
        normalized["grade"] = _first_present(student, "grade")
        _copy_trait_scores(normalized, student)
        normalized_students.append(normalized)

    return normalized_students


def load_matching_output() -> Dict[str, Any]:
    matching_output = fetch_matching_result()
    if matching_output:
        return matching_output

    if MATCHING_OUTPUT_PATH.exists() and MATCHING_OUTPUT_PATH.stat().st_size > 0:
        with open(MATCHING_OUTPUT_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    raise HTTPException(status_code=404, detail="MySQL 또는 matching_output.json에 매칭 결과가 없습니다.")


def get_final_result(matching_output: Dict[str, Any]) -> Dict[str, Any]:
    return matching_output.get("final_result") or matching_output


def get_final_teams(final_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    final_teams = final_result.get("final_teams", [])

    if isinstance(final_teams, dict):
        return final_teams.get("final_teams", [])

    return final_teams


def get_algorithm_teams(final_result: Dict[str, Any], matching_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    return final_result.get("algorithm_teams") or matching_output.get("teams", [])


def get_team_evaluations(final_result: Dict[str, Any], matching_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    balance_result = final_result.get("balance_result") or matching_output.get("balance_result", {})
    llm_result = balance_result.get("llm_result", {})
    return llm_result.get("team_evaluations", [])


def parse_stack_score(stack_score: str) -> List[Dict[str, Any]]:
    matches = re.findall(
        r"([A-Za-z가-힣0-9#.+/\- ]{1,30})\s*:\s*(\d{1,2})점",
        stack_score or "",
    )

    return [
        {
            "stack": stack.strip(),
            "score": int(score),
        }
        for stack, score in matches
        if stack.strip()
    ]


def build_student_map(matching_output: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        student.get("name"): ensure_trait_profile(student)
        for student in matching_output.get("analyzed_students", [])
        if student.get("name")
    }


def build_algorithm_team_map(
    final_result: Dict[str, Any],
    matching_output: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return {
        team.get("team_name"): team
        for team in get_algorithm_teams(final_result, matching_output)
        if team.get("team_name")
    }


def build_evaluation_map(
    final_result: Dict[str, Any],
    matching_output: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return {
        evaluation.get("team_name"): evaluation
        for evaluation in get_team_evaluations(final_result, matching_output)
        if evaluation.get("team_name")
    }


def get_top_stack_scores(
    member_names: List[str],
    student_map: Dict[str, Dict[str, Any]],
    limit: int = 2,
) -> List[Dict[str, Any]]:
    scores = []

    for member_name in member_names:
        student = student_map.get(member_name, {})
        for stack_score in parse_stack_score(student.get("stack_score", "")):
            scores.append({
                "student": member_name,
                "stack": stack_score["stack"],
                "score": stack_score["score"],
            })

    return sorted(scores, key=lambda item: item["score"], reverse=True)[:limit]


def get_student_top_stack_scores(student: Dict[str, Any], limit: int = 2) -> List[Dict[str, Any]]:
    return sorted(
        parse_stack_score(student.get("stack_score", "")),
        key=lambda item: item["score"],
        reverse=True,
    )[:limit]


def get_display_role_group(role: str) -> str:
    role_text = (role or "").lower()
    if any(keyword in role_text for keyword in ["frontend", "front", "프론트"]):
        return "frontend"
    if any(keyword in role_text for keyword in ["backend", "back", "server", "서버", "백엔드"]):
        return "backend"
    if any(keyword in role_text for keyword in ["ai", "data", "데이터", "머신러닝", "ml"]):
        return "ai_data"
    if any(keyword in role_text for keyword in ["app", "android", "ios", "mobile", "앱", "모바일"]):
        return "app"
    if any(keyword in role_text for keyword in ["unity", "game", "게임"]):
        return "game"
    return "etc"


def normalize_role_counts(role_groups: Any) -> List[Dict[str, Any]]:
    if isinstance(role_groups, list):
        return [
            {
                "role_group": item.get("role_group"),
                "count": int(item.get("count", 0) or 0),
            }
            for item in role_groups
            if isinstance(item, dict) and item.get("role_group")
        ]

    if isinstance(role_groups, dict):
        return [
            {"role_group": role_group, "count": int(count or 0)}
            for role_group, count in role_groups.items()
            if role_group
        ]

    return []


def get_member_name(member: Any) -> Optional[str]:
    if isinstance(member, dict):
        return member.get("name")
    return member


def get_member_names(members: Any) -> List[str]:
    if not isinstance(members, list):
        return []

    return [
        member_name
        for member_name in (get_member_name(member) for member in members)
        if member_name
    ]


def build_role_counts(
    member_names: List[str],
    student_map: Dict[str, Dict[str, Any]],
    final_team: Optional[Dict[str, Any]] = None,
    algorithm_team: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    for source in (final_team, algorithm_team):
        role_counts = normalize_role_counts((source or {}).get("role_groups"))
        if role_counts:
            return role_counts

    counts: Dict[str, int] = {}
    for member_name in member_names:
        role_group = get_display_role_group(student_map.get(member_name, {}).get("role", ""))
        counts[role_group] = counts.get(role_group, 0) + 1

    return [
        {"role_group": role_group, "count": count}
        for role_group, count in counts.items()
    ]


def normalize_skill_level_label(skill_level: str) -> str:
    return {
        "높음": "상",
        "보통": "중",
        "낮음": "하",
        "상": "상",
        "중": "중",
        "하": "하",
    }.get(skill_level, skill_level or "")


def get_skill_level_counts(members: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"상": 0, "중": 0, "하": 0}

    for member in members:
        level = normalize_skill_level_label(member.get("skill_level"))
        if level:
            counts[level] += 1

    return counts


def build_member_summaries(
    member_names: List[str],
    algorithm_members: List[Dict[str, Any]],
    student_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    member_map = {
        member.get("name"): member
        for member in algorithm_members
        if member.get("name")
    }

    summaries = []
    for member_name in member_names:
        member = member_map.get(member_name) or student_map.get(member_name, {"name": member_name})
        student = student_map.get(member_name, {})
        enriched_member = ensure_trait_profile({**student, **member})
        summaries.append({
            "name": enriched_member.get("name"),
            "role": enriched_member.get("role"),
            "role_group": enriched_member.get("role_group") or get_display_role_group(enriched_member.get("role", "")),
            "skill_level": normalize_skill_level_label(enriched_member.get("skill_level")),
            "top_stack_scores": get_student_top_stack_scores(student),
        })

    return summaries


def build_team_summary(matching_output: Dict[str, Any] = None) -> Dict[str, Any]:
    if matching_output is None:
        matching_output = load_matching_output()

    final_result = get_final_result(matching_output)
    final_teams = get_final_teams(final_result)

    student_map = build_student_map(matching_output)
    algorithm_team_map = build_algorithm_team_map(final_result, matching_output)
    evaluation_map = build_evaluation_map(final_result, matching_output)

    teams = []
    for final_team in final_teams:
        team_name = final_team.get("team_name")
        member_names = get_member_names(final_team.get("members", []))
        algorithm_team = algorithm_team_map.get(team_name, {})
        algorithm_members = algorithm_team.get("members", [])
        evaluation = evaluation_map.get(team_name, {})
        member_summaries = build_member_summaries(member_names, algorithm_members, student_map)
        leader_member = choose_team_leader(member_summaries)

        teams.append({
            "total_people": len(member_names),
            "team_name": team_name,
            "role_counts": build_role_counts(member_names, student_map, final_team, algorithm_team),
            "leader": final_team.get("leader") or leader_member.get("name"),
            "matching_reason": evaluation.get("matching_reason") or final_team.get("reason", ""),
            "strengths": evaluation.get("strengths", ""),
            "weaknesses": evaluation.get("risks", ""),
            "skill_level_counts": get_skill_level_counts(member_summaries),
            "members": member_summaries,
        })

    return {
        "total_students": len(student_map),
        "total_teams": len(teams),
        "teams": teams,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/teams/summary")
def teams_summary():
    return build_team_summary()


@app.post("/analysis/run")
def run_analysis(students: Optional[List[Dict[str, Any]]] = Body(default=None)):
    from student_analysis.analysis_llm import get_analyze_stu

    results = get_analyze_stu(normalize_request_students(students))
    return {
        "status": "ok",
        "total_students": len(results),
    }


@app.post("/matching/run")
def run_matching(students: Optional[List[Dict[str, Any]]] = Body(default=None)):
    from student_analysis.analysis_llm import get_analyze_stu
    # from matching_student.workflow_matching_student import run_workflow #open_ai_api로 할때 이거 밑에 주석치고 이거하셈
    from matching_student.upstage_matching import run_workflow

    request_students = normalize_request_students(students)
    if request_students is not None:
        get_analyze_stu(request_students)

    result = run_workflow(force_rematch=True)
    return build_team_summary(result)
#uvicorn api.main:app --reload
