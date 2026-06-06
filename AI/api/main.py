#총인원, 팀이름, 직군별 사람수, 팀장, 학생당 스택점수 제일 높은거 2개, 팀 배정 이유,팀마다 강점약점, 학생마다 skill_level : 상중하
#팀 재생성 프롬포트 넣어서 팀 재생성 누르면 가능하도록 최종 팀에서 재생성 프롬포트넣어서 llm이 수정하도록 하기.
import json
import re
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException

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


def build_role_counts(member_names: List[str], student_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        member_names = final_team.get("members", [])
        algorithm_team = algorithm_team_map.get(team_name, {})
        algorithm_members = algorithm_team.get("members", [])
        evaluation = evaluation_map.get(team_name, {})
        member_summaries = build_member_summaries(member_names, algorithm_members, student_map)
        leader_member = choose_team_leader(member_summaries)

        teams.append({
            "total_people": len(member_names),
            "team_name": team_name,
            "role_counts": build_role_counts(member_names, student_map),
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
def run_analysis():
    from student_analysis.analysis_llm import get_analyze_stu

    results = get_analyze_stu()
    return {
        "status": "ok",
        "total_students": len(results),
    }


@app.post("/matching/run")
def run_matching():
    # from matching_student.workflow_matching_student import run_workflow #open_ai_api로 할때 이거 밑에 주석치고 이거하셈
    from matching_student.upstage_matching import run_workflow

    result = run_workflow(force_rematch=True)
    return build_team_summary(result)
#uvicorn api.main:app --reload
