#총인원, 팀이름, 직군별 사람수, 팀장, 학생당 스택점수 제일 높은거 2개, 팀 배정 이유,팀마다 강점약점, 학생마다 skill_level : 상/중상/중/중하/하
#팀 재생성 프롬포트 넣어서 팀 재생성 누르면 가능하도록 최종 팀에서 재생성 프롬포트넣어서 llm이 수정하도록 하기.
import json
import os
import re
from queue import Empty, Queue
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, List, Optional
from dotenv import load_dotenv
import requests

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from capteam_db import fetch_matching_result, save_matching_result
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

load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="CapTeam Matching API")

SSE_HEARTBEAT_SECONDS = 15
STREAM_END = object()


def create_batch_completion_callback(
    job_id: Optional[str],
    analyzed_students: List[Dict[str, Any]],
) -> Optional[Callable[[str, List[Dict[str, Any]]], None]]:
    """워크플로우 배치 완료를 백엔드 내부 API로 전달한다.

    백엔드가 호출하는 일반 /matching/run 경로에서도 같은 콜백을 사용한다.
    콜백 실패가 AI 매칭 전체 실패로 이어지지 않도록 호출부에서 예외를 로그만 남기고
    계속 진행한다.
    """
    normalized_job_id = str(job_id or "").strip()
    backend_base_url = str(os.getenv("BACKEND_BASE_URL") or "").strip().rstrip("/")
    internal_api_key = str(os.getenv("INTERNAL_MATCHING_API_KEY") or "").strip()

    if not normalized_job_id or not backend_base_url or not internal_api_key:
        return None

    state = {"total_teams": 0, "sent_team_names": set()}

    def on_progress(event_type: str, teams: List[Dict[str, Any]]) -> None:
        # 팀 구성이 확정된 직후의 preview를 팀 단위로 저장한다.
        # team_ready까지 기다리면 모든 설명 생성이 끝난 뒤 콜백이 몰려
        # 프론트가 첫 팀 도착을 놓칠 수 있다.
        if event_type != "team_preview" or not teams:
            return

        state["total_teams"] = len(teams)
        total_teams = state["total_teams"]
        for team in teams:
            team_name = str(team.get("team_name") or "").strip()
            if not team_name or team_name in state["sent_team_names"]:
                continue

            callback_team = build_stream_team_summary(team, analyzed_students)
            if not callback_team:
                continue

            try:
                response = requests.post(
                    f"{backend_base_url}/internal/matching/jobs/{normalized_job_id}/batch-complete",
                    headers={
                        "X-Internal-Api-Key": internal_api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "batch_index": len(state["sent_team_names"]),
                        "total_batches": total_teams,
                        "teams": [callback_team],
                    },
                    timeout=10,
                )
                response.raise_for_status()
                state["sent_team_names"].add(team_name)
            except requests.RequestException as error:
                print(
                    "백엔드 팀 완료 콜백 실패: "
                    f"{type(error).__name__}: {error}, team={team_name}"
                )
                # 앞 팀이 저장되지 않았는데 뒤 팀을 먼저 보내면 화면 순서가 깨진다.
                # 재호출 시 같은 팀부터 다시 시도할 수 있도록 중단한다.
                break

    return on_progress


# data에서 여러 후보 key 중 처음 존재하는 값을 반환한다.
# 요청/응답마다 다른 camelCase, snake_case 필드를 통일할 때 사용한다.
def _first_present(data: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data.get(key) is not None:
            return data.get(key)
    return None


# AI 내부 5단계 skill_level 값을 백엔드 StudentLevel enum 문자열로 변환한다.
def to_backend_student_level(skill_level: str) -> str:
    return {
        "높음": "UPPER",
        "상": "UPPER",
        "중상": "UPPER_MIDDLE",
        "보통": "MIDDLE",
        "중": "MIDDLE",
        "중하": "LOWER_MIDDLE",
        "낮음": "LOWER",
        "하": "LOWER",
        "UPPER": "UPPER",
        "UPPER_MIDDLE": "UPPER_MIDDLE",
        "MIDDLE_UPPER": "UPPER_MIDDLE",
        "MIDDLE": "MIDDLE",
        "LOWER_MIDDLE": "LOWER_MIDDLE",
        "MIDDLE_LOWER": "LOWER_MIDDLE",
        "LOWER": "LOWER",
    }.get(skill_level, "MIDDLE")


# 학생 분석 결과를 백엔드가 읽기 쉬운 응답 dict로 변환한다.
# 기존 분석 필드는 유지하고 user_id, analysis_result, student_level을 명시적으로 보강한다.
def build_analysis_response_result(result: Dict[str, Any]) -> Dict[str, Any]:
    skill_level = result.get("skill_level") or result.get("student_level") or result.get("level")
    analysis_result = (
        result.get("analysis_result")
        or result.get("analysisResult")
        or result.get("reason")
        or result.get("strength")
        or ""
    )
    analysis_status = str(result.get("analysis_status") or "SUCCESS").strip().upper()
    if analysis_status not in {"SUCCESS", "FAILED"}:
        analysis_status = "FAILED"
    if analysis_status == "SUCCESS" and (
        not str(analysis_result).strip() or not str(skill_level or "").strip()
    ):
        analysis_status = "FAILED"
        analysis_result = "분석 응답에 필수 결과가 누락되었습니다."
    return {
        **result,
        "user_id": (
            result.get("user_id")
            or result.get("userId")
            or result.get("student_id")
            or result.get("studentId")
        ),
        "analysis_result": analysis_result,
        "student_level": to_backend_student_level(skill_level),
        "analysis_status": analysis_status,
    }


# source의 성향 점수를 target에 표준 필드명으로 복사한다.
# 입력은 원본 학생 dict이고, 출력은 target dict에 성격/개발 성향 key를 채우는 방식이다.
def _copy_trait_scores(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    personality = source.get("personality_scores") or source.get("personalityScores") or {}
    development = source.get("development_scores") or source.get("developmentScores") or {}
    hackathon_personality = (
        source.get("hackathon_personality_scores")
        or source.get("hackathonPersonalityScores")
        or source.get("personalityScores")
        or {}
    )
    hackathon_development = (
        source.get("hackathon_development_scores")
        or source.get("hackathonDevelopmentScores")
        or source.get("developmentScores")
        or {}
    )

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
    hackathon_personality_keys = {
        "ideaPlanning", "communication", "roleFlexibility", "timePressure", "staminaFocus"
    }
    hackathon_development_keys = {
        "implementation", "problemSolving", "completionQuality", "presentation", "leadership"
    }
    if hackathon_personality_keys.issubset(hackathon_personality):
        target["hackathon_personality_scores"] = {
            key: hackathon_personality[key] for key in hackathon_personality_keys
        }
    if hackathon_development_keys.issubset(hackathon_development):
        target["hackathon_development_scores"] = {
            key: hackathon_development[key] for key in hackathon_development_keys
        }


# API 요청으로 받은 학생 목록을 AI 분석 함수가 쓰는 표준 구조로 변환한다.
# None이면 그대로 None을 반환하고, 있으면 이름/역할/스택/경험/성향 필드를 정규화한다.
def normalize_request_students(students: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if students is None:
        return None

    normalized_students = []
    for student in students:
        normalized = dict(student)
        normalized["user_id"] = _first_present(student, "user_id", "userId", "student_id", "studentId")
        normalized["name"] = _first_present(student, "name", "studentName")
        if isinstance(normalized["name"], str):
            normalized["name"] = normalized["name"].strip()
        if not normalized["name"]:
            user_id = normalized.get("user_id") or "unknown"
            raise HTTPException(status_code=400, detail=f"학생 이름이 비어 있습니다: {user_id}")
        normalized["role"] = _first_present(student, "role", "studentRole")
        normalized["stack"] = _first_present(student, "stack", "skill", "skills") or []
        normalized["experience"] = _first_present(student, "experience", "experiences") or []
        normalized["wants_leader"] = _first_present(student, "wants_leader", "wantsLeader")
        normalized["preferred_members"] = (
            _first_present(student, "preferred_members", "preferredMembers", "preferredTeammates")
            or []
        )
        normalized["response_reliability"] = (
            _first_present(student, "response_reliability", "responseReliability")
            or "HIGH"
        )
        normalized["grade"] = _first_present(student, "grade")
        _copy_trait_scores(normalized, student)
        normalized_students.append(normalized)

    return normalized_students


# /matching/run 요청이 리스트 또는 {students, regeneration_prompt} 둘 다 가능하도록 분리한다.
# 백엔드 재생성 요청의 prompt/current_teams를 AI 워크플로우 입력으로 넘길 때 사용한다.
def parse_matching_request(payload: Any) -> Dict[str, Any]:
    if payload is None or isinstance(payload, list):
        return {
            "students": payload,
            "prompt": "",
            "current_teams": None,
        }
    if isinstance(payload, dict):
        return {
            "students": payload.get("students"),
            "prompt": (
                payload.get("regeneration_prompt")
                or payload.get("regenerationPrompt")
                or payload.get("prompt")
                or ""
            ).strip(),
            "current_teams": payload.get("current_teams") or payload.get("currentTeams"),
        }

    raise HTTPException(status_code=400, detail="매칭 요청 형식이 올바르지 않습니다.")


# MySQL에 저장된 최신 매칭 결과를 우선 읽고, 없으면 로컬 matching_output.json을 읽는다.
# 둘 다 없으면 API 응답용 404 예외를 발생시킨다.
def load_matching_output(matching_type: str = "CAPSTONE") -> Dict[str, Any]:
    normalized_type = str(matching_type or "CAPSTONE").strip().upper()
    try:
        matching_output = fetch_matching_result(normalized_type)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if matching_output:
        return matching_output

    if normalized_type == "CAPSTONE" and MATCHING_OUTPUT_PATH.exists() and MATCHING_OUTPUT_PATH.stat().st_size > 0:
        with open(MATCHING_OUTPUT_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    raise HTTPException(status_code=404, detail=f"저장된 {normalized_type} 매칭 결과가 없습니다.")


# 워크플로우 전체 결과에서 화면에 쓸 final_result 부분만 꺼낸다.
# final_result 래퍼가 없으면 입력 dict 자체를 최종 결과로 사용한다.
def get_final_result(matching_output: Dict[str, Any]) -> Dict[str, Any]:
    return matching_output.get("final_result") or matching_output


# final_result에서 최종 팀 목록을 리스트 형태로 꺼낸다.
# final_teams가 중첩 dict로 저장된 경우 내부 final_teams 리스트를 반환한다.
def get_final_teams(final_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    final_teams = final_result.get("final_teams", [])

    if isinstance(final_teams, dict):
        return final_teams.get("final_teams", [])

    return final_teams


# 최종 결과에 저장된 알고리즘 초안 팀을 가져온다.
# final_result에 없으면 워크플로우의 teams 값을 fallback으로 사용한다.
def get_algorithm_teams(final_result: Dict[str, Any], matching_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    return final_result.get("algorithm_teams") or matching_output.get("teams", [])


# 팀별 평가 결과를 final_result, matching_output, balance_result 순서로 찾아 반환한다.
# 화면에서 강점/약점 fallback 정보를 만들 때 사용한다.
def get_team_evaluations(final_result: Dict[str, Any], matching_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    if final_result.get("team_evaluations"):
        return final_result.get("team_evaluations", [])
    if matching_output.get("team_evaluations"):
        return matching_output.get("team_evaluations", [])

    balance_result = final_result.get("balance_result") or matching_output.get("balance_result", {})
    llm_result = balance_result.get("llm_result", {})
    return llm_result.get("team_evaluations", [])


# "React: 7점" 형태의 stack_score 문자열을 [{stack, score}] 리스트로 변환한다.
# 화면에 학생별/팀별 대표 기술 점수를 보여주기 위한 파서다.
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


# 분석된 학생 목록을 이름 기준 dict로 만든다.
# 각 학생은 성향 프로필을 보강해서 이후 팀 요약 계산에 바로 사용할 수 있게 한다.
def build_student_map(matching_output: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        student.get("name"): ensure_trait_profile(student)
        for student in matching_output.get("analyzed_students", [])
        if student.get("name")
    }


# 알고리즘 초안 팀 목록을 team_name 기준 dict로 만든다.
# 최종 팀과 원래 초안 팀의 멤버/역할 정보를 비교하거나 보강할 때 사용한다.
def build_algorithm_team_map(
    final_result: Dict[str, Any],
    matching_output: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return {
        team.get("team_name"): team
        for team in get_algorithm_teams(final_result, matching_output)
        if team.get("team_name")
    }


# 팀 평가 목록을 team_name 기준 dict로 만든다.
# 최종 요약에서 평가 결과의 강점/리스크를 팀별로 빠르게 찾기 위해 사용한다.
def build_evaluation_map(
    final_result: Dict[str, Any],
    matching_output: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return {
        evaluation.get("team_name"): evaluation
        for evaluation in get_team_evaluations(final_result, matching_output)
        if evaluation.get("team_name")
    }


# 팀원 이름 목록과 학생 map을 받아 팀 내 상위 기술 점수를 뽑는다.
# 여러 학생의 stack_score를 합쳐 점수 높은 순으로 limit개 반환한다.
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


# 학생 한 명의 stack_score에서 상위 기술 점수 limit개를 반환한다.
# 멤버 카드에 표시할 대표 기술을 만들 때 사용한다.
def get_student_top_stack_scores(student: Dict[str, Any], limit: int = 2) -> List[Dict[str, Any]]:
    return sorted(
        parse_stack_score(student.get("stack_score", "")),
        key=lambda item: item["score"],
        reverse=True,
    )[:limit]


# 자유 입력 역할 문자열을 화면/집계용 role_group으로 분류한다.
# 프론트, 백엔드, AI, 앱, 디자인, 게임, 풀스택, DevOps, 보안, 기타 중 하나를 반환한다.
def get_display_role_group(role: str) -> str:
    role_text = (role or "").lower()
    if any(keyword in role_text for keyword in ["fullstack", "full-stack", "풀스택"]):
        return "fullstack"
    if any(keyword in role_text for keyword in ["devops", "dev ops", "인프라", "배포", "ci/cd", "cicd"]):
        return "devops"
    if any(keyword in role_text for keyword in ["security", "secure", "보안", "owasp"]):
        return "security"
    if any(keyword in role_text for keyword in ["frontend", "front", "프론트"]):
        return "frontend"
    if any(keyword in role_text for keyword in ["backend", "back", "server", "서버", "백엔드"]):
        return "backend"
    if any(keyword in role_text for keyword in ["ai", "data", "데이터", "머신러닝", "ml"]):
        return "ai_data"
    if any(keyword in role_text for keyword in ["app", "android", "ios", "mobile", "앱", "모바일"]):
        return "app"
    if any(keyword in role_text for keyword in ["design", "designer", "figma", "ui/ux", "ui", "ux", "디자인", "디자이너"]):
        return "design"
    if any(keyword in role_text for keyword in ["unity", "game", "게임"]):
        return "game"
    return "etc"


# role_groups 값을 리스트든 dict든 [{role_group, count}] 형태로 통일한다.
# AI 결과와 알고리즘 결과의 역할 분포 저장 형식 차이를 흡수한다.
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


# 멤버 값이 dict면 name을 꺼내고, 문자열이면 그대로 이름으로 반환한다.
# 최종 팀 members가 dict/문자열 어느 형태로 와도 처리하기 위한 보조 함수다.
def get_member_name(member: Any) -> Optional[str]:
    if isinstance(member, dict):
        return member.get("name")
    return member


# members 목록에서 유효한 학생 이름만 리스트로 정리한다.
# 팀 요약 계산 전 멤버 표현 형식을 이름 문자열 목록으로 맞춘다.
def get_member_names(members: Any) -> List[str]:
    if not isinstance(members, list):
        return []

    return [
        member_name
        for member_name in (get_member_name(member) for member in members)
        if member_name
    ]


# LLM 이유 문자열 뒤에 붙은 강점/약점 같은 섹션을 잘라낸다.
# 화면에 표시할 핵심 배정 이유만 깔끔하게 남기기 위해 사용한다.
def clean_reason_sections(reason: str) -> str:
    if not reason:
        return ""

    text = str(reason).strip()
    internal_labels = {
        "team_name": "팀 이름",
        "reason_cards": "배정 이유 카드",
        "role_groups": "역할 구성",
        "matching_evidence": "배정 근거",
        "member_profiles": "팀원 정보",
        "personality_evidence": "성향 참고",
        "role_distribution": "역할 구성",
        "implementation_connections": "역할 간 연결",
        "leader_selection": "팀장 선정 근거",
        "key_placements": "주요 배치 근거",
        "trait_complements": "성향 보완 관계",
        "response_reliability": "설문 응답 신뢰도",
        "skill_level": "기술 수준",
        "role_group": "역할 분야",
        "top_skills": "주요 기술",
        "roleGroup": "역할 분야",
        "skillLevel": "기술 수준",
    }
    for internal_name, display_name in internal_labels.items():
        text = text.replace(internal_name, display_name)

    return re.split(r"\s*\[(?:강점|보완점|약점|리스크)\]", text, maxsplit=1)[0].strip(" \"'")


# final_team의 reason_cards를 화면용 카드 리스트로 정규화한다.
# 카드가 없으면 matching_reason을 단일 카드로 변환하고, 둘 다 없으면 빈 리스트를 반환한다.
def normalize_reason_cards(final_team: Dict[str, Any], matching_reason: str) -> List[Dict[str, str]]:
    reason_cards = final_team.get("reason_cards") or final_team.get("reasonCards") or []
    if isinstance(reason_cards, list):
        normalized_cards = []
        for card in reason_cards:
            if not isinstance(card, dict):
                continue

            title = clean_reason_sections((card.get("title") or "").strip())
            description = clean_reason_sections((card.get("description") or "").strip())
            if title and description:
                normalized_cards.append({
                    "title": title,
                    "description": description,
                })

        if normalized_cards:
            return normalized_cards

    if matching_reason:
        return [{
            "title": "팀 배정 이유",
            "description": matching_reason,
        }]

    return []


# 최종 팀/알고리즘 팀의 role_groups를 우선 사용해 역할 분포를 만든다.
# 저장된 역할 분포가 없으면 학생 role을 다시 분류해서 count를 계산한다.
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


# AI 내부 skill_level과 백엔드 enum을 화면 표기용 5단계로 변환한다.
def normalize_skill_level_label(skill_level: str) -> str:
    return {
        "높음": "상",
        "보통": "중",
        "낮음": "하",
        "상": "상",
        "중상": "중상",
        "중": "중",
        "중하": "중하",
        "하": "하",
        "UPPER": "상",
        "UPPER_MIDDLE": "중상",
        "MIDDLE_UPPER": "중상",
        "MIDDLE": "중",
        "LOWER_MIDDLE": "중하",
        "MIDDLE_LOWER": "중하",
        "LOWER": "하",
    }.get(skill_level, skill_level or "")


# 팀원 목록에서 상/중상/중/중하/하 실력 분포 개수를 계산한다.
# 관리자 팀 요약 화면의 skill_level_counts 필드를 만든다.
def get_skill_level_counts(members: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"상": 0, "중상": 0, "중": 0, "중하": 0, "하": 0}

    for member in members:
        level = normalize_skill_level_label(member.get("skill_level"))
        if level in counts:
            counts[level] += 1

    return counts


# 여러 후보 텍스트 중 처음 비어 있지 않은 값을 반환한다.
# 문자열 리스트는 쉼표로 합쳐 팀 강점/약점 fallback 문구로 사용한다.
def first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            text = ", ".join(str(item).strip() for item in value if str(item).strip())
            if text:
                return text
    return ""


# 팀원들의 분석 필드(strength/weakness 등)를 모아 짧은 팀 요약 문장으로 만든다.
# 해당 필드가 없으면 fallback 문장을 반환한다.
def summarize_team_analysis(
    member_names: List[str],
    student_map: Dict[str, Dict[str, Any]],
    field: str,
    fallback: str,
) -> str:
    summaries = [
        student_map.get(member_name, {}).get(field, "").strip()
        for member_name in member_names
        if student_map.get(member_name, {}).get(field)
    ]
    if not summaries:
        return fallback

    return " ".join(summaries[:2])


# 최종 팀 멤버 이름을 화면에 필요한 멤버 요약 dict 목록으로 변환한다.
# 알고리즘 초안과 학생 분석 정보를 합쳐 역할, 실력, 점수, 대표 기술을 채운다.
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
            "user_id": (
                enriched_member.get("user_id")
                or enriched_member.get("userId")
                or enriched_member.get("student_id")
            ),
            "name": enriched_member.get("name"),
            "role": enriched_member.get("role"),
            "role_group": enriched_member.get("role_group") or get_display_role_group(enriched_member.get("role", "")),
            "skill_level": normalize_skill_level_label(enriched_member.get("skill_level")),
            "score": round(float(enriched_member.get("score", 0) or 0), 2),
            "top_stack_scores": get_student_top_stack_scores(student),
        })

    return summaries


# 매칭 결과 전체를 프론트 팀 요약 API 응답 형태로 변환한다.
# 최종 팀, 알고리즘 초안, 학생 분석, 팀 평가를 합쳐 total/teams 구조를 만든다.
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
        evaluation = evaluation_map.get(team_name, {})
        algorithm_members = algorithm_team.get("members", [])
        member_summaries = build_member_summaries(member_names, algorithm_members, student_map)
        leader_member = choose_team_leader(member_summaries)
        leader_name = final_team.get("leader") or leader_member.get("name")
        matching_reason = clean_reason_sections(final_team.get("reason", ""))
        reason_cards = normalize_reason_cards(final_team, matching_reason)
        strengths = first_text(
            final_team.get("strengths"),
            final_team.get("strength"),
            evaluation.get("strengths"),
            summarize_team_analysis(
                member_names,
                student_map,
                "strength",
                matching_reason,
            ),
        )
        weaknesses = first_text(
            final_team.get("weaknesses"),
            final_team.get("weakness"),
            final_team.get("risks"),
            evaluation.get("weaknesses"),
            evaluation.get("risks"),
            summarize_team_analysis(
                member_names,
                student_map,
                "weakness",
                "뚜렷한 팀 약점은 확인되지 않았습니다.",
            ),
        )

        teams.append({
            "total_people": len(member_names),
            "team_name": team_name,
            "role_counts": build_role_counts(member_names, student_map, final_team, algorithm_team),
            "leader": leader_name,
            "matching_reason": matching_reason,
            "reason_cards": reason_cards,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "skill_level_counts": get_skill_level_counts(member_summaries),
            "members": member_summaries,
        })

    return {
        "total_students": len(student_map),
        "total_teams": len(teams),
        "changed": final_result.get("changed", False),
        "change_summary": final_result.get("change_summary", ""),
        "validation_notes": final_result.get("validation_notes", ""),
        "teams": teams,
    }


def normalize_stream_job_id(job_id: str) -> str:
    normalized = str(job_id or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="X-Matching-Job-Id 헤더가 필요합니다.")
    return normalized


def parse_stream_students(payload: Any, required: bool) -> Optional[List[Dict[str, Any]]]:
    if isinstance(payload, list):
        students = payload
    elif isinstance(payload, dict):
        students = payload.get("students")
    elif payload is None and not required:
        return None
    else:
        raise HTTPException(status_code=400, detail="students 목록이 필요합니다.")

    if students is None and not required:
        return None
    if not isinstance(students, list) or not students:
        raise HTTPException(status_code=400, detail="students 목록이 필요합니다.")
    return normalize_request_students(students)


def format_sse_event(event_id: int, event_type: str, data: Dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event_id}\nevent: {event_type}\ndata: {payload}\n\n"


def build_stream_team_summary(
    team: Dict[str, Any],
    analyzed_students: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary = build_team_summary({
        "analyzed_students": analyzed_students,
        "final_result": {"final_teams": [team]},
    })
    return summary["teams"][0] if summary.get("teams") else {}


def create_team_progress_callback(
    analyzed_students: List[Dict[str, Any]],
    emit: Callable[[str, Dict[str, Any]], None],
    set_stage: Callable[[str], None],
):
    versions = {"team_preview": 1, "team_update": 2, "team_ready": 3}
    state = {
        "team_index_by_name": {},
        "total_teams": 0,
        "seen": set(),
        "validation_reported": False,
        "explanation_reported": False,
    }

    def callback(event_type: str, teams: List[Dict[str, Any]]) -> None:
        if event_type not in versions or not teams:
            return

        if event_type == "team_preview" and not state["validation_reported"]:
            set_stage("VALIDATING")
            state["validation_reported"] = True
            state["total_teams"] = len(teams)
            state["team_index_by_name"] = {
                team.get("team_name"): index
                for index, team in enumerate(teams, start=1)
            }

        if not state["team_index_by_name"]:
            state["total_teams"] = len(teams)
            state["team_index_by_name"] = {
                team.get("team_name"): index
                for index, team in enumerate(teams, start=1)
            }

        for team in teams:
            team_name = team.get("team_name") or ""
            event_key = (event_type, team_name)
            if event_key in state["seen"]:
                continue
            emit(event_type, {
                "team_name": team_name,
                "team_index": state["team_index_by_name"].get(team_name, 0),
                "total_teams": state["total_teams"],
                "version": versions[event_type],
                "team": build_stream_team_summary(team, analyzed_students),
            })
            state["seen"].add(event_key)

        if event_type == "team_preview" and not state["explanation_reported"]:
            set_stage("EXPLAINING")
            state["explanation_reported"] = True

    return callback


def run_capstone_stream_job(
    mode: str,
    matching_type: str,
    request_students: Optional[List[Dict[str, Any]]],
    stored_students: Optional[List[Dict[str, Any]]],
    current_teams: Optional[List[Dict[str, Any]]],
    prompt: str,
    emit: Callable[[str, Dict[str, Any]], None],
    set_stage: Callable[[str], None],
) -> None:
    from student_analysis.analysis_llm import get_analyze_stu
    from matching_student.workflow_matching_student import (
        build_public_workflow_result,
        run_regenerate_workflow,
        run_workflow,
    )

    set_stage("ANALYZING")
    analyzed_students = (
        get_analyze_stu(request_students)
        if request_students is not None
        else list(stored_students or [])
    )
    if not analyzed_students:
        raise ValueError("팀 매칭에 사용할 학생 분석 결과가 없습니다.")

    progress_callback = create_team_progress_callback(analyzed_students, emit, set_stage)
    set_stage("MATCHING")
    if mode == "REGENERATE":
        result = run_regenerate_workflow(
            prompt=prompt,
            current_teams=current_teams,
            analyzed_students=analyzed_students,
            persist_result=False,
            progress_callback=progress_callback,
        )
    else:
        result = run_workflow(
            force_rematch=True,
            analyzed_students=analyzed_students,
            persist_result=False,
            progress_callback=progress_callback,
        )

    final_teams = get_final_teams(get_final_result(result))
    for event_type in ("team_preview", "team_update", "team_ready"):
        progress_callback(event_type, final_teams)

    set_stage("SAVING")
    save_matching_result(build_public_workflow_result(result), matching_type=matching_type)
    emit("completed", {"result": build_team_summary(result)})


def create_matching_stream_response(
    job_id: str,
    mode: str,
    worker: Callable[[Callable[[str, Dict[str, Any]], None], Callable[[str], None]], None],
) -> StreamingResponse:
    event_queue = Queue()
    current_stage = {"value": "STARTING"}

    def emit(event_type: str, data: Dict[str, Any]) -> None:
        event_queue.put({
            "event": event_type,
            "data": {"job_id": job_id, **data},
        })

    def set_stage(stage: str) -> None:
        current_stage["value"] = stage
        emit("progress", {"stage": stage})

    def run_worker() -> None:
        try:
            worker(emit, set_stage)
        except Exception as error:
            print(f"팀 매칭 스트림 실패: {type(error).__name__}: {error}")
            emit("error", {
                "stage": current_stage["value"],
                "message": str(error) or "팀 매칭 스트림 처리 중 오류가 발생했습니다.",
                "retryable": False,
            })
        finally:
            event_queue.put(STREAM_END)

    event_queue.put({
        "event": "started",
        "data": {"job_id": job_id, "mode": mode},
    })
    Thread(target=run_worker, name=f"matching-stream-{job_id}", daemon=True).start()

    def event_generator():
        event_id = 1
        while True:
            try:
                item = event_queue.get(timeout=SSE_HEARTBEAT_SECONDS)
            except Empty:
                yield ": keep-alive\n\n"
                continue
            if item is STREAM_END:
                return
            yield format_sse_event(event_id, item["event"], item["data"])
            event_id += 1

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def build_hackathon_summary(matching_output: Dict[str, Any]) -> Dict[str, Any]:
    final_result = matching_output.get("final_result") or matching_output
    final_teams = final_result.get("final_teams", [])
    teams = []
    for team in final_teams:
        members = team.get("members", [])
        role_groups = team.get("role_groups", {})
        role_counts = [
            {"role_group": role_group, "count": count}
            for role_group, count in role_groups.items()
            if count
        ]
        teams.append({
            "total_people": len(members),
            "team_name": team.get("team_name"),
            "role_counts": normalize_role_counts(team.get("role_groups")),
            "leader": team.get("leader", ""),
            "presentation_candidate": team.get("presentation_candidate", ""),
            "planning_candidate": team.get("planning_candidate", ""),
            "flexible_supporter": team.get("flexible_supporter", ""),
            "matching_reason": team.get("reason", ""),
            "reason_cards": team.get("reason_cards", []),
            "strengths": team.get("strengths", ""),
            "weaknesses": team.get("weaknesses", ""),
            "technical_average": team.get("technical_average", 0),
            "execution_average": team.get("execution_average", 0),
            "personality_averages": team.get("personality_averages", {}),
            "development_averages": team.get("development_averages", {}),
            "warnings": team.get("warnings", []),
            "members": members,
        })
    return {
        "matching_type": "HACKATHON",
        "total_students": len(matching_output.get("analyzed_students", [])),
        "total_teams": len(teams),
        "changed": final_result.get("changed", False),
        "change_summary": final_result.get("change_summary", ""),
        "teams": teams,
        "balance_result": final_result.get("balance_result", {}),
        "iteration_count": final_result.get("iteration_count", 0),
        "finalized_by": final_result.get("finalized_by", ""),
    }


@app.get("/health")
# 서버가 살아 있는지 확인하는 헬스체크 API다.
# 입력 없이 {"status": "ok"}를 반환한다.
def health():
    return {"status": "ok"}


@app.get("/teams/summary")
# 저장된 최신 매칭 결과를 프론트 요약 응답으로 반환하는 API다.
# 입력 없이 MySQL/파일 결과를 읽어 build_team_summary 결과를 반환한다.
def teams_summary(matching_type: str = "CAPSTONE"):
    normalized_type = str(matching_type or "CAPSTONE").strip().upper()
    result = load_matching_output(normalized_type)
    if normalized_type == "HACKATHON":
        return build_team_summary(result)
    return build_team_summary(result)


@app.post("/analysis/run")
# 요청 학생 목록을 정규화해 학생 분석 LLM을 실행하는 API다.
# 분석된 학생 수, 성공 상태, 백엔드 저장용 분석 결과 목록을 반환한다.
def run_analysis(students: Optional[List[Dict[str, Any]]] = Body(default=None)):
    from student_analysis.analysis_llm import get_analyze_stu

    results = get_analyze_stu(
        normalize_request_students(students),
        force_reanalyze=True,
    )
    response_results = [build_analysis_response_result(result) for result in results]
    failed_results = [
        result
        for result in response_results
        if result.get("analysis_status") == "FAILED"
    ]
    if failed_results:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "FAILED",
                "message": "학생 설문 분석을 완료하지 못했습니다.",
                "results": response_results,
            },
        )
    return {
        "status": "ok",
        "total_students": len(results),
        "results": response_results,
    }


@app.post("/matching/run")
# 요청 학생이 있으면 재분석한 뒤 팀 매칭 워크플로우를 새로 실행하는 API다.
# force_rematch=True로 캐시를 무시하고 새 추천안을 만든 뒤 요약 응답을 반환한다.
def run_matching(
    payload: Any = Body(default=None),
    matching_job_id: Optional[str] = Header(default=None, alias="X-Matching-Job-Id"),
):
    from student_analysis.analysis_llm import get_analyze_stu
    from matching_student.workflow_matching_student import (
        FinalAssignmentValidationError,
        run_regenerate_workflow,
        run_workflow,
    ) #open_ai_api로 할때 이거 밑에 주석치고 이거하셈
    # from matching_student.upstage_matching import run_regenerate_workflow, run_workflow

    matching_request = parse_matching_request(payload)
    request_students = normalize_request_students(matching_request["students"])
    analyzed_students = None
    if request_students is not None:
        analyzed_students = get_analyze_stu(request_students)
    callback_students = analyzed_students or load_matching_output().get("analyzed_students", [])
    batch_completion_callback = create_batch_completion_callback(
        matching_job_id,
        callback_students,
    )

    if matching_request["prompt"]:
        result = run_regenerate_workflow(
            prompt=matching_request["prompt"],
            current_teams=matching_request["current_teams"],
            analyzed_students=analyzed_students,
            progress_callback=batch_completion_callback,
        )
        return build_team_summary(result)

    try:
        result = run_workflow(
            force_rematch=True,
            analyzed_students=analyzed_students,
            progress_callback=batch_completion_callback,
        )
    except FinalAssignmentValidationError as error:
        raise HTTPException(
            status_code=422,
            detail={"status": "VALIDATION_FAILED", "message": str(error)},
        ) from error
    return build_team_summary(result)


@app.post("/matching/run/stream")
def stream_matching(
    payload: Any = Body(default=None),
    matching_job_id: str = Header(..., alias="X-Matching-Job-Id"),
):
    job_id = normalize_stream_job_id(matching_job_id)
    matching_request = parse_matching_request(payload)
    request_students = parse_stream_students(payload, required=True)
    prompt = matching_request["prompt"]

    return create_matching_stream_response(
        job_id=job_id,
        mode="REGENERATE" if prompt else "INITIAL",
        worker=lambda emit, set_stage: run_capstone_stream_job(
            mode="REGENERATE" if prompt else "INITIAL",
            matching_type="CAPSTONE",
            request_students=request_students,
            stored_students=None,
            current_teams=matching_request["current_teams"],
            prompt=prompt,
            emit=emit,
            set_stage=set_stage,
        ),
    )


@app.post("/matching/hackathon/run")
# 백엔드 호환을 위해 해커톤 URL을 유지하되 캡스톤 기준으로 팀을 생성한다.
# 결과는 HACKATHON 저장 영역에 따로 보관해 일반 CAPSTONE 캐시를 덮어쓰지 않는다.
def run_hackathon_matching(payload: Any = Body(default=None)):
    from student_analysis.analysis_llm import get_analyze_stu
    from matching_student.workflow_matching_student import build_public_workflow_result, run_workflow

    if isinstance(payload, list):
        students = payload
    elif isinstance(payload, dict):
        students = payload.get("students")
    else:
        raise HTTPException(status_code=400, detail="students 목록이 필요합니다.")

    try:
        request_students = normalize_request_students(students)
        analyzed_students = get_analyze_stu(request_students)
        result = run_workflow(
            force_rematch=True,
            analyzed_students=analyzed_students,
            persist_result=False,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    save_matching_result(build_public_workflow_result(result), matching_type="HACKATHON")
    return build_team_summary(result)


@app.post("/matching/hackathon/run/stream")
def stream_hackathon_matching(
    payload: Any = Body(default=None),
    matching_job_id: str = Header(..., alias="X-Matching-Job-Id"),
):
    job_id = normalize_stream_job_id(matching_job_id)
    request_students = parse_stream_students(payload, required=True)
    return create_matching_stream_response(
        job_id=job_id,
        mode="INITIAL",
        worker=lambda emit, set_stage: run_capstone_stream_job(
            mode="INITIAL",
            matching_type="HACKATHON",
            request_students=request_students,
            stored_students=None,
            current_teams=None,
            prompt="",
            emit=emit,
            set_stage=set_stage,
        ),
    )


@app.get("/matching/hackathon/summary")
def hackathon_matching_summary():
    return build_team_summary(load_matching_output("HACKATHON"))


@app.post("/matching/hackathon/regenerate")
def regenerate_hackathon_matching(payload: Optional[Dict[str, Any]] = Body(default=None)):
    from student_analysis.analysis_llm import get_analyze_stu
    from matching_student.workflow_matching_student import (
        build_public_workflow_result,
        run_regenerate_workflow,
    )

    payload = payload or {}
    prompt = str(
        payload.get("prompt")
        or payload.get("regeneration_prompt")
        or payload.get("regenerationPrompt")
        or ""
    ).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="재생성 프롬프트가 비어 있습니다.")

    saved_result = load_matching_output("HACKATHON")
    request_students = normalize_request_students(payload.get("students"))
    if request_students is not None:
        analyzed_students = get_analyze_stu(request_students)
    else:
        analyzed_students = saved_result.get("analyzed_students", [])
    current_teams = (
        payload.get("current_teams")
        or payload.get("currentTeams")
        or (saved_result.get("final_result") or {}).get("final_teams")
    )
    try:
        result = run_regenerate_workflow(
            prompt=prompt,
            current_teams=current_teams,
            analyzed_students=analyzed_students,
            persist_result=False,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    save_matching_result(build_public_workflow_result(result), matching_type="HACKATHON")
    return build_team_summary(result)


@app.post("/matching/hackathon/regenerate/stream")
def stream_regenerate_hackathon_matching(
    payload: Optional[Dict[str, Any]] = Body(default=None),
    matching_job_id: str = Header(..., alias="X-Matching-Job-Id"),
):
    job_id = normalize_stream_job_id(matching_job_id)
    payload = payload or {}
    prompt = str(
        payload.get("prompt")
        or payload.get("regeneration_prompt")
        or payload.get("regenerationPrompt")
        or ""
    ).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="재생성 프롬프트가 비어 있습니다.")

    saved_result = load_matching_output("HACKATHON")
    request_students = parse_stream_students(payload, required=False)
    current_teams = (
        payload.get("current_teams")
        or payload.get("currentTeams")
        or (saved_result.get("final_result") or {}).get("final_teams")
    )
    if not current_teams:
        raise HTTPException(status_code=400, detail="재생성할 현재 팀이 없습니다.")

    return create_matching_stream_response(
        job_id=job_id,
        mode="REGENERATE",
        worker=lambda emit, set_stage: run_capstone_stream_job(
            mode="REGENERATE",
            matching_type="HACKATHON",
            request_students=request_students,
            stored_students=saved_result.get("analyzed_students", []),
            current_teams=current_teams,
            prompt=prompt,
            emit=emit,
            set_stage=set_stage,
        ),
    )


@app.post("/matching/regenerate")
# 사용자가 입력한 재생성 프롬프트로 현재 추천안을 다시 조정하는 API다.
# prompt, current_teams, 선택적 students를 받아 재생성 결과 요약을 반환한다.
def regenerate_matching(
    payload: Optional[Dict[str, Any]] = Body(default=None),
    matching_job_id: Optional[str] = Header(default=None, alias="X-Matching-Job-Id"),
):
    from student_analysis.analysis_llm import get_analyze_stu
    from matching_student.workflow_matching_student import run_regenerate_workflow

    payload = payload or {}
    prompt = (
        payload.get("prompt")
        or payload.get("regeneration_prompt")
        or payload.get("regenerationPrompt")
        or ""
    ).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="재생성 프롬프트가 비어 있습니다.")

    request_students = normalize_request_students(payload.get("students"))
    analyzed_students = get_analyze_stu(request_students) if request_students is not None else None
    callback_students = analyzed_students or load_matching_output().get("analyzed_students", [])
    batch_completion_callback = create_batch_completion_callback(
        matching_job_id,
        callback_students,
    )

    try:
        result = run_regenerate_workflow(
            prompt=prompt,
            current_teams=payload.get("current_teams") or payload.get("currentTeams"),
            analyzed_students=analyzed_students,
            progress_callback=batch_completion_callback,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return build_team_summary(result)
#uvicorn api.main:app --reload
