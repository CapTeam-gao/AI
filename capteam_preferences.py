#팀장선호와 선호팀원 코드
from typing import Any, Dict, Iterable, List, Optional, Set

from capteam_traits import (
    CORE_RISK_TRAITS,
    TRAIT_LABELS,
    NEUTRAL_SCORE,
    choose_team_leader,
    get_all_trait_scores,
    get_leader_score,
)


MAX_PREFERRED_MEMBERS = 3


def normalize_wants_leader(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "팀장", "원함"}
    return False


def _as_name_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",")]
    if isinstance(value, list):
        names = []
        for item in value:
            if isinstance(item, dict):
                names.append(str(item.get("name") or item.get("student_name") or "").strip())
            else:
                names.append(str(item).strip())
        return names
    return []


def normalize_preferred_members(
    student: Dict[str, Any],
    allowed_names: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    own_name = student.get("name")
    raw_names = _as_name_list(
        student.get("preferred_members")
        or student.get("preferredMembers")
        or student.get("matching_preferences", {}).get("preferred_members")
    )
    normalized = []
    ignored = []
    seen = set()

    for raw_name in raw_names:
        if not raw_name:
            ignored.append("빈 선호 이름은 제외했습니다.")
            continue
        if raw_name == own_name:
            ignored.append(f"{raw_name} 선호는 자기 자신이라 제외했습니다.")
            continue
        if raw_name in seen:
            ignored.append(f"{raw_name} 선호는 중복이라 제외했습니다.")
            continue
        if allowed_names is not None and raw_name not in allowed_names:
            ignored.append(f"{raw_name} 선호는 분석 데이터에 없는 학생이라 제외했습니다.")
            continue
        if len(normalized) >= MAX_PREFERRED_MEMBERS:
            ignored.append(f"{raw_name} 선호는 최대 3명 제한으로 제외했습니다.")
            continue

        seen.add(raw_name)
        normalized.append(raw_name)

    return {
        "preferred_members": normalized,
        "ignored_preferences": ignored,
    }


def ensure_preference_profile(
    student: Dict[str, Any],
    allowed_names: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    normalized = normalize_preferred_members(student, allowed_names=allowed_names)
    wants_leader = normalize_wants_leader(
        student.get("wants_leader")
        if "wants_leader" in student
        else student.get("wantsLeader", student.get("matching_preferences", {}).get("wants_leader"))
    )

    enriched = dict(student)
    enriched["preferred_members"] = normalized["preferred_members"]
    enriched["wants_leader"] = wants_leader
    enriched["matching_preferences"] = {
        "preferred_members": normalized["preferred_members"],
        "wants_leader": wants_leader,
        "ignored_preferences": normalized["ignored_preferences"],
    }
    return enriched


def ensure_preference_profiles(students: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    allowed_names = {student.get("name") for student in students if student.get("name")}
    return [
        ensure_preference_profile(student, allowed_names=allowed_names)
        for student in students
    ]


def get_preferred_members(student: Dict[str, Any]) -> List[str]:
    return student.get("matching_preferences", {}).get("preferred_members", student.get("preferred_members", []))


def wants_leader(student: Dict[str, Any]) -> bool:
    return bool(student.get("matching_preferences", {}).get("wants_leader", student.get("wants_leader", False)))


def choose_preference_aware_leader(members: List[Dict[str, Any]]) -> Dict[str, Any]:
    leader_candidates = [member for member in members if wants_leader(member)]
    if leader_candidates:
        return max(
            leader_candidates,
            key=lambda member: (
                get_leader_score(member),
                float(member.get("technical_score", member.get("score", 0)) or 0),
            ),
        )
    return choose_team_leader(members)


def preference_bonus(student: Dict[str, Any], members: List[Dict[str, Any]]) -> int:
    if not members:
        return 0

    student_name = student.get("name")
    student_preferences = set(get_preferred_members(student))
    bonus = 0

    for member in members:
        member_name = member.get("name")
        member_preferences = set(get_preferred_members(member))

        if member_name in student_preferences:
            bonus += 12
        if student_name in member_preferences:
            bonus += 12
        if member_name in student_preferences and student_name in member_preferences:
            bonus += 8

    return bonus


def _has_single_role_group(members: List[Dict[str, Any]]) -> bool:
    role_groups = {member.get("role_group") for member in members if member.get("role_group")}
    return len(members) > 1 and len(role_groups) == 1


def _has_core_trait_risk(members: List[Dict[str, Any]]) -> bool:
    if not members:
        return False

    for trait in CORE_RISK_TRAITS:
        values = [get_all_trait_scores(member).get(trait, NEUTRAL_SCORE) for member in members]
        if sum(values) / len(values) < 3:
            return True
        if sum(1 for value in values if value <= 2) >= 2:
            return True
    return False


def breaks_preference_constraints(candidate_members: List[Dict[str, Any]]) -> List[str]:
    reasons = []

    if _has_single_role_group(candidate_members):
        reasons.append("같은 역할군이 한 팀에 몰립니다.")
    if _has_core_trait_risk(candidate_members):
        reasons.append("핵심 성향 평균 또는 낮은 성향 분포가 기준을 벗어납니다.")

    return reasons


def team_preference_notes(members: List[Dict[str, Any]]) -> List[str]:
    member_names = {member.get("name") for member in members}
    notes = []

    for member in members:
        preferred = [name for name in get_preferred_members(member) if name in member_names]
        if preferred:
            notes.append(f"{member.get('name')}의 선호 팀원 {', '.join(preferred)}을 함께 배치했습니다.")

    return notes


def build_preference_rejections(
    teams: List[Dict[str, Any]],
    students: Iterable[Dict[str, Any]],
) -> List[str]:
    team_by_student = {}
    for team in teams:
        for member in team.get("members", []):
            name = member.get("name") if isinstance(member, dict) else member
            if name:
                team_by_student[name] = team.get("team_name")

    rejections = []
    for student in students:
        student_name = student.get("name")
        student_team = team_by_student.get(student_name)
        for preferred_name in get_preferred_members(student):
            preferred_team = team_by_student.get(preferred_name)
            if student_team and preferred_team and student_team != preferred_team:
                rejections.append(
                    f"{student_name}-{preferred_name} 선호는 반영하지 않았습니다. "
                    "점수, 역할군, 성향 균형 기준을 우선했습니다."
                )

    return rejections
