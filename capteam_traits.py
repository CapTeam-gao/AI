#학생 설문 성향 점수를 정리하고 분석과, 매칭에 쓸수있도록 만듬.
from typing import Any, Dict, Iterable, List, Optional


PERSONALITY_TRAITS = [
    ("communication", "소통"),
    ("responsibility", "책임감"),
    ("collaboration", "협업"),
    ("flexibility", "유연성"),
    ("emotionalStability", "감정 안정성"),
]

DEVELOPMENT_TRAITS = [
    ("leadership", "리더십"),
    ("problemSolving", "문제 해결력"),
    ("implementation", "구현 실행력"),
    ("learningAbility", "학습 성장성"),
    ("planning", "기획 정리력"),
]

CORE_RISK_TRAITS = ["communication", "responsibility", "implementation", "problemSolving"]
TRAIT_LABELS = dict(PERSONALITY_TRAITS + DEVELOPMENT_TRAITS)
NEUTRAL_SCORE = 3


def normalize_trait_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return NEUTRAL_SCORE

    if score <= 0:
        return NEUTRAL_SCORE
    return min(max(score, 1), 5)


def _raw_trait_value(data: Dict[str, Any], key: str) -> Any:
    for group_key in ("personality_scores", "development_scores"):
        group = data.get(group_key)
        if isinstance(group, dict) and key in group:
            return group.get(key)
    return data.get(key)


def _build_scores(data: Dict[str, Any], traits: Iterable[tuple]) -> Dict[str, int]:
    return {
        key: normalize_trait_score(_raw_trait_value(data, key))
        for key, _ in traits
    }


def get_defaulted_traits(data: Dict[str, Any]) -> List[str]:
    defaulted = []
    for key, _ in PERSONALITY_TRAITS + DEVELOPMENT_TRAITS:
        value = _raw_trait_value(data, key)
        if value in (None, "", 0, "0"):
            defaulted.append(key)
    return defaulted


def build_trait_summary(scores: Dict[str, int]) -> Dict[str, Any]:
    high_traits = [
        {"trait": key, "label": TRAIT_LABELS[key], "score": score}
        for key, score in scores.items()
        if score >= 4
    ]
    low_traits = [
        {"trait": key, "label": TRAIT_LABELS[key], "score": score}
        for key, score in scores.items()
        if score <= 2
    ]

    if high_traits:
        strength_text = ", ".join(item["label"] for item in high_traits)
    else:
        strength_text = "두드러진 고점 성향은 없고 전반적으로 중립 수준입니다."

    if low_traits:
        risk_text = ", ".join(item["label"] for item in low_traits)
    else:
        risk_text = "낮은 리스크 성향은 없습니다."

    return {
        "strengths": high_traits,
        "risks": low_traits,
        "summary": f"강점: {strength_text} / 리스크: {risk_text}",
    }


def build_matching_traits(personality_scores: Dict[str, int], development_scores: Dict[str, int]) -> Dict[str, Any]:
    all_scores = {**personality_scores, **development_scores}
    trait_average = round(sum(all_scores.values()) / len(all_scores), 2)
    leader_score = round(
        development_scores["leadership"] * 0.4
        + development_scores["planning"] * 0.3
        + development_scores["problemSolving"] * 0.3,
        2,
    )

    low_traits = [
        {"trait": key, "label": TRAIT_LABELS[key], "score": score}
        for key, score in all_scores.items()
        if score <= 2
    ]
    high_traits = [
        {"trait": key, "label": TRAIT_LABELS[key], "score": score}
        for key, score in all_scores.items()
        if score >= 4
    ]

    return {
        "trait_average": trait_average,
        "leader_score": leader_score,
        "low_traits": low_traits,
        "high_traits": high_traits,
        "defaulted_traits": [],
    }


def ensure_trait_profile(student: Dict[str, Any], source: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = {**(source or {}), **student}
    personality_scores = _build_scores(data, PERSONALITY_TRAITS)
    development_scores = _build_scores(data, DEVELOPMENT_TRAITS)
    matching_traits = build_matching_traits(personality_scores, development_scores)
    matching_traits["defaulted_traits"] = get_defaulted_traits(data)

    enriched = dict(student)
    enriched["personality_scores"] = personality_scores
    enriched["development_scores"] = development_scores
    enriched["personality_summary"] = build_trait_summary(personality_scores)
    enriched["development_summary"] = build_trait_summary(development_scores)
    enriched["matching_traits"] = matching_traits
    return enriched


def get_leader_score(student: Dict[str, Any]) -> float:
    return float(student.get("matching_traits", {}).get("leader_score", 3.0))


def get_trait_average(student: Dict[str, Any]) -> float:
    return float(student.get("matching_traits", {}).get("trait_average", 3.0))


def get_trait_score(student: Dict[str, Any]) -> float:
    return get_trait_average(student) * 6


def get_all_trait_scores(student: Dict[str, Any]) -> Dict[str, int]:
    return {
        **student.get("personality_scores", {}),
        **student.get("development_scores", {}),
    }


def get_low_trait_names(student: Dict[str, Any]) -> set:
    return {
        item["trait"]
        for item in student.get("matching_traits", {}).get("low_traits", [])
    }


def get_high_trait_names(student: Dict[str, Any]) -> set:
    return {
        item["trait"]
        for item in student.get("matching_traits", {}).get("high_traits", [])
    }


def calculate_trait_averages(members: List[Dict[str, Any]], score_key: str) -> Dict[str, float]:
    if not members:
        return {}

    keys = list(members[0].get(score_key, {}).keys())
    averages = {}
    for key in keys:
        values = [member.get(score_key, {}).get(key, NEUTRAL_SCORE) for member in members]
        averages[key] = round(sum(values) / len(values), 2)
    return averages


def build_team_trait_risks(members: List[Dict[str, Any]]) -> List[str]:
    if not members:
        return ["팀원이 없어 성향 리스크를 계산할 수 없습니다."]

    all_scores = [get_all_trait_scores(member) for member in members]
    risks = []

    for trait in CORE_RISK_TRAITS:
        values = [scores.get(trait, NEUTRAL_SCORE) for scores in all_scores]
        average = sum(values) / len(values)
        low_count = sum(1 for value in values if value <= 2)
        high_count = sum(1 for value in values if value >= 4)

        if average < 3:
            risks.append(f"{TRAIT_LABELS[trait]} 평균이 3 미만입니다.")
        if low_count >= 2:
            risks.append(f"{TRAIT_LABELS[trait]} 낮은 학생이 한 팀에 몰렸습니다.")
        if low_count and not high_count:
            risks.append(f"{TRAIT_LABELS[trait]} 보완 역할을 할 고점 학생이 없습니다.")

    if max(get_leader_score(member) for member in members) < 4.0:
        risks.append("leader_score 4.0 이상 팀장 후보가 없습니다.")

    return risks


def choose_team_leader(members: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not members:
        return {}
    return max(
        members,
        key=lambda member: (
            get_leader_score(member),
            float(member.get("technical_score", member.get("score", 0)) or 0),
        ),
    )


def build_leader_reason(leader: Dict[str, Any]) -> str:
    if not leader:
        return ""
    scores = leader.get("development_scores", {})
    return (
        f"리더십 {scores.get('leadership', NEUTRAL_SCORE)}점, "
        f"기획 정리력 {scores.get('planning', NEUTRAL_SCORE)}점, "
        f"문제 해결력 {scores.get('problemSolving', NEUTRAL_SCORE)}점을 기준으로 "
        f"leader_score {get_leader_score(leader)}점입니다."
    )
