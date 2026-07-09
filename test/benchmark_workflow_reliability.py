import argparse
import copy
import json
import random
import re
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import patch

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capteam_traits import ensure_trait_profile
from matching_student.workflow_matching_student import (
    MAX_ITERATION,
    TeamMatchingResult,
    adjust_team_node,
    build_initial_state,
    build_reason_context,
    create_initial_teams,
    evaluate_balance_node,
    get_llm,
    get_member_names,
    get_matching_prompt_chain,
    get_role_group,
    get_student_score,
    get_student_names,
    repair_team_matching_result,
    should_adjust,
    validation_balance_team,
)


FIXTURE_PATH = PROJECT_ROOT / "data/student_analysis_data/matching_output.json"
DEFAULT_RESULT_PATH = PROJECT_ROOT / "data/benchmark_results/workflow_effect_20260701.json"


def load_fixture():
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return data["analyzed_students"], data["final_result"]["final_teams"]


def build_cohort(label, students, scenario):
    cohort = copy.deepcopy(students)
    name_map = {
        student.get("name"): f"{label}-{index + 1:02d}"
        for index, student in enumerate(cohort)
    }

    for index, student in enumerate(cohort):
        student["name"] = name_map[student.get("name")]
        student["user_id"] = f"{label.lower()}-{index + 1:02d}"
        student["preferred_members"] = []
        if str(student.get("role") or student.get("goal")).upper() == "GAME":
            student["role"] = "DEVOPS"
            student["goal"] = "DEVOPS"

    if scenario == "role_skew":
        roles = ["BACKEND"] * 10 + ["FRONTEND"] * 6 + ["AI"] * 2 + ["APP"] * 2
        for student, role in zip(cohort, roles):
            student["role"] = role
            student["goal"] = role
    elif scenario == "skill_skew":
        levels = ["상"] * 4 + ["중상"] * 4 + ["중"] * 4 + ["중하"] * 4 + ["하"] * 4
        for student, level in zip(cohort, levels):
            student["skill_level"] = level
    elif scenario == "preference_conflict":
        for index, student in enumerate(cohort):
            student["preferred_members"] = [cohort[(index + 7) % len(cohort)]["name"]]
    elif scenario == "trait_reliability":
        reliabilities = ["HIGH", "MEDIUM", "LOW", "HIGH", "MEDIUM"]
        for index, student in enumerate(cohort):
            student["communication"] = 1 + index % 5
            student["responsibility"] = 5 - index % 5
            student["collaboration"] = 1 + (index * 2) % 5
            student["flexibility"] = 1 + (index * 3) % 5
            student["response_reliability"] = reliabilities[index % len(reliabilities)]

    cohort = [ensure_trait_profile(student) for student in cohort]
    initial_teams = create_initial_teams(cohort)
    base_teams = [
        {
            "team_name": team.get("team_name"),
            "members": get_member_names(team),
            "total_score": team.get("total_score"),
            "leader": team.get("leader"),
        }
        for team in initial_teams
    ]
    return {
        "name": scenario,
        "students": cohort,
        "teams": base_teams,
    }


def build_cohorts(students):
    scenarios = (
        "balanced",
        "role_skew",
        "skill_skew",
        "preference_conflict",
        "trait_reliability",
    )
    return [
        build_cohort(f"C{index + 1}", students, scenario)
        for index, scenario in enumerate(scenarios)
    ]


def choose_two_teams(teams, rng):
    source_index, target_index = rng.sample(range(len(teams)), 2)
    return teams[source_index], teams[target_index]


def inject_error(base_teams, error_type, rng):
    teams = copy.deepcopy(base_teams)
    source, target = choose_two_teams(teams, rng)

    if error_type == "missing":
        source["members"].pop(rng.randrange(len(source["members"])))
    elif error_type == "duplicate":
        target["members"].append(rng.choice(source["members"]))
    elif error_type == "unknown":
        source["members"][rng.randrange(len(source["members"]))] = "존재하지않는학생"
    elif error_type == "oversized":
        target["members"].append(source["members"].pop())
    elif error_type == "team_count":
        teams.pop()
    elif error_type == "empty_team":
        target["members"].extend(source["members"])
        source["members"] = []
    elif error_type == "mixed_structural":
        removed_name = source["members"].pop()
        target["members"].append(rng.choice(target["members"]))
        source["members"].append("존재하지않는학생")
        if removed_name == "존재하지않는학생":
            raise AssertionError("fixture unexpectedly contains the synthetic unknown name")
    else:
        raise ValueError(f"지원하지 않는 오류 유형입니다: {error_type}")

    return teams


def is_valid(teams, analyzed_students, base_teams):
    balance_result, _ = validation_balance_team(
        teams,
        analyzed_students,
        base_teams=base_teams,
    )
    return balance_result["is_balanced"] and not balance_result["need_adjustment"]


def harmless_permutation(base_teams, rng):
    teams = copy.deepcopy(base_teams)
    rng.shuffle(teams)
    for team in teams:
        rng.shuffle(team["members"])
    return teams


def run_benchmark(trials, seed):
    fixture_students, _ = load_fixture()
    cohorts = build_cohorts(fixture_students)
    rng = random.Random(seed)
    detection_types = (
        "missing",
        "duplicate",
        "unknown",
        "oversized",
        "team_count",
    )
    repair_types = (
        "missing",
        "duplicate",
        "unknown",
        "team_count",
    )

    invalid_cohorts = []
    for cohort in cohorts:
        if not is_valid(cohort["teams"], cohort["students"], cohort["teams"]):
            invalid_cohorts.append(cohort["name"])
    if invalid_cohorts:
        raise AssertionError(f"정상 코호트가 검증을 통과하지 못했습니다: {invalid_cohorts}")

    detection = {}
    for error_type in detection_types:
        detected = 0
        for trial in range(trials):
            cohort = cohorts[trial % len(cohorts)]
            corrupted = inject_error(cohort["teams"], error_type, rng)
            detected += not is_valid(corrupted, cohort["students"], cohort["teams"])
        detection[error_type] = {
            "detected": detected,
            "trials": trials,
            "rate_percent": round(detected / trials * 100, 2),
        }

    recovery = {}
    for error_type in repair_types:
        recovered = 0
        for trial in range(trials):
            cohort = cohorts[trial % len(cohorts)]
            corrupted = inject_error(cohort["teams"], error_type, rng)
            repaired = repair_team_matching_result(
                {"final_teams": corrupted},
                cohort["students"],
                cohort["teams"],
            )["final_teams"]
            recovered += is_valid(repaired, cohort["students"], cohort["teams"])
        recovery[error_type] = {
            "recovered": recovered,
            "trials": trials,
            "rate_percent": round(recovered / trials * 100, 2),
        }

    false_positives = 0
    for trial in range(trials):
        cohort = cohorts[trial % len(cohorts)]
        valid_variant = harmless_permutation(cohort["teams"], rng)
        false_positives += not is_valid(valid_variant, cohort["students"], cohort["teams"])

    total_detected = sum(item["detected"] for item in detection.values())
    total_detection_trials = trials * len(detection_types)
    total_recovered = sum(item["recovered"] for item in recovery.values())
    total_recovery_trials = trials * len(repair_types)

    return {
        "cohorts": [
            {
                "name": cohort["name"],
                "students": len(cohort["students"]),
                "teams": len(cohort["teams"]),
                "members_per_team": [len(team["members"]) for team in cohort["teams"]],
            }
            for cohort in cohorts
        ],
        "seed": seed,
        "trials_per_case": trials,
        "detection": detection,
        "detection_total": {
            "detected": total_detected,
            "trials": total_detection_trials,
            "rate_percent": round(total_detected / total_detection_trials * 100, 2),
        },
        "recovery": recovery,
        "recovery_total": {
            "recovered": total_recovered,
            "trials": total_recovery_trials,
            "rate_percent": round(total_recovered / total_recovery_trials * 100, 2),
        },
        "valid_input_false_positive": {
            "false_positives": false_positives,
            "trials": trials,
            "rate_percent": round(false_positives / trials * 100, 2),
        },
    }


def aggregate_benchmarks(results):
    detection = {}
    recovery = {}
    for error_type in results[0]["detection"]:
        detected = sum(result["detection"][error_type]["detected"] for result in results)
        trials = sum(result["detection"][error_type]["trials"] for result in results)
        detection[error_type] = {
            "detected": detected,
            "trials": trials,
            "rate_percent": round(detected / trials * 100, 2),
        }
    for error_type in results[0]["recovery"]:
        recovered = sum(result["recovery"][error_type]["recovered"] for result in results)
        trials = sum(result["recovery"][error_type]["trials"] for result in results)
        recovery[error_type] = {
            "recovered": recovered,
            "trials": trials,
            "rate_percent": round(recovered / trials * 100, 2),
        }

    detected_total = sum(item["detected"] for item in detection.values())
    detection_trials = sum(item["trials"] for item in detection.values())
    recovered_total = sum(item["recovered"] for item in recovery.values())
    recovery_trials = sum(item["trials"] for item in recovery.values())
    false_positives = sum(
        result["valid_input_false_positive"]["false_positives"]
        for result in results
    )
    valid_trials = sum(
        result["valid_input_false_positive"]["trials"]
        for result in results
    )

    return {
        "cohorts": results[0]["cohorts"],
        "seeds": [result["seed"] for result in results],
        "trials_per_case_per_seed": results[0]["trials_per_case"],
        "detection": detection,
        "detection_total": {
            "detected": detected_total,
            "trials": detection_trials,
            "rate_percent": round(detected_total / detection_trials * 100, 2),
        },
        "recovery": recovery,
        "recovery_total": {
            "recovered": recovered_total,
            "trials": recovery_trials,
            "rate_percent": round(recovered_total / recovery_trials * 100, 2),
        },
        "valid_input_false_positive": {
            "false_positives": false_positives,
            "trials": valid_trials,
            "rate_percent": round(false_positives / valid_trials * 100, 2),
        },
    }


class BenchmarkUsageHandler(UsageMetadataCallbackHandler):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def on_llm_end(self, response, **kwargs):
        self.call_count += 1
        return super().on_llm_end(response, **kwargs)


def summarize_usage(handler):
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for usage in handler.usage_metadata.values():
        input_tokens += int(usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("output_tokens", 0) or 0)
        total_tokens += int(usage.get("total_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens or input_tokens + output_tokens,
        "llm_calls": handler.call_count,
    }


def calculate_preference_rate(teams, students):
    team_by_name = {}
    for team_index, team in enumerate(teams):
        for name in get_member_names(team):
            team_by_name[name] = team_index
    pairs = []
    for student in students:
        for preferred_name in student.get("preferred_members", []):
            pair = tuple(sorted((student.get("name"), preferred_name)))
            if all(pair) and pair not in pairs:
                pairs.append(pair)
    if not pairs:
        return None
    satisfied = sum(
        team_by_name.get(first) is not None
        and team_by_name.get(first) == team_by_name.get(second)
        for first, second in pairs
    )
    return round(satisfied / len(pairs) * 100, 2)


def calculate_role_diversity(teams, students):
    roles = {
        student.get("name"): get_role_group(student.get("role") or student.get("goal"))
        for student in students
    }
    values = [
        len({roles.get(name) for name in get_member_names(team) if roles.get(name)})
        for team in teams
    ]
    return round(sum(values) / len(values), 2) if values else 0


def calculate_score_gap(teams, students):
    scores = {
        student.get("name"): get_student_score(student)
        for student in students
    }
    totals = [
        sum(scores.get(name, 0) for name in get_member_names(team))
        for team in teams
    ]
    return round(max(totals) - min(totals), 2) if totals else 0


def structural_metrics(candidate_result, students, base_teams):
    result, _ = validation_balance_team(candidate_result, students, base_teams=base_teams)
    teams = candidate_result.get("final_teams", []) if isinstance(candidate_result, dict) else candidate_result
    return {
        "valid": result["is_balanced"] and not result["need_adjustment"],
        "error_count": len(result.get("errors", [])),
        "missing_count": len(result.get("missing_names", [])),
        "duplicate_count": len(result.get("duplicate_names", [])),
        "unknown_count": len(result.get("unknown_names", [])),
        "team_count": result.get("team_count", 0),
        "member_counts": result.get("member_counts", []),
        "score_gap": calculate_score_gap(teams, students),
        "average_role_diversity": calculate_role_diversity(teams, students),
        "preference_satisfaction_percent": calculate_preference_rate(teams, students),
    }


def get_legacy_prompt_chain():
    system_prompt = """
당신은 학생들의 코딩 실력 분석 결과를 바탕으로 캡스톤 프로젝트 팀을 매칭하는 챗봇이다.
아래 학생 분석 결과와 1차 팀 배정안을 참고해서 균형 잡힌 최종 팀을 만들어라.

매칭 기준:
- 각 팀의 전체 실력 합계가 너무 차이 나지 않게 한다.
- 가능하면 프론트엔드, 백엔드, AI/데이터, 기타 역할이 한 팀에 몰리지 않게 한다.
- 낮음 학생은 보통 또는 높음 학생과 함께 배치한다.
- 같은 역할만 모인 팀은 피한다.

이름 사용 규칙:
- 아래 allowed_student_names에 있는 이름만 사용한다.
- 학생 이름은 한 글자도 바꾸지 말고 그대로 복사한다.
- 모든 학생은 정확히 한 팀에만 들어가야 한다.

출력 형식:
팀 1:
- 팀원: 이름 목록
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
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])


def parse_legacy_teams(text, allowed_names):
    matches = list(re.finditer(r"(?m)^\s*(?:#+\s*)?팀\s*(\d+)\s*[:：]", text or ""))
    teams = []
    parse_errors = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end():end]
        member_match = re.search(
            r"(?mi)^\s*[-*]?\s*(?:팀원|구성원)\s*[:：]\s*(.+)$",
            block,
        )
        if not member_match:
            parse_errors.append(f"팀 {match.group(1)}의 팀원 줄을 찾지 못했습니다.")
            members = []
        else:
            member_line = member_match.group(1)
            members = [name for name in allowed_names if name in member_line]
        teams.append({
            "team_name": f"팀 {match.group(1)}",
            "members": members,
        })
    if not matches:
        parse_errors.append("팀 구분을 찾지 못했습니다.")
    return teams, parse_errors


def invoke_legacy_matching(students, base_teams, callback, run_tag):
    allowed_names = get_student_names(students)
    chain = get_legacy_prompt_chain() | get_llm()
    inputs = {
        "allowed_student_names": json.dumps(allowed_names, ensure_ascii=False),
        "student_analysis": json.dumps(students, ensure_ascii=False, indent=0),
        "initial_teams": json.dumps(base_teams, ensure_ascii=False, indent=0),
        "input": "1차 팀 배정안을 바탕으로 균형 잡힌 최종 프로젝트 팀을 만들어줘.",
    }
    response = chain.invoke(
        inputs,
        config={"callbacks": [callback], "tags": [run_tag, "legacy-matching"]},
    )
    text = response.content
    first_pass_missing = [name for name in allowed_names if name not in text]
    retried = False
    if first_pass_missing:
        retried = True
        inputs["input"] = (
            "이전 응답에서 학생 이름이 누락되거나 다르게 작성되었다. "
            f"누락된 이름: {', '.join(first_pass_missing)}. "
            "allowed_student_names의 이름만 정확히 사용해서 다시 작성해줘."
        )
        response = chain.invoke(
            inputs,
            config={"callbacks": [callback], "tags": [run_tag, "legacy-retry"]},
        )
        text = response.content
    teams, parse_errors = parse_legacy_teams(text, allowed_names)
    return {
        "candidate": {"final_teams": teams},
        "raw_output": text,
        "first_pass_valid": not first_pass_missing and not parse_errors,
        "retried": retried,
        "parse_errors": parse_errors,
    }


def invoke_current_draft(students, base_teams, callback, run_tag):
    allowed_names = get_student_names(students)
    chain = get_matching_prompt_chain() | get_llm().with_structured_output(TeamMatchingResult)
    response = chain.invoke(
        {
            "allowed_student_names": json.dumps(allowed_names, ensure_ascii=False),
            "student_analysis": json.dumps(students, ensure_ascii=False, indent=0),
            "initial_teams": json.dumps(base_teams, ensure_ascii=False, indent=0),
            "reason_context": json.dumps(build_reason_context(base_teams, students), ensure_ascii=False, indent=0),
        },
        config={"callbacks": [callback], "tags": [run_tag, "workflow-draft"]},
    )
    return response.model_dump() if hasattr(response, "model_dump") else response


def invoke_workflow_core(students, base_teams, callback, run_tag):
    original_get_llm = get_llm

    def instrumented_get_llm(model=None):
        llm = original_get_llm(model=model)
        llm.callbacks = [callback]
        llm.tags = [run_tag, "with-workflow"]
        return llm

    state = build_initial_state(analyzed_students=students)
    with patch("matching_student.workflow_matching_student.get_llm", instrumented_get_llm):
        raw_draft = invoke_current_draft(students, base_teams, callback, run_tag)
        initial_metrics = structural_metrics(raw_draft, students, base_teams)
        repaired_draft = repair_team_matching_result(raw_draft, students, base_teams)
        state = {**state, "teams": base_teams, "llm_result": repaired_draft}
        state = {**state, **evaluate_balance_node(state)}
        while should_adjust(state) == "adjust_team_node" and state["iteration_count"] < MAX_ITERATION:
            state = {**state, **adjust_team_node(state)}
            state = {**state, **evaluate_balance_node(state)}
    candidate = state.get("llm_result") or state.get("teams")
    final_metrics = structural_metrics(candidate, students, base_teams)
    finalized_by = "validation_passed"
    if not final_metrics["valid"]:
        candidate = {"final_teams": [
            {
                "team_name": team.get("team_name"),
                "members": get_member_names(team),
            }
            for team in base_teams
        ]}
        finalized_by = "algorithm_after_validation_failure"
    return {
        **state,
        "raw_draft": raw_draft,
        "initial_metrics": initial_metrics,
        "candidate": candidate,
        "finalized_by": finalized_by,
    }


def empty_failure_metrics():
    return {
        "valid": False,
        "error_count": 1,
        "missing_count": 0,
        "duplicate_count": 0,
        "unknown_count": 0,
        "team_count": 0,
        "member_counts": [],
        "score_gap": 0,
        "average_role_diversity": 0,
        "preference_satisfaction_percent": None,
    }


def average_optional(rows, key):
    values = [row["metrics"][key] for row in rows if row["metrics"][key] is not None]
    return round(sum(values) / len(values), 2) if values else None


def improvement_percent(before, after, lower_is_better=False):
    if before in (None, 0):
        return None
    change = (after - before) / before * 100
    return round(-change if lower_is_better else change, 2)


def summarize_ab_rows(rows):
    summary = {}
    for mode in ("legacy_llm", "workflow"):
        selected = [row for row in rows if row["mode"] == mode]
        successes = sum(row["metrics"]["valid"] for row in selected)
        first_passes = sum(row["first_pass_valid"] for row in selected)
        recovered = sum(
            not row["first_pass_valid"] and row["metrics"]["valid"]
            for row in selected
        )
        initially_invalid = len(selected) - first_passes
        summary[mode] = {
            "runs": len(selected),
            "complete_assignment_success_rate_percent": round(successes / len(selected) * 100, 2),
            "structural_error_rate_percent": round((len(selected) - successes) / len(selected) * 100, 2),
            "first_pass_rate_percent": round(first_passes / len(selected) * 100, 2),
            "automatic_recovery_rate_percent": (
                round(recovered / initially_invalid * 100, 2) if initially_invalid else None
            ),
            "stable_termination_rate_percent": round(
                sum(row["stable_termination"] for row in selected) / len(selected) * 100,
                2,
            ),
            "average_preference_satisfaction_percent": average_optional(selected, "preference_satisfaction_percent"),
            "average_role_diversity": average_optional(selected, "average_role_diversity"),
            "average_team_score_gap": average_optional(selected, "score_gap"),
            "average_latency_seconds": round(sum(row["latency_seconds"] for row in selected) / len(selected), 3),
            "average_tokens": round(sum(row["usage"]["total_tokens"] for row in selected) / len(selected), 1),
            "average_llm_calls": round(sum(row["usage"]["llm_calls"] for row in selected) / len(selected), 2),
            "runtime_failures": sum(bool(row["runtime_error"]) for row in selected),
        }
    legacy = summary["legacy_llm"]
    workflow = summary["workflow"]
    summary["improvement"] = {
        "complete_assignment_success_percentage_points": round(
            workflow["complete_assignment_success_rate_percent"]
            - legacy["complete_assignment_success_rate_percent"], 2
        ),
        "structural_error_reduction_percent": improvement_percent(
            legacy["structural_error_rate_percent"],
            workflow["structural_error_rate_percent"],
            lower_is_better=True,
        ),
        "preference_satisfaction_percentage_points": (
            round(
                workflow["average_preference_satisfaction_percent"]
                - legacy["average_preference_satisfaction_percent"], 2
            )
            if legacy["average_preference_satisfaction_percent"] is not None
            and workflow["average_preference_satisfaction_percent"] is not None
            else None
        ),
        "team_score_gap_reduction_percent": improvement_percent(
            legacy["average_team_score_gap"],
            workflow["average_team_score_gap"],
            lower_is_better=True,
        ),
    }
    return summary


def run_fault_injection_from_llm(rows, cohorts, trials_per_type, seed):
    workflow_rows = [row for row in rows if row["mode"] == "workflow" and row.get("final_teams")]
    cohort_lookup = {cohort["name"]: cohort for cohort in cohorts}
    base_teams_lookup = {
        cohort["name"]: create_initial_teams(cohort["students"])
        for cohort in cohorts
    }
    error_types = (
        "missing", "duplicate", "unknown", "team_count",
        "oversized", "empty_team", "mixed_structural",
    )
    rng = random.Random(seed)
    details = {}
    for error_type in error_types:
        detected = 0
        recovered = 0
        blocked = 0
        for trial in range(trials_per_type):
            source = workflow_rows[trial % len(workflow_rows)]
            cohort = cohort_lookup[source["cohort"]]
            base_teams = base_teams_lookup[source["cohort"]]
            corrupted = inject_error(source["final_teams"], error_type, rng)
            corrupted_valid = is_valid(corrupted, cohort["students"], base_teams)
            detected += not corrupted_valid
            repaired = repair_team_matching_result(
                {"final_teams": corrupted}, cohort["students"], base_teams
            )["final_teams"]
            repaired_valid = is_valid(repaired, cohort["students"], base_teams)
            recovered += repaired_valid
            blocked += repaired_valid or is_valid(base_teams, cohort["students"], base_teams)
        details[error_type] = {
            "trials": trials_per_type,
            "detected": detected,
            "detection_rate_percent": round(detected / trials_per_type * 100, 2),
            "directly_recovered": recovered,
            "direct_recovery_rate_percent": round(recovered / trials_per_type * 100, 2),
            "blocked_from_user": blocked,
            "user_exposure_block_rate_percent": round(blocked / trials_per_type * 100, 2),
        }
    valid_false_positives = 0
    for trial in range(trials_per_type):
        source = workflow_rows[trial % len(workflow_rows)]
        cohort = cohort_lookup[source["cohort"]]
        base_teams = base_teams_lookup[source["cohort"]]
        valid_false_positives += not is_valid(
            harmless_permutation(source["final_teams"], rng),
            cohort["students"],
            base_teams,
        )
    total_trials = trials_per_type * len(error_types)
    return {
        "source": "actual gpt-5.4 workflow drafts with controlled synthetic errors",
        "error_types": details,
        "total": {
            "trials": total_trials,
            "detection_rate_percent": round(
                sum(item["detected"] for item in details.values()) / total_trials * 100, 2
            ),
            "direct_recovery_rate_percent": round(
                sum(item["directly_recovered"] for item in details.values()) / total_trials * 100, 2
            ),
            "user_exposure_block_rate_percent": round(
                sum(item["blocked_from_user"] for item in details.values()) / total_trials * 100, 2
            ),
            "valid_input_false_positive_rate_percent": round(
                valid_false_positives / trials_per_type * 100, 2
            ),
        },
    }


def run_llm_ab(samples_per_cohort, seed):
    fixture_students, _ = load_fixture()
    cohorts = build_cohorts(fixture_students)
    run_tag = f"workflow-ab-{uuid.uuid4().hex[:10]}"
    rows = []

    for cohort_index, cohort in enumerate(cohorts):
        for sample_index in range(samples_per_cohort):
            for mode in ("legacy_llm", "workflow"):
                callback = BenchmarkUsageHandler()
                started = time.perf_counter()
                base_teams = create_initial_teams(cohort["students"])
                try:
                    if mode == "legacy_llm":
                        legacy = invoke_legacy_matching(
                            cohort["students"], base_teams, callback, run_tag
                        )
                        candidate = legacy["candidate"]
                        first_pass_valid = legacy["first_pass_valid"]
                        raw_output = legacy["raw_output"]
                        iterations = 0
                        finalized_by = "legacy_response"
                    else:
                        state = invoke_workflow_core(
                            cohort["students"], base_teams, callback, run_tag
                        )
                        candidate = state["candidate"]
                        first_pass_valid = state["initial_metrics"]["valid"]
                        raw_output = state["raw_draft"]
                        iterations = state.get("iteration_count", 0)
                        finalized_by = state["finalized_by"]
                    metrics = structural_metrics(
                        candidate, cohort["students"], base_teams
                    )
                    error = ""
                except Exception as exc:
                    metrics = empty_failure_metrics()
                    first_pass_valid = False
                    raw_output = None
                    iterations = 0
                    finalized_by = "runtime_error"
                    error = f"{type(exc).__name__}: {exc}"
                rows.append({
                    "mode": mode,
                    "cohort": cohort["name"],
                    "sample": sample_index + 1,
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    "iterations": iterations,
                    "first_pass_valid": first_pass_valid,
                    "stable_termination": not error and iterations <= MAX_ITERATION,
                    "finalized_by": finalized_by,
                    "usage": summarize_usage(callback),
                    "metrics": metrics,
                    "final_teams": candidate.get("final_teams", []) if not error else [],
                    "raw_output": raw_output,
                    "runtime_error": error,
                })
                print(
                    f"[{len(rows)}/{len(cohorts) * samples_per_cohort * 2}] "
                    f"{mode} {cohort['name']} sample={sample_index + 1} "
                    f"valid={metrics['valid']} latency={rows[-1]['latency_seconds']}s",
                    file=sys.stderr,
                )

    summary = summarize_ab_rows(rows)
    fault_injection = None
    if (
        summary["legacy_llm"]["complete_assignment_success_rate_percent"] == 100
        and summary["workflow"]["complete_assignment_success_rate_percent"] == 100
    ):
        fault_injection = run_fault_injection_from_llm(rows, cohorts, 50, seed)
    return {
        "benchmark": "historical legacy LLM matching vs validation-repair workflow",
        "model": "gpt-5.4 unless OPENAI_MATCHING_MODEL overrides it",
        "students_per_run": len(cohorts[0]["students"]),
        "cohorts": [cohort["name"] for cohort in cohorts],
        "samples_per_cohort": samples_per_cohort,
        "langsmith_tag": run_tag,
        "summary": summary,
        "fault_injection_if_normal_ab_tied": fault_injection,
        "runs": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--seed-runs", type=int, default=5)
    parser.add_argument("--llm-ab", action="store_true")
    parser.add_argument("--samples-per-cohort", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.llm_ab:
        result = run_llm_ab(args.samples_per_cohort, args.seed)
        rendered = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered)
        return
    results = [
        run_benchmark(args.trials, args.seed + offset)
        for offset in range(args.seed_runs)
    ]
    print(json.dumps(aggregate_benchmarks(results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
