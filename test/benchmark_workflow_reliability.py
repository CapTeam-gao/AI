import argparse
import copy
import json
import random
from pathlib import Path

from matching_student.workflow_matching_student import (
    repair_team_matching_result,
    validation_balance_team,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "data/student_analysis_data/matching_output.json"


def load_fixture():
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return data["analyzed_students"], data["final_result"]["final_teams"]


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
    analyzed_students, base_teams = load_fixture()
    rng = random.Random(seed)
    detection_types = (
        "missing",
        "duplicate",
        "unknown",
        "oversized",
        "team_count",
        "empty_team",
        "mixed_structural",
    )
    repair_types = (
        "missing",
        "duplicate",
        "unknown",
        "mixed_structural",
    )

    detection = {}
    for error_type in detection_types:
        detected = 0
        for _ in range(trials):
            corrupted = inject_error(base_teams, error_type, rng)
            detected += not is_valid(corrupted, analyzed_students, base_teams)
        detection[error_type] = {
            "detected": detected,
            "trials": trials,
            "rate_percent": round(detected / trials * 100, 2),
        }

    recovery = {}
    for error_type in repair_types:
        recovered = 0
        for _ in range(trials):
            corrupted = inject_error(base_teams, error_type, rng)
            repaired = repair_team_matching_result(
                {"final_teams": corrupted},
                analyzed_students,
                base_teams,
            )["final_teams"]
            recovered += is_valid(repaired, analyzed_students, base_teams)
        recovery[error_type] = {
            "recovered": recovered,
            "trials": trials,
            "rate_percent": round(recovered / trials * 100, 2),
        }

    false_positives = 0
    for _ in range(trials):
        valid_variant = harmless_permutation(base_teams, rng)
        false_positives += not is_valid(valid_variant, analyzed_students, base_teams)

    total_detected = sum(item["detected"] for item in detection.values())
    total_detection_trials = trials * len(detection_types)
    total_recovered = sum(item["recovered"] for item in recovery.values())
    total_recovery_trials = trials * len(repair_types)

    return {
        "fixture": {
            "students": len(analyzed_students),
            "teams": len(base_teams),
            "members_per_team": [len(team["members"]) for team in base_teams],
        },
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260630)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.trials, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
