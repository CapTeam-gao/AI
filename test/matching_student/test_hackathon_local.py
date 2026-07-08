"""Local smoke tests for the hackathon matcher.

Default execution is deterministic and does not use MySQL or OpenAI:
    python3 test/matching_student/test_hackathon_local.py -v

To include real OpenAI final-description calls:
    RUN_HACKATHON_LLM_TEST=true python3 test/matching_student/test_hackathon_local.py -v
"""

import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HACKATHON_MATCHING_LLM_ENABLED", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

import matching_student.hackerton_matching as matcher


PERSONALITY_KEYS = [
    "ideaPlanning",
    "communication",
    "roleFlexibility",
    "timePressure",
    "staminaFocus",
]
DEVELOPMENT_KEYS = [
    "implementation",
    "problemSolving",
    "completionQuality",
    "presentation",
    "leadership",
]


def build_students(count=20):
    levels = ["상", "중상", "중", "중하", "하"]
    roles = ["frontend", "backend", "ai", "design", "app"]
    students = []
    for index in range(count):
        students.append({
            "user_id": f"local-{index + 1}",
            "name": f"로컬학생{index + 1:02d}",
            "skill_level": levels[index % len(levels)],
            "stack_score": f"Python: {5 + index % 5}점",
            "role": roles[index % len(roles)],
            "personalityScores": {
                key: 1 + ((index + offset) % 5)
                for offset, key in enumerate(PERSONALITY_KEYS)
            },
            "developmentScores": {
                key: 1 + ((index + offset + 2) % 5)
                for offset, key in enumerate(DEVELOPMENT_KEYS)
            },
        })
    return students


class HackathonLocalTest(unittest.TestCase):
    def setUp(self):
        os.environ["HACKATHON_MATCHING_LLM_ENABLED"] = "false"

    def test_full_workflow_without_db_or_openai(self):
        students = build_students(20)
        result = matcher.run_workflow(students, team_size=5)
        final_result = result["final_result"]
        teams = final_result["final_teams"]
        names = [member["name"] for team in teams for member in team["members"]]

        self.assertEqual([5, 5, 5, 5], [len(team["members"]) for team in teams])
        self.assertEqual(20, len(names))
        self.assertEqual(20, len(set(names)))
        self.assertLessEqual(final_result["balance_result"]["technical_spread"], 2.0)
        self.assertLessEqual(final_result["balance_result"]["execution_spread"], 1.0)

        required_fields = {
            "leader",
            "presentation_candidate",
            "planning_candidate",
            "flexible_supporter",
            "strengths",
            "weaknesses",
            "reason_cards",
            "reason",
        }
        for team in teams:
            self.assertTrue(required_fields.issubset(team))
            self.assertTrue(team["strengths"])
            self.assertTrue(team["weaknesses"])
            self.assertGreaterEqual(len(team["reason_cards"]), 2)

    def test_missing_hackathon_score_is_rejected(self):
        student = build_students(1)[0]
        del student["personalityScores"]["timePressure"]
        with self.assertRaisesRegex(ValueError, "timePressure"):
            matcher.run_workflow([student])

    def test_skill_unbalanced_candidate_is_rejected(self):
        students = matcher.validate_and_normalize_students(build_students(10))
        draft = matcher.create_initial_teams(students, team_size=5)
        baseline = matcher.validate_teams(draft, students)
        members = sorted(
            [member for team in draft for member in team["members"]],
            key=lambda member: member["technical_score"],
            reverse=True,
        )
        unbalanced = [
            {"team_name": "팀 1", "capacity": 5, "members": members[:5]},
            {"team_name": "팀 2", "capacity": 5, "members": members[5:]},
        ]
        validation = matcher.validate_teams(
            unbalanced,
            students,
            {
                "technical_spread": baseline["technical_spread"],
                "execution_spread": baseline["execution_spread"],
            },
        )
        self.assertFalse(validation["is_valid"])
        self.assertFalse(validation["technical_preserved"])

    def test_parallel_batch_merge_keeps_team_order(self):
        teams = [{"team_name": f"팀 {index}"} for index in range(1, 8)]

        def worker(batch):
            first_index = int(batch[0]["team_name"].split()[-1])
            time.sleep((8 - first_index) * 0.002)
            return [{**team, "processed": True} for team in batch]

        os.environ["LOCAL_TEST_WORKERS"] = "3"
        os.environ["LOCAL_TEST_BATCH_SIZE"] = "2"
        result = matcher._run_parallel_team_batches(
            teams,
            worker,
            "LOCAL_TEST_WORKERS",
            "LOCAL_TEST_BATCH_SIZE",
            "로컬 테스트",
        )
        expected = [team["team_name"] for team in teams]
        self.assertEqual(expected, [team["team_name"] for team in result])
        self.assertTrue(all(team["processed"] for team in result))

    def test_local_api_run_contract_without_db(self):
        import api.main as api
        import student_analysis.analysis_llm as analysis

        saved = []
        with patch.object(analysis, "get_analyze_stu", side_effect=lambda students: students), patch.object(
            api,
            "save_matching_result",
            side_effect=lambda result, matching_type="CAPSTONE": saved.append(matching_type),
        ):
            response = api.run_hackathon_matching({
                "students": build_students(10),
                "teamSize": 5,
            })

        self.assertEqual("HACKATHON", response["matching_type"])
        self.assertEqual(10, response["total_students"])
        self.assertEqual([5, 5], [team["total_people"] for team in response["teams"]])
        self.assertEqual(["HACKATHON"], saved)

    @unittest.skipUnless(
        os.getenv("RUN_HACKATHON_LLM_TEST", "false").lower() == "true",
        "RUN_HACKATHON_LLM_TEST=true일 때만 실제 OpenAI 호출을 검사합니다.",
    )
    def test_real_openai_final_explanations(self):
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY가 없습니다.")
        os.environ["HACKATHON_MATCHING_LLM_ENABLED"] = "true"
        result = matcher.run_workflow(build_students(5), team_size=5)
        team = result["final_result"]["final_teams"][0]
        self.assertTrue(team["strengths"])
        self.assertTrue(team["weaknesses"])
        self.assertGreaterEqual(len(team["reason_cards"]), 2)
        self.assertNotIn("analysis_generation_error", team)
        self.assertNotIn("reason_generation_error", team)


if __name__ == "__main__":
    unittest.main()
