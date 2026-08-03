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

    def test_regenerate_groups_five_game_students_when_requested(self):
        students = build_students(15)
        for index in range(5):
            students[index]["role"] = "game"
        normalized = matcher.validate_and_normalize_students(students)
        summaries = [matcher.make_student_summary(student) for student in normalized]
        current_teams = [
            {
                "team_name": "팀 1",
                "capacity": 5,
                "members": [summaries[0], summaries[5], summaries[6], summaries[7], summaries[8]],
            },
            {
                "team_name": "팀 2",
                "capacity": 5,
                "members": [summaries[1], summaries[2], summaries[9], summaries[10], summaries[11]],
            },
            {
                "team_name": "팀 3",
                "capacity": 5,
                "members": [summaries[3], summaries[4], summaries[12], summaries[13], summaries[14]],
            },
        ]
        raw_same_as_current = {
            "teams": [
                {"team_name": team["team_name"], "members": [member["name"] for member in team["members"]]}
                for team in current_teams
            ],
            "change_summary": "기존 팀을 유지했습니다.",
        }

        with (
            patch("matching_student.hackerton_matching.is_llm_enabled", return_value=True),
            patch("matching_student.hackerton_matching._request_user_regeneration", return_value=raw_same_as_current),
        ):
            result = matcher.run_regenerate_workflow(
                prompt="게임 역할 학생들은 5명씩 붙여줘",
                current_teams=current_teams,
                analyzed_students=normalized,
            )

        final_teams = result["final_result"]["final_teams"]
        game_counts = [
            sum(member["role_group"] == "game" for member in team["members"])
            for team in final_teams
        ]
        self.assertIn(5, game_counts)
        self.assertTrue(result["final_result"]["changed"])
        self.assertEqual("validated_regeneration", result["final_result"]["finalized_by"])

    def test_initial_matching_groups_mutual_preferred_members_when_balance_allows(self):
        students = build_students(10)
        students[0]["preferred_members"] = [students[9]["name"]]
        students[9]["preferred_members"] = [students[0]["name"]]

        normalized = matcher.validate_and_normalize_students(students)
        teams = matcher.create_initial_teams(normalized, team_size=5)
        team_by_name = {
            member["name"]: team["team_name"]
            for team in teams
            for member in team["members"]
        }

        self.assertEqual(team_by_name[students[0]["name"]], team_by_name[students[9]["name"]])
        self.assertEqual(
            0,
            matcher._unmet_preference_count(
                teams,
                {student["name"] for student in map(matcher.make_student_summary, normalized)},
            ),
        )

    def test_preferred_member_ids_are_normalized_like_capstone(self):
        students = build_students(10)
        students[0]["preferred_members"] = [students[9]["user_id"]]
        students[9]["preferredMembers"] = [students[0]["user_id"]]
        students[0]["wantsLeader"] = True

        normalized = matcher.validate_and_normalize_students(students)
        teams = matcher.create_initial_teams(normalized, team_size=5)
        team_by_name = {
            member["name"]: team["team_name"]
            for team in teams
            for member in team["members"]
        }

        self.assertEqual([students[9]["name"]], normalized[0]["preferred_members"])
        self.assertEqual([students[0]["name"]], normalized[9]["preferred_members"])
        self.assertEqual(team_by_name[students[0]["name"]], team_by_name[students[9]["name"]])

        result = matcher.run_workflow(students, team_size=5)
        preferred_team = next(
            team
            for team in result["final_result"]["final_teams"]
            if students[0]["name"] in {member["name"] for member in team["members"]}
        )
        self.assertTrue(preferred_team["preference_notes"])
        self.assertEqual(students[0]["name"], preferred_team["leader"])

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

    def test_legacy_hackathon_matcher_contract_without_db(self):
        import api.main as api

        result = matcher.run_workflow(build_students(10), team_size=5)
        response = api.build_hackathon_summary(result)

        self.assertEqual("HACKATHON", response["matching_type"])
        self.assertEqual(10, response["total_students"])
        self.assertEqual([5, 5], [team["total_people"] for team in response["teams"]])

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
