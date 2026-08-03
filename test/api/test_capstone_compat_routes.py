import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api.main as api
import matching_student.workflow_matching_student as capstone_workflow
import student_analysis.analysis_llm as analysis


def build_analyzed_students():
    return [
        {
            "user_id": "stu-1",
            "name": "학생1",
            "role": "FRONTEND",
            "skill_level": "중상",
            "stack_score": "React: 7점",
        },
        {
            "user_id": "stu-2",
            "name": "학생2",
            "role": "BACKEND",
            "skill_level": "중",
            "stack_score": "Spring: 6점",
        },
    ]


def build_workflow_result():
    students = build_analyzed_students()
    return {
        "analyzed_students": students,
        "final_result": {
            "final_teams": [
                {
                    "team_name": "팀 1",
                    "members": ["학생1", "학생2"],
                    "leader": "학생1",
                    "role_groups": [
                        {"role_group": "frontend", "count": 1},
                        {"role_group": "backend", "count": 1},
                    ],
                    "strengths": "역할이 균형 있게 구성되었습니다.",
                    "weaknesses": "협업 규칙을 먼저 정해야 합니다.",
                    "reason_cards": [
                        {
                            "title": "역할 균형",
                            "description": "프론트엔드와 백엔드 역할을 함께 배치했습니다.",
                        }
                    ],
                }
            ],
            "changed": False,
            "change_summary": "",
            "validation_notes": "",
        },
    }


class CapstoneCompatibilityRouteTest(unittest.TestCase):
    def test_capstone_workflow_can_skip_default_capstone_storage(self):
        workflow_result = build_workflow_result()

        with (
            patch.object(capstone_workflow, "load_cached_matching_result", return_value=None),
            patch.object(capstone_workflow, "build_initial_state", return_value={"analyzed_students": []}),
            patch.object(capstone_workflow.app, "invoke", return_value=workflow_result),
            patch.object(capstone_workflow, "save_workflow_result") as save_workflow_result,
        ):
            result = capstone_workflow.run_workflow(force_rematch=True, persist_result=False)

        self.assertIs(workflow_result, result)
        save_workflow_result.assert_not_called()

    def test_capstone_regeneration_can_skip_default_capstone_storage(self):
        initial_state = {
            "analyzed_students": [],
            "iteration_count": capstone_workflow.MAX_ITERATION,
        }
        final_result = {"final_result": {"final_teams": []}}

        with (
            patch.object(capstone_workflow, "build_regenerate_state", return_value=initial_state),
            patch.object(capstone_workflow, "finalize_node", return_value=final_result),
            patch.object(capstone_workflow, "save_workflow_result") as save_workflow_result,
        ):
            result = capstone_workflow.run_regenerate_workflow(
                prompt="역할 균형을 다시 확인해줘",
                persist_result=False,
            )

        self.assertEqual(final_result["final_result"], result["final_result"])
        save_workflow_result.assert_not_called()

    def test_hackathon_run_uses_capstone_workflow_and_hackathon_storage(self):
        analyzed_students = build_analyzed_students()
        workflow_result = build_workflow_result()

        with (
            patch.object(analysis, "get_analyze_stu", return_value=analyzed_students),
            patch.object(capstone_workflow, "run_workflow", return_value=workflow_result) as run_workflow,
            patch.object(api, "save_matching_result") as save_result,
        ):
            response = api.run_hackathon_matching({"students": analyzed_students, "teamSize": 5})

        run_workflow.assert_called_once_with(
            force_rematch=True,
            analyzed_students=analyzed_students,
            persist_result=False,
        )
        save_result.assert_called_once_with(
            capstone_workflow.build_public_workflow_result(workflow_result),
            matching_type="HACKATHON",
        )
        self.assertEqual(2, response["total_students"])
        self.assertEqual(1, response["total_teams"])
        self.assertEqual(["학생1", "학생2"], [member["name"] for member in response["teams"][0]["members"]])

    def test_hackathon_regenerate_uses_capstone_workflow_and_current_teams(self):
        analyzed_students = build_analyzed_students()
        workflow_result = build_workflow_result()
        saved_result = build_workflow_result()
        current_teams = saved_result["final_result"]["final_teams"]

        with (
            patch.object(api, "load_matching_output", return_value=saved_result),
            patch.object(analysis, "get_analyze_stu", return_value=analyzed_students),
            patch.object(
                capstone_workflow,
                "run_regenerate_workflow",
                return_value=workflow_result,
            ) as regenerate,
            patch.object(api, "save_matching_result") as save_result,
        ):
            response = api.regenerate_hackathon_matching({
                "students": analyzed_students,
                "prompt": "역할 균형을 다시 확인해줘",
            })

        regenerate.assert_called_once_with(
            prompt="역할 균형을 다시 확인해줘",
            current_teams=current_teams,
            analyzed_students=analyzed_students,
            persist_result=False,
        )
        save_result.assert_called_once_with(
            capstone_workflow.build_public_workflow_result(workflow_result),
            matching_type="HACKATHON",
        )
        self.assertEqual(1, response["total_teams"])

    def test_hackathon_summary_uses_common_capstone_summary_shape(self):
        saved_result = build_workflow_result()

        with patch.object(api, "load_matching_output", return_value=saved_result) as load_result:
            response = api.hackathon_matching_summary()

        load_result.assert_called_once_with("HACKATHON")
        self.assertEqual(1, response["total_teams"])
        self.assertIn("skill_level_counts", response["teams"][0])
        self.assertIn("reason_cards", response["teams"][0])


if __name__ == "__main__":
    unittest.main()
