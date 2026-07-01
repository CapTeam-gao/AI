import unittest
from unittest.mock import patch

from matching_student.workflow_matching_student import (
    MAX_ITERATION,
    build_candidate_quality_score,
    evaluate_normal_workflow_node,
    finalize_normal_workflow_node,
    run_regenerate_workflow,
    should_adjust_normal_workflow,
)


def balance_result(*, errors=None, warnings=None, score_gap=0, hard_score_gap=30, unmet=None):
    algorithm_result = {
        "is_balanced": not errors,
        "need_adjustment": bool(errors),
        "errors": errors or [],
        "warnings": warnings or [],
        "missing_names": [],
        "duplicate_names": [],
        "unknown_names": [],
        "unmet_preference_pairs": unmet or [],
        "team_count": 1,
        "member_counts": [3],
        "score_gap": score_gap,
        "hard_score_gap": hard_score_gap,
    }
    return {
        **algorithm_result,
        "algorithm_result": algorithm_result,
    }


class NormalWorkflowIterationTest(unittest.TestCase):
    def test_candidate_score_prioritizes_structure_then_errors_and_preferences(self):
        valid = balance_result(errors=["game 분산"], unmet=[{"student": "A"}])
        structural = balance_result(errors=["누락"])
        structural["algorithm_result"]["missing_names"] = ["A"]

        self.assertLess(
            build_candidate_quality_score(valid, expected_team_count=1),
            build_candidate_quality_score(structural, expected_team_count=1),
        )

        fewer_preferences = balance_result(errors=["game 분산"], unmet=[])
        self.assertLess(
            build_candidate_quality_score(fewer_preferences, expected_team_count=1),
            build_candidate_quality_score(valid, expected_team_count=1),
        )

    def test_normal_workflow_stops_when_adjustment_does_not_improve(self):
        state = {
            "balance_result": balance_result(errors=["game 분산"]),
            "iteration_count": 1,
            "last_adjustment_improved": False,
        }
        self.assertEqual(should_adjust_normal_workflow(state), "finalize_node")

        state["last_adjustment_improved"] = True
        self.assertEqual(should_adjust_normal_workflow(state), "adjust_team_node")

        state["iteration_count"] = MAX_ITERATION
        self.assertEqual(should_adjust_normal_workflow(state), "finalize_node")

    def test_normal_workflow_continues_only_while_each_iteration_improves(self):
        state = {
            "balance_result": balance_result(errors=["game 분산"]),
            "last_adjustment_improved": True,
        }
        for iteration in range(1, MAX_ITERATION):
            state["iteration_count"] = iteration
            self.assertEqual(should_adjust_normal_workflow(state), "adjust_team_node")

        state["iteration_count"] = MAX_ITERATION
        self.assertEqual(should_adjust_normal_workflow(state), "finalize_node")

    def test_warnings_only_do_not_start_adjustment_loop(self):
        state = {
            "balance_result": balance_result(errors=[], warnings=["소통 평균이 낮습니다."]),
            "iteration_count": 0,
            "last_adjustment_improved": False,
        }
        self.assertEqual(should_adjust_normal_workflow(state), "finalize_node")

    def test_normal_evaluation_keeps_best_candidate_when_new_result_is_worse(self):
        best = {"final_teams": [{"team_name": "팀 1", "members": ["A", "B", "C"]}]}
        current = {"final_teams": [{"team_name": "팀 1", "members": ["A", "B", "C"]}]}
        best_balance = balance_result(errors=["game 분산"])
        worse_balance = balance_result(errors=["game 분산", "점수 격차"])
        state = {
            "analyzed_students": [],
            "teams": [{"team_name": "팀 1", "members": []}],
            "llm_result": current,
            "best_candidate": best,
            "best_balance_result": best_balance,
            "best_team_evaluations": [{"team_name": "팀 1"}],
            "best_candidate_score": list(build_candidate_quality_score(best_balance, 1)),
            "iteration_count": 1,
        }

        with patch(
            "matching_student.workflow_matching_student.evaluate_balance_node",
            return_value={"balance_result": worse_balance, "team_evaluations": []},
        ):
            update = evaluate_normal_workflow_node(state)

        self.assertFalse(update["last_adjustment_improved"])
        self.assertEqual(update["best_candidate"], best)

    def test_normal_finalize_uses_best_candidate_and_marks_no_improvement(self):
        best = {
            "final_teams": [{"team_name": "팀 1", "members": ["A", "B", "C"]}],
            "changed": True,
        }
        state = {
            "best_candidate": best,
            "best_balance_result": balance_result(errors=["game 분산"]),
            "best_team_evaluations": [],
            "best_candidate_score": [0, 1, 0, 0, 0],
            "balance_result": balance_result(errors=["game 분산", "점수 격차"]),
            "iteration_count": 1,
            "last_adjustment_improved": False,
            "llm_result": {"final_teams": []},
            "teams": [],
        }
        captured = {}

        def fake_finalize(finalize_state):
            captured.update(finalize_state)
            return {"final_result": {"iteration_count": 0, "finalized_by": "manual_finalize"}}

        with patch("matching_student.workflow_matching_student.finalize_node", side_effect=fake_finalize):
            result = finalize_normal_workflow_node(state)

        self.assertEqual(captured["llm_result"], best)
        self.assertEqual(result["final_result"]["iteration_count"], 1)
        self.assertEqual(result["final_result"]["finalized_by"], "no_improvement")

    def test_regenerate_workflow_does_not_use_normal_workflow_nodes(self):
        initial_state = {
            "iteration_count": 0,
            "balance_result": {"is_balanced": False, "need_adjustment": True},
        }

        with (
            patch("matching_student.workflow_matching_student.build_regenerate_state", return_value=initial_state),
            patch(
                "matching_student.workflow_matching_student.adjust_team_node",
                return_value={"iteration_count": 1},
            ),
            patch(
                "matching_student.workflow_matching_student.evaluate_balance_node",
                return_value={"balance_result": {"is_balanced": True, "need_adjustment": False}},
            ),
            patch(
                "matching_student.workflow_matching_student.finalize_node",
                return_value={"final_result": {"finalized_by": "validation_passed"}},
            ),
            patch(
                "matching_student.workflow_matching_student.evaluate_normal_workflow_node",
                side_effect=AssertionError("재생성에서 일반 평가 노드를 호출하면 안 됩니다."),
            ),
            patch(
                "matching_student.workflow_matching_student.finalize_normal_workflow_node",
                side_effect=AssertionError("재생성에서 일반 확정 노드를 호출하면 안 됩니다."),
            ),
            patch("matching_student.workflow_matching_student.save_workflow_result"),
        ):
            result = run_regenerate_workflow(prompt="팀을 다시 조정해줘")

        self.assertEqual(result["iteration_count"], 1)
        self.assertEqual(result["final_result"]["finalized_by"], "validation_passed")


if __name__ == "__main__":
    unittest.main()
