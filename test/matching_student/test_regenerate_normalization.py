import os
import sys
import unittest
from pathlib import Path


os.environ.setdefault("OPENAI_API_KEY", "test-key")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matching_student.workflow_matching_student import (
    build_adjustment_scope_rules,
    build_regenerate_state,
    create_regeneration_seed_teams,
    evaluate_balance_node,
    extract_reference_teams_from_prompt,
    get_adjust_team_prompt_chain,
    normalize_current_teams,
    parse_regeneration_constraints,
    validate_regeneration_constraints,
)


class RegenerateNormalizationTest(unittest.TestCase):
    def test_regenerate_prompt_locks_unrelated_teams_and_balance(self):
        scope_rules = build_adjustment_scope_rules(regeneration_mode=True)
        messages = get_adjust_team_prompt_chain().format_messages(
            adjustment_scope_rules=scope_rules,
            regeneration_constraints="{}",
            allowed_student_names="[]",
            student_analysis="[]",
            algorithm_teams="[]",
            regeneration_base_teams="[]",
            current_candidate="[]",
            reason_context="{}",
            balance_result="{}",
            adjustment_history="[]",
        )
        prompt_text = "\n".join(message.content for message in messages)

        request_priority = prompt_text.index("2) 사용자 요청 반영")
        balance_priority = prompt_text.index("3) 역할·점수·성향 균형")
        minimal_change_priority = prompt_text.index("4) 최소 변경")
        self.assertLess(request_priority, balance_priority)
        self.assertLess(balance_priority, minimal_change_priority)
        self.assertIn("사용자 요청보다 앞서는 거절 기준으로 사용하지 않는다", prompt_text)
        self.assertIn("요청과 무관한 팀은 팀원, 팀장, 팀 이름", prompt_text)
        self.assertIn("역할 다양성을 낮추지 않는다", prompt_text)
        self.assertIn("전체 팀 점수 격차를 변경 전보다 키우지 않는", prompt_text)

    def test_teacher_rebuild_prompt_is_compiled_into_measurable_constraints(self):
        prompt = """
        아래 팀 편성안은 참고용 가안이다. 그대로 유지하지 말고 약 20% 정도만 참고해서
        새롭게 팀을 재생성해줘. 기존 가안과 완전히 똑같은 팀이 나오면 안 된다.
        대부분의 학생은 다른 팀원과 새롭게 섞고 약 80%는 새 조합으로 재편성해줘.
        같은 반/같은 번호 학생이 한 팀에 과도하게 몰리지 않게 하고,
        프론트엔드/백엔드/앱/보안/게임/DevOps/풀스택 역할을 균형 있게 섞어줘.
        """

        constraints = parse_regeneration_constraints(prompt)

        self.assertEqual("balanced_rebuild", constraints["scope"])
        self.assertEqual(0.2, constraints["max_preserved_pair_ratio"])
        self.assertEqual(0.8, constraints["min_changed_student_ratio"])
        self.assertTrue(constraints["forbid_unchanged_teams"])
        self.assertTrue(constraints["spread_class_or_number"])
        self.assertTrue(constraints["distribute_roles"])

        scope_rules = build_adjustment_scope_rules(True, constraints)
        self.assertIn("대규모 재편성", scope_rules)
        self.assertIn("최소 수정 방향으로 축소하지 않는다", scope_rules)
        self.assertNotIn("요청과 무관한 팀은 팀원", scope_rules)

    def test_reference_team_table_in_prompt_becomes_overlap_baseline(self):
        students = [
            {"name": f"학생{index}", "role": "BACKEND", "skill_level": "중"}
            for index in range(1, 11)
        ]
        prompt = """
        아래 참고용 기존 가안은 그대로 유지하지 말고 새롭게 재편성해줘.
        참고용 기존 가안:
        1   학생1 학생2 학생3 학생4 학생5
        2   학생6 학생7 학생8 학생9 학생10
        """

        reference_teams = extract_reference_teams_from_prompt(prompt, students)

        self.assertEqual(2, len(reference_teams))
        self.assertEqual(
            ["학생1", "학생2", "학생3", "학생4", "학생5"],
            reference_teams[0]["members"],
        )
        self.assertEqual("팀 2", reference_teams[1]["team_name"])

    def test_rebuild_seed_satisfies_teacher_overlap_request_for_44_students(self):
        roles = ["FRONTEND", "BACKEND", "APP", "SECURITY", "GAME", "DEVOPS", "FULLSTACK"]
        students = []
        for index in range(44):
            class_number = (index % 4) + 1
            student_number = (index // 4) + 1
            students.append({
                "user_id": f"stu2{class_number}{student_number:02d}",
                "name": f"학생{index + 1}",
                "role": roles[index % len(roles)],
                "skill_level": ["상", "중상", "중", "중하", "하"][index % 5],
                "stack_score": "기술: 5점",
            })

        base_teams = []
        cursor = 0
        for team_index, size in enumerate([5, 5, 5, 5, 5, 5, 5, 5, 4]):
            names = [student["name"] for student in students[cursor:cursor + size]]
            cursor += size
            base_teams.append({"team_name": f"팀 {team_index + 1}", "members": names})

        constraints = parse_regeneration_constraints(
            "기존 가안은 20%만 참고하고 80%는 새 조합으로 재편성해줘. "
            "완전히 똑같은 팀은 금지하고 같은 반과 같은 번호가 몰리지 않게 역할도 균형 있게 분산해줘."
        )
        seed_teams = create_regeneration_seed_teams(students, base_teams, constraints)
        compliance = validate_regeneration_constraints(
            seed_teams,
            base_teams,
            students,
            constraints,
        )

        assigned_names = [
            name
            for team in seed_teams
            for name in team["members"]
        ]
        self.assertEqual(44, len(assigned_names))
        self.assertEqual(44, len(set(assigned_names)))
        self.assertEqual([], compliance["errors"])
        self.assertLessEqual(compliance["preserved_pair_ratio"], 0.2)
        self.assertGreaterEqual(compliance["changed_student_ratio"], 0.8)
        self.assertEqual([], compliance["unchanged_teams"])

    def test_unchanged_candidate_fails_teacher_rebuild_constraints(self):
        students = [
            {"user_id": f"stu21{index:02d}", "name": f"학생{index}", "role": "BACKEND", "skill_level": "중"}
            for index in range(1, 11)
        ]
        base_teams = [
            {"team_name": "팀 1", "members": [f"학생{index}" for index in range(1, 6)]},
            {"team_name": "팀 2", "members": [f"학생{index}" for index in range(6, 11)]},
        ]
        constraints = parse_regeneration_constraints(
            "기존 조합은 최대 20%만 유지하고 80%는 새 조합으로 바꿔. 완전히 똑같은 팀은 안 돼."
        )

        compliance = validate_regeneration_constraints(
            base_teams,
            base_teams,
            students,
            constraints,
        )

        self.assertEqual(3, len(compliance["errors"]))
        self.assertEqual(1.0, compliance["preserved_pair_ratio"])
        self.assertEqual(0.0, compliance["changed_student_ratio"])

    def test_qualitative_request_rejection_forces_another_adjustment(self):
        students = [
            {"user_id": "stu2101", "name": "학생1", "role": "BACKEND", "skill_level": "중"},
            {"user_id": "stu2201", "name": "학생2", "role": "FRONTEND", "skill_level": "중"},
        ]
        teams = [{"team_name": "팀 1", "members": ["학생1", "학생2"]}]
        state = {
            "analyzed_students": students,
            "teams": teams,
            "llm_result": {"final_teams": teams},
            "regeneration_mode": True,
            "regeneration_base_teams": teams,
            "regeneration_constraints": {
                **parse_regeneration_constraints("발표 경험이 있는 학생을 팀장으로 지정해줘"),
                "min_changed_student_ratio": None,
            },
        }

        from unittest.mock import patch

        with (
            patch(
                "matching_student.workflow_matching_student.evaluate_regeneration_request_with_llm",
                return_value={
                    "is_satisfied": False,
                    "satisfied_requirements": [],
                    "unmet_requirements": ["발표 경험이 있는 학생이 팀장으로 지정되지 않았습니다."],
                    "summary": "팀장 조건 미반영",
                },
            ),
            patch(
                "matching_student.workflow_matching_student.llm_validation_balance_team",
                return_value={
                    "is_balanced": False,
                    "need_adjustment": True,
                    "overall_reason": "사용자 요청 미반영",
                    "adjustment_request": "팀장 조건을 반영하세요.",
                    "team_evaluations": [],
                },
            ),
        ):
            result = evaluate_balance_node(state)

        algorithm_result = result["balance_result"]["algorithm_result"]
        self.assertTrue(algorithm_result["request_errors"])
        self.assertTrue(result["balance_result"]["need_adjustment"])

    def test_request_current_teams_drops_old_reason_fields(self):
        normalized = normalize_current_teams({
            "teams": [
                {
                    "team_name": "팀 1",
                    "members": [{"name": "학생1"}, {"name": "학생2"}],
                    "leader": "학생1",
                    "reason": "이전 배정 이유",
                    "reason_cards": [{"title": "old", "description": "old"}],
                    "matching_reason": "예전 요약",
                }
            ]
        })

        self.assertEqual(normalized[0]["members"], ["학생1", "학생2"])
        self.assertEqual(normalized[0]["reason"], "")
        self.assertEqual(normalized[0]["reason_cards"], [])

    def test_cached_matching_result_is_sanitized_before_regenerate(self):
        students = [
            {
                "name": "학생1",
                "role": "BACKEND",
                "skill_level": "중",
                "stack_score": "Python: 6점",
                "communication": 3,
                "responsibility": 3,
                "collaboration": 3,
                "flexibility": 3,
                "emotionalStability": 3,
                "leadership": 3,
                "problemSolving": 3,
                "implementation": 3,
                "learningAbility": 3,
                "planning": 3,
            },
            {
                "name": "학생2",
                "role": "FRONTEND",
                "skill_level": "중",
                "stack_score": "React: 6점",
                "communication": 3,
                "responsibility": 3,
                "collaboration": 3,
                "flexibility": 3,
                "emotionalStability": 3,
                "leadership": 3,
                "problemSolving": 3,
                "implementation": 3,
                "learningAbility": 3,
                "planning": 3,
            },
        ]

        from unittest.mock import patch

        with patch(
            "matching_student.workflow_matching_student.load_cached_matching_result",
            return_value={
                "final_result": {
                    "final_teams": [
                        {
                            "team_name": "팀 1",
                            "members": ["학생1", "학생2"],
                            "leader": "학생1",
                            "reason": "이전 이유가 남아 있음",
                            "reason_cards": [{"title": "old", "description": "old"}],
                        }
                    ]
                }
            },
        ):
            state = build_regenerate_state(
                analyzed_students=students,
                prompt="팀을 다시 조정해줘",
                current_teams=None,
            )

        current_candidate = state["llm_result"]["final_teams"]
        self.assertEqual(current_candidate[0]["reason"], "")
        self.assertEqual(current_candidate[0]["reason_cards"], [])
        self.assertTrue(state["regeneration_mode"])
        self.assertEqual(state["regeneration_base_teams"], current_candidate)
        self.assertIsNot(state["regeneration_base_teams"], current_candidate)


if __name__ == "__main__":
    unittest.main()
