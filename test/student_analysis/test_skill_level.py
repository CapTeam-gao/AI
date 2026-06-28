import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import get_skill_level_counts, to_backend_student_level
from matching_student.workflow_matching_student import (
    SKILL_LEVEL_SCORE,
    get_technical_score,
    validation_balance_team,
)
from student_analysis.analysis_llm import StudentAnalysis, cap_stack_score, normalize_analysis


class StudentSkillLevelTest(unittest.TestCase):
    def test_new_skill_levels_are_supported_by_structured_output(self):
        for level in ("상", "중상", "중", "중하", "하"):
            result = StudentAnalysis(
                name="테스트 학생",
                strength="구현 경험",
                weakness="운영 경험 부족",
                reason="학생 데이터의 전체 문맥을 기준으로 판단",
                stack_score="Python: 5점",
                skill_level=level,
                role="백엔드",
                suggestion="프론트엔드 역할과 협업",
            )
            self.assertEqual(result.skill_level, level)

    def test_stack_score_caps_support_new_and_legacy_levels(self):
        expected_caps = {
            "상": 10,
            "중상": 8,
            "중": 7,
            "중하": 6,
            "하": 4,
            "높음": 10,
            "보통": 7,
            "낮음": 4,
        }

        for level, expected in expected_caps.items():
            with self.subTest(level=level):
                self.assertEqual(
                    cap_stack_score("Python: 10점", level),
                    f"Python: {expected}점",
                )

    def test_low_collaboration_does_not_override_technical_level(self):
        response = {
            "name": "테스트 학생",
            "strength": "응답 시간을 줄인 구현 경험",
            "weakness": "운영 경험 부족",
            "reason": "문장 전체의 수행 내용을 기준으로 판단",
            "stack_score": "Python: 9점",
            "skill_level": "상",
            "role": "백엔드",
            "suggestion": "프론트엔드 역할과 협업",
        }
        student = {
            "name": "테스트 학생",
            "goal": "백엔드",
            "stack": ["Python"],
            "experience": ["처리 시간을 3초에서 0.8초로 줄이고 병목 원인을 해결함"],
            "collaboration": 0,
        }

        normalized = normalize_analysis(response, student)

        self.assertEqual(normalized["skill_level"], "상")
        self.assertEqual(normalized["stack_score"], "Python: 9점")

    def test_legacy_middle_cache_is_preserved(self):
        response = {
            "name": "기존 학생",
            "strength": "프로젝트 경험",
            "weakness": "운영 경험 부족",
            "reason": "기존 분석 결과",
            "stack_score": "Java: 9점",
            "skill_level": "보통",
            "role": "백엔드",
            "suggestion": "프론트엔드 역할과 협업",
        }
        student = {
            "name": "기존 학생",
            "goal": "백엔드",
            "stack": ["Java"],
            "experience": ["기존 프로젝트 구현 경험"],
            "collaboration": 0,
        }

        normalized = normalize_analysis(response, student)

        self.assertEqual(normalized["skill_level"], "보통")
        self.assertEqual(normalized["stack_score"], "Java: 7점")

    def test_backend_level_mapping_uses_five_level_contract(self):
        self.assertEqual(to_backend_student_level("상"), "UPPER")
        self.assertEqual(to_backend_student_level("중상"), "MIDDLE_UPPER")
        self.assertEqual(to_backend_student_level("중"), "MIDDLE")
        self.assertEqual(to_backend_student_level("중하"), "MIDDLE_LOWER")
        self.assertEqual(to_backend_student_level("하"), "LOWER")
        self.assertEqual(to_backend_student_level("보통"), "MIDDLE")

    def test_matching_score_order_supports_new_and_legacy_levels(self):
        self.assertEqual(
            [SKILL_LEVEL_SCORE[level] for level in ("상", "중상", "중", "중하", "하")],
            [5, 4, 3, 2, 1],
        )
        self.assertEqual(SKILL_LEVEL_SCORE["높음"], SKILL_LEVEL_SCORE["상"])
        self.assertEqual(SKILL_LEVEL_SCORE["보통"], SKILL_LEVEL_SCORE["중"])
        self.assertEqual(SKILL_LEVEL_SCORE["낮음"], SKILL_LEVEL_SCORE["하"])
        self.assertGreater(
            get_technical_score({"skill_level": "중상", "stack_score": "Python: 5점"}),
            get_technical_score({"skill_level": "중", "stack_score": "Python: 5점"}),
        )

    def test_team_summary_counts_use_five_levels(self):
        members = [
            {"skill_level": "상"},
            {"skill_level": "MIDDLE_UPPER"},
            {"skill_level": "보통"},
            {"skill_level": "중하"},
            {"skill_level": "낮음"},
        ]

        self.assertEqual(
            get_skill_level_counts(members),
            {"상": 1, "중상": 1, "중": 1, "중하": 1, "하": 1},
        )

    def test_all_lower_level_team_validation_supports_new_and_legacy_labels(self):
        students = [
            {"name": "학생1", "role": "프론트엔드", "skill_level": "하", "stack_score": ""},
            {"name": "학생2", "role": "백엔드", "skill_level": "낮음", "stack_score": ""},
            {"name": "학생3", "role": "디자인", "skill_level": "하", "stack_score": ""},
        ]
        teams = [{"team_name": "1팀", "members": ["학생1", "학생2", "학생3"]}]

        balance_result, _ = validation_balance_team(teams, students, base_teams=teams)

        self.assertTrue(
            any("하위 등급 학생만" in error for error in balance_result["errors"])
        )


if __name__ == "__main__":
    unittest.main()
