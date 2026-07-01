import os
import sys
import unittest
from pathlib import Path


os.environ.setdefault("OPENAI_API_KEY", "test-key")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matching_student.workflow_matching_student import build_regenerate_state, normalize_current_teams


class RegenerateNormalizationTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
