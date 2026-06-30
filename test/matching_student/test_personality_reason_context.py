import unittest

from capteam_traits import ensure_trait_profile
from matching_student.workflow_matching_student import (
    build_personality_reason_evidence,
    get_final_reason_cards_prompt_chain,
)


def make_student(name, communication, responsibility, reliability="HIGH"):
    return ensure_trait_profile({
        "name": name,
        "response_reliability": reliability,
        "communication": communication,
        "responsibility": responsibility,
        "collaboration": 3,
        "flexibility": 3,
        "emotionalStability": 3,
        "leadership": 3,
        "problemSolving": 3,
        "implementation": 3,
        "learningAbility": 3,
        "planning": 3,
    })


class PersonalityReasonContextTest(unittest.TestCase):
    def test_prompt_requires_personality_and_technical_reasons_together(self):
        prompt = get_final_reason_cards_prompt_chain()
        prompt_text = "\n".join(
            getattr(message.prompt, "template", "")
            for message in prompt.messages
        )

        self.assertIn("성향 강점 조합을 설명하는 카드를 반드시 1개", prompt_text)
        self.assertIn("implementation_connections를 반영", prompt_text)

    def test_builds_named_personality_complement_without_scores(self):
        evidence = build_personality_reason_evidence([
            make_student("소통지원필요", communication=2, responsibility=3),
            make_student("소통강점", communication=5, responsibility=3),
        ])

        self.assertEqual(evidence["complements"], [{
            "trait": "소통",
            "member_to_support": "소통지원필요",
            "supporters": ["소통강점"],
        }])
        self.assertNotIn("score", str(evidence))

    def test_excludes_low_reliability_students(self):
        evidence = build_personality_reason_evidence([
            make_student("책임감강점", communication=3, responsibility=5),
            make_student("신뢰도낮음", communication=5, responsibility=5, reliability="LOW"),
        ])

        names = {item["name"] for item in evidence["strengths"]}
        self.assertIn("책임감강점", names)
        self.assertNotIn("신뢰도낮음", names)


if __name__ == "__main__":
    unittest.main()
