import unittest

from pydantic import ValidationError

from matching_student.workflow_matching_student import FinalTeamReasonCards


def make_reason_cards(count):
    return [
        {
            "title": f"배정 이유 {index}",
            "description": f"서로 다른 배정 근거 {index}입니다.",
        }
        for index in range(1, count + 1)
    ]


class FinalTeamReasonCardsCountTest(unittest.TestCase):
    def test_accepts_two_to_four_reason_cards(self):
        for count in (2, 3, 4):
            with self.subTest(count=count):
                result = FinalTeamReasonCards(
                    team_name="테스트 팀",
                    reason_cards=make_reason_cards(count),
                    reason="배정 이유 요약입니다.",
                )

                self.assertEqual(len(result.reason_cards), count)

    def test_rejects_reason_card_count_outside_range(self):
        for count in (1, 5):
            with self.subTest(count=count):
                with self.assertRaises(ValidationError):
                    FinalTeamReasonCards(
                        team_name="테스트 팀",
                        reason_cards=make_reason_cards(count),
                        reason="배정 이유 요약입니다.",
                    )


if __name__ == "__main__":
    unittest.main()
