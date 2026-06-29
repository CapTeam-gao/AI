import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("OPENAI_API_KEY", "test-key")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from student_analysis.analysis_llm import get_analyze_stu


class StudentAnalysisReuseTest(unittest.TestCase):
    @patch("student_analysis.analysis_llm.save_analysis_results")
    @patch("student_analysis.analysis_llm.get_llm")
    @patch("student_analysis.analysis_llm.fetch_analysis_results_for_students")
    def test_matching_reuses_cache_and_embedded_backend_analysis(
        self,
        fetch_cached,
        get_llm,
        save_results,
    ):
        students = [
            {
                "user_id": "student-1",
                "name": "캐시학생",
                "role": "BACKEND",
                "stack": ["Spring Boot"],
                "experience": ["게시판 API 구현"],
            },
            {
                "user_id": "student-2",
                "name": "백엔드분석학생",
                "role": "FRONTEND",
                "stack": ["React"],
                "experience": ["목록 화면 구현"],
                "analysis_result": "React 화면 구현 경험이 확인되었습니다.",
                "student_level": "MIDDLE",
            },
        ]
        fetch_cached.return_value = [{
            "user_id": "student-1",
            "name": "캐시학생",
            "strength": "Spring API 구현 경험",
            "weakness": "운영 경험 부족",
            "reason": "기존 캐시 분석",
            "stack_score": "Spring Boot: 6점",
            "skill_level": "중",
            "role": "BACKEND",
            "suggestion": "프론트엔드와 협업",
            "analysis_status": "SUCCESS",
        }]

        results = get_analyze_stu(students)

        get_llm.assert_not_called()
        save_results.assert_called_once()
        self.assertEqual([result["user_id"] for result in results], ["student-1", "student-2"])
        self.assertEqual(results[0]["reason"], "기존 캐시 분석")
        self.assertEqual(results[1]["reason"], "React 화면 구현 경험이 확인되었습니다.")
        self.assertEqual(results[1]["skill_level"], "중")


if __name__ == "__main__":
    unittest.main()
