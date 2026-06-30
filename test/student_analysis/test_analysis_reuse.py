import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("OPENAI_API_KEY", "test-key")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import run_analysis
from student_analysis.analysis_llm import _resolve_reusable_analyses, get_analyze_stu


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
            "stack": ["Spring Boot"],
            "experience": ["게시판 API 구현"],
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

    def test_changed_survey_input_invalidates_same_student_cache(self):
        student = {
            "user_id": "stu2399",
            "name": "테스트학생",
            "role": "FRONTEND",
            "grade": "GRADE_2",
            "stack": ["React", "TypeScript", "CSS"],
            "experience": [
                "관리자 대시보드 구현",
                "React Query 공지 목록 캐싱",
                "CSS Modules 반응형 UI",
            ],
        }
        stale_cache = {
            "user_id": "stu2399",
            "name": "테스트학생",
            "role": "FRONTEND",
            "grade": "GRADE_2",
            "stack": ["React", "JavaScript"],
            "experience": ["로그인 기능 구현 - axios와 zustand 활용"],
            "reason": "예전 분석 결과",
            "skill_level": "중",
        }

        self.assertEqual(_resolve_reusable_analyses([student], [stale_cache]), {})

    @patch("student_analysis.analysis_llm.get_analyze_stu")
    def test_analysis_endpoint_always_forces_fresh_analysis(self, analyze):
        analyze.return_value = []
        student = {
            "user_id": "stu2399",
            "name": "테스트학생",
            "role": "FRONTEND",
            "stack": ["React"],
            "experience": ["대시보드 구현"],
        }

        run_analysis([student])

        _, kwargs = analyze.call_args
        self.assertTrue(kwargs["force_reanalyze"])


if __name__ == "__main__":
    unittest.main()
