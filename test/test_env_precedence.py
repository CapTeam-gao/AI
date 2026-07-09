import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_LOADING_FILES = (
    "student_analysis/analysis_llm.py",
    "matching_student/workflow_matching_student.py",
    "matching_student/upstage_matching.py",
)


class EnvironmentPrecedenceTest(unittest.TestCase):
    def test_local_dotenv_never_overrides_runtime_environment(self):
        for relative_path in ENV_LOADING_FILES:
            with self.subTest(path=relative_path):
                source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
                self.assertNotIn("override=True", source)
                self.assertIn("override=False", source)

    def test_docker_image_excludes_local_environment_file(self):
        dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

        self.assertIn(".env", dockerignore)
        self.assertIn(".env.*", dockerignore)


if __name__ == "__main__":
    unittest.main()
