import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DockerComposeDatabaseConfigTest(unittest.TestCase):
    def test_ai_server_uses_backend_database_defaults(self):
        compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertRegex(compose, re.compile(r"DB_HOST:\s+gao-mysql"))
        self.assertRegex(compose, re.compile(r"DB_NAME:\s+\$\{DB_NAME:-gao_db\}"))
        self.assertRegex(compose, re.compile(r"DB_USER:\s+\$\{DB_USER:-gao_user\}"))
        self.assertRegex(compose, re.compile(r"DB_PASSWORD:\s+\$\{DB_PASSWORD:-1234\}"))


if __name__ == "__main__":
    unittest.main()
