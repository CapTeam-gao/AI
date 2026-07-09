import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DockerComposeDatabaseConfigTest(unittest.TestCase):
    def test_ai_server_uses_backend_database_defaults(self):
        compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertRegex(compose, re.compile(r"DB_HOST:\s+gao-mysql"))
        self.assertRegex(compose, re.compile(r"DB_PORT:\s+3306"))
        self.assertRegex(compose, re.compile(r"DB_NAME:\s+\$\{DB_NAME:-gao_db\}"))
        self.assertRegex(compose, re.compile(r"DB_USER:\s+\$\{DB_USER:-gao_user\}"))
        self.assertRegex(compose, re.compile(r"DB_PASSWORD:\s+\$\{DB_PASSWORD:-1234\}"))

    def test_ai_compose_does_not_create_a_duplicate_mysql(self):
        compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertNotRegex(compose, re.compile(r"(?m)^  mysql:\s*$"))
        self.assertNotIn("container_name: mysql", compose)
        self.assertNotIn("mysql_data:", compose)
        self.assertRegex(
            compose,
            re.compile(r"backend_internal:\s*\n\s+external:\s+true"),
        )


if __name__ == "__main__":
    unittest.main()
