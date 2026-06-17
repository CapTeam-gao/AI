import json
import os
from typing import Any, Dict, List, Optional

import pymysql
from pymysql.err import MySQLError
from pymysql.cursors import DictCursor


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def get_connection():
    return pymysql.connect(
        #위에 _env함수를 써서 DB_HOST라는 환경변수가 설정되어 있는지 확인 없으면 지정해준 값으로 근데 docker나 sever에서 환경변수를 주면 그값으로 db에 연결
        host=_env("DB_HOST", "localhost"),
        port=int(_env("DB_PORT", "3306")),
        user=_env("DB_USER", "root"),
        password=_env("DB_PASSWORD", "1234"),
        database=_env("DB_NAME", "mydb"),
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=True,
    )


def _json_load(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)): #isinstance : 객체가 어떤 타입인지 확인하는 함수.
        return value #dict이나 list면 변환할 필요없이 그냥 반환
    if isinstance(value, (bytes, bytearray)): #bytes, bytearray면 디코딩해서 value에저장
        value = value.decode("utf-8")
    if isinstance(value, str): #문자열이면 앞뒤 공백제거
        value = value.strip()
        if not value:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _as_list(value: Any) -> List[Any]:
    loaded = _json_load(value)
    if loaded is None:
        return []
    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict):
        return list(loaded.keys()) #딕이면 키만 뽑음
    if isinstance(loaded, str):
        return [item.strip() for item in loaded.split(",") if item.strip()]
    return [loaded]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (bytes, bytearray)):
        return any(byte != 0 for byte in value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "팀장", "원함"}
    return False


def _normalize_student(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = None
    for key in ("student_data", "payload", "data", "json_data", "raw_json"):
        if key in row:
            loaded = _json_load(row[key])
            if isinstance(loaded, dict):
                payload = loaded
                break

    student = dict(payload or row)
    student.setdefault("name", row.get("name") or row.get("student_name"))
    student.setdefault("goal", row.get("goal") or row.get("desired_role"))

    skills = student.get("skills", row.get("skills"))
    stack = student.get("stack", row.get("stack"))
    if stack is None and skills is not None:
        stack = skills
    student["stack"] = _as_list(stack)

    experience = student.get("experience", row.get("experience"))
    student["experience"] = _as_list(experience)
    student["collaboration"] = int(student.get("collaboration", row.get("collaboration", 0)) or 0)
    student["preferred_members"] = _as_list(
        student.get("preferred_members")
        or student.get("preferredMembers")
        or row.get("preferred_members")
    )
    student["wants_leader"] = _as_bool(
        student.get("wants_leader")
        if "wants_leader" in student
        else student.get("wantsLeader", row.get("wants_leader"))
    )

    return student


def fetch_students() -> List[Dict[str, Any]]:
    sql = os.getenv("STUDENT_SOURCE_SQL")
    table = _env("STUDENT_SOURCE_TABLE", "users")
    order_by = os.getenv("STUDENT_SOURCE_ORDER_BY", "id")

    if not sql:
        if table == "users":
            sql = """
            SELECT
                u.user_id,
                u.name,
                u.student_role AS goal,
                u.grade,
                u.wants_leader,
                u.personality_communication AS communication,
                u.personality_responsibility AS responsibility,
                u.personality_collaboration AS collaboration,
                u.personality_flexibility AS flexibility,
                u.personality_emotional_stability AS emotionalStability,
                u.development_leadership AS leadership,
                u.development_problem_solving AS problemSolving,
                u.development_implementation AS implementation,
                u.development_learning_ability AS learningAbility,
                u.development_planning AS planning,
                (SELECT JSON_ARRAYAGG(skill) FROM user_skill WHERE user_user_id = u.user_id) AS stack,
                (SELECT JSON_ARRAYAGG(experience) FROM user_experience WHERE user_user_id = u.user_id) AS experience,
                (SELECT JSON_ARRAYAGG(preferred_teammates) FROM user_preferred_teammates WHERE user_user_id = u.user_id) AS preferred_members
            FROM users u
            WHERE u.account_role = 'STUDENT'
              AND u.survey_completed = b'1'
            ORDER BY u.user_id
            """
        else:
            sql = f"SELECT * FROM `{table}`"
            if order_by:
                sql += f" ORDER BY `{order_by}`"
    elif order_by and f"ORDER BY `{order_by}`" not in sql:
        if table != "users":
            sql += f" ORDER BY `{order_by}`"

    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
    except MySQLError:
        if sql == f"SELECT * FROM `{table}` ORDER BY `{order_by}`":
            with get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f"SELECT * FROM `{table}`")
                    rows = cursor.fetchall()
        else:
            raise

    return [_normalize_student(row) for row in rows]


def _extract_json_from_row(row: Dict[str, Any]) -> Any:
    for key in ("result_json", "payload", "data", "json_data", "matching_output", "analysis_output"):
        if key in row:
            loaded = _json_load(row[key])
            if loaded is not None:
                return loaded
    return row


def fetch_latest_json_result(table_env: str, default_table: str, sql_env: str) -> Optional[Any]:
    sql = os.getenv(sql_env)
    table = _env(table_env, default_table)

    if not sql:
        sql = f"SELECT * FROM `{table}` ORDER BY id DESC LIMIT 1"

    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                row = cursor.fetchone()
    except MySQLError:
        return None

    if not row:
        return None
    return _extract_json_from_row(row)


def ensure_result_table(table: str) -> None:
    sql = f"""
    CREATE TABLE IF NOT EXISTS `{table}` (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        result_type VARCHAR(64) NOT NULL,
        result_json JSON NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql)


def save_json_result(table_env: str, default_table: str, result_type: str, payload: Any) -> None:
    table = _env(table_env, default_table)
    ensure_result_table(table)
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO `{table}` (result_type, result_json) VALUES (%s, %s)",
                (result_type, json.dumps(payload, ensure_ascii=False)),
            )


def fetch_analysis_results() -> Optional[List[Dict[str, Any]]]:
    result = fetch_latest_json_result(
        "ANALYSIS_RESULT_TABLE",
        "student_analysis_results",
        "ANALYSIS_RESULT_SQL",
    )
    if result is None:
        return None
    if isinstance(result, dict) and "results" in result:
        return result["results"]
    if isinstance(result, list):
        return result
    return None


def save_analysis_results(results: List[Dict[str, Any]]) -> None:
    save_json_result("ANALYSIS_RESULT_TABLE", "student_analysis_results", "analysis", results)


def fetch_matching_result() -> Optional[Dict[str, Any]]:
    result = fetch_latest_json_result(
        "MATCHING_RESULT_TABLE",
        "team_matching_results",
        "MATCHING_RESULT_SQL",
    )
    return result if isinstance(result, dict) else None


def save_matching_result(result: Dict[str, Any]) -> None:
    save_json_result("MATCHING_RESULT_TABLE", "team_matching_results", "matching", result)
