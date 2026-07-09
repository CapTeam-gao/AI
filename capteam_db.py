import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymysql
from dotenv import load_dotenv
from pymysql.err import MySQLError
from pymysql.cursors import DictCursor


load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def get_connection():
    return pymysql.connect(
        #위에 _env함수를 써서 DB_HOST라는 환경변수가 설정되어 있는지 확인 없으면 지정해준 값으로 근데 docker나 sever에서 환경변수를 주면 그값으로 db에 연결
        host=_env("DB_HOST", "localhost"),
        port=int(_env("DB_PORT", "3306")),
        user=_env("DB_USER", "gao_user"),
        password=_env("DB_PASSWORD", "1234"),
        database=_env("DB_NAME", "gao_db"),
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
    # users 전체 컬럼을 읽을 때 AI 분석과 JSON 저장에 불필요하거나
    # 직렬화할 수 없는 인증/감사 필드는 학생 입력에서 제거한다.
    for private_key in ("password", "password_encoded", "created_at", "updated_at"):
        student.pop(private_key, None)
    student.setdefault("name", row.get("name") or row.get("student_name"))
    student.setdefault("goal", row.get("goal") or row.get("desired_role"))
    student["survey_completed"] = _as_bool(
        student.get("survey_completed", row.get("survey_completed"))
    )

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
    student["response_reliability"] = (
        student.get("response_reliability")
        or student.get("responseReliability")
        or row.get("response_reliability")
        or "HIGH"
    )
    student["wants_leader"] = _as_bool(
        student.get("wants_leader")
        if "wants_leader" in student
        else student.get("wantsLeader", row.get("wants_leader"))
    )

    def first_score(*keys: str) -> Any:
        for key in keys:
            if student.get(key) is not None:
                return student.get(key)
            if row.get(key) is not None:
                return row.get(key)
        return None

    hackathon_personality = student.get("hackathon_personality_scores") or student.get("personalityScores")
    if not isinstance(hackathon_personality, dict):
        hackathon_personality = {
            "ideaPlanning": first_score("ideaPlanning", "idea_planning", "personality_idea_planning"),
            "communication": first_score("communication", "personality_communication"),
            "roleFlexibility": first_score("roleFlexibility", "role_flexibility", "personality_role_flexibility"),
            "timePressure": first_score("timePressure", "time_pressure", "personality_time_pressure"),
            "staminaFocus": first_score("staminaFocus", "stamina_focus", "personality_stamina_focus"),
        }
    personality_keys = {"ideaPlanning", "communication", "roleFlexibility", "timePressure", "staminaFocus"}
    if personality_keys.issubset(hackathon_personality) and all(
        hackathon_personality[key] is not None for key in personality_keys
    ):
        student["hackathon_personality_scores"] = hackathon_personality

    hackathon_development = student.get("hackathon_development_scores") or student.get("developmentScores")
    if not isinstance(hackathon_development, dict):
        hackathon_development = {
            "implementation": first_score("implementation", "development_implementation"),
            "problemSolving": first_score("problemSolving", "problem_solving", "development_problem_solving"),
            "completionQuality": first_score("completionQuality", "completion_quality", "development_completion_quality"),
            "presentation": first_score("presentation", "development_presentation"),
            "leadership": first_score("leadership", "development_leadership"),
        }
    development_keys = {"implementation", "problemSolving", "completionQuality", "presentation", "leadership"}
    if development_keys.issubset(hackathon_development) and all(
        hackathon_development[key] is not None for key in development_keys
    ):
        student["hackathon_development_scores"] = hackathon_development

    return student


def fetch_students() -> List[Dict[str, Any]]:
    sql = os.getenv("STUDENT_SOURCE_SQL")
    table = _env("STUDENT_SOURCE_TABLE", "users")
    order_by = os.getenv("STUDENT_SOURCE_ORDER_BY", "id")

    if not sql:
        if table == "users":
            sql = """
            SELECT
                u.*,
                u.student_role AS goal,
                ua.analysis_result AS analysis_result,
                ua.student_level AS student_level,
                COALESCE(ua.response_reliability, 'HIGH') AS response_reliability,
                (SELECT JSON_ARRAYAGG(skill) FROM user_skill WHERE user_user_id = u.user_id) AS stack,
                (SELECT JSON_ARRAYAGG(experience) FROM user_experience WHERE user_user_id = u.user_id) AS experience,
                (SELECT JSON_ARRAYAGG(preferred_teammates) FROM user_preferred_teammates WHERE user_user_id = u.user_id) AS preferred_members
            FROM users u
            LEFT JOIN user_analysis ua ON ua.user_id = u.user_id
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


def fetch_analysis_results_for_students(
    students: List[Dict[str, Any]],
    row_limit: int = 100,
) -> List[Dict[str, Any]]:
    """Return the newest cached analysis for each requested student."""
    if not students:
        return []

    requested_ids = {
        student_id
        for student in students
        if (student_id := (
            student.get("user_id")
            or student.get("userId")
            or student.get("student_id")
            or student.get("studentId")
        ))
    }
    requested_names = {
        student.get("name")
        for student in students
        if student.get("name")
    }

    if os.getenv("ANALYSIS_RESULT_SQL"):
        batches = [fetch_analysis_results() or []]
    else:
        table = _env("ANALYSIS_RESULT_TABLE", "student_analysis_results")
        sql = f"""
        SELECT result_json
        FROM `{table}`
        WHERE result_type = 'analysis'
        ORDER BY id DESC
        LIMIT %s
        """
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, (max(1, int(row_limit)),))
                rows = cursor.fetchall()
        batches = []
        for row in rows:
            payload = _extract_json_from_row(row)
            if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                payload = payload["results"]
            if isinstance(payload, list):
                batches.append(payload)

    cached_by_id: Dict[str, Dict[str, Any]] = {}
    cached_by_name: Dict[str, Dict[str, Any]] = {}
    for batch in batches:
        for result in batch:
            if not isinstance(result, dict):
                continue
            result_id = (
                result.get("user_id")
                or result.get("userId")
                or result.get("student_id")
                or result.get("studentId")
            )
            result_name = result.get("name")
            if result_id in requested_ids:
                cached_by_id.setdefault(result_id, result)
            if result_name in requested_names:
                cached_by_name.setdefault(result_name, result)

    ordered_results = []
    for student in students:
        student_id = (
            student.get("user_id")
            or student.get("userId")
            or student.get("student_id")
            or student.get("studentId")
        )
        result = cached_by_id.get(student_id) or cached_by_name.get(student.get("name"))
        if result:
            ordered_results.append(result)
    return ordered_results


def save_analysis_results(results: List[Dict[str, Any]]) -> None:
    save_json_result("ANALYSIS_RESULT_TABLE", "student_analysis_results", "analysis", results)


MATCHING_RESULT_TYPES = {
    "CAPSTONE": "matching",
    "HACKATHON": "hackathon_matching",
}


def fetch_matching_result(matching_type: str = "CAPSTONE") -> Optional[Dict[str, Any]]:
    normalized_type = str(matching_type or "CAPSTONE").strip().upper()
    result_type = MATCHING_RESULT_TYPES.get(normalized_type)
    if result_type is None:
        raise ValueError(f"지원하지 않는 매칭 유형입니다: {matching_type}")

    custom_sql = os.getenv("MATCHING_RESULT_SQL")
    if custom_sql and normalized_type == "CAPSTONE":
        result = fetch_latest_json_result(
            "MATCHING_RESULT_TABLE",
            "team_matching_results",
            "MATCHING_RESULT_SQL",
        )
    else:
        table = _env("MATCHING_RESULT_TABLE", "team_matching_results")
        sql = f"SELECT result_json FROM `{table}` WHERE result_type = %s ORDER BY id DESC LIMIT 1"
        try:
            with get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql, (result_type,))
                    row = cursor.fetchone()
        except MySQLError:
            return None
        result = _extract_json_from_row(row) if row else None
    return result if isinstance(result, dict) else None


def save_matching_result(result: Dict[str, Any], matching_type: str = "CAPSTONE") -> None:
    normalized_type = str(matching_type or "CAPSTONE").strip().upper()
    result_type = MATCHING_RESULT_TYPES.get(normalized_type)
    if result_type is None:
        raise ValueError(f"지원하지 않는 매칭 유형입니다: {matching_type}")
    save_json_result("MATCHING_RESULT_TABLE", "team_matching_results", result_type, result)
