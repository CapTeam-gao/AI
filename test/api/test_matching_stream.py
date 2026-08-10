import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api.main as api
import matching_student.workflow_matching_student as workflow
import student_analysis.analysis_llm as analysis


def build_students():
    return [
        {
            "user_id": "stu-1",
            "name": "학생1",
            "role": "FRONTEND",
            "skill_level": "중상",
            "stack_score": "React: 7점",
        },
        {
            "user_id": "stu-2",
            "name": "학생2",
            "role": "BACKEND",
            "skill_level": "중",
            "stack_score": "Spring: 6점",
        },
    ]


def build_team(strengths="", weaknesses="", reason_cards=None):
    return {
        "team_name": "팀 1",
        "members": ["학생1", "학생2"],
        "leader": "학생1",
        "role_groups": [
            {"role_group": "frontend", "count": 1},
            {"role_group": "backend", "count": 1},
        ],
        "strengths": strengths,
        "weaknesses": weaknesses,
        "reason_cards": reason_cards or [],
        "reason": " ".join(card["description"] for card in (reason_cards or [])),
    }


def build_result(team=None):
    return {
        "analyzed_students": build_students(),
        "final_result": {
            "final_teams": [team or build_team()],
            "changed": False,
            "change_summary": "",
            "validation_notes": "",
        },
    }


async def read_stream(response, first_event_only=False):
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        if first_event_only:
            await response.body_iterator.aclose()
            break
    return "".join(chunks)


def parse_events(stream_text):
    events = []
    for frame in stream_text.split("\n\n"):
        if not frame or frame.startswith(":"):
            continue
        fields = {}
        for line in frame.splitlines():
            key, _, value = line.partition(":")
            fields[key] = value.lstrip()
        events.append({
            "id": int(fields["id"]),
            "event": fields["event"],
            "data": json.loads(fields["data"]),
        })
    return events


class MatchingStreamTest(unittest.TestCase):
    def test_batch_completion_callback_posts_progress_stage(self):
        callback_response = Mock()

        with patch.dict(
            os.environ,
            {
                "BACKEND_BASE_URL": "http://backend.test/",
                "INTERNAL_MATCHING_API_KEY": "test-key",
            },
        ), patch.object(api.requests, "post", return_value=callback_response) as post:
            callback = api.create_batch_completion_callback(
                "job-stage-1",
                build_students(),
            )
            callback("collaboration_matching", [])

        post.assert_called_once()
        self.assertEqual(
            "http://backend.test/internal/matching/jobs/job-stage-1/stage",
            post.call_args.args[0],
        )
        self.assertEqual(2, post.call_args.kwargs["json"]["progress_step"])

    def test_batch_completion_callback_posts_public_team_shape(self):
        callback_response = Mock()
        students = build_students()

        with patch.dict(
            os.environ,
            {
                "BACKEND_BASE_URL": "http://backend.test/",
                "INTERNAL_MATCHING_API_KEY": "test-key",
            },
        ), patch.object(api.requests, "post", return_value=callback_response) as post:
            callback = api.create_batch_completion_callback("job-callback-1", students)
            team = build_team(
                "역할이 연결됩니다.",
                "검토가 필요합니다.",
                [{"title": "역할 연결", "description": "구현 흐름을 연결했습니다."}],
            )
            callback("team_preview", [team])

        post.assert_called_once()
        request = post.call_args.kwargs
        self.assertEqual(
            "http://backend.test/internal/matching/jobs/job-callback-1/batch-complete",
            post.call_args.args[0],
        )
        self.assertEqual("test-key", request["headers"]["X-Internal-Api-Key"])
        self.assertEqual(0, request["json"]["batch_index"])
        self.assertEqual(1, request["json"]["total_batches"])
        self.assertEqual(
            {
                "team_name",
                "total_people",
                "role_counts",
                "leader",
                "matching_reason",
                "reason_cards",
                "strengths",
                "weaknesses",
                "skill_level_counts",
                "members",
            },
            set(request["json"]["teams"][0]),
        )

    def test_blocking_regenerate_route_passes_job_callback_to_workflow(self):
        students = build_students()
        callback = Mock()
        result = build_result()

        with (
            patch.object(analysis, "get_analyze_stu", return_value=students),
            patch.object(workflow, "run_regenerate_workflow", return_value=result) as regenerate,
            patch.object(api, "create_batch_completion_callback", return_value=callback) as callback_factory,
            patch.object(api, "build_team_summary", return_value={}),
        ):
            api.regenerate_matching(
                {"prompt": "현재 팀을 유지하고 설명만 보완해줘", "students": students},
                matching_job_id="job-blocking-regenerate",
            )

        callback_factory.assert_called_once_with("job-blocking-regenerate", students)
        self.assertIs(regenerate.call_args.kwargs["progress_callback"], callback)
        self.assertEqual("현재 팀을 유지하고 설명만 보완해줘", regenerate.call_args.kwargs["prompt"])

    def test_blocking_run_route_passes_job_callback_to_workflow(self):
        students = build_students()
        callback = Mock()
        result = build_result()

        with (
            patch.object(analysis, "get_analyze_stu", return_value=students),
            patch.object(workflow, "run_workflow", return_value=result) as run_workflow,
            patch.object(api, "create_batch_completion_callback", return_value=callback) as callback_factory,
            patch.object(api, "build_team_summary", return_value={}),
        ):
            api.run_matching(
                {"students": students},
                matching_job_id="job-blocking-run",
            )

        callback_factory.assert_called_once_with("job-blocking-run", students)
        self.assertIs(run_workflow.call_args.kwargs["progress_callback"], callback)

    def test_stream_route_requires_matching_job_header_before_starting_worker(self):
        client = TestClient(api.app)
        response = client.post(
            "/matching/hackathon/run/stream",
            json={"students": build_students()},
        )

        self.assertEqual(422, response.status_code)
        self.assertIn("X-Matching-Job-Id", response.text)

    def test_initial_stream_emits_ordered_team_versions_and_completed_result(self):
        students = build_students()
        preview = build_team()
        updated = build_team("구현 역할이 연결됩니다.", "협업 규칙이 필요합니다.")
        ready = build_team(
            updated["strengths"],
            updated["weaknesses"],
            [{"title": "역할 연결", "description": "두 역할의 구현 흐름을 연결했습니다."}],
        )
        result = build_result(ready)

        def fake_workflow(**kwargs):
            callback = kwargs["progress_callback"]
            callback("team_preview", [preview])
            callback("team_update", [updated])
            callback("team_ready", [ready])
            return result

        with (
            patch.object(analysis, "get_analyze_stu", return_value=students),
            patch.object(workflow, "run_workflow", side_effect=fake_workflow),
            patch.object(api, "save_matching_result") as save_result,
        ):
            response = api.stream_hackathon_matching(
                {"students": students},
                matching_job_id="job-stream-1",
            )
            events = parse_events(asyncio.run(read_stream(response)))

        self.assertTrue(response.media_type.startswith("text/event-stream"))
        self.assertEqual(
            [
                "started",
                "progress",
                "progress",
                "progress",
                "team_preview",
                "progress",
                "team_update",
                "team_ready",
                "progress",
                "completed",
            ],
            [event["event"] for event in events],
        )
        self.assertEqual(list(range(1, len(events) + 1)), [event["id"] for event in events])
        self.assertEqual(
            ["ANALYZING", "MATCHING", "VALIDATING", "EXPLAINING", "SAVING"],
            [event["data"]["stage"] for event in events if event["event"] == "progress"],
        )
        team_events = [event for event in events if event["event"].startswith("team_")]
        self.assertEqual([1, 2, 3], [event["data"]["version"] for event in team_events])
        self.assertTrue(all(event["data"]["team_name"] == "팀 1" for event in team_events))
        self.assertTrue(all(event["data"]["job_id"] == "job-stream-1" for event in events))
        self.assertEqual(1, events[-1]["data"]["result"]["total_teams"])
        completed_members = [
            member["name"]
            for team in events[-1]["data"]["result"]["teams"]
            for member in team["members"]
        ]
        self.assertEqual({"학생1", "학생2"}, set(completed_members))
        self.assertEqual(len(completed_members), len(set(completed_members)))
        save_result.assert_called_once_with(
            workflow.build_public_workflow_result(result),
            matching_type="HACKATHON",
        )

    def test_capstone_stream_emits_teams_and_saves_capstone_result(self):
        students = build_students()
        ready = build_team(
            "역할이 연결됩니다.",
            "협업 규칙이 필요합니다.",
            [{"title": "역할 연결", "description": "두 역할의 구현 흐름을 연결했습니다."}],
        )
        result = build_result(ready)

        def fake_workflow(**kwargs):
            callback = kwargs["progress_callback"]
            callback("team_preview", [ready])
            callback("team_update", [ready])
            callback("team_ready", [ready])
            return result

        with (
            patch.object(analysis, "get_analyze_stu", return_value=students),
            patch.object(workflow, "run_workflow", side_effect=fake_workflow),
            patch.object(api, "save_matching_result") as save_result,
        ):
            response = api.stream_matching(
                {"students": students},
                matching_job_id="job-grade-3-stream",
            )
            events = parse_events(asyncio.run(read_stream(response)))

        self.assertEqual("INITIAL", events[0]["data"]["mode"])
        self.assertEqual("completed", events[-1]["event"])
        self.assertEqual(
            ["team_preview", "team_update", "team_ready"],
            [event["event"] for event in events if event["event"].startswith("team_")],
        )
        save_result.assert_called_once_with(
            workflow.build_public_workflow_result(result),
            matching_type="CAPSTONE",
        )

    def test_regeneration_stream_uses_saved_teams_and_capstone_regeneration(self):
        students = build_students()
        ready = build_team(
            "역할이 연결됩니다.",
            "검토가 필요합니다.",
            [{"title": "재조정", "description": "요청한 역할 균형을 반영했습니다."}],
        )
        saved_result = build_result(build_team())
        result = build_result(ready)

        def fake_regenerate(**kwargs):
            callback = kwargs["progress_callback"]
            callback("team_preview", [ready])
            callback("team_update", [ready])
            callback("team_ready", [ready])
            return result

        with (
            patch.object(api, "load_matching_output", return_value=saved_result),
            patch.object(workflow, "run_regenerate_workflow", side_effect=fake_regenerate) as regenerate,
            patch.object(api, "save_matching_result"),
        ):
            response = api.stream_regenerate_hackathon_matching(
                {"prompt": "역할 균형을 다시 확인해줘"},
                matching_job_id="job-stream-2",
            )
            events = parse_events(asyncio.run(read_stream(response)))

        self.assertEqual("completed", events[-1]["event"])
        kwargs = regenerate.call_args.kwargs
        self.assertEqual("역할 균형을 다시 확인해줘", kwargs["prompt"])
        self.assertEqual(saved_result["final_result"]["final_teams"], kwargs["current_teams"])
        self.assertFalse(kwargs["persist_result"])

    def test_runtime_error_is_sent_as_error_event_without_saving(self):
        students = build_students()
        with (
            patch.object(analysis, "get_analyze_stu", return_value=students),
            patch.object(workflow, "run_workflow", side_effect=RuntimeError("LLM 연결 실패")),
            patch.object(api, "save_matching_result") as save_result,
        ):
            response = api.stream_hackathon_matching(
                {"students": students},
                matching_job_id="job-stream-error",
            )
            events = parse_events(asyncio.run(read_stream(response)))

        self.assertEqual("error", events[-1]["event"])
        self.assertEqual("MATCHING", events[-1]["data"]["stage"])
        self.assertNotIn("completed", [event["event"] for event in events])
        save_result.assert_not_called()

    def test_worker_continues_after_stream_consumer_disconnects(self):
        students = build_students()
        release_worker = Event()
        save_finished = Event()
        result = build_result(build_team())

        def delayed_workflow(**kwargs):
            release_worker.wait(timeout=2)
            callback = kwargs["progress_callback"]
            for event_type in ("team_preview", "team_update", "team_ready"):
                callback(event_type, result["final_result"]["final_teams"])
            return result

        def mark_saved(*args, **kwargs):
            save_finished.set()

        with (
            patch.object(analysis, "get_analyze_stu", return_value=students),
            patch.object(workflow, "run_workflow", side_effect=delayed_workflow),
            patch.object(api, "save_matching_result", side_effect=mark_saved),
        ):
            response = api.stream_hackathon_matching(
                {"students": students},
                matching_job_id="job-stream-disconnect",
            )
            first_frame = asyncio.run(read_stream(response, first_event_only=True))
            release_worker.set()
            self.assertTrue(save_finished.wait(timeout=2))

        self.assertEqual("started", parse_events(first_frame)[0]["event"])

    def test_failed_batch_still_reports_fallback_teams(self):
        teams = [{"team_name": "팀 1"}, {"team_name": "팀 2"}]
        completed_batches = []

        def failing_worker(batch, analyzed_students):
            raise RuntimeError("설명 생성 실패")

        with patch.dict(os.environ, {"STREAM_TEST_WORKERS": "1", "STREAM_TEST_BATCH": "1"}):
            result = workflow.run_parallel_team_batches(
                final_teams=teams,
                analyzed_students=[],
                worker_fn=failing_worker,
                worker_env_name="STREAM_TEST_WORKERS",
                batch_env_name="STREAM_TEST_BATCH",
                error_fields={"strengths": "", "weaknesses": ""},
                error_key="analysis_generation_error",
                on_batch_complete=completed_batches.append,
            )

        self.assertEqual(2, len(completed_batches))
        self.assertEqual(["팀 1", "팀 2"], [team["team_name"] for team in result])
        self.assertTrue(all("analysis_generation_error" in team for team in result))


if __name__ == "__main__":
    unittest.main()
