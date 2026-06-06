from ortools.sat.python import cp_model
import math


def create_optimized_teams(student_summaries, team_size=4):

    student_count = len(student_summaries)
    team_count = math.ceil(student_count / team_size)

    model = cp_model.CpModel()

    x = {}

    for i in range(student_count):
        for j in range(team_count):
            x[i, j] = model.NewBoolVar(f"x_{i}_{j}")

    for i in range(student_count):
        model.Add(
            sum(x[i, j] for j in range(team_count))
            == 1
        )

    # 팀 인원 균형 제한
    min_team_size = student_count // team_count
    max_team_size = math.ceil(student_count / team_count)

    for j in range(team_count):

        member_count = sum(
            x[i, j]
            for i in range(student_count)
        )

        model.Add(member_count >= min_team_size)
        model.Add(member_count <= max_team_size)

    
    
    # 역할군 정보 추출 (frontend, backend, ai_data, game 등)

    scores = [
        int(student["score"] * 100)
        for student in student_summaries
    ]

    def normalize_role(role: str) -> str:
        role = role.lower()

        if "frontend" in role:
            return "Frontend"

        if "unity" in role or "game" in role:
            return "Game"

        if "ai" in role:
            return "AI"

        if "데이터" in role or "머신러닝" in role:
            return "AI"

        if "backend" in role or "백엔드" in role:
            return "Backend"

        return "Unknown"
    
    role_groups = [
        normalize_role(student.get("role", ""))
        for student in student_summaries
    ]

    # 균형 배치를 적용할 핵심 역할군
    core_roles = ["Frontend", "Backend", "AI"]

    # 역할군 편차 패널티 저장
    role_penalties = []

    for role in core_roles:

        role_counts = []

        for j in range(team_count):

            role_count = model.NewIntVar(
                0,
                team_size,
                f"{role}_count_{j}"
            )

            model.Add(
                role_count == sum(
                    x[i, j]
                    for i in range(student_count)
                    if role_groups[i] == role
                )
            )

            role_counts.append(role_count)

        role_max = model.NewIntVar(
            0,
            team_size,
            f"{role}_max"
        )

        role_min = model.NewIntVar(
            0,
            team_size,
            f"{role}_min"
        )

        model.AddMaxEquality(role_max, role_counts)
        model.AddMinEquality(role_min, role_counts)

        # 역할군 최대 인원 팀과 최소 인원 팀의 차이
        penalty = model.NewIntVar(
            0,
            team_size,
            f"{role}_penalty"
        )

        model.Add(penalty == role_max - role_min)

        role_penalties.append(penalty)

    team_scores = []

    for j in range(team_count):

        team_score = model.NewIntVar(
            0,
            100000,
            f"team_score_{j}"
        )

        model.Add(
            team_score ==
            sum(
                scores[i] * x[i, j]
                for i in range(student_count)
            )
        )

        team_scores.append(team_score)

    max_score = model.NewIntVar(
        0,
        100000,
        "max_score"
    )

    min_score = model.NewIntVar(
        0,
        100000,
        "min_score"
    )

    model.AddMaxEquality(
        max_score,
        team_scores
    )

    model.AddMinEquality(
        min_score,
        team_scores
    )

    score_gap = model.NewIntVar(
        0,
        100000,
        "score_gap"
    )

    model.Add(
        score_gap == max_score - min_score
    )

    # 역할군 분포가 한쪽 팀에 몰리지 않도록 패널티 적용 (ex: FE 5 혹은 BE 4, AI 1 같은 상황 대비)
    total_role_penalty = sum(role_penalties)

    # 1순위: 팀 점수 균형
    # 2순위: FE/BE/AI 역할군 균형
    model.Minimize(
        score_gap * 100
        + total_role_penalty * 10
    )

    solver = cp_model.CpSolver()

    status = solver.Solve(model)

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE
    ):
        return []

    teams = []

    for j in range(team_count):

        members = []
        total_score = 0

        # 추후 게임 팀 생성을 위해 게임 개발자 수 집계
        game_count = 0
        role_groups = {}


        for i in range(student_count):

            if solver.Value(x[i, j]):

                student = student_summaries[i]

                members.append(student)

                total_score += student["score"]

                role = normalize_role(student.get("role", ""))
                # 게임 역할 인원 수 계산
                if role == "Game":
                    game_count += 1

                role_groups[role] = (
                    role_groups.get(role, 0)
                    + 1
                )

        teams.append({
            "team_name": f"팀 {j+1}",
            "members": members,
            "total_score": round(total_score, 2),
            "role_groups": role_groups,
            "game_count": game_count,
        })

    return teams
