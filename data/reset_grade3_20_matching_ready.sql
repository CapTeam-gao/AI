SET NAMES utf8mb4;
START TRANSACTION;

-- GRADE_3 팀에서 파생된 운영 데이터를 먼저 제거한다.
DELETE crs
FROM chat_read_status crs
JOIN chat_channels cc ON cc.id = crs.channel_id
JOIN chat_rooms cr ON cr.id = cc.chat_room_id
JOIN teams t ON t.id = cr.team_id
WHERE t.grade = 'GRADE_3';

DELETE cm
FROM chat_messages cm
JOIN chat_channels cc ON cc.id = cm.channel_id
JOIN chat_rooms cr ON cr.id = cc.chat_room_id
JOIN teams t ON t.id = cr.team_id
WHERE t.grade = 'GRADE_3';

DELETE cc
FROM chat_channels cc
JOIN chat_rooms cr ON cr.id = cc.chat_room_id
JOIN teams t ON t.id = cr.team_id
WHERE t.grade = 'GRADE_3';

DELETE cr
FROM chat_rooms cr
JOIN teams t ON t.id = cr.team_id
WHERE t.grade = 'GRADE_3';

DELETE je
FROM journal_entries je
JOIN journals j ON j.id = je.journal_id
JOIN teams t ON t.id = j.team_id
WHERE t.grade = 'GRADE_3';

DELETE j
FROM journals j
JOIN teams t ON t.id = j.team_id
WHERE t.grade = 'GRADE_3';

DELETE FROM team_projects WHERE team_id IN (SELECT id FROM teams WHERE grade = 'GRADE_3');
DELETE FROM team_members WHERE team_id IN (SELECT id FROM teams WHERE grade = 'GRADE_3');
DELETE FROM teams WHERE grade = 'GRADE_3';

DELETE FROM recommendation_reasons
WHERE recommendation_id IN (SELECT id FROM team_recommendations WHERE grade = 'GRADE_3');

DELETE FROM team_recommendation_members
WHERE recommendation_id IN (SELECT id FROM team_recommendations WHERE grade = 'GRADE_3');

DELETE FROM team_recommendations WHERE grade = 'GRADE_3';
DELETE FROM matching_jobs WHERE grade = 'GRADE_3';

-- GRADE_3 학생이 다른 공용 데이터에 남긴 참조도 함께 제거한다.
DELETE nr
FROM notice_reads nr
JOIN notices n ON n.id = nr.notice_id
WHERE n.result_grade = 'GRADE_3'
   OR n.writer_id IN (
       SELECT user_id FROM users WHERE account_role = 'STUDENT' AND grade = 'GRADE_3'
   );

DELETE FROM notices
WHERE result_grade = 'GRADE_3'
   OR writer_id IN (
       SELECT user_id FROM users WHERE account_role = 'STUDENT' AND grade = 'GRADE_3'
   );

DELETE crs
FROM chat_read_status crs
JOIN users u ON u.user_id = crs.user_id
WHERE u.account_role = 'STUDENT' AND u.grade = 'GRADE_3';

DELETE cm
FROM chat_messages cm
JOIN users u ON u.user_id = cm.sender_id
WHERE u.account_role = 'STUDENT' AND u.grade = 'GRADE_3';

DELETE je
FROM journal_entries je
JOIN users u ON u.user_id = je.writer_id
WHERE u.account_role = 'STUDENT' AND u.grade = 'GRADE_3';

DELETE nr
FROM notice_reads nr
JOIN users u ON u.user_id = nr.user_id
WHERE u.account_role = 'STUDENT' AND u.grade = 'GRADE_3';

DELETE rt
FROM refresh_tokens rt
JOIN users u ON u.user_id = rt.user_id
WHERE u.account_role = 'STUDENT' AND u.grade = 'GRADE_3';

DELETE trm
FROM team_recommendation_members trm
JOIN users u ON u.user_id = trm.user_id
WHERE u.account_role = 'STUDENT' AND u.grade = 'GRADE_3';

DELETE tm
FROM team_members tm
JOIN users u ON u.user_id = tm.user_id
WHERE u.account_role = 'STUDENT' AND u.grade = 'GRADE_3';

DELETE ua
FROM user_analysis ua
JOIN users u ON u.user_id = ua.user_id
WHERE u.account_role = 'STUDENT' AND u.grade = 'GRADE_3';

DELETE us
FROM user_skill us
JOIN users u ON u.user_id = us.user_user_id
WHERE u.account_role = 'STUDENT' AND u.grade = 'GRADE_3';

DELETE ue
FROM user_experience ue
JOIN users u ON u.user_id = ue.user_user_id
WHERE u.account_role = 'STUDENT' AND u.grade = 'GRADE_3';

DELETE up
FROM user_preferred_teammates up
JOIN users u ON u.user_id = up.user_user_id
WHERE u.account_role = 'STUDENT' AND u.grade = 'GRADE_3';

DELETE FROM users WHERE account_role = 'STUDENT' AND grade = 'GRADE_3';

DELETE FROM student_analysis_results
WHERE JSON_UNQUOTE(JSON_EXTRACT(result_json, '$[0].grade')) = 'GRADE_3';

DELETE FROM team_matching_results
WHERE JSON_UNQUOTE(JSON_EXTRACT(result_json, '$.analyzed_students[0].grade')) = 'GRADE_3';

-- 설문 완료 학생 20명: 높음 5명, 보통 11명, 낮음 4명으로 구성한다.
INSERT INTO users (
    user_id, created_at, updated_at, account_role,
    development_implementation, development_leadership, development_learning_ability,
    development_planning, development_problem_solving,
    grade, name, password, password_encoded,
    personality_collaboration, personality_communication, personality_emotional_stability,
    personality_flexibility, personality_responsibility,
    student_role, survey_completed, wants_leader
) VALUES
('stu3test01', NOW(6), NOW(6), 'STUDENT', 4.8, 4.4, 4.5, 4.2, 4.7, 'GRADE_3', '3학년테스트01', '1234', b'0', 4.3, 4.2, 4.1, 4.0, 4.6, 'BACKEND',   b'1', b'1'),
('stu3test02', NOW(6), NOW(6), 'STUDENT', 4.0, 3.5, 4.0, 3.7, 4.1, 'GRADE_3', '3학년테스트02', '1234', b'0', 3.9, 4.1, 3.8, 4.0, 4.1, 'FRONTEND', b'1', b'0'),
('stu3test03', NOW(6), NOW(6), 'STUDENT', 3.9, 3.4, 4.2, 3.6, 4.0, 'GRADE_3', '3학년테스트03', '1234', b'0', 3.8, 3.9, 3.7, 4.1, 3.9, 'AI',       b'1', b'0'),
('stu3test04', NOW(6), NOW(6), 'STUDENT', 2.5, 2.8, 3.1, 2.7, 2.6, 'GRADE_3', '3학년테스트04', '1234', b'0', 3.2, 3.0, 3.1, 3.3, 3.4, 'APP',      b'1', b'0'),
('stu3test05', NOW(6), NOW(6), 'STUDENT', 3.7, 4.1, 3.8, 4.4, 3.8, 'GRADE_3', '3학년테스트05', '1234', b'0', 4.3, 4.4, 4.0, 4.2, 4.1, 'DESIGN',   b'1', b'1'),
('stu3test06', NOW(6), NOW(6), 'STUDENT', 4.7, 4.1, 4.5, 4.0, 4.6, 'GRADE_3', '3학년테스트06', '1234', b'0', 4.0, 4.1, 4.2, 4.3, 4.5, 'DEVOPS',   b'1', b'0'),
('stu3test07', NOW(6), NOW(6), 'STUDENT', 3.8, 3.6, 4.0, 3.5, 4.1, 'GRADE_3', '3학년테스트07', '1234', b'0', 4.0, 3.8, 3.9, 3.7, 4.2, 'GAME',     b'1', b'0'),
('stu3test08', NOW(6), NOW(6), 'STUDENT', 4.1, 3.7, 4.0, 3.8, 4.2, 'GRADE_3', '3학년테스트08', '1234', b'0', 3.8, 4.0, 3.7, 4.1, 4.0, 'FULLSTACK',b'1', b'0'),
('stu3test09', NOW(6), NOW(6), 'STUDENT', 2.6, 2.7, 3.2, 2.8, 2.9, 'GRADE_3', '3학년테스트09', '1234', b'0', 3.4, 3.2, 3.5, 3.1, 3.6, 'SECURITY', b'1', b'0'),
('stu3test10', NOW(6), NOW(6), 'STUDENT', 4.0, 4.0, 3.9, 3.8, 4.1, 'GRADE_3', '3학년테스트10', '1234', b'0', 4.1, 4.0, 3.9, 3.8, 4.2, 'BACKEND',  b'1', b'0'),
('stu3test11', NOW(6), NOW(6), 'STUDENT', 4.8, 4.5, 4.4, 4.3, 4.6, 'GRADE_3', '3학년테스트11', '1234', b'0', 4.4, 4.6, 4.1, 4.2, 4.5, 'FRONTEND', b'1', b'1'),
('stu3test12', NOW(6), NOW(6), 'STUDENT', 3.8, 3.3, 4.3, 3.5, 4.0, 'GRADE_3', '3학년테스트12', '1234', b'0', 3.7, 3.8, 3.6, 4.0, 3.9, 'AI',       b'1', b'0'),
('stu3test13', NOW(6), NOW(6), 'STUDENT', 4.0, 3.6, 4.1, 3.7, 4.0, 'GRADE_3', '3학년테스트13', '1234', b'0', 3.9, 4.0, 3.8, 4.1, 4.0, 'APP',      b'1', b'0'),
('stu3test14', NOW(6), NOW(6), 'STUDENT', 2.4, 2.9, 3.0, 3.1, 2.7, 'GRADE_3', '3학년테스트14', '1234', b'0', 3.5, 3.3, 3.4, 3.2, 3.7, 'DESIGN',   b'1', b'0'),
('stu3test15', NOW(6), NOW(6), 'STUDENT', 3.9, 4.2, 4.0, 4.1, 4.1, 'GRADE_3', '3학년테스트15', '1234', b'0', 4.2, 4.1, 4.0, 3.9, 4.3, 'DEVOPS',   b'1', b'1'),
('stu3test16', NOW(6), NOW(6), 'STUDENT', 4.7, 4.0, 4.5, 4.1, 4.6, 'GRADE_3', '3학년테스트16', '1234', b'0', 4.1, 4.0, 4.2, 4.3, 4.4, 'GAME',     b'1', b'0'),
('stu3test17', NOW(6), NOW(6), 'STUDENT', 4.1, 3.8, 4.2, 3.9, 4.2, 'GRADE_3', '3학년테스트17', '1234', b'0', 4.0, 4.1, 3.8, 4.0, 4.1, 'FULLSTACK',b'1', b'0'),
('stu3test18', NOW(6), NOW(6), 'STUDENT', 2.7, 2.8, 3.1, 2.6, 2.8, 'GRADE_3', '3학년테스트18', '1234', b'0', 3.1, 3.3, 3.2, 3.4, 3.5, 'SECURITY', b'1', b'0'),
('stu3test19', NOW(6), NOW(6), 'STUDENT', 4.8, 4.2, 4.6, 4.1, 4.7, 'GRADE_3', '3학년테스트19', '1234', b'0', 4.2, 4.3, 4.1, 4.0, 4.6, 'BACKEND',  b'1', b'0'),
('stu3test20', NOW(6), NOW(6), 'STUDENT', 4.0, 3.7, 4.1, 3.8, 4.0, 'GRADE_3', '3학년테스트20', '1234', b'0', 4.1, 4.2, 3.9, 4.0, 4.2, 'FRONTEND', b'1', b'0');

INSERT INTO user_skill (user_user_id, skill) VALUES
('stu3test01', 'Spring Boot'), ('stu3test01', 'JPA'), ('stu3test01', 'MySQL'), ('stu3test01', 'Redis'),
('stu3test02', 'React'), ('stu3test02', 'TypeScript'), ('stu3test02', 'TanStack Query'), ('stu3test02', 'CSS Modules'),
('stu3test03', 'Python'), ('stu3test03', 'Pandas'), ('stu3test03', 'scikit-learn'), ('stu3test03', 'FastAPI'),
('stu3test04', 'Flutter'), ('stu3test04', 'Dart'), ('stu3test04', 'Firebase'),
('stu3test05', 'Figma'), ('stu3test05', 'User Flow'), ('stu3test05', 'Wireframe'), ('stu3test05', 'Design System'),
('stu3test06', 'Docker'), ('stu3test06', 'GitHub Actions'), ('stu3test06', 'Nginx'), ('stu3test06', 'AWS EC2'),
('stu3test07', 'Unity'), ('stu3test07', 'C#'), ('stu3test07', 'Photon'),
('stu3test08', 'React'), ('stu3test08', 'Node.js'), ('stu3test08', 'Express'), ('stu3test08', 'MySQL'),
('stu3test09', 'Linux'), ('stu3test09', 'JWT'), ('stu3test09', 'OWASP'),
('stu3test10', 'Spring Boot'), ('stu3test10', 'JPA'), ('stu3test10', 'JWT'), ('stu3test10', 'MySQL'),
('stu3test11', 'Next.js'), ('stu3test11', 'React'), ('stu3test11', 'TypeScript'), ('stu3test11', 'Zustand'),
('stu3test12', 'Python'), ('stu3test12', 'PyTorch'), ('stu3test12', 'OpenCV'), ('stu3test12', 'FastAPI'),
('stu3test13', 'Kotlin'), ('stu3test13', 'Android Jetpack'), ('stu3test13', 'Room'), ('stu3test13', 'Retrofit'),
('stu3test14', 'Figma'), ('stu3test14', 'Prototyping'), ('stu3test14', 'Usability Test'),
('stu3test15', 'Docker'), ('stu3test15', 'Jenkins'), ('stu3test15', 'Linux'), ('stu3test15', 'Nginx'),
('stu3test16', 'Unity'), ('stu3test16', 'C#'), ('stu3test16', 'Photon'), ('stu3test16', 'Addressables'),
('stu3test17', 'Next.js'), ('stu3test17', 'NestJS'), ('stu3test17', 'PostgreSQL'), ('stu3test17', 'Docker'),
('stu3test18', 'Linux'), ('stu3test18', 'OWASP'), ('stu3test18', 'Burp Suite'),
('stu3test19', 'Spring Security'), ('stu3test19', 'Redis'), ('stu3test19', 'Spring Batch'), ('stu3test19', 'PostgreSQL'),
('stu3test20', 'Vue'), ('stu3test20', 'TypeScript'), ('stu3test20', 'Pinia'), ('stu3test20', 'Vite');

INSERT INTO user_experience (user_user_id, experience) VALUES
('stu3test01', 'Spring Boot와 JPA로 팀 관리 API를 설계하고 Redis 캐시를 적용해 반복 조회 성능을 개선했습니다.'),
('stu3test01', '로그와 쿼리 실행 계획을 확인해 N+1 문제를 찾고 fetch join으로 해결했습니다.'),
('stu3test02', 'React와 TypeScript로 관리자 목록, 검색, 페이지네이션 화면을 구현했습니다.'),
('stu3test02', 'TanStack Query를 사용해 서버 상태와 로딩 및 오류 화면을 관리했습니다.'),
('stu3test03', 'Pandas로 설문 데이터를 정제하고 간단한 분류 모델을 학습했습니다.'),
('stu3test03', 'FastAPI로 모델 추론 결과를 반환하는 API를 구현했습니다.'),
('stu3test04', 'Flutter로 로그인 화면과 게시글 목록 화면을 구현했습니다.'),
('stu3test04', 'Firebase 예제를 따라 데이터를 저장하고 조회해 본 경험이 있습니다.'),
('stu3test05', 'Figma로 사용자 흐름과 핵심 화면 와이어프레임을 설계했습니다.'),
('stu3test05', '개발자와 컴포넌트 규칙을 정리하고 구현 화면을 디자인 QA했습니다.'),
('stu3test06', 'Docker Compose로 백엔드와 DB 환경을 구성하고 GitHub Actions 배포를 자동화했습니다.'),
('stu3test06', 'Nginx 리버스 프록시와 로그를 설정하고 배포 장애 원인을 분석했습니다.'),
('stu3test07', 'Unity로 2D 미니게임의 캐릭터 이동과 점수 시스템을 구현했습니다.'),
('stu3test07', 'Photon을 사용해 방 생성과 플레이어 입장 기능을 연결했습니다.'),
('stu3test08', 'React와 Express로 게시판 CRUD를 만들고 MySQL 데이터까지 연동했습니다.'),
('stu3test08', '프론트와 백엔드의 API 명세를 맞추며 오류 응답을 처리했습니다.'),
('stu3test09', 'JWT 로그인 흐름과 OWASP 기본 항목을 학습하고 예제 코드를 점검했습니다.'),
('stu3test09', 'Linux에서 권한 설정과 간단한 보안 명령어를 실습했습니다.'),
('stu3test10', 'Spring Boot와 JPA로 회원 및 게시판 CRUD API를 구현했습니다.'),
('stu3test10', 'JWT 인증과 공통 예외 응답을 적용해 프론트엔드와 연동했습니다.'),
('stu3test11', 'Next.js 관리자 대시보드를 구현하고 렌더링 병목을 분석해 불필요한 재렌더링을 줄였습니다.'),
('stu3test11', '공통 컴포넌트와 상태 구조를 설계하고 코드 리뷰 기준을 정리했습니다.'),
('stu3test12', 'PyTorch로 이미지 분류 모델을 학습하고 OpenCV 전처리를 적용했습니다.'),
('stu3test12', 'FastAPI 추론 API를 만들고 confidence 값을 함께 반환했습니다.'),
('stu3test13', 'Kotlin과 Android Jetpack으로 로그인, 목록, 상세 화면을 구현했습니다.'),
('stu3test13', 'Retrofit API 연동과 Room 임시 저장 기능을 적용했습니다.'),
('stu3test14', 'Figma로 간단한 앱 화면과 클릭 가능한 프로토타입을 만들었습니다.'),
('stu3test14', '사용자 피드백을 받아 버튼 위치와 문구를 수정했습니다.'),
('stu3test15', 'Docker와 Jenkins로 빌드 및 배포 파이프라인을 구성했습니다.'),
('stu3test15', 'Linux 서버 로그를 확인하고 Nginx 설정 오류를 해결했습니다.'),
('stu3test16', 'Unity와 Photon으로 협동 게임의 상태 동기화와 재입장 흐름을 구현했습니다.'),
('stu3test16', 'Addressables로 리소스 로딩을 분리하고 메모리 사용량을 점검했습니다.'),
('stu3test17', 'Next.js와 NestJS로 로그인부터 게시글 작성까지 전체 흐름을 구현했습니다.'),
('stu3test17', 'PostgreSQL 스키마와 REST API를 설계하고 Docker 실행 환경을 구성했습니다.'),
('stu3test18', 'OWASP 문서를 참고해 입력값 검증 항목을 확인했습니다.'),
('stu3test18', 'Burp Suite 기본 기능으로 예제 요청을 확인해 본 경험이 있습니다.'),
('stu3test19', 'Spring Security와 Redis로 인증 및 토큰 재발급 구조를 구현했습니다.'),
('stu3test19', 'Spring Batch 처리 성능을 측정하고 대량 저장 방식을 개선했습니다.'),
('stu3test20', 'Vue와 Pinia로 통계 대시보드와 검색 필터를 구현했습니다.'),
('stu3test20', 'TypeScript로 API 응답 타입을 정의하고 로딩 및 오류 상태를 처리했습니다.');

INSERT INTO user_preferred_teammates (user_user_id, preferred_teammates) VALUES
('stu3test01', 'stu3test02'),
('stu3test03', 'stu3test08'),
('stu3test06', 'stu3test15'),
('stu3test11', 'stu3test20'),
('stu3test16', 'stu3test07');

-- 백엔드 화면과 매칭 준비 단계에서 사용하는 학생별 분석 완료 상태를 저장한다.
INSERT INTO user_analysis (
    user_id, created_at, updated_at, analysis_result, student_level,
    development_inconsistent_count, inconsistent_answers,
    personality_inconsistent_count, response_reliability
)
SELECT
    u.user_id,
    NOW(6),
    NOW(6),
    CASE
        WHEN u.user_id IN ('stu3test01', 'stu3test06', 'stu3test11', 'stu3test16', 'stu3test19')
            THEN CONCAT(u.student_role, ' 분야에서 설계, 구현, 개선 경험이 확인되어 높은 수준으로 분석되었습니다.')
        WHEN u.user_id IN ('stu3test04', 'stu3test09', 'stu3test14', 'stu3test18')
            THEN CONCAT(u.student_role, ' 분야의 기초 구현 경험이 있으며 팀원의 지원을 통해 성장할 수 있는 수준으로 분석되었습니다.')
        ELSE CONCAT(u.student_role, ' 분야에서 여러 기능을 직접 구현한 경험이 있어 보통 수준으로 분석되었습니다.')
    END,
    CASE
        WHEN u.user_id IN ('stu3test01', 'stu3test06', 'stu3test11', 'stu3test16', 'stu3test19') THEN 'UPPER'
        WHEN u.user_id IN ('stu3test04', 'stu3test09', 'stu3test14', 'stu3test18') THEN 'LOWER'
        ELSE 'MIDDLE'
    END,
    0, 0, 0, 'HIGH'
FROM users u
WHERE u.account_role = 'STUDENT' AND u.grade = 'GRADE_3';

-- AI 서버가 분석을 다시 호출하지 않고 사용할 수 있는 최신 분석 캐시를 생성한다.
INSERT INTO student_analysis_results (result_type, result_json)
SELECT
    'analysis',
    JSON_ARRAYAGG(
        JSON_OBJECT(
            'user_id', u.user_id,
            'name', u.name,
            'role', u.student_role,
            'grade', u.grade,
            'stack', COALESCE(
                (SELECT JSON_ARRAYAGG(us.skill) FROM user_skill us WHERE us.user_user_id = u.user_id),
                JSON_ARRAY()
            ),
            'experience', COALESCE(
                (SELECT JSON_ARRAYAGG(ue.experience) FROM user_experience ue WHERE ue.user_user_id = u.user_id),
                JSON_ARRAY()
            ),
            'preferred_members', COALESCE(
                (SELECT JSON_ARRAYAGG(up.preferred_teammates) FROM user_preferred_teammates up WHERE up.user_user_id = u.user_id),
                JSON_ARRAY()
            ),
            'wants_leader', IF(u.wants_leader = b'1', TRUE, FALSE),
            'skill_level', CASE ua.student_level WHEN 'UPPER' THEN '높음' WHEN 'LOWER' THEN '낮음' ELSE '보통' END,
            'student_level', ua.student_level,
            'stack_score', CONCAT(
                COALESCE(
                    (SELECT GROUP_CONCAT(
                        CONCAT(us.skill, ': ', CASE ua.student_level WHEN 'UPPER' THEN 5 WHEN 'LOWER' THEN 2 ELSE 4 END, '점')
                        SEPARATOR '\n'
                    ) FROM user_skill us WHERE us.user_user_id = u.user_id),
                    ''
                )
            ),
            'analysis_result', ua.analysis_result,
            'reason', ua.analysis_result,
            'strength', CONCAT(u.student_role, ' 역할의 기술 스택과 구현 경험을 보유하고 있습니다.'),
            'weakness', CASE ua.student_level
                WHEN 'UPPER' THEN '특정 역할에 업무가 집중되지 않도록 팀 내 지식 공유가 필요합니다.'
                WHEN 'LOWER' THEN '복잡한 기능은 경험이 많은 팀원의 리뷰와 지원이 필요합니다.'
                ELSE '설계와 운영 경험을 추가로 쌓으면 더 안정적인 구현이 가능합니다.'
            END,
            'suggestion', CONCAT(u.student_role, ' 역할을 중심으로 다른 역할군과 협업하는 구성이 적합합니다.'),
            'analysis_status', 'SUCCESS',
            'response_reliability', ua.response_reliability,
            'communication', u.personality_communication,
            'responsibility', u.personality_responsibility,
            'collaboration', u.personality_collaboration,
            'flexibility', u.personality_flexibility,
            'emotionalStability', u.personality_emotional_stability,
            'leadership', u.development_leadership,
            'problemSolving', u.development_problem_solving,
            'implementation', u.development_implementation,
            'learningAbility', u.development_learning_ability,
            'planning', u.development_planning
        )
    )
FROM users u
JOIN user_analysis ua ON ua.user_id = u.user_id
WHERE u.account_role = 'STUDENT' AND u.grade = 'GRADE_3';

COMMIT;
