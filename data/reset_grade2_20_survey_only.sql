SET FOREIGN_KEY_CHECKS=0;

DELETE crs
FROM chat_read_status crs
JOIN chat_channels cc ON crs.channel_id = cc.id
JOIN chat_rooms cr ON cc.chat_room_id = cr.id
JOIN teams t ON cr.team_id = t.id
WHERE t.grade = 'GRADE_2';

DELETE cm
FROM chat_messages cm
JOIN chat_channels cc ON cm.channel_id = cc.id
JOIN chat_rooms cr ON cc.chat_room_id = cr.id
JOIN teams t ON cr.team_id = t.id
WHERE t.grade = 'GRADE_2';

DELETE cc
FROM chat_channels cc
JOIN chat_rooms cr ON cc.chat_room_id = cr.id
JOIN teams t ON cr.team_id = t.id
WHERE t.grade = 'GRADE_2';

DELETE cr
FROM chat_rooms cr
JOIN teams t ON cr.team_id = t.id
WHERE t.grade = 'GRADE_2';

DELETE FROM team_projects WHERE team_id IN (SELECT id FROM teams WHERE grade = 'GRADE_2');
DELETE FROM team_members WHERE team_id IN (SELECT id FROM teams WHERE grade = 'GRADE_2');
DELETE FROM teams WHERE grade = 'GRADE_2';

DELETE FROM recommendation_reasons
WHERE recommendation_id IN (SELECT id FROM team_recommendations WHERE grade = 'GRADE_2');

DELETE FROM team_recommendation_members
WHERE recommendation_id IN (SELECT id FROM team_recommendations WHERE grade = 'GRADE_2');

DELETE FROM team_recommendations WHERE grade = 'GRADE_2';
DELETE FROM matching_jobs WHERE grade = 'GRADE_2';

DELETE FROM user_analysis WHERE user_id LIKE 'stu23%';
DELETE FROM user_skill WHERE user_user_id LIKE 'stu23%';
DELETE FROM user_experience WHERE user_user_id LIKE 'stu23%';
DELETE FROM user_preferred_teammates WHERE user_user_id LIKE 'stu23%';
DELETE FROM users WHERE user_id LIKE 'stu23%';

DELETE FROM student_analysis_results;
DELETE FROM team_matching_results;

SET FOREIGN_KEY_CHECKS=1;

INSERT INTO users (
    user_id, created_at, updated_at, account_role,
    development_implementation, development_leadership, development_learning_ability,
    development_planning, development_problem_solving,
    grade, name, password, password_encoded,
    personality_collaboration, personality_communication, personality_emotional_stability,
    personality_flexibility, personality_responsibility,
    student_role, survey_completed, wants_leader
) VALUES
('stu2301', NOW(6), NOW(6), 'STUDENT', 4.6, 4.1, 4.2, 3.8, 4.4, 'GRADE_2', '서강준', '1234', b'0', 4.0, 4.1, 3.7, 3.9, 4.3, 'BACKEND', b'1', b'1'),
('stu2302', NOW(6), NOW(6), 'STUDENT', 4.3, 3.2, 4.0, 3.7, 4.1, 'GRADE_2', '한유림', '1234', b'0', 3.8, 4.2, 3.6, 4.0, 3.9, 'FRONTEND', b'1', b'0'),
('stu2303', NOW(6), NOW(6), 'STUDENT', 4.1, 3.4, 4.5, 3.5, 4.3, 'GRADE_2', '오태민', '1234', b'0', 3.7, 3.8, 3.5, 4.1, 3.8, 'AI', b'1', b'0'),
('stu2304', NOW(6), NOW(6), 'STUDENT', 3.8, 3.6, 3.9, 4.4, 3.7, 'GRADE_2', '이다솜', '1234', b'0', 4.2, 4.3, 3.8, 4.5, 4.0, 'DESIGN', b'1', b'0'),
('stu2305', NOW(6), NOW(6), 'STUDENT', 4.0, 3.9, 4.1, 3.6, 4.0, 'GRADE_2', '문태준', '1234', b'0', 3.9, 3.7, 3.6, 4.0, 4.1, 'APP', b'1', b'0'),
('stu2306', NOW(6), NOW(6), 'STUDENT', 4.5, 3.8, 4.0, 3.4, 4.2, 'GRADE_2', '권서준', '1234', b'0', 3.6, 3.8, 3.4, 3.7, 4.2, 'DEVOPS', b'1', b'1'),
('stu2307', NOW(6), NOW(6), 'STUDENT', 3.9, 3.3, 4.2, 3.8, 4.0, 'GRADE_2', '정예나', '1234', b'0', 4.1, 4.4, 3.9, 4.3, 4.0, 'FRONTEND', b'1', b'0'),
('stu2308', NOW(6), NOW(6), 'STUDENT', 4.2, 3.1, 4.5, 3.3, 4.4, 'GRADE_2', '신우진', '1234', b'0', 3.5, 3.7, 3.6, 3.9, 3.8, 'AI', b'1', b'0'),
('stu2309', NOW(6), NOW(6), 'STUDENT', 3.7, 3.5, 3.8, 4.2, 3.6, 'GRADE_2', '김아린', '1234', b'0', 4.3, 4.5, 3.8, 4.2, 4.1, 'DESIGN', b'1', b'0'),
('stu2310', NOW(6), NOW(6), 'STUDENT', 4.4, 3.6, 3.9, 3.4, 4.1, 'GRADE_2', '민하준', '1234', b'0', 3.7, 3.8, 3.5, 3.9, 4.0, 'APP', b'1', b'0'),
('stu2311', NOW(6), NOW(6), 'STUDENT', 4.7, 4.2, 4.1, 3.9, 4.5, 'GRADE_2', '나현우', '1234', b'0', 4.0, 4.1, 3.9, 3.8, 4.4, 'BACKEND', b'1', b'1'),
('stu2312', NOW(6), NOW(6), 'STUDENT', 4.1, 3.7, 4.0, 4.0, 4.2, 'GRADE_2', '노시현', '1234', b'0', 3.9, 4.0, 3.7, 4.1, 4.0, 'FULLSTACK', b'1', b'0'),
('stu2313', NOW(6), NOW(6), 'STUDENT', 3.8, 3.4, 4.4, 3.7, 4.0, 'GRADE_2', '이건우', '1234', b'0', 3.6, 3.8, 3.5, 3.9, 3.7, 'SECURITY', b'1', b'0'),
('stu2314', NOW(6), NOW(6), 'STUDENT', 4.0, 3.5, 4.3, 3.8, 4.1, 'GRADE_2', '박서윤', '1234', b'0', 4.0, 4.2, 3.8, 4.0, 3.9, 'AI', b'1', b'0'),
('stu2315', NOW(6), NOW(6), 'STUDENT', 3.9, 3.3, 3.9, 4.3, 3.7, 'GRADE_2', '최도윤', '1234', b'0', 4.2, 4.1, 3.9, 4.4, 4.0, 'DESIGN', b'1', b'0'),
('stu2316', NOW(6), NOW(6), 'STUDENT', 4.2, 3.4, 4.0, 3.5, 4.0, 'GRADE_2', '윤지호', '1234', b'0', 3.7, 3.9, 3.6, 4.1, 3.9, 'GAME', b'1', b'0'),
('stu2317', NOW(6), NOW(6), 'STUDENT', 4.5, 3.7, 4.1, 3.6, 4.3, 'GRADE_2', '강민재', '1234', b'0', 3.8, 3.9, 3.6, 3.8, 4.1, 'BACKEND', b'1', b'0'),
('stu2318', NOW(6), NOW(6), 'STUDENT', 4.0, 3.2, 4.1, 3.7, 3.9, 'GRADE_2', '송하린', '1234', b'0', 4.1, 4.3, 3.8, 4.2, 4.0, 'FRONTEND', b'1', b'0'),
('stu2319', NOW(6), NOW(6), 'STUDENT', 3.9, 4.0, 4.0, 4.2, 3.8, 'GRADE_2', '배준호', '1234', b'0', 4.0, 4.0, 3.7, 3.9, 4.3, 'FULLSTACK', b'1', b'1'),
('stu2320', NOW(6), NOW(6), 'STUDENT', 4.1, 3.4, 3.9, 3.5, 4.0, 'GRADE_2', '홍다은', '1234', b'0', 3.9, 4.1, 3.8, 4.0, 3.9, 'APP', b'1', b'0');

INSERT INTO user_skill (user_user_id, skill) VALUES
('stu2301', 'Spring Boot'), ('stu2301', 'JPA'), ('stu2301', 'MySQL'), ('stu2301', 'JWT'),
('stu2302', 'React'), ('stu2302', 'TypeScript'), ('stu2302', 'TanStack Query'), ('stu2302', 'CSS Modules'),
('stu2303', 'Python'), ('stu2303', 'Pandas'), ('stu2303', 'scikit-learn'), ('stu2303', 'FastAPI'),
('stu2304', 'Figma'), ('stu2304', 'User Flow'), ('stu2304', 'Design System'), ('stu2304', 'Prototyping'),
('stu2305', 'Flutter'), ('stu2305', 'Dart'), ('stu2305', 'Firebase'), ('stu2305', 'REST API'),
('stu2306', 'Docker'), ('stu2306', 'GitHub Actions'), ('stu2306', 'Nginx'), ('stu2306', 'AWS EC2'),
('stu2307', 'Vue'), ('stu2307', 'Pinia'), ('stu2307', 'Vite'), ('stu2307', 'Chart.js'),
('stu2308', 'PyTorch'), ('stu2308', 'OpenCV'), ('stu2308', 'NumPy'), ('stu2308', 'FastAPI'),
('stu2309', 'Figma'), ('stu2309', 'MUI'), ('stu2309', 'Responsive UI'), ('stu2309', 'Usability Test'),
('stu2310', 'Kotlin'), ('stu2310', 'Android Jetpack'), ('stu2310', 'Room'), ('stu2310', 'Retrofit'),
('stu2311', 'Spring Security'), ('stu2311', 'Redis'), ('stu2311', 'Spring Batch'), ('stu2311', 'PostgreSQL'),
('stu2312', 'React'), ('stu2312', 'Node.js'), ('stu2312', 'Express'), ('stu2312', 'MongoDB'),
('stu2313', 'Linux'), ('stu2313', 'OWASP'), ('stu2313', 'JWT'), ('stu2313', 'Burp Suite'),
('stu2314', 'Python'), ('stu2314', 'LangChain'), ('stu2314', 'RAG'), ('stu2314', 'Vector DB'),
('stu2315', 'Figma'), ('stu2315', 'UX Research'), ('stu2315', 'Wireframe'), ('stu2315', 'Design QA'),
('stu2316', 'Unity'), ('stu2316', 'C#'), ('stu2316', 'Photon'), ('stu2316', 'Game UI'),
('stu2317', 'NestJS'), ('stu2317', 'TypeORM'), ('stu2317', 'PostgreSQL'), ('stu2317', 'Swagger'),
('stu2318', 'Next.js'), ('stu2318', 'React'), ('stu2318', 'Zustand'), ('stu2318', 'Tailwind CSS'),
('stu2319', 'Next.js'), ('stu2319', 'Spring Boot'), ('stu2319', 'Docker'), ('stu2319', 'MySQL'),
('stu2320', 'SwiftUI'), ('stu2320', 'iOS'), ('stu2320', 'CoreData'), ('stu2320', 'REST API');

INSERT INTO user_experience (user_user_id, experience) VALUES
('stu2301', '캡스톤 예비 프로젝트에서 Spring Boot로 회원가입, 로그인, JWT 인증 API를 구현하고 예외 응답 형식을 통일했습니다.'),
('stu2301', 'JPA 연관관계를 이용해 팀, 사용자, 신청 내역을 저장하는 구조를 만들고 N+1 문제를 줄이기 위해 fetch join을 적용했습니다.'),
('stu2302', 'React와 TypeScript로 관리자 대시보드 화면을 만들고 TanStack Query로 목록 조회, 검색, 상태 변경 후 캐시 갱신 흐름을 구현했습니다.'),
('stu2302', '폼 검증, 모달, 페이지네이션 컴포넌트를 분리해 여러 화면에서 재사용할 수 있도록 정리한 경험이 있습니다.'),
('stu2303', 'Pandas로 설문 CSV를 정제하고 결측값 처리, 범주형 응답 인코딩, 간단한 군집 분석 결과를 FastAPI 엔드포인트로 제공했습니다.'),
('stu2303', '모델 성능을 정확도만 보지 않고 혼동 행렬과 feature importance로 설명하는 실험 노트를 작성한 경험이 있습니다.'),
('stu2304', 'Figma로 사용자 가입부터 팀 추천 확인까지의 플로우를 와이어프레임으로 설계하고, 개발자가 구현하기 쉬운 컴포넌트 단위로 화면을 정리했습니다.'),
('stu2304', '사용자 테스트 피드백을 바탕으로 버튼 위치, 필터 구조, 빈 상태 문구를 개선해 화면 이해도를 높인 경험이 있습니다.'),
('stu2305', 'Flutter로 일정 관리 앱을 만들며 Firebase Auth, Firestore 저장, REST API 연동 화면을 구현했습니다.'),
('stu2305', 'Provider 기반 상태 관리를 적용해 로그인 상태와 서버 응답 로딩 상태를 분리하고, 네트워크 오류 화면을 처리했습니다.'),
('stu2306', 'Docker Compose로 Spring Boot, MySQL, Nginx 개발 환경을 구성하고 GitHub Actions에서 빌드 후 EC2 배포까지 자동화했습니다.'),
('stu2306', '환경변수 분리, 로그 확인, 포트 충돌 해결 경험이 있어 팀 프로젝트 초기 개발 환경 세팅을 맡을 수 있습니다.'),
('stu2307', 'Vue와 Pinia로 통계 차트 화면을 구현하고 Chart.js 데이터를 필터 조건에 따라 갱신하는 기능을 만들었습니다.'),
('stu2307', '사용자 입력 상태와 API 응답 상태를 분리해 화면 깜빡임을 줄이고, 검색 조건을 URL query로 유지한 경험이 있습니다.'),
('stu2308', 'PyTorch로 이미지 분류 모델을 학습하고 OpenCV 전처리, 데이터 증강, FastAPI 기반 추론 API를 연결했습니다.'),
('stu2308', '모델 결과를 confidence와 함께 반환해 프론트엔드에서 신뢰도 표시가 가능하도록 응답 형식을 설계했습니다.'),
('stu2309', 'MUI 기반 테이블 화면을 디자인하고 반응형 레이아웃, 필터 UI, 상세 패널의 사용성을 개선한 경험이 있습니다.'),
('stu2309', '디자인 시스템 색상과 spacing 규칙을 정리해 프론트엔드 구현 시 화면 간 일관성을 유지하도록 도왔습니다.'),
('stu2310', 'Kotlin과 Android Jetpack으로 로그인, 목록, 상세 화면을 구현하고 Retrofit으로 백엔드 API를 연동했습니다.'),
('stu2310', 'Room을 이용해 오프라인 임시 저장을 구현하고 서버 동기화 실패 시 재시도 흐름을 처리했습니다.'),
('stu2311', 'Spring Security와 Redis를 이용해 refresh token 저장, 로그아웃 처리, 권한별 접근 제한을 구현했습니다.'),
('stu2311', 'Spring Batch로 매일 새벽 통계 데이터를 집계하는 작업을 만들고 실패 로그를 확인해 재실행하는 흐름을 정리했습니다.'),
('stu2312', 'React와 Express로 게시판형 서비스를 만들고 파일 업로드, 댓글, 좋아요 기능을 한 흐름으로 연결했습니다.'),
('stu2312', '프론트와 백엔드를 모두 다뤄 API 명세 변경이 화면 상태에 미치는 영향을 빠르게 조정한 경험이 있습니다.'),
('stu2313', 'JWT 인증 흐름에서 토큰 저장 위치, 만료 처리, 권한 체크 누락 가능성을 점검하고 보완안을 정리했습니다.'),
('stu2313', 'OWASP Top 10 기준으로 입력값 검증, CORS 설정, 파일 업로드 확장자 제한을 점검한 경험이 있습니다.'),
('stu2314', 'LangChain과 벡터 DB를 이용해 문서 검색 기반 Q&A 기능을 만들고, 검색 결과와 답변 근거를 함께 반환했습니다.'),
('stu2314', '프롬프트 템플릿을 여러 버전으로 비교하고 답변 누락 사례를 기록해 개선한 경험이 있습니다.'),
('stu2315', '사용자 인터뷰 결과를 기능 우선순위로 정리하고 핵심 화면 wireframe과 clickable prototype을 제작했습니다.'),
('stu2315', '개발 완료 화면을 디자인 시안과 비교해 여백, 버튼 크기, 텍스트 길이 문제를 QA한 경험이 있습니다.'),
('stu2316', 'Unity로 2D 협동 미니게임을 만들고 Photon을 이용해 방 생성, 입장, 플레이어 동기화 기능을 구현했습니다.'),
('stu2316', '게임 UI 상태와 네트워크 이벤트를 분리해 접속 끊김, 재입장, 결과 화면 전환을 처리했습니다.'),
('stu2317', 'NestJS와 TypeORM으로 REST API를 만들고 Swagger 문서, DTO 검증, PostgreSQL 마이그레이션을 적용했습니다.'),
('stu2317', '권한별 API 접근 제어와 에러 응답 포맷을 통일해 프론트엔드가 처리하기 쉬운 서버 구조를 만든 경험이 있습니다.'),
('stu2318', 'Next.js로 검색과 필터가 있는 상품 목록 화면을 만들고 Zustand로 필터 상태를 관리했습니다.'),
('stu2318', 'Tailwind CSS로 반응형 카드 레이아웃을 구현하고 skeleton loading과 빈 결과 화면을 처리했습니다.'),
('stu2319', 'Next.js 프론트와 Spring Boot 백엔드를 함께 구성해 로그인 후 사용자별 데이터 조회 흐름을 직접 연결했습니다.'),
('stu2319', 'Docker로 로컬 실행 환경을 통일하고 MySQL 스키마 변경이 화면에 미치는 영향을 확인하며 개발했습니다.'),
('stu2320', 'SwiftUI로 iOS 일정 관리 화면을 만들고 CoreData 임시 저장, REST API 동기화, 로딩 상태 처리를 구현했습니다.'),
('stu2320', 'MVVM 구조로 View와 API 호출 로직을 분리해 화면 테스트와 기능 수정이 쉽도록 정리한 경험이 있습니다.');

INSERT INTO user_preferred_teammates (user_user_id, preferred_teammates) VALUES
('stu2301', 'stu2302'), ('stu2301', 'stu2306'),
('stu2302', 'stu2301'),
('stu2303', 'stu2308'), ('stu2303', 'stu2314'),
('stu2304', 'stu2309'),
('stu2305', 'stu2310'), ('stu2305', 'stu2320'),
('stu2306', 'stu2301'),
('stu2307', 'stu2318'),
('stu2308', 'stu2303'),
('stu2309', 'stu2304'), ('stu2309', 'stu2315'),
('stu2310', 'stu2305'),
('stu2311', 'stu2312'), ('stu2311', 'stu2317'),
('stu2312', 'stu2311'),
('stu2313', 'stu2317'),
('stu2314', 'stu2303'),
('stu2315', 'stu2309'),
('stu2316', 'stu2320'),
('stu2317', 'stu2311'),
('stu2318', 'stu2307'),
('stu2319', 'stu2312'),
('stu2320', 'stu2305'), ('stu2320', 'stu2316');
