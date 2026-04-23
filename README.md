## Git Workflow

### 1. 프로젝트 가져오기
git clone <repo-url>

### 2. 작업 전 최신화
git checkout main
git pull origin main

### 3. 브랜치 생성
git checkout -b feature-branch

### 4. 작업 후 커밋
git add .
git commit -m "작업 내용"

### 5. 원격 브랜치 업로드
git push origin feature-branch

### 6. Pull Request 생성
- feature-branch → main PR 생성

### 7. 코드 리뷰 후 Merge

### 8. Merge 이후 동기화
git checkout main
git pull origin main

git checkout feature-branch
git merge main
