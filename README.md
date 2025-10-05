# 🀄 AI 마작 분석 웹 서비스 (AI Mahjong Analyzer)


딥러닝 비전 모델(YOLO)을 활용하여 사용자가 업로드한 마작 패 이미지를 분석하고, 점수 계산과 함께 샤텐(Shanten) 수 기반의 최적 타패를 추천해주는 웹 서비스입니다. 실시간 영상 인식을 통해 현재 손패를 분석하는 기능도 제공합니다.

---

## 🌟 주요 기능 (Features)

* **🎴 타패 추천 기능 (`/discard`)**:
    * 사용자가 손패 이미지를 업로드하면 AI가 14개의 패를 자동으로 인식합니다.
    * 인식된 패를 사용자가 직접 수정하거나 추가/삭제할 수 있습니다.
    * **샤텐(Shanten) 수**를 기반으로 현재 패에서 어떤 패를 버리는 것이 가장 효율적인지 추천합니다.
    * 각 패를 버렸을 때의 **샹텐 수 변화**와 **유효패 개수**를 시각적으로 보여줍니다.
    * 텐파이(Tenpai) 상태일 경우 **대기패(기다리는 패)** 목록을 알려줍니다.

* **💯 점수 계산 기능 (`/score`)**:
    * 화료(완성)한 패 이미지를 업로드하면 AI가 패를 인식합니다.
    * **쯔모/론, 리치, 도라, 장풍/자풍** 등 다양한 게임 상황을 옵션으로 선택할 수 있습니다.
    * 입력된 정보를 바탕으로 **역(Yaku), 부수(Fu), 판수(Han)**를 계산하여 최종 점수를 알려줍니다.
    * 치또이츠, 국사무쌍 등 특수패도 자동으로 인식하고 계산합니다.

* **📹 실시간 패 인식 기능 (`/realtime`)**:
    * 웹캠을 통해 실시간으로 사용자의 손패를 인식하고 화면에 표시합니다.
    * 현재 어떤 패를 들고 있는지 지속적으로 트래킹합니다.

* **📖 마작 가이드 (`/guide`)**:
    * 마작의 기본적인 규칙이나 용어를 안내하는 페이지를 제공합니다.

---

## 🛠️ 기술 스택 (Tech Stack)

* **Backend**: Python, Flask
* **Deep Learning**: PyTorch, YOLOv8 (via `ultralytics`)
* **Computer Vision**: OpenCV, Pillow
* **Frontend**: HTML, CSS, JavaScript
* **Deployment**: (서버에 배포했다면 내용을 적어주세요. 예: AWS, GCP)

---

## ⚙️ 설치 및 실행 방법 (Installation & Usage)

1.  **GitHub 리포지토리 복제 (Clone)**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repository-Name].git
    cd [Your-Repository-Name]
    ```

2.  **가상 환경 생성 및 활성화**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **필요한 패키지 설치**
    ```bash
    pip install -r requirements.txt
    ```

4.  **웹 애플릭케이션 실행**
    ```bash
    python app.py
    ```
    > **참고**: `app.py` 코드에 `ssl_context='adhoc'` 설정이 있어 HTTPS로 실행됩니다. 처음 실행 시 브라우저에서 안전하지 않다는 경고가 나올 수 있으나, 로컬 테스트용이므로 '고급' 또는 '계속 진행'을 눌러 접속하면 됩니다.

5.  **서비스 접속**
    -   웹 브라우저를 열고 `https://127.0.0.1:5000` 주소로 접속합니다.

---

## 📁 프로젝트 구조 (Project Structure)
.
├── apis/                 # 기능별 API (Blueprints)
│   ├── discard/          # 타패 추천 기능 (로직, 라우트)
│   ├── guide/            # 가이드 페이지
│   ├── realtime/         # 실시간 인식 기능
│   └── score/            # 점수 계산 기능
├── model/
│   └── best.pt           # 학습된 YOLOv8 마작 패 인식 모델
├── static/               # CSS, JavaScript, 이미지 등 정적 파일
├── templates/            # 기능별 HTML 템플릿 파일
├── uploads/              # 사용자가 업로드한 이미지 임시 저장
├── app.py                # Flask 메인 애플리케이션
├── detected_tiles.json   # AI가 탐지한 타일 결과 (임시 저장용)
├── requirements.txt      # 프로젝트 의존성 목록
└── README.md             # 프로젝트 설명서