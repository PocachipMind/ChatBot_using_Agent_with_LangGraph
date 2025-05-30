# Project

프로젝트에 쓰인 코드가 있는 공간입니다.

## 환경 설정 및 실행 방법

### Anaconda 등 Environment 설정

해당 git에 있는 env_yaml이나 requirements.txt를 통해 환경 설정합니다. 저의 경우 Anaconda 환경을 사용했습니다.

### 1. 해당 git clone

- 해당 Repository Clone ( git bash )
```
$ git clone https://github.com/PocachipMind/ChatBot_using_Agent_with_LangGraph.git
```

### 2. 로컬 환경 (OS)에 맞는 Anaconda 실행 파일 다운

링크 : http://www.anaconda.com/

![image](https://github.com/user-attachments/assets/eef74a9d-5c5b-4746-9fc0-fe6839dc87ca)


### 2. Anaconda 설치 후 Terminal 실행 ( Windows OS 기준 Powershell Prompt )

![image](https://github.com/user-attachments/assets/e50acb0e-1f3c-43d4-8700-df1366890a45)

### 3. Terminal 실행 후 Python 가상환경 활성화

- cd 명령어를 통해 git clone한 폴더의 Project폴더로 진입
```
$ cd ( 깃 클론한 곳 위치 )
```

- 첨부된 env_yaml.yaml을 활용
- 하기와 같은 Command를 사용하여 가상환경 설치 진행
```
$ conda env create -f env_yaml.yaml
```

- 환경 설치가 완료되면 하기와 같은 Command를 사용하여 가상환경 활성화 진행
```
$ conda activate AgentChat
```

### 4. 앱 실행
- 앱을 실행하기 전 환경 설정
  
해당 폴더 안에 ".env" 파일을 생성하고 GPT API KEY, TAVILY API KEY를 다음과 같은 형식으로 저장
![image](https://github.com/user-attachments/assets/ed5cd250-2920-409d-83f8-1ed639ebeea3)


- Streamlit을 통한 앱 실행
```
$ streamlit run app.py
```
