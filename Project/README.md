# Project

프로젝트에 쓰인 코드가 있는 공간입니다.

## 환경 설정 

### Anaconda 등 Environment 설정

해당 git에 있는 env_yaml이나 requirements.txt를 통해 환경 설정합니다. 저의 경우 Anaconda 환경을 사용했습니다.

### 1. 로컬 환경 (OS)에 맞는 Anaconda 실행 파일 다운

링크 : http://www.anaconda.com/

![image](https://github.com/user-attachments/assets/eef74a9d-5c5b-4746-9fc0-fe6839dc87ca)


### 2. Anaconda 설치 후 Terminal 실행 ( Windows OS 기준 Powershell Prompt )

![image](https://github.com/user-attachments/assets/e50acb0e-1f3c-43d4-8700-df1366890a45)

### 3. Terminal 실행 후 Python 가상환경 활성화

- 첨부된 env_yaml.yaml을 활용
  
- 하기와 같은 Command를 사용하여 가상환경 설치 진행
```
$ conda env create -f env_yaml.yaml
```
- 환경 설치가 완료되면 하기와 같은 Command를 사용하여 가상환경 활성화 진행
```
$ conda activate nvidia_openCV_pj_env
```
## 실행 환경 : Visual Studio (VS) Code

### 1. 로컬 환경 (OS)에 맞는VS Code 실행 파일 다운

링크 : http://code.visualstudio.com/

![image](https://github.com/user-attachments/assets/e6e241c4-765a-4b4b-a886-549211a247c7)

### 2. VS Code 설치 후 Python Extension 설치
좌측이 Extension 아이콘 클릭 후 Python Extension Pack 검색 및 설치
![image](https://github.com/user-attachments/assets/76ebe31a-4b97-4db2-bd17-2eabf1a1c383)

### 3. Windows OS 기준 Control + P 후 Python: Select Interpreter 검색

### 4. 실습에서 사용할 환경 선택

예시 이미지의 경우 이름이 조금 다름. 우리의 경우 nvidia_openCV_pj_env 로 나와야함.

![image](https://github.com/user-attachments/assets/be946505-b5ab-4eb4-ae7f-0309a441d6a1)

### 5. 우측 하단에 3.10.14('nvidia_openCV_pj_env': conda)로 표기 되어있는지 확인
