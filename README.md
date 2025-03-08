# ChatBot_using_Agent_with_LangGraph
LangGraph를 통해 AI Agent를 구현합니다. 그리고 이를 Streamlit으로 사용해봅니다.

![image](https://github.com/user-attachments/assets/4df6a34a-214d-441e-9f3d-c023cae89a19)

<br>

전체 프로그램 구조 및 사용 시연 : https://youtu.be/dr989e1u4uE

<br>

## 1. AI Agent Work Flow

프로젝트 내부 구현되있는 Agent Work Flow는 다음과 같습니다.

![image](https://github.com/user-attachments/assets/d30c5862-dce3-4d03-98a1-eff1688d4319)



해당 구조는 Adaptive RAG를 기반에서 

![image](https://github.com/user-attachments/assets/86bacbc8-d5e7-43f9-9724-36888a4baa4e)

Self RAG 부분으로 되어있는 부분을 Corrective RAG로 변경 및 일부 커스텀 하여 구성되었습니다.

![image](https://github.com/user-attachments/assets/9ffb3fe6-046a-44ca-bdd2-33d3a9f3a7f8)
