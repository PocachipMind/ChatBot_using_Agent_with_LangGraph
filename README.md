# ChatBot_using_Agent_with_LangGraph
LangGraph를 통해 AI Agent를 구현합니다. 그리고 이를 Streamlit으로 사용해봅니다.

![image](https://github.com/user-attachments/assets/4df6a34a-214d-441e-9f3d-c023cae89a19)

<br>

전체 프로그램 구조 및 사용 시연 : https://youtu.be/dr989e1u4uE

<br>

# 1. AI Agent Work Flow

### 1. Work Flow

프로젝트 내부 구현되있는 Agent Work Flow는 다음과 같습니다.

![image](https://github.com/user-attachments/assets/d30c5862-dce3-4d03-98a1-eff1688d4319)

<br>

해당 구조는 다음과 같은 Adaptive RAG 구조를 기반으로 합니다.

![image](https://github.com/user-attachments/assets/86bacbc8-d5e7-43f9-9724-36888a4baa4e)

Adaptive RAG의 Self RAG 로 되있는 부분을 Corrective RAG로 변경 및 일부 커스텀 하여 구성하였습니다.

![image](https://github.com/user-attachments/assets/9ffb3fe6-046a-44ca-bdd2-33d3a9f3a7f8)

관련 정보 : 

- Adaptive RAG :
    - https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/
    - https://arxiv.org/abs/2403.14403
- Self RAG :
    - https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/?h=self
    - https://arxiv.org/abs/2310.11511
- Corrective RAG :
    - https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/?h=corrective+retrieval
    - https://arxiv.org/abs/2401.15884

<br>

### 2. 동작 방식

먼저, 사용자에게 질문이 들어오면 질문이 어떤 유형인지 routing 합니다.

![image](https://github.com/user-attachments/assets/0bceb4f7-5b59-403f-903f-ce2ae6f6f32f)

```python
def router(state: AgentState) -> Literal['Dall-E', 'vector_store', 'web_search', 'Just_GPT']:
    """
    사용자의 질문에 기반하여 적절한 경로를 결정합니다.

    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state.

    Returns:
        Literal['Dall-E', 'vector_store', 'web_search']: 질문을 처리하기 위한 적절한 경로를 나타내는 문자열.
    """
    
    # state에서 질문과 벡터 정보를 추출합니다
    query = state['query_message']
    inf = state['vector_info']
    
    # 프롬프트와 구조화된 라우터 LLM을 연결하여 체인을 생성합니다
    router_chain = router_prompt | structured_router_llm 
    
    # 체인을 사용하여 경로를 결정합니다
    route = router_chain.invoke({'query': query , 'vector_inf' : inf})
    
    print(f"call : router > {route.target}")
    # 결정된 경로의 타겟을 반환합니다
    return route.target
```
