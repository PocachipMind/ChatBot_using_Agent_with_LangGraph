# ChatBot_using_Agent_with_LangGraph
LangGraph를 통해 AI Agent를 구현합니다. 그리고 이를 Streamlit으로 사용해봅니다.

![image](https://github.com/user-attachments/assets/4df6a34a-214d-441e-9f3d-c023cae89a19)

<br>

전체 프로그램 구조 및 사용 시연 : https://youtu.be/dr989e1u4uE

<br>

# 1. AI Agent Work Flow

## 1. Work Flow

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

## 2. 동작 방식

먼저, 사용자에게 질문이 들어오면 질문이 어떤 유형인지 routing 합니다.

해당 프롬프트를 사용합니다. 여기에서 {vector_inf}는 사용자에게 UI에서 입력 받는 Vector DB의 정보입니다.

```python
class Route(BaseModel):
    target: Literal['Dall-E', 'vector_store', 'web_search', 'Just_GPT'] = Field(
        description="The target for the query to answer"
    )

router_system_prompt = """
You are an expert at routing a user's question to 'Dall-E', 'vector_store', 'Just_GPT' or 'web_search'.
'vector_store' contains {vector_inf}.
if you think to create an image use 'Dall-E'.
if you think you need to search the web to answer the question use 'web_search'
else use 'Just_GPT'
"""
```

![image](https://github.com/user-attachments/assets/4e806e82-2f16-40bb-8541-888d8144a878)

- ① : 이미지 생성이 필요하다 판단
- ② : 사용자가 올려준 문서로 부터 RAG를 해야한다 판단
- ③ : 웹 검색이 필요한 질문이라 판단
- ④ : 단순하게 그냥 대답할 수 있는 질문이라 판단

### 1. 이미지 생성 WorkFlow

사용자의 질문이 이미지 생성이 필요하다고 판단되었을 때 해당 WorkFlow를 타게 됩니다.

![image](https://github.com/user-attachments/assets/de1a079b-bace-4d6f-b368-89e5d46ee322)

이미지 제너레이터 모델인 Dall-E를 사용.

LangChain 공식 사이트의 프롬프트만 사용했을 경우, 이미지 생성 요구 프롬프트가 1000자를 자꾸 넘어서 오류가 생기기에 글자수 제한 프롬프트를 추가함.

```python
prompt = PromptTemplate(
    input_variables=["image_desc"],
    template="Generate a prompt to generate an image based on the following description. Prompt must be length 1000 or less : {image_desc}",
)
```

참고 : https://python.langchain.com/docs/integrations/tools/dalle_image_generator/

### 2. RAG 활용 WorkFlow

사용자의 질문이 Vector DB 내부에 정보가 있을 것 같다고 판단되었을 때 해당 WorkFlow를 타게 됩니다.

![image](https://github.com/user-attachments/assets/97ebdd2f-b960-4ea5-9cf0-e40d72dd0b6d)

**- ① : DB에서 Retrieve 한 다음 해당 내용이 사용자의 질문과 관련이 있는지 파악**

해당 프롬프트를 사용합니다.
```
####### system ######

You are a teacher grading a quiz. 

You will be given a QUESTION and a set of FACTS provided by the student. 

Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question (2) is met 

Score:
A score of 1 means that the FACT contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant. This is the highest (best) score. 
A score of 0 means that the FACTS are completely unrelated to the QUESTION. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
 
Avoid simply stating the correct answer at the outset.

###### human ######

FACTS: {{documents}} 
QUESTION: {{question}}
```

만약 가져온 문서가 질문과 유사성이 높다고 판단된다면 ② Work Flow를 통해 답변을 내놓고, 아니면 Web Search WorkFlow를 타게됩니다.

참고 : https://smith.langchain.com/hub/langchain-ai/rag-document-relevance

**- ② : Retrieve 정보를 기반하여 답변 생성**

해당 프롬프트를 사용합니다.
```
##### human #####

You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
```

참고 : https://smith.langchain.com/hub/rlm/rag-prompt

### 3. Web 활용 WorkFlow

사용자의 질문이 Vector DB 내부에도 없을 것 같고 일반적인 답이 아니라 웹 검색을 해야 알 수 있을 것 같다고 판단되었을 때 해당 WorkFlow를 타게 됩니다.

![image](https://github.com/user-attachments/assets/6da2096c-5b64-4f7e-a035-fd3d3ba5a1a1)


