from dotenv import load_dotenv

load_dotenv()




########################### State 정의 ##################################

from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

class AgentState(TypedDict):
    query_message: str
    query_img: str
    context: list
    answer: dict
    vector_store : VectorStoreRetriever
    vector_info : str
    
graph_builder = StateGraph(AgentState)










###################################### 입력 들어오면 첫 라우팅 #######################################

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain_openai import ChatOpenAI

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

router_prompt = ChatPromptTemplate.from_messages([
    ('system', router_system_prompt),
    ('user', '{query}')
])

router_llm = ChatOpenAI(model="gpt-4o-mini") # 간단한 질문이므로 mini 사용
structured_router_llm = router_llm.with_structured_output(Route) # 해당 정해진 문자열만 받도록 리턴하게함

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


######################################################################################################




############################################### Dall-E ################################################

from langchain.chains import LLMChain
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI




def DallE_Create(state: AgentState) -> AgentState:
    """
    주어진 state의 query_message기반으로 이미지 생성을 수행합니다.

    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 생성된 답변을 포함한 state를 반환합니다.
    """
    Dall_llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["image_desc"],
        template="Generate a prompt to generate an image based on the following description. Prompt must be length 1000 or less : {image_desc}",
    )
    chain = LLMChain(llm=Dall_llm, prompt=prompt)

    query = state['query_message']
    
    image_url = DallEAPIWrapper().run(chain.run(query))

    answer = { "type" : "image", "message" : image_url}

    print("call : DallE_Create")
    return {'answer': answer}



######################################################################################################







############################################### web_search ################################################

from langchain_community.tools import TavilySearchResults

tavily_search_tool = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

def web_search(state: AgentState) -> AgentState:
    """
    주어진 state를 기반으로 웹 검색을 수행합니다.

    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 웹 검색 결과가 추가된 state를 반환합니다.
    """
    query = state['query_message']
    results = tavily_search_tool.invoke(query)
    print("call : web_search")
    
    return {'context': results}



from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# LangChain 허브에서 프롬프트를 가져옵니다
generate_prompt = hub.pull("rlm/rag-prompt")
# OpenAI의 GPT-4o 모델을 사용합니다
generate_llm = ChatOpenAI(model="gpt-4o")

def web_generate(state: AgentState) -> AgentState:
    """
    주어진 문맥과 질문을 기반으로 답변을 생성합니다.

    Args:
        state (AgentState): 문맥과 질문을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 생성된 답변을 포함한 state를 반환합니다.
    """
    # state에서 문맥과 질문을 추출합니다
    context = state['context']
    query = state['query_message']
    
    # 프롬프트와 모델, 출력 파서를 연결하여 체인을 생성합니다
    rag_chain = generate_prompt | generate_llm | StrOutputParser()
    
    # 체인을 사용하여 답변을 생성합니다 ( tavily_search_tool의 경우 이미지를 검색으로 받지 않으므로 text만 적용하겠음. )
    response = rag_chain.invoke({'question': query, 'context': context})
    answer = { "type" : "Webtext", "message" : response}
    
    print("call : web_generate")
    
    # 생성된 답변을 'answer'로 반환합니다
    return {'answer': answer}


###########################################################################################################





############################################### vector_store ################################################

def retrieve(state: AgentState) -> AgentState:
    """
    사용자의 질문에 기반하여 벡터 스토어에서 관련 문서를 검색합니다.

    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 검색된 문서가 추가된 state를 반환합니다.
    """
    query = state['query_message']
    docs = state['vector_store'].invoke(query)
    
    print("call : retrieve")
    
    return {'context': docs}






# set the LANGCHAIN_API_KEY environment variable (create key in settings)
from langchain import hub
from typing import Literal
doc_relevance_prompt = hub.pull("langchain-ai/rag-document-relevance")

def check_doc_relevance(state: AgentState) -> Literal['relevant', 'irrelvant']:
    """
    주어진 state를 기반으로 문서의 관련성을 판단합니다.

    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state.

    Returns:
        Literal['relevant', 'irrelevant']: 문서가 관련성이 높으면 'relevant', 그렇지 않으면 'irrelevant'를 반환합니다.
    """
    query = state['query_message']
    context = state['context']

    doc_relevance_chain = doc_relevance_prompt | generate_llm
    response = doc_relevance_chain.invoke({'question': query, 'documents': context})
    
    print(f"call : check_doc_relevance > {response['Score']} (1 = 'relevant' , 0 = 'irrelvant')")

    if response['Score'] == 1:
        return 'relevant'
    
    return 'irrelvant'




from langchain_openai import ChatOpenAI
from langchain import hub

# 허브에서 RAG 프롬프트를 가져옵니다
generate_prompt = hub.pull("rlm/rag-prompt")

# 지정된 매개변수로 언어 모델을 초기화합니다
generate_llm2 = ChatOpenAI(model='gpt-4o')

def generate(state: AgentState) -> AgentState:
    """
    주어진 state를 기반으로 RAG 체인을 사용하여 응답을 생성합니다.

    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 생성된 응답을 포함하는 state를 반환합니다.
    """
    
    context = state['context']
    query = state['query_message']
    
    make_massage = []
    massage = generate_prompt.invoke({'question': query, 'context': context})
    messages = massage.to_messages()
    # print(massage)
    # print("--------------------------------")
    # print(messages)
    # print("--------------------------------")
    # print(messages[0])
    # print("--------------------------------")
    # print(messages[0].content)
    # print("--------------------------------")
    make_massage.append({"type": "text", "text": messages[0].content})


    if state['query_img'] != "":
        make_massage.append({"type": "image_url", 'image_url': {"url": state['query_img']}})
    

    final_massage = HumanMessage(content=make_massage)
    response = generate_llm2.invoke([final_massage])

    
    answer = { "type" : "RAGtext", "message" : response.content}
    
    print(f"call : generate")
    
    return {'answer': answer}



################################################################################################

##################################### JUST GPT ###############################################

def Nomal_GPT(state: AgentState) -> AgentState:
    """
    주어진 state를 기반으로 GPT를 사용하여 응답을 생성합니다.

    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 생성된 응답을 포함하는 state를 반환합니다.
    """
    query = state['query_message']
    
    make_massage = []
    make_massage.append({"type": "text", "text": query})
    
    if state['query_img'] != "":
        make_massage.append({"type": "image_url", 'image_url': {"url": state['query_img']}})
    
    final_massage = HumanMessage(content=make_massage)
    
    response = generate_llm.invoke([final_massage])
    
    answer = { "type" : "Notext", "message" : response.content}
    
    print(f"call : Nomal_GPT")
    
    return {'answer': answer}

################################################################################################



####################################  노드 구성 ################################################

graph_builder.add_node('DallE_Create', DallE_Create)
graph_builder.add_node('web_search', web_search)
graph_builder.add_node('web_generate', web_generate)
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_node('Nomal_GPT', Nomal_GPT)

from langgraph.graph import START, END

graph_builder.add_conditional_edges(
    START, 
    router,
    {
        'Dall-E': 'DallE_Create',
        'vector_store': 'retrieve',
        'web_search': 'web_search',
        'Just_GPT': 'Nomal_GPT',
    }
)

graph_builder.add_conditional_edges(
    'retrieve',
    check_doc_relevance,
    {
        'relevant': 'generate',
        'irrelvant': 'web_search'
    }
)


graph_builder.add_conditional_edges(
    'web_search',
    check_doc_relevance,
    {
        'relevant': 'web_generate',
        'irrelvant': 'Nomal_GPT'
    }
)
graph_builder.add_edge('DallE_Create', END)
graph_builder.add_edge('generate', END)
graph_builder.add_edge('Nomal_GPT', END)
graph_builder.add_edge('web_generate', END)
graph = graph_builder.compile()








################################################################




# from IPython.display import Image, display
# from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

# display(Image(graph.get_graph().draw_mermaid_png()))

if __name__ == '__main__':
    ######################  기본 벡터 DB 불러오기 ######################

    from langchain_openai import OpenAIEmbeddings


    # OpenAI에서 제공하는 Embedding Model을 활용해서 `chunk`를 vector화
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')

    from langchain_chroma import Chroma

    # 이미 저장된 데이터를 사용할 때 
    vector_store = Chroma(collection_name='chroma-income_tax', persist_directory="./chroma", embedding_function=embedding)

    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    
    initial_state = {'query_message': '연봉 5천만원 거주자의 소득세는 얼마인가요?', 'query_img' : "" , "vector_store" : retriever , "vector_info" : "소득세관련 벡터베이스"}
    graph.invoke(initial_state)