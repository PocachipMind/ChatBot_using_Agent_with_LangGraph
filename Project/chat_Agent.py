from dotenv import load_dotenv

load_dotenv()




from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

small_llm = ChatOpenAI(model="gpt-4o-mini")


########################################## Tools ###############################################

##### 검색 툴 ##


from langchain_community.tools import TavilySearchResults

search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

## dalle 툴 ##

from langchain.agents import load_tools

dalletools = load_tools(["dalle-image-generator"])




## retriever 툴 만들기 함수 ( ui에서 실행될 때 생성해야함. )

from langchain_core.tools.retriever import create_retriever_tool

def get_retriever_tool(retriever, infor):
    
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name='retriever',
        description=f'Contains information about {infor}',
    )
    return retriever_tool



########################################################################





######################## Agent 여기부터  #########################################


from langgraph.graph import MessagesState, StateGraph
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage


class AgentState(MessagesState):
    summary: str
    llm_with_tools : Runnable[LanguageModelInput, BaseMessage]


################################ Agent ##################################################

from langchain_core.messages import SystemMessage

def agent(state: AgentState) -> AgentState:
    """
    주어진 `state`에서 메시지를 가져와
    LLM과 도구를 사용하여 응답 메시지를 생성합니다.

    Args:
        state (AgentState): 메시지 기록과 요약을 포함하는 state.

    Returns:
        MessagesState: 응답 메시지를 포함하는 새로운 state.
    """
    # 메시지와 요약을 state에서 가져옵니다.
    messages = state['messages']
    summary = state['summary']
    llm_with_tools = state['llm_with_tools']
    
    # 요약이 비어있지 않으면, 요약을 메시지 앞에 추가합니다.
    if summary != '':
        messages = [SystemMessage(content=f'Here is the summary of the earlier conversation: {summary}')] + messages
    
    # LLM과 도구를 사용하여 메시지에 대한 응답을 생성합니다.
    response = llm_with_tools.invoke(messages)
    
    # 응답 메시지를 포함하는 새로운 state를 반환합니다.
    return {'messages': [response]}



################################### summarize_messages 메세지 요약 #########################

def summarize_messages(state: AgentState) -> AgentState:
    """
    주어진 state의 메시지를 요약합니다.

    Args:
        state (AgentState): 메시지와 요약을 포함하는 state.

    Returns:
        AgentState: 요약된 메시지를 포함하는 딕셔너리.
    """
    # state에서 메시지와 요약을 가져옵니다.
    messages = state['messages']
    summary = state['summary']
    
    # 요약 프롬프트를 생성합니다.
    summary_prompt = f'summarize this chat history below: \n\nchat_history:{messages}'
    
    # 기존 요약이 있으면, 요약을 포함한 프롬프트를 생성합니다.
    if summary != '':
        summary_prompt = f'''summarize this chat history below while looking at the summary of earlier conversations
chat_history:{messages}
summary:{summary}'''
    
    # LLM을 사용하여 요약을 생성합니다.
    summary = small_llm.invoke(summary_prompt)
    
    # 요약된 메시지를 반환합니다.
    return {'summary': summary.content}




############################ delete_messages ######################################

from langchain_core.messages import RemoveMessage

def delete_messages(state: AgentState) -> AgentState:
    """
    주어진 state에서 오래된 메시지를 삭제합니다.

    Args:
        state (AgentState): 메시지를 포함하는 state.

    Returns:
        AgentState: 삭제된 메시지를 포함하는 새로운 state.
    """
    # state에서 메시지를 가져옵니다.
    messages = state['messages']
    # 마지막 세 개의 메시지를 제외한 나머지 메시지를 삭제합니다.
    delete_messages = [RemoveMessage(id=message.id) for message in messages[:-3]]
    # 삭제된 메시지를 포함하는 새로운 state를 반환합니다.
    return {'messages': delete_messages}



############################### should_continue ############################################

from typing import Literal

def should_continue(state: AgentState) -> Literal['tools', 'summarize_messages']:
    """
    주어진 state에 따라 다음 단계로 진행할지를 결정합니다.

    Args:
        state (AgentState): 메시지와 도구 호출 정보를 포함하는 state.

    Returns:
        Literal['tools', 'summarize_messages']: 다음 단계로 'tools' 또는 'summarize_messages'를 반환합니다.
    """
    # state에서 메시지를 가져옵니다.
    messages = state['messages']
    # 마지막 AI 메시지를 확인합니다.
    last_ai_message = messages[-1]
    
    # 마지막 AI 메시지가 도구 호출을 포함하고 있는지 확인합니다.
    if last_ai_message.tool_calls:
        # 도구 호출이 있으면 'tools'를 반환합니다.
        return 'tools'
    
    # 도구 호출이 없으면 'summarize_messages'를 반환합니다.
    return 'summarize_messages'








from langgraph.prebuilt import ToolNode

def make_agent(retriever, infor):
    graph_builder = StateGraph(AgentState)

    tool_list = [ search , get_retriever_tool(retriever, infor) ] + dalletools
    
    llm_with_tools = small_llm.bind_tools(tool_list)
    tool_node = ToolNode(tool_list)



    graph_builder.add_node('agent', 
                           
                           agent)
    graph_builder.add_node('tools', tool_node)
    graph_builder.add_node(delete_messages)
    graph_builder.add_node(summarize_messages)







    from langgraph.graph import START, END


    graph_builder.add_edge(START, 'agent')
    graph_builder.add_conditional_edges(
        'agent',
        should_continue,
        ['tools', 'summarize_messages']
    )
    graph_builder.add_edge('tools', 'agent')
    graph_builder.add_edge('summarize_messages', 'delete_messages')
    graph_builder.add_edge('delete_messages', END)




    # - 히스토리 관리를 위해 checkpointer를 사용합니다  MemorySaver는 메모리에 저장하는 방법입니다

    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()

    graph= graph_builder.compile(checkpointer=checkpointer)
    
    return graph , llm_with_tools



if __name__ == '__main__':

    from langchain_openai import OpenAIEmbeddings
    from langchain_core.messages import HumanMessage


    # OpenAI에서 제공하는 Embedding Model을 활용해서 `chunk`를 vector화
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')

    from langchain_chroma import Chroma

    # 이미 저장된 데이터를 사용할 때 
    vector_store = Chroma(collection_name='chroma-income_tax', persist_directory="./chroma", embedding_function=embedding)

    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    
    graph , llm_with_tools = make_agent(retriever,"income_tax")
    
    config = {
        'configurable': {
            'thread_id': 'summarize_paper'
        }
    }
    
    query = '안녕'
    for chunk in graph.stream({'messages': [HumanMessage(query)], 'summary': '' , 'llm_with_tools' : llm_with_tools}, config=config, stream_mode='values'):
        chunk['messages'][-1].pretty_print()
    
    test = graph.invoke({'messages': [HumanMessage(query)], 'summary': '' , 'llm_with_tools' : llm_with_tools}, config=config)
    
    
    print(type(test['messages'][-1]))
    
    
    