import streamlit as st
from openai import OpenAI
import base64
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import sys
from dotenv import load_dotenv
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import chat_LangGraph
import chat_Agent

load_dotenv()


#Chroma tenant 오류 방지 위한 코드
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

#cache_resource로 한번 실행한 결과 캐싱해두기
@st.cache_resource
def load_pdf(_file):
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
        tmp_file.write(_file.getvalue())
        tmp_file_path = tmp_file.name
        #PDF 파일 업로드
        loader = PyPDFLoader(file_path=tmp_file_path)
        pages = loader.load_and_split()
    return pages

#텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(_docs)
    vectorstore = Chroma.from_documents(split_docs, OpenAIEmbeddings(model='text-embedding-3-large'))
    return vectorstore

st.title("Chat with Image Support")
st.caption("왼쪽에서 문서 설정시 채팅이 가능합니다.")
with st.sidebar:
    st.header("🔧 RAG 쓰일 문서 설정")

    option = st.selectbox("Select Model Type", ("Just GPT", "LangGraph","AI Agent"))
    vectorinfo = st.text_input("Enter your PDF's Information")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])





# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []




# PDF 안올리면 안나옴
if uploaded_file is not None and vectorinfo:
    
    
    if "retriever" not in st.session_state:
        pages = load_pdf(uploaded_file)
        vectorstore = create_vector_store(pages)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        st.session_state.retriever = retriever
        
        graph , llm_with_tools = chat_Agent.make_agent(retriever,vectorinfo)
        st.session_state.graph = graph
        st.session_state.llm_with_tools = llm_with_tools
        st.session_state.config = {
            'configurable': {
                'thread_id': 'summarize_paper'
            }
        }
    



    # 기존 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 유저 인풋 폼 만들기
    with st.form("chat_form"):
        prompt = st.text_input("Enter your message:")
        uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
        col1, col2, col3 = st.columns([1, 4, 1]) 
        with col1:
            submit_button = st.form_submit_button("Send")
        with col3:
            clear_button = st.form_submit_button("clear")
            
    if clear_button:
        st.session_state.messages = []
        st.rerun()

    if submit_button:
        # 사용자 메시지 콘텐츠
        message_content = []
        img = ""
        if prompt:
            message_content.append({"type": "text", "text": prompt})
        if uploaded_file is not None:
            # base64로 이미지 읽기
            image_bytes = uploaded_file.read()
            image_type = uploaded_file.type  # e.g., 'image/jpeg'
            image_data = base64.b64encode(image_bytes).decode("utf-8")
            # 메세지 콘텐츠에 이미지 넣기
            message_content.append(
                {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{image_data}"}}
            )
            img = f"data:{image_type};base64,{image_data}"
        
        
        # session state에 사용자 메세지 추가 ( 메세지 전달 기록 추가 )
        st.session_state.messages.append({"role": "user", "content": prompt+"\n"+f"![]({img})"})
        
        with st.chat_message("user"):
            if prompt:
                st.markdown(prompt)
            if uploaded_file is not None:
                st.image(uploaded_file)
        
        
        
        
        
        
        if option == "Just GPT":
            # HumanMessage 만들기
            message = HumanMessage(content=message_content)

            # 응답받기
            client = ChatOpenAI(model='gpt-4o')
            response = client.invoke([message])
            
            # session state에 AI메세지 추가 ( 메세지 전달 기록 추가 )
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            with st.chat_message("assistant"):
                st.markdown(response.content)
                
                
        elif option == "LangGraph":
            initial_state = {'query_message': prompt, 'query_img' : img, "vector_store" : st.session_state.retriever , "vector_info" : vectorinfo }
            print(initial_state)
            print()
            answer = chat_LangGraph.graph.invoke(initial_state)
            # print(answer)
            print()
            print(answer["answer"]["type"])
            

            
            if answer["answer"]["type"] == "RAGtext":
                st.session_state.messages.append({"role": "assistant", "content": answer["answer"]["message"]})
                
                with st.chat_message("assistant"):
                    st.write(answer["answer"]["message"])
                    
                    with st.expander("참고 문서 확인"):
                        for doc in answer['context']:
                            st.markdown(doc.metadata['source'], help=doc.page_content)
            
            elif answer["answer"]["type"] == "Webtext":
                st.session_state.messages.append({"role": "assistant", "content": answer["answer"]["message"]})
                
                with st.chat_message("assistant"):
                    st.write(answer["answer"]["message"])
                    
                    with st.expander("참고 문서 확인"):
                        for doc in answer['context']:
                            st.markdown(doc['title'], help=doc['url'])
           
            elif answer["answer"]["type"] == "Notext":
                st.session_state.messages.append({"role": "assistant", "content": answer["answer"]["message"]})
                
                with st.chat_message("assistant"):
                    st.write(answer["answer"]["message"])
            
            else:
                text = f"""여기 요청하신 이미지입니다!

![]({answer["answer"]["message"]})"""
                st.session_state.messages.append({"role": "assistant", "content": text })
                with st.chat_message("assistant"):
                    st.markdown(text)

 
 
        
        elif option == "AI Agent":
            
            message = HumanMessage(content=message_content)
            
            for chunk in st.session_state.graph.stream({'messages': [message], 'summary': '' , 'llm_with_tools' : st.session_state.llm_with_tools}, config=st.session_state.config, stream_mode='values'):
                chunk['messages'][-1].pretty_print()
                
            # 디버깅을 위해 stream으로 출력    
            # response = st.session_state.graph.invoke({'messages': [message], 'summary': '' , 'llm_with_tools' : st.session_state.llm_with_tools}, config=st.session_state.config)
              
            response = chunk['messages'][-1]
            
            # session state에 AI메세지 추가 ( 메세지 전달 기록 추가 )
            st.session_state.messages.append({"role": "assistant", "content": response.content})

            with st.chat_message("assistant"):
                st.markdown(response.content)

