import streamlit as st
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from senti import run_sentiment
import time

load_dotenv()
#강의 5주차 1 Streamlit에서 RAG 구현하기 (feat. 멀티턴) solar_rag.py 코드의 내용을 많이 참조하였음.

st.set_page_config(page_title="1조_권용진")
st.title("중간평가용 멀티턴 챗봇")

# 시작하면 아무것도 없으므로 시스템 프롬프트로 messages라는 리스트 영구 변수 생성 및 저장. 여기에 모든 대화 내용이 축적
if "messages" not in st.session_state:
    # 가장 첫 시스템 프롬프트
    st.session_state["messages"] = [{"role": "system", "content": "반말로 답변해줘. 그리고 너의 이름은 앞으로 '거북이'야. 기억해둬."}]

# 이전 대화 모두 상태창에 출력
for memo in st.session_state["messages"][1:]:  
    # messages 리스트의 첫 딕셔너리[0]는 시스템 프롬프트이므로 제외하고 나머지 것들만 for로 꺼내 계속 쌓아서 출력.
    #for 돌때마다 role을 구분하고 그 role이 말한 대화 내용을 쌓음.
    #st.chat_message 안에는 role이 누구인지에 따라서 카카오톡 대화처럼 구분해주는 기능이 있다.
    with st.chat_message(memo["role"]):
        st.markdown(memo["content"])

# 유저의 입력을 받아 user_input에 저장. 입력이 되면, 즉 input이 True면 아래를 실행
# messages 변수에 role은 user, 받아온 입력값은 content로 리스트 추가하고 [이전 대화 출력] 구문처럼 카톡 메세지 형태로 출력 
if user_input := st.chat_input("여기에 메세지 입력!"):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Solar LLM 준비 (API 키는 .env에서 자동). 그냥 mini가 익숙함.
    llm = ChatUpstage(model="solar-mini")
    
    # 멀티턴 구현.. 이전 대화 모두 LLM에게 전달
    #system 프롬프트를 맥락 중에도 기억할 수 있게 꺼내와서 전달. 고정이니 굳이 for 하지 않아도 되고.
    #과거 대화 내용을 user와 assistant로 분리. 파이썬 임시 변수를 만들어서 messages 에서 꺼내와 저장한다.
    history_memory = [SystemMessage(content=st.session_state["messages"][0]["content"])]
    #user가 입력했던 모든내용 꺼내오기, assistant가 답했던 모든 내용 꺼내오기.
    #history_memory에 리스트 요소로 저장.
    for x in st.session_state["messages"][1:]:
        if x["role"] == "user":
            history_memory.append(HumanMessage(content=x["content"]))
        elif x["role"] == "assistant":
            history_memory.append(AIMessage(content=x["content"]))

    # Solar에게 이전 대화(멀티턴) 한 번에 전달.
   
    #답변 출력 부분. 좀 더 멋있게 하고 싶다.
    # invoke가 있었지만 역시 LLM은 stream이지! 
    # 길정현님 코드 참조함.
    with st.chat_message("assistant"):
        with st.spinner("답변 생성중..."):
            msg_box = st.empty()
            txt_space = ""
            for chunk in llm.stream(history_memory):
                txt_space += chunk.content
                msg_box.markdown(txt_space)
                time.sleep(0.3)

            #감정분류 부분. 완성된 txt_space를 text 타입으로 받아와 함수에 넣고 돌려 결과를 반환받는다.
        sentiment = run_sentiment(txt_space)
        st.markdown(f"감정분석: **{sentiment}**")
    #solar의 답변 내용을 messages 변수에 session_state로 영구 저장
    st.session_state["messages"].append({"role": "assistant", "content": txt_space})



