import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 🔹 환경 변수 로드
load_dotenv()

# 🔹 API 키 확인
def get_api_key():
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        st.error(".env 파일에 OPENAI_API_KEY 또는 AZURE_OPENAI_ENDPOINT가 누락되었습니다.")
        st.stop()

# 🔹 벡터스토어 로드
@st.cache_resource
def load_json_file():
    with open("accident_data_a.json", "r", encoding="utf-8") as file:
        accident_data = json.load(file)

    documents = []
    for case in accident_data:
        doc = Document(
            page_content=(
                f"사고유형: {case['사고유형']}\n"
                f"자동차 A: {case['자동차 A']}\n"
                f"자동차 B: {case['자동차 B']}\n"
                f"사고 설명: {case['사고 설명']}\n"
                f"과실 비율: {case['과실 비율']}\n"
                f"사고 링크: {case['사고 링크']}"
            )
        )
        documents.append(doc)

    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# 🔹 유사 문서 검색
def get_selected_docs(user_query):
    search_result = retriever.invoke(user_query)
    return "\n\n".join([doc.page_content for doc in search_result])

# 🔸 페이지 설정 및 스타일
st.set_page_config(page_title="차분해(車分解) - 교통사고 과실비율 챗봇", layout="centered")

st.markdown("""
<style>
html, body, .main {
    background-color: #121212;
    color: white;
    font-family: 'Pretendard', sans-serif;
}
h1, h2, h3 {
    text-align: center;
    font-weight: 700;
    color: #FAFAFA;
}
textarea, .stTextArea>div>div>textarea {
    background-color: #1E1E1E !important;
    color: white !important;
    border: 1px solid #333 !important;
    border-radius: 10px;
    font-size: 18px;
    padding: 1.2rem;
}
button[kind="primary"] {
    background-color: #4A90E2;
    color: white;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-weight: bold;
    font-size: 16px;
    border: none;
}
.stButton>button:hover {
    background-color: #357ABD;
}
.result-box {
    background-color: #1A1A1A;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #444;
    margin-top: 1rem;
    white-space: pre-wrap;
    font-size: 16px;
}
.footer {
    font-size: 13px;
    color: #888;
    text-align: center;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# 🔹 타이틀 및 설명
st.markdown("# 🚗 차분해 (車分解)")
st.markdown("#### 사고 원인과 과실 비율을 **차분히 해석**해드립니다.")

# 🔹 사용자 입력
user_input = st.text_area("사고 상황을 입력해주세요 (예: 신호 위반으로 교차로에서 충돌한 상황 등)", "", label_visibility="collapsed", height=150)


# 🔹 시스템 준비
get_api_key()
vectorstore = load_json_file()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 🔹 프롬프트
prompt = PromptTemplate.from_template("""
당신은 사고 상황을 설명하면 과실비율을 알려주는 챗봇입니다.
질문을 보고 참고 문서의 사고 설명과 가장 유사한 사고유형을 찾고, 먼저 사고 설명을 말해줍니다.
그다음 과실비율을 알려줍니다.
또한 마지막에 관례 판결을 보고 싶으면 사고 링크를 제공해줍니다.
사고 링크는 반드시 포함되어야 합니다.
사고 링크는 "사고 링크: [링크]" 형식으로 작성해주세요.
유사한 사례가 없다면 "유사한 사례가 없습니다. 좀 더 구체적인 상황 설명이 필요합니다." 라고 말하세요.
대답은 한국어로 해주세요.

# 참고 문서: {documents}

# 질문: {question}
""")

# 🔹 LLM 체인 구성
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.3
)
chain = prompt | llm | StrOutputParser()

# 🔹 분석 버튼 (스피너 포함)
if st.button("🚀 과실비율 분석하기", use_container_width=True):
    if not user_input.strip():
        st.warning("사고 상황을 입력해주세요.")
    else:
        with st.spinner("🧠 잠시만 기다려 주세요... 판례를 찾아보는 중입니다."):
            selected_docs = get_selected_docs(user_input)
            result = chain.invoke({"documents": selected_docs, "question": user_input})
        st.markdown(f"<div class='result-box'>📌 <strong>AI 분석 결과:</strong>\n\n{result}</div>", unsafe_allow_html=True)

# 🔹 푸터
st.markdown("<div class='footer'>© 2025 차분해. 모든 사고는 차분히 분석되어야 합니다.</div>", unsafe_allow_html=True)
