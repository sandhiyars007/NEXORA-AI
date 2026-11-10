
import streamlit as st
import uuid

from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

st.set_page_config(
    page_title="NEXORA | AI Assistant",
    page_icon="✨",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background: #10111B;
    color: #F5F5F5;
}

[data-testid="stSidebar"] {
    background: #191A29;
}

.main-title {
    font-size: 45px;
    font-weight: 800;
    color: #A78BFA;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #A1A1AA;
    font-size: 15px;
    margin-bottom: 30px;
}

.stChatMessage {
    border-radius: 14px;
    border: 1px solid #303044;
    padding: 12px;
}

.stChatInput textarea {
    background: #202133;
    color: white;
}

div.stButton > button {
    border-radius: 10px;
    border: 1px solid #8B5CF6;
    background: #25223A;
    color: white;
}

div.stButton > button:hover {
    background: #6D28D9;
    color: white;
}

</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "history_store" not in st.session_state:
    st.session_state.history_store = {}

def get_session_history(session_id):

    if session_id not in st.session_state.history_store:

        st.session_state.history_store[session_id] = (
            InMemoryChatMessageHistory()
        )

    return st.session_state.history_store[session_id]

def create_chatbot(model_name):

    llm = ChatOllama(
        model=model_name,
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
            You are NEXORA, a friendly, intelligent and helpful
            personal AI assistant.

            Your responsibilities:

            1. Answer questions clearly and accurately.
            2. Remember previous conversations using chat memory.
            3. Maintain conversation context.
            4. Give simple explanations and examples.
            5. Help with programming, education, technology,
               artificial intelligence and general knowledge.
            6. Be friendly, professional and supportive.
            7. If you do not know an answer, be honest.

            Always provide useful and easy-to-understand responses.
            """
        ),

        MessagesPlaceholder(variable_name="history"),

        ("human", "{input}")

    ])

    chain = prompt | llm

    chatbot = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    return chatbot

with st.sidebar:

    st.markdown("## ✨ NEXORA")

    st.caption("Your Intelligent AI Companion")

    st.divider()

    st.markdown("### ⚙️ Settings")

    model_name = st.selectbox(
        "Choose AI Model",
        [
            "gemma3:1b",
            "llama3.2:1b",
            "qwen2.5:1.5b"
        ]
    )

    st.caption(
        "Select an installed Ollama model."
    )

    st.divider()

    st.markdown("### 🧠 Memory Status")

    st.success("LangChain Memory Active")

    history = get_session_history(
        st.session_state.session_id
    )

    st.write(
        f"Messages remembered: {len(history.messages)}"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        old_session = st.session_state.session_id

        st.session_state.history_store.pop(
            old_session, None
        )

        st.session_state.session_id = str(uuid.uuid4())

        st.session_state.messages = []

        st.rerun()

    st.markdown("---")

    st.caption("Powered by Python • Streamlit • LangChain • Ollama")

    st.caption("NEXORA AI © 2026")

st.markdown(
    '<div class="main-title">✨ NEXORA AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Think. Learn. Remember. Respond. ✨'
    '</div>',
    unsafe_allow_html=True
)

if not st.session_state.messages:

    st.markdown("### 👋 Welcome to NEXORA!")

    st.markdown(
        "Hello! I'm NEXORA, your personal AI assistant. "
        "I'm here to answer your questions and remember "
        "our conversation."
    )

    st.markdown("#### 💡 Try asking me")

    suggestions = [

        "Explain Artificial Intelligence simply",

        "Give me Python project ideas",

        "What is LangChain?",

        "How does machine learning work?"

    ]

    cols = st.columns(2)

    for i, suggestion in enumerate(suggestions):

        with cols[i % 2]:

            if st.button(
                suggestion,
                use_container_width=True
            ):

                st.session_state.pending_prompt = suggestion

                st.rerun()

for message in st.session_state.messages:

    with st.chat_message(
        message["role"],
        avatar="👤" if message["role"] == "user" else "✨"
    ):

        st.markdown(message["content"])

prompt_text = st.chat_input(
    "Message NEXORA..."
)

if "pending_prompt" in st.session_state:

    prompt_text = st.session_state.pop(
        "pending_prompt"
    )

if prompt_text:

    st.session_state.messages.append({

        "role": "user",

        "content": prompt_text

    })

    with st.chat_message("user", avatar="👤"):

        st.markdown(prompt_text)

    with st.chat_message("assistant", avatar="✨"):

        try:

            chatbot = create_chatbot(model_name)

            with st.spinner("NEXORA is thinking..."):

                response = chatbot.invoke(

                    {"input": prompt_text},

                    config={

                        "configurable": {

                            "session_id": st.session_state.session_id

                        }

                    }

                )

            answer = response.content

            st.markdown(answer)

            st.session_state.messages.append({

                "role": "assistant",

                "content": answer

            })

            st.rerun()

        except Exception as e:

            st.error(
                "Unable to connect to the AI model. "
                "Please check whether Ollama is running "
                "and the selected model is installed."
            )

            st.caption(str(e))
