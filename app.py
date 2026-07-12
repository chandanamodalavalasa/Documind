import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
import tempfile
import os
import json
import uuid
from datetime import datetime

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

HISTORY_FILE = "chat_history.json"
CONFIG_FILE = "config.yaml"

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500&display=swap');

    .stApp {
        background: linear-gradient(160deg, #1C1B29 0%, #2B2A3D 45%, #1E1D2B 100%);
    }

    /* Constrain actual content width instead of using columns */
    .block-container {
        max-width: 720px;
        padding-top: 3rem;
        margin: 0 auto;
    }

    h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        font-style: italic;
        font-size: 4rem;
        color: #FFFFFF;
        letter-spacing: -1.5px;
        margin-bottom: 0.3rem;
        text-align: center;
    }
    .eyebrow {
        display: none;
    }
    h4, .stMarkdown h4 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: #FFFFFF;
        letter-spacing: -0.3px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1.05rem;
        color: rgba(255,255,255,0.75);
    }
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
    }
    p, .stCaption, div[data-testid="stCaptionContainer"], label {
        font-family: 'Inter', sans-serif;
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.05rem !important;
    }
    div[data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: none;
        border-radius: 18px;
        padding: 1.8rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    }
    div[data-testid="stFileUploader"] section { background: transparent; }
    div[data-testid="stFileUploader"] label { color: #2B2A3D !important; }
    div[data-testid="stFileUploader"] p, div[data-testid="stFileUploader"] span { color: #6B6472 !important; }
    .stButton button {
        border-radius: 12px;
        border: none;
        background: #FFFFFF;
        color: #2B2A3D;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
        margin-bottom: 0.4rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    .stButton button:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.35);
        transform: translateY(-1px);
    }
    .stTextInput input {
        background: rgba(255,255,255,0.12) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.25) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    .stAlert {
        border-radius: 14px;
        background: rgba(255,255,255,0.15) !important;
        backdrop-filter: blur(15px);
        font-family: 'Inter', sans-serif;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2);
    }
    [data-testid="stExpander"] {
        background: #24233362;
        border-radius: 14px;
        border: none;
        box-shadow: 0 4px 18px rgba(0,0,0,0.2);
    }
    [data-testid="stExpander"] summary { color: white !important; font-family: 'Inter', sans-serif; }
    div[data-testid="stChatInput"] {
        background: #FFFFFF !important;
        border-radius: 16px !important;
        border: none !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3) !important;
    }
   /* Nuke every possible outer wrapper around chat_input */
div[data-testid="stBottomBlockContainer"],
div[data-testid="stBottom"],
section[data-testid="stBottom"],
div[data-testid="stChatInputContainer"],
.stChatFloatingInputContainer,
.stChatInputContainer {
    padding: 0 !important;
    margin: 0 !important;
    background: #2E2D42 !important;
    border: none !important;
}

/* Cover the true page background so nothing shows through below the bar */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: #1C1B29 !important;
}

div[data-testid="stBottomBlockContainer"],
div[data-testid="stBottom"] {
    background: transparent !important;
    padding: 0 !important;
}

div[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 20px !important;
    left: calc(21rem + 40px) !important;
    right: 30px !important;
    width: auto !important;
    max-width: none !important;
    transform: none !important;
    background: #2E2D42 !important;
    border-radius: 25px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3) !important;
    z-index: 999 !important;
    padding: 0.5rem 1rem !important;
    margin: 0 !important;
}
div[data-testid="stChatInput"] textarea {
    color: #EDEBF5 !important;
}

.block-container {
    padding-bottom: 100px !important;
}
    hr { border-color: rgba(255,255,255,0.2); }
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: rgba(255,255,255,0.65);
        background: #24233362;
        border-radius: 18px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.25);
        margin-top: 1.5rem;
    }
    .glass-bubble-user {
        background: rgba(255,255,255,0.9);
        color: #2B2420;
        padding: 0.9rem 1.3rem;
        border-radius: 18px 18px 4px 18px;
        margin: 1rem 0;
        max-width: 78%;
        margin-left: auto;
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .glass-bubble-ai {
        background: #2E2D42;
        border: none;
        color: #EDEBF5;
        padding: 0.9rem 1.3rem;
        border-radius: 18px 18px 18px 4px;
        margin: 1rem 0;
        max-width: 85%;
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        line-height: 1.6;
        box-shadow: 0 6px 24px rgba(0,0,0,0.3);
    }
    ::placeholder { color: rgba(255,255,255,0.6) !important; }
    section[data-testid="stSidebar"] {
        background: #16151F;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] * { color: #9D99AC !important; }
    section[data-testid="stSidebar"] h4 { color: #D4D1DD !important; }
    .auth-title {
        text-align: center;
        margin-bottom: 1.5rem;
    }
            .st-key-chat_row {
    position: relative;
}
.st-key-chat_row div[data-testid="stAudioInput"] {
    position: absolute;
    top: 50%;
    right: 55px;
    transform: translateY(-50%);
    width: 32px !important;
    height: 32px !important;
    background: transparent !important;
    box-shadow: none !important;
    z-index: 10;
    overflow: hidden;
}
.st-key-chat_row div[data-testid="stAudioInput"] * {
    box-shadow: none !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD CONFIG ----------
with open(CONFIG_FILE) as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# ---------- LOGIN / SIGNUP FLOW ----------
if not st.session_state.get("authentication_status"):
    st.markdown("<h1 class='auth-title'>DocuMind</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; margin-top:-0.5rem;'>A quiet space to read, question, and understand your documents.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

    with tab_login:
        authenticator.login()
        if st.session_state.get("authentication_status") is False:
            st.error("Username or password is incorrect")

    with tab_signup:
        st.caption("Create a new account to get started.")
        try:
            email, username_new, name_new = authenticator.register_user(pre_authorized=None)
            if email:
                with open(CONFIG_FILE, "w") as file:
                    yaml.dump(config, file, default_flow_style=False)
                st.success("Account created. Please log in from the 'Log In' tab.")
        except Exception as e:
            st.error(str(e))

    st.stop()

# ---------- LOGGED IN FROM HERE ----------
username = st.session_state["username"]

# ---------- CHAT HISTORY STORAGE ----------
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_history(all_history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(all_history, f, indent=2)

all_history = load_history()
user_history = all_history.get(username, {})

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("<h4 style='margin-bottom:0;'>DocuMind</h4>", unsafe_allow_html=True)
    st.caption(f"Signed in as {st.session_state.get('name', username)}")
    authenticator.logout("Log out", "sidebar")
    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.rerun()

    st.markdown("<h4 style='font-size:0.95rem; margin-top:0.5rem;'>Previous Chats</h4>", unsafe_allow_html=True)
    if len(user_history) == 0:
        st.caption("No chats yet.")
    else:
        for sid, sdata in sorted(user_history.items(), key=lambda x: x[1]["created"], reverse=False):
            label = sdata.get("title", "Untitled chat")
            if st.button(label, key=f"hist_{sid}", use_container_width=True):
                st.session_state.current_session_id = sid
                st.session_state.messages = sdata["messages"]
                st.session_state.vectorstore = None
                st.rerun()

    st.divider()
    if st.button("🗑️ Clear All History", use_container_width=True):
        all_history[username] = {}
        save_history(all_history)
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.rerun()

# ---------- MAIN AREA ----------
st.markdown("<h1 style='text-align:center;'>DocuMind</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='margin-top:-0.3rem; text-align:center;'>A quiet space to read, question, and understand your documents.</p>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if not uploaded_file and len(st.session_state.messages) == 0:
    st.markdown("""
    <div class='empty-state'>
        <p style='font-size:0.95rem;'>Upload a PDF to begin — ask questions, get answers grounded in the actual text, with page references.</p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='glass-bubble-user'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='glass-bubble-ai'>{msg['content']}</div>", unsafe_allow_html=True)

if uploaded_file:
    st.caption(f"Reading: {uploaded_file.name}")

    with st.spinner("Reading document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(pages)

        if len(chunks) == 0:
            st.error("No text could be extracted from this PDF. It may be a scanned/image-based file.")
            st.stop()

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.session_state.vectorstore = vectorstore

    st.markdown("<br>", unsafe_allow_html=True)

# Chat input now always renders, no matter what
user_question = st.chat_input("Ask something about this document...")

if user_question:
    if st.session_state.vectorstore is None:
        st.warning("Please upload a PDF first — this chat's document isn't loaded. Upload the same file again to continue asking questions.")
        st.stop()

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(user_question)

    context_text = ""
    for doc in relevant_docs:
        page_num = doc.metadata.get("page", "unknown")
        context_text += f"\n[Page {page_num}]: {doc.page_content}\n"

    prompt = f"""You are a helpful assistant answering questions based only on the provided document context.
Always cite the page number(s) you used, like [Page 3].
If the answer isn't in the context, say you don't know.

Context:
{context_text}

Question: {user_question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    answer = response.choices[0].message.content

    st.session_state.messages.append({"role": "user", "content": user_question})
    st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.current_session_id is None:
        st.session_state.current_session_id = str(uuid.uuid4())

    sid = st.session_state.current_session_id
    if sid in user_history and "title" in user_history[sid]:
        title = user_history[sid]["title"]
    else:
        title = user_question[:40] + ("..." if len(user_question) > 40 else "")

    user_history[sid] = {
        "title": title,
        "created": user_history.get(sid, {}).get("created", datetime.now().isoformat()),
        "messages": st.session_state.messages
    }
    all_history[username] = user_history
    save_history(all_history)

    st.rerun()