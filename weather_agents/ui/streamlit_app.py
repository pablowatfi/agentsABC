import uuid
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Weather Agent", page_icon="🌤")
st.title("🌤 Weather Agent")
st.caption("Ask me anything weather-related!")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask a weather question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(BACKEND_URL, json={
                    "message": user_input,
                    "session_id": st.session_state.session_id,
                }, timeout=180)
                resp.raise_for_status()
                data = resp.json()
                answer = data["response"]
                cache_hit = data.get("cache_hit", False)
                cache_score = data.get("cache_score", 0.0)
                st.session_state.session_id = data["session_id"]
            except Exception as e:
                answer = f"Error contacting backend: {e}"
                cache_hit = False
                cache_score = 0.0

        st.write(answer)
        if cache_hit:
            st.caption(f"_Answered from cache (similarity {cache_score:.2f})_")
        st.session_state.messages.append({"role": "assistant", "content": answer})
