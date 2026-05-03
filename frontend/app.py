import streamlit as st
import requests
import base64
import json
from pathlib import Path

API_BASE = "http://localhost:8000"
HISTORY_FILE = Path("chat_history.json")

# ---------------- HISTORY ----------------
def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except:
            return []
    return []

def save_history(messages):
    try:
        HISTORY_FILE.write_text(json.dumps(messages[-50:], indent=2))
    except:
        pass

# ---------------- UI ----------------
st.set_page_config(page_title="Livestock Assistant")
st.title(" Livestock AI Assistant")

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = load_history()

if "pending_image" not in st.session_state:
    st.session_state.pending_image = None

# ---------------- CLEAR BUTTON ----------------
if st.button(" Clear Chat"):
    st.session_state.messages = []
    save_history([])
    st.rerun()

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.session_state.pending_image = uploaded_file
    st.image(uploaded_file, caption="📸 Image ready")

if st.session_state.pending_image:
    st.info("📸 Image will be used when you send your message")

# ---------------- USER INPUT ----------------
user_input = st.chat_input("Ask something...")

# ---------------- MAIN LOGIC ----------------
if user_input:
    try:
        # ---------- USER MESSAGE ----------
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        # ---------- IMAGE MODE ----------
        if st.session_state.pending_image:

            with st.spinner(" Analyzing image..."):

                file_bytes = st.session_state.pending_image.getvalue()

                if not file_bytes:
                    st.error("Empty image file")
                    st.stop()

                files = {
                    "file": (
                        st.session_state.pending_image.name,
                        file_bytes,
                        st.session_state.pending_image.type
                    )
                }

                data = {"query": user_input}

                res = requests.post(
                    f"{API_BASE}/detect_and_explain",
                    files=files,
                    data=data
                )

                try:
                    result = res.json()
                except:
                    st.error(f"Invalid response: {res.text}")
                    st.stop()

                if "error" in result:
                    st.error(result["error"])
                    st.stop()

                if result.get("image"):
                    st.image(base64.b64decode(result["image"]))

                text = result.get("agent_response", {}).get("text", "No response")

                disease = result.get("disease", "Unknown")
                confidence = result.get("confidence", 0)

                prediction_info = f"**Disease:** {disease}  \n**Confidence:** {confidence:.2f}"

                # clear image after use
                st.session_state.pending_image = None

        # ---------- TEXT MODE ----------
        else:
            with st.spinner(" Thinking..."):

                res = requests.post(
                    f"{API_BASE}/text_query",
                    data={"query": user_input}
                )

                try:
                    result = res.json()
                except:
                    st.error(f"Invalid response: {res.text}")
                    st.stop()

                text = result.get("agent_response", {}).get("text", "No response")
                prediction_info = ""

        # ---------- RESPONSE ----------
        formatted = (
            f"###  Assistant\n\n{prediction_info}\n\n{text}"
            if prediction_info
            else f"### Assistant\n\n{text}"
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": formatted
        })

        with st.chat_message("assistant"):
            st.markdown(formatted)

        save_history(st.session_state.messages)

    except Exception as e:
        st.error(f"Error: {str(e)}")
