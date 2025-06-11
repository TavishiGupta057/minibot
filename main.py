import streamlit as st
from llama_cpp import Llama

@st.cache_resource
def load_llama_model():
    return Llama(
        model_path="/mnt/c/Users/tavis/OneDrive/linux/chatbot/model/meta-llama-3-8b-instruct.Q5_K_M.gguf",
        n_ctx=2048,
        n_threads=6,
        temperature=0.2,
        top_p=0.7,
        stop=["###", "<|assistant|>"],
        verbose=False
    )

llm = load_llama_model()

st.title("🦙 LLaMA 3 Chatbot (Offline)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything…")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    prompt = f"<|begin_of_text|><|user|>\n{user_input.strip()}\n<|assistant|>"

    output = llm(prompt, max_tokens=200)
    response = output["choices"][0]["text"].strip()

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
