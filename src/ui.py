import streamlit as st
import requests

st.set_page_config(page_title="Kök RAG", page_icon="👨‍🍳")

st.title("👨‍🍳 The AI Sous-Chef")
st.markdown("Ask me anything about your recipes. I only know what's in the book!")

# Initialise chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("E.g., How long do I simmer the beef?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            # Call the FastAPI container (Streamlit runs server-side, so 'kok-api' resolves via Docker DNS)
            response = requests.post("http://kok-api:8000/ask", json={"question": prompt})

            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]

                # Display the answer
                message_placeholder.markdown(answer)

                # Show the sources in an expandable dropdown
                with st.expander("View Source Chunks"):
                    for i, source in enumerate(data["sources"]):
                        st.text(f"Chunk {i+1}:\n{source}")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                message_placeholder.error(f"API Error: {response.status_code}")
        except Exception as e:
            message_placeholder.error(f"Failed to connect to API: {str(e)}")