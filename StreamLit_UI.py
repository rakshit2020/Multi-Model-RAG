import streamlit as st
import requests
import io
import base64
from PIL import Image
import os

st.set_page_config(
    page_title="Multi-Model RAG Assistant",
    layout="wide"
)

API_BASE_URL = "http://localhost:5000"

UPLOADS_DIR = "uploads"
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

def check_api_connection():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_document(file):
    try:
        file_path = os.path.join(UPLOADS_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error uploading document: {str(e)}")
        return None

def query_document(question):
    try:
        data = {"question": question, "return_sources": True}
        response = requests.post(f"{API_BASE_URL}/query", json=data)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error querying document: {str(e)}")
        return None

def display_images(images_b64):
    if not images_b64:
        return

    for idx, img_b64 in enumerate(images_b64):
        try:
            img_data = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_data))
            st.image(img, width=400, caption=f"Related Image {idx + 1}")
        except Exception as e:
            continue  # Skip problematic images silently

def main():
    st.title("Multi-Model RAG Assistant")

    if not check_api_connection():
        st.error("Cannot connect to the API server. Please make sure the server is running.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Upload Document")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'pptx', 'xlsx', 'txt', 'csv']
        )

        if uploaded_file is not None:
            if st.button("Upload", type="primary"):
                with st.spinner("Processing..."):
                    result = upload_document(uploaded_file)

                if result:
                    st.success("Document uploaded successfully!")
                    st.session_state.document_ready = True
                    st.session_state.document_name = uploaded_file.name
                else:
                    st.error("Failed to upload document.")

    with col2:
        st.header("Chat")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "images" in message:
                    display_images(message["images"])

        if prompt := st.chat_input("Type your question..."):
            if not st.session_state.get('document_ready', False):
                st.warning("Please upload a document first.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = query_document(prompt)

                    if response:
                        answer = response.get('answer', 'No answer found.')
                        st.markdown(answer)

                        assistant_message = {"role": "assistant", "content": answer}

                        if response.get('sources') and response['sources'].get('images'):
                            images = response['sources']['images']
                            assistant_message["images"] = images
                            display_images(images)

                        st.session_state.messages.append(assistant_message)
                    else:
                        error_msg = "Sorry, I couldn't process your question."
                        st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()