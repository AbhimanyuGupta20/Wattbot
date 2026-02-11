import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import TextNode, ImageNode
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
import base64
import json

@st.cache_resource
def load_data():
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    return index
index = load_data()        

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])   

with st.chat_message("user"):
    st.write("Hello! Feel free to ask any questions you may have about the environmental impact of Artificial Intelligence. Before we begin, please enter a valid OpenAI API key.")

api_key = st.sidebar.text_input("Enter your OpenAI API Key here", type="password")
if not api_key:
    st.info("Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

def encode_image(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def retrieval(question):
    openai_api_key = api_key
    client = OpenAI(api_key=openai_api_key)
    input_question = question
    retriever = index.as_retriever(similarity_top_k=1, image_similarity_top_k=1)
    retrieval_results = retriever.retrieve(input_question)
    result = retrieval_results[0]
    if type(result.node).__name__ == "TextNode":
        metadata = result.metadata
        ref_url = metadata["ref_url"]

        context = result.text
        question = input_question
        qc = f"context:{context},question:{question}"
        with open("textInterpretation.prompt", "r") as f:
            textInterpretationPrompt = f.read()
        resp = client.responses.create(
            model="gpt-5.2",
            input=[
                {"role": "developer", "content": textInterpretationPrompt},
                {"role": "user", "content": qc},
            ],
        )
        output = resp.output_text
        try:
            data = json.loads(output)
            answer = data["answer"]
            evidence = data["evidence"]
        except:
            answer = "Not found (error in processing)"
            evidence = "Not found (error in processing)"
        return {
            "answer" : answer,
            "evidence" : evidence,
            "source document" : ref_url
        }

    elif type(result.node).__name__ == "ImageNode":
        metadata = result.metadata
        ref_url = metadata["ref_url"]

        path = result.node.image_path
        context = encode_image(path)
        question = input_question
        qc = f"context:(Look at the attached encoded image),question:{question}"
        with open("imageInterpretation.prompt", "r") as f:
            imageInterpretationPrompt = f.read()
        resp = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "developer", "content": imageInterpretationPrompt},
                {"role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": qc
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{context}"
                        }
                    }
                ]
                },
            ],
        )
        output = resp.choices[0].message.content
        try:
            data = json.loads(output)
            answer = data["answer"]
        except:
            answer = "Not found (error in processing)"
        return {
            "answer": answer,
            "source document of image used to answer this question": ref_url
        }

prompt = st.chat_input("Say something")  
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Processing question..."):
            answer = retrieval(prompt)
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

