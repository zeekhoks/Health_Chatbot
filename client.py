import json
import csv
import os
import pandas as pd
import boto3
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock
)


def get_claude_llm():
    llm = BedrockLLM(
        model_id="anthropic.claude-instant-v1",
        client=bedrock,
        model_kwargs=({"max_tokens_to_sample": 512}),
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("ACCESS_KEY_ID", default=None),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY", default=None),
    )
    return llm


prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    answer = qa({"query": query})
    return answer["result"]


def main():
    st.set_page_config("Health Bot")

    st.header("Chat with this health bot and find a basic diagnosis for your symptoms!")

    user_question = st.text_input("Ask a question about your symptoms.")

    if st.button("Claude Output"):
        with st.spinner("Processing...."):
            faiss_index = FAISS.load_local(
                "faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True
            )
            llm = get_claude_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")


if __name__ == "__main__":
    main()
