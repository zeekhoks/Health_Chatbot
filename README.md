# Health Bot - RAG LLM Project

## Description
The project is a RAG LLM project called **Health Bot**, which provides a primitive diagnosis based on symptoms entered by the user. It can also answer questions related to various diseases. 

The dataset used is the [MASHQA dataset](https://github.com/mingzhu0527/MASHQA), which contains forum-based questions and answers from WebMD and PubMed. The dataset has been processed and saved in a JSONL file, which is then split and converted into chunks of documents using Langchain's `CharacterTextSplitter`. The data is vectorized using Bedrock embeddings from Amazon Titan via Langchain, and the indices are stored in a FAISS vector store. 

A retriever is employed to search the vector store for relevant information using a combination of the prompt and user query. The similarity search results are then utilized by the Claude LLM to draw inferences and provide responses. The Claude LLM is accessed through Amazon Bedrock. Bedrock also includes a model invocation setting that generates logs of embeddings, prompts, and responses. The final result is displayed using a Streamlit UI app, allowing seamless interaction with the LLM.

## Tech Stack
- Python
- AWS Bedrock
- Langchain
- Streamlit

## Process Overview
1. **Dataset Processing**: The MASHQA dataset is processed and saved in a JSONL format, ready for chunking.
2. **Chunking**: The data is split into manageable document chunks using Langchain's `CharacterTextSplitter`.
3. **Vectorization**: The document chunks are vectorized using Bedrock embeddings from Amazon Titan, storing the indices in a FAISS vector store.
4. **Retrieval**: A retriever searches the FAISS vector store for relevant information using the combined query from the user.
5. **Inference**: The Claude LLM draws inferences based on the similarity search results and provides a response.
6. **UI Display**: The response is displayed through a Streamlit UI app for user interaction.

## FAISS
FAISS (Facebook AI Similarity Search) is a library that enables efficient similarity search and clustering of dense vectors. In this project, it helps store and retrieve the vectorized document chunks quickly, facilitating fast and accurate response generation.

## Amazon Bedrock
Amazon Bedrock provides access to foundational models for various tasks, allowing easy integration and deployment of machine learning capabilities.

## Installation Guide
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```
To run the project, execute:
```bash
streamlit run client.py
```
## Roadmap
- Future scope includes hosting the application on Elastic Beanstalk or an EC2 instance, along with creating a workflow using GitHub Actions to achieve the deployment.

## Contact Information
For any questions or suggestions, feel free to reach out:

- Email: khokawala.z@northeastern.edu
- LinkedIn: Zainab Khokawala

## License Information
This project is licensed under the terms of the MIT License.

