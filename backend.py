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


def convert_json_csv(json_filename, csv_filename):
    with open(json_filename, 'r') as file:
        data=json.load(file)


    qa_pairs = []
    for item in data['data']:
        for paragraph in item['paragraphs']:
            for qa in paragraph['qas']:
                question=qa['question']
                answer=qa['answers'][0]['text'] if qa['answers'] else None
                qa_pairs.append((question,answer))

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Question','Answer'])
        csv_writer.writerows(qa_pairs)

current_working_directory = os.getcwd()

for filename in os.listdir(current_working_directory):
    if(filename.endswith('.json')):
        json_filepath=os.path.join(current_working_directory, filename)
        csv_filename=os.path.splitext(filename)[0]+'_output.csv'
        csv_filepath=os.path.join(current_working_directory, csv_filename)

        convert_json_csv(json_filepath, csv_filepath)

def merge_all_csvs(input_folder, output_filename):
    csv_files= [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    if os.path.exists(output_filename):
        raise FileExistsError(f"The output file `{output_filename}` already exists. Please choose a different name.")

    dfs = [pd.read_csv(os.path.join(input_folder, csv_file)) for csv_file in csv_files]
    merged_df = pd.concat(dfs, ignore_index=True)

    merged_df.to_csv(output_filename, index=False, encoding='utf-8')

    print(f'Merged csv file `{output_filename}` created successfully!')

input_folder_path=os.getcwd()
output_merged_file='merged_output_all.csv'
merge_all_csvs(input_folder_path, output_merged_file)

df=pd.read_csv('merged_output_all.csv')
json_data=df.to_json("merged.jsonl", orient='records',lines=True)

with open("merged.jsonl", "r") as file:
    json_list = [line.strip("{}\n") for line in file]
# print(json_list[0])

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=10000,
    chunk_overlap=1000,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents(json_list[:500])
# print(len(texts))

load_dotenv()

bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock
)


def get_vector_store(docs):
    print("In vector store function")
    print(texts[1])
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    print(vectorstore_faiss)
    print("Vector store created")
    vectorstore_faiss.save_local("faiss_index")


def main():
    get_vector_store(texts)
 

if __name__ == "__main__":
    main()
