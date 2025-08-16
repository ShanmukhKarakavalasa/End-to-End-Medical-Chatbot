from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from dotenv import load_dotenv
from src.prompt import *
from src.llm import *
import os

app=Flask(__name__)

load_dotenv()

os.environ["PINECONE_API_KEY"]= "pcsk_3vQhR1_2nE1eSdCSbcX1bBvk2ffnzuqiSDjwst14U2s5AhFp6Vx3jbQSuMMN52wBv5H4LF"


embeddings=download_hugging_face_embeddings()

index_name = "medicalbot"

docsearch= PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":5})

# Local LLM
model_name = "gpt2-medium"  # Stronger than distilgpt2, no auth required
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,  # Increased for detailed answers
    truncation=True,
    pad_token_id=tokenizer.eos_token_id,
    temperature=0.7,  # Balanced creativity
    top_p=0.9  # Improve response quality
)
llm = HuggingFacePipeline(pipeline=pipe)


# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg=request.form["user-input"]
    input=msg
    print(input)
    response=rag_chain.invoke({"input":msg})
    print("Response: ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)



