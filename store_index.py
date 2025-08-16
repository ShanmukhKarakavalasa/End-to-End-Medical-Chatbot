from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

os.environ["PINECONE_API_KEY"]= "pcsk_3vQhR1_2nE1eSdCSbcX1bBvk2ffnzuqiSDjwst14U2s5AhFp6Vx3jbQSuMMN52wBv5H4LF"

extracted_data=load_pdf_file(data='Data/')
text_chunks=text_split(extracted_data)
embeddings=download_hugging_face_embeddings()


pc = Pinecone(api_key="pcsk_3vQhR1_2nE1eSdCSbcX1bBvk2ffnzuqiSDjwst14U2s5AhFp6Vx3jbQSuMMN52wBv5H4LF")

index_name = "medicalbot"


pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)


docsearch= PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
