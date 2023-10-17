import json

import chromadb
from chromadb.utils import embedding_functions
import openai

open_ai_key = "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"
chunk_tokens_size = 512


openai.api_key = open_ai_key
client = chromadb.PersistentClient(path="./db")

document = open("climate_test.txt", "rb").read()

document = document.decode("utf-8", errors="ignore")

document = document.replace("\r", "")

# print(len(document))

paragraphs = document.split("\n\n")
while '' in paragraphs:
    paragraphs.remove('')
    

print(paragraphs)        


openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=open_ai_key,
                model_name="text-embedding-ada-002"
            )


collection = client.get_or_create_collection(name="test", embedding_function=openai_ef)

print("computing embeddings...")
collection.add(
    documents=paragraphs,
    ids = [str(i) for i in list(range(0, len(paragraphs)))]
)

print(paragraphs)