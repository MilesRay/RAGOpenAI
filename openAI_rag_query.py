import requests
import json
import sseclient

import chromadb
from chromadb.utils import embedding_functions
import openai


model_name = "gpt-3.5-turbo"
open_ai_key = "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"

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=open_ai_key,
                model_name="text-embedding-ada-002"
            )

client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection(name="test", embedding_function=openai_ef)


while 1:
    query = input("question >")
    if len(query) < 3:
        continue

    print()

    results = collection.query(
        query_texts=[query],
        n_results=2
    )


    context = "\n\n".join(results['documents'][0])

    print(context)
    print("---------")
    prompt = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. The information given is about climate change and is on a portfolio assessment. If you don't know the answer, just say that you don't know.
    Question: {query}
    Context: {context}
    Answer:"""

    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    response_message = response["choices"][0]["message"]["content"]

    print(response_message)
