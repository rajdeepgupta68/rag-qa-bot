import os
import sys
from dotenv import load_dotenv

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(root_dir, ".env"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from huggingface_hub import InferenceClient
from retrieve import retrieve

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

client = InferenceClient(
    token=HF_TOKEN,
    provider="groq",
    timeout=30
)

PROMPT_TEMPLATE = """You are an intelligent assistant helping users understand documents deeply.

Using ONLY the context below, provide a detailed and analytical answer to the question.
- If the question asks for analysis or opinion, reason through the evidence in the context.
- Use specific details from the context to support your answer.
- If multiple perspectives exist in the context, discuss all of them.
- If the context is insufficient, say exactly what information is missing.
- At the end, cite sources like [1], [2], [3].

Context:
{context}

Question: {query}

Detailed Answer:"""

#Generate answer 
def generate_answer(query):
    chunks, sources = retrieve(query)

    context = ""
    for i, (chunk, source) in enumerate(zip(chunks, sources)):
        context += f"[{i+1}] (Source: {source})\n{chunk}\n\n"

    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=MODEL,
        max_tokens=500,
        temperature=0.3
)

    return response.choices[0].message.content, sources


if __name__ == "__main__":
    print("=== RAG Q&A Bot ===\n")
    query = input("Ask a question: ")
    answer, sources = generate_answer(query)
    print("\n--- Answer ---")
    print(answer)
    print("\n--- Sources ---")
    for i, source in enumerate(sources):
        print(f"[{i+1}] {source}")