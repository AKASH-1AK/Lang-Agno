import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"   # or llama3:8b-instruct-q4_K_M for speed


def generate_answer(context: str, question: str) -> str:
    """
    Generate an answer using Ollama LLM based strictly on retrieved context.
    Produces paragraph-style answers for descriptive questions.
    """

    # Safety check: no context
    if not context or len(context.strip()) < 30:
        return "Not found in the document."

    prompt = f"""
You are a helpful academic assistant.

Using ONLY the information given in the context below,
write a clear and concise paragraph answering the question.

Rules:
- Combine related points if needed
- Rephrase the content in your own words
- Do NOT add information outside the context
- Do NOT guess or hallucinate
- If the answer is not present at all, say exactly:
  Not found in the document.

Context:
{context}

Question:
{question}

Answer (in paragraph form):
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": True
            },
            stream=True,
            timeout=120
        )

        answer = ""

        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line.decode("utf-8"))

            if "response" in data:
                answer += data["response"]

        final_answer = answer.strip()

        if not final_answer:
            return "Not found in the document."

        return final_answer

    except Exception as e:
        print("❌ LLM Error:", e)
        return "⚠️ The system took too long to respond. Please try again."
