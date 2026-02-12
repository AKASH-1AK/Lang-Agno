from rag.query import retrieve
from llm.llm import generate_answer

print("📘 Student Handbook RAG Chatbot")
print("Type 'exit' to stop\n")

while True:
    question = input("Ask a question from the handbook: ")

    if question.lower() in ["exit", "quit"]:
        print("👋 Exiting chatbot")
        break

    context = retrieve(question)

    answer = generate_answer(context, question)

    print("\n--- Answer ---")
    print(answer)
    print("\n----------------\n")