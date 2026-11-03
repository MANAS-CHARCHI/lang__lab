from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from pinecone import Pinecone, EmbedQuery

load_dotenv()

llm = ChatAnthropic(model="claude-haiku-5-5")

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "lang-lab-rag")
if not api_key:
    raise ValueError("PINECONE_API_KEY is missing from .env")

pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)
namespace = "__default__"

def retrieve_documents(query: str, top_k: int = 5):
    results = index.documents.search(
        namespace=namespace,
        top_k=top_k,
        score_by=[
            EmbedQuery(
                field="chunk_text",
                query=query
            )
        ],
        include_fields=[
            "chunk_text",
            "title",
            "source",
            "page"
        ]
    )

    documents = []

    for hit in results.matches:
        document = {
            "text": hit.get("chunk_text"),
            "score": hit.score,
            "metadata": {
                "title": hit.get("title"),
                "source": hit.get("source"),
                "page": hit.get("page")
            }          
        }
        documents.append(document)
    return documents

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant.

        Answer the question using only the provided context.
        If the answer is not available in the context,
        explicitly say that you don't know.
        Try to answer in a concise manner, and avoid repeating the context verbatim.
        Try to answer direct with only important information, and avoid unnecessary details.

        Context:
        {context}
        """
    ),
    ("human", "{question}")
])

def ask_questions(question: str):
    documents = retrieve_documents(question)
    if not documents:
        return "No relevant documents found for the question."
    
    context = "\n\n".join([
            f"Page: {doc['metadata']['page']}\n{doc['text']}"
            for doc in documents
    ])
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    if isinstance(response.content, str):
        answer = response.content

    else:
        answer = "\n".join(
            block["text"]
            for block in response.content
            if isinstance(block, dict)
            and block.get("type") == "text"
        )
    return answer

if __name__ == "__main__":
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        answer = ask_questions(question)

        print(f"Answer: {answer}\n")