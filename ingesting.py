
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone
import os
import hashlib

load_dotenv()

if __name__ == "__main__":
    print("Ingesting PDF...")

    pdf_path = (
        "/home/meritech/Manas_research/lang_understand/"
        "Build-a-large-language-models.pdf"
    )

    # 1. Load PDF
    loader = UnstructuredLoader(
        pdf_path,
        chunking_strategy="basic",
        max_characters=10000,
    )

    documents = loader.load()
    print(f"Loaded {len(documents)} document elements")

    if not documents:
        raise ValueError("No content extracted from the PDF.")

    # 2. Split documents into chunks
    print("Splitting documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks")

    # 3. Initialize Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "lang-lab-rag")

    if not api_key:
        raise ValueError("PINECONE_API_KEY is missing from .env")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    namespace = "__default__"
    title = os.path.basename(pdf_path)

    # 4. Upload text only; Pinecone embeds chunk_text
    batch_size = 32
    print("Uploading chunks to Pinecone...")

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        records = [
            {
                "_id": f"pdf-chunk-{start + i}",
                "chunk_text": doc.page_content,
                "title": title,
                "source": pdf_path,
                "page": doc.metadata.get("page_number", 0),
            }
            for i, doc in enumerate(batch)
        ]

        index.documents.upsert(namespace=namespace, documents=records)
        print(f"Uploaded {start + len(batch)}/{len(texts)} chunks")

    print("Finished ingesting successfully!")
