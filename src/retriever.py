# src/retriever.py

import os
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

class ComplaintRetriever:
    def __init__(
        self,
        vector_store_path: str = "vector_store/chroma/",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        default_top_k: int = 5,
    ):
        """
        Initialize the retriever with a Chroma vector store and embedding model.
        :param vector_store_path: Path to the persisted Chroma vector DB
        :param embedding_model_name: HuggingFace embedding model
        :param default_top_k: Default number of documents to return
        """
        self.vector_store_path = vector_store_path
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.default_top_k = default_top_k

        if not os.path.exists(vector_store_path):
            raise ValueError(f" Vector store not found at: {vector_store_path}")

        self.db = Chroma(
            collection_name="complaints",
            embedding_function=self.embedding_model,
            persist_directory=self.vector_store_path
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        product_filter: Optional[str] = None
    ) -> List[Document]:
        """
        Perform semantic search and return top-k documents.
        :param query: User's natural language question
        :param top_k: Number of top matches to return
        :param product_filter: Optional product to filter on
        :return: List of top-k LangChain Documents
        """
        top_k = top_k or self.default_top_k

        if product_filter:
            print(f"🔍 Searching with filter: {product_filter}")
            return self.db.similarity_search(query=query, k=top_k, filter={"product": product_filter})
        else:
            print("🔍 Searching without product filter")
            return self.db.similarity_search(query=query, k=top_k)


# Optional test mode
def main():
    VECTOR_DB_PATH = "vector_store/chroma/"
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    try:
        retriever = ComplaintRetriever(
            vector_store_path=VECTOR_DB_PATH,
            embedding_model_name=MODEL_NAME,
            default_top_k=5
        )

        query = "Why are customers complaining about savings accounts?"
        product = "Savings account"

        results = retriever.retrieve(query=query, product_filter=product)

        for i, doc in enumerate(results, 1):
            print(f"\n Result {i}")
            print("Product:", doc.metadata.get("product", "N/A"))
            print(" Complaint ID:", doc.metadata.get("complaint_id", "N/A"))
            print(" Text Preview:\n", doc.page_content[:300], "...")

    except Exception as e:
        print(" Error:", str(e))


if __name__ == "__main__":
    main()
