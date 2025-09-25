import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI

def create_vector_index(data_directory="./data"):
    """
    Loads documents, creates embeddings, and builds a vector index.

    Args:
        data_directory (str): The path to the directory containing documents.

    Returns:
        VectorStoreIndex: The created index object.
    """
    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # 1. Load documents from the specified directory
    # SimpleDirectoryReader automatically handles various file types, including PDF.
    print(f"Loading documents from '{data_directory}'...")
    try:
        documents = SimpleDirectoryReader(data_directory).load_data()
        if not documents:
            print("No documents found in the directory.")
            return None
    except Exception as e:
        print(f"Error loading documents: {e}")
        return None

    # 2. Configure the service context
    # This bundles the LLM and embedding model. LlamaIndex uses OpenAI models by default.
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    service_context = ServiceContext.from_defaults(llm=llm)

    # 3. Create the VectorStoreIndex
    # This step handles text splitting, embedding, and storing in a simple in-memory vector store.
    print("Creating vector index... This may take some time depending on document size.")
    index = VectorStoreIndex.from_documents(
        documents, service_context=service_context
    )
    print("Index created successfully.")

    return index

if __name__ == "__main__":
    # Create a dummy data directory and file for testing
    if not os.path.exists("./data"):
        os.makedirs("./data")
    with open("./data/sample_research_paper.txt", "w") as f:
        f.write("Artificial intelligence (AI) is intelligence demonstrated by machines, "
                "in contrast to the natural intelligence displayed by humans and animals. "
                "Retrieval-Augmented Generation (RAG) is an advanced AI framework that "
                "improves the quality of LLM responses by grounding the model on external "
                "sources of knowledge.")

    vector_index = create_vector_index()
    if vector_index:
        print("\nVector index is ready for querying.")