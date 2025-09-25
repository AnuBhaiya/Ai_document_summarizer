from rag_indexer import create_vector_index

def run_qa_system(index):
    """
    Runs a command-line Q&A interface using the provided index.
    """
    if not index:
        print("Index not available. Exiting.")
        return

    # Create a query engine from the index
    # This is the simplest way to get a functional RAG pipeline.
    query_engine = index.as_query_engine()

    print("\nQ&A System is ready. Type 'exit' to quit.")
    while True:
        query = input("Ask a question: ")
        if query.lower() == 'exit':
            break

        print("Querying...")
        response = query_engine.query(query)

        print("\nAnswer:")
        print(str(response))

        # LlamaIndex can also return the source nodes used for the answer
        print("\nSources:")
        for node in response.source_nodes:
            # The metadata usually contains the file name
            print(f"- Source: {node.metadata.get('file_name', 'N/A')}, Score: {node.score:.4f}")

if __name__ == "__main__":
    # This assumes the indexer has been run or is run here
    index = create_vector_index()
    run_qa_system(index)