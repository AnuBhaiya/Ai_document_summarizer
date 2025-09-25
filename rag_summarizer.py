from llama_index.core import DocumentSummaryIndex
from llama_index.core import SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI

def run_summarization_system():
    """
    Builds a DocumentSummaryIndex and performs summarization.
    """
    # Load documents
    documents = SimpleDirectoryReader("./data").load_data()

    # Configure service context
    llm = OpenAI(model="gpt-3.5-turbo")
    service_context = ServiceContext.from_defaults(llm=llm)

    # Build the DocumentSummaryIndex
    # This will generate a summary for each document during indexing.
    doc_summary_index = DocumentSummaryIndex.from_documents(
        documents,
        service_context=service_context,
    )

    # Create a query engine specifically for summarization
    query_engine = doc_summary_index.as_query_engine(
        response_mode="tree_summarize",  # Uses a hierarchical summarization method
        use_async=True
    )

    # Query for a summary of all documents related to a topic
    summary_query = "Summarize the key findings related to Retrieval-Augmented Generation."
    response = query_engine.query(summary_query)

    print("Summary:")
    print(str(response))

if __name__ == "__main__":
    run_summarization_system()