from datetime import datetime
from dotenv import load_dotenv
import json
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv("../.env")


def load_data() -> list[str]:
    file_path = "../data/links.txt"
    base_url = "https://www.paulgraham.com/"

    with open(file_path) as f:
        links = json.load(f)

    return [base_url + link for link in links]


def initialize_rag_system():
    """Initialize and return the RAG system components."""
    print("Initializing RAG system...")
    
    # Initialize components
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

    hf = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
    )

    vector_store = Chroma(
        collection_name="PG_essays",
        embedding_function=hf,
        persist_directory="../chroma_langchain_db",
    )

    # Load and process documents
    loader = WebBaseLoader(
        web_paths=load_data(),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(["title", "body"])),
    )

    try:
        docs = loader.load()
        for doc in docs:
            doc.metadata["chunk_source"] = "web scraping"
            doc.metadata["processing_date"] = str(datetime.now())
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        all_splits = text_splitter.split_documents(docs)
        print(f"Loaded {len(all_splits)} chunks from {len(docs)} documents.")

        # Index chunks
        vector_store.add_documents(documents=all_splits)
    except Exception as e:
        print(f"An error occurred while loading or processing documents: {e}")

    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query."""
        retrieved_docs = vector_store.similarity_search(query, k=5)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def generate(state: MessagesState):
        """Generate answer."""
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks from Paul Graham's essays and are called AskPG. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        response = llm.invoke(prompt)
        return {"messages": [response]}

    # Build graph
    graph_builder = StateGraph(MessagesState)
    tools = ToolNode([retrieve])
    memory = MemorySaver()

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile(checkpointer=memory)
    
    print("RAG system initialized!")
    return graph


# Initialize the RAG system immediately when module is imported
_rag_graph = initialize_rag_system()


def ask_pg(input_message: str, userId: int = 0) -> str:
    """
    Ask a question to the RAG system and return the answer.

    Args:
        input_message (str): The question to ask
        userId (int): User ID for conversation threading

    Returns:
        str: The answer from the RAG system
    """
    config = {"configurable": {"thread_id": userId}}

    # Get the final response
    final_response = None
    for step in _rag_graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        final_response = step["messages"][-1]

    return final_response.content if final_response else "No response generated."


if __name__ == "__main__":
    question = "How to make sure my startup won't fail?"
    answer = ask_pg(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
