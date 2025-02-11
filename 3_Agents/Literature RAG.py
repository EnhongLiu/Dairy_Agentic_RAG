import os
import logging
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field

from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import DatabricksEmbeddings
from langchain.chat_models import ChatDatabricks
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.tools import Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

from langchain_core.messages import HumanMessage


def jds_rag(state):
    """
    Search information via JDS retrieval to answer user's question

    Args:
        state (messages): The current state

    Returns:
        str: Message to be sent to the next agent, grade_jds_answer. If passed, it will be the final answer, otherwise, web_search agent will be called
    """
    
    # persist_dir = "/Workspace/Users/el839@cornell.edu/1_DairyLLM/Data/faiss_index"
    persist_dir = "/dbfs/FileStore/jds_faiss_index"

    def load_retriever(persist_dir):
        """
        Load a FAISS retriever from a local directory.
        """
        embeddings = db_embeddings
        vectorstore = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore.as_retriever(search_type="mmr",
                                        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5})

    # Define prompt for RAG
    RAG_PROMPT = """
    Using the context provided, answer the user's query concisely in 100-200 words. Following the response, list the full titles, publication years, and DOIs of the top 5 sources referenced. If the query cannot be answered, reply with "I don't know." When you do not know the anser, you DO NOT provide the sources. 

    Question:
    {question}

    Context:
    {context}
    """

    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    question = state["messages"][0].content

    # RAG-based response generation
    def generate_response_with_rag(question):
        """
        Generate a response using RAG with retrieved documents from the retriever.
        """

        # Initialize retriever
        # persist_dir = "/Workspace/Users/el839@cornell.edu/1_DairyLLM/Data/faiss_index"
        persist_dir = "/dbfs/FileStore/jds_faiss_index"
        retriever = load_retriever(persist_dir) # This step might need to be optimized, otherwise FAISS database need to be loaded for each query
        
        retrieved_docs = retriever.invoke(question)
        
        if not retrieved_docs:
            return "No relevant information found."
        
        # Format retrieved documents
        formatted_context = "\n".join([
            f"Title: {doc.metadata.get('Title', 'Unknown')}\n"
            f"Authors: {doc.metadata.get('Authors', 'Unknown')}\n"
            f"Publication Date: {doc.metadata.get('Publication Date', 'Unknown')}\n"
            f"DOI: {doc.metadata.get('DOI', 'Unknown')}\n"
            f"Link: {doc.metadata.get('Link', 'Unknown')}\n"
            f"Affiliations: {doc.metadata.get('Affiliations', 'Unknown')}\n"
            f"PubMed ID: {doc.metadata.get('PubMed ID', 'Unknown')}\n"
            f"Publication Year: {doc.metadata.get('Publication Year', 'Unknown')}\n"
            f"Country: {doc.metadata.get('Country', 'Unknown')}\n"
            f"Content: {doc.page_content}" for doc in retrieved_docs
        ])
        
        llm_chain = rag_prompt | llama70b
        response = llm_chain.invoke({"question": question, "context": formatted_context})
        
        return response

    response = generate_response_with_rag(question)    
    # return {'messages': [response]}

    state["messages"].append(response)

    
    # Return the updated state
    return {"messages": state["messages"]}
