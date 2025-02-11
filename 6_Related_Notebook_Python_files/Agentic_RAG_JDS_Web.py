# omit this line if directly creating this file; this command is purely for running within Jupyter

import re
import os
from typing import List
# import torch

from databricks_langchain import ChatDatabricks
from langchain_google_genai import ChatGoogleGenerativeAI
from databricks_langchain import DatabricksEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
# from operator import itemgetter
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnableLambda

import mlflow
from mlflow.exceptions import RestException
import tempfile
import langchain_databricks

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.chains import LLMChain

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain.tools import Tool

from langgraph.prebuilt import tools_condition
from typing import Annotated, Literal, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field



os.environ['GOOGLE_API_KEY'] = 'AIzaSyD-jNfRVCijtlDpOWcPr4uoDuSUfpdekmE' ################## Hide ##########################
os.environ['HF_TOKEN'] = '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' ######################### Hide ##########################
os.environ["DATABRICKS_TOKEN"]= 'dapi9051c57c202f3ad21a44d756ab422d2f-3' ################## Hide ##########################


# Retriever Model
llama70b = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct",
                        temperature=0.0,
                        max_tokens=512)

# Judge Model (Use gemini2.0 for now)
gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",
                             temperature=0.0,
                             max_tokens=512)
                                    
# Final Response Model
mixtral = ChatDatabricks(endpoint="databricks-mixtral-8x7b-instruct",
                        temperature=0.0,
                        max_tokens=512)

# Embedding model
db_embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# Vector Store
persist_dir = "/dbfs/FileStore/jds_faiss_index"


def load_retriever(persist_dir=persist_dir):
    vectorstore = FAISS.load_local(
        persist_dir,
        db_embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
    )

jds_retriever = load_retriever(persist_dir)


def jds_rag(state):
    """
    Search information via JDS retrieval to answer user's question

    Args:
        state (messages): The current state

    Returns:
        str: Message to be sent to the next agent, grade_jds_answer. If passed, it will be the final answer, otherwise, web_search agent will be called
    """
    
    # persist_dir = "/Workspace/Users/el839@cornell.edu/1_DairyLLM/Data/faiss_index"
    # persist_dir = "/dbfs/FileStore/jds_faiss_index"
    # persist_dir = "dbfs:/FileStore/jds_faiss_index"
    

    # def load_retriever(persist_dir):
    #     """
    #     Load a FAISS retriever from a local directory.
    #     """
    #     embeddings = db_embeddings
    #     vectorstore = FAISS.load_local(
    #         persist_dir,
    #         embeddings,
    #         allow_dangerous_deserialization=True,
    #     )
    #     return vectorstore.as_retriever(search_type="mmr",
    #                                     search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5})

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
        # persist_dir = "/dbfs/FileStore/jds_faiss_index"
        # persist_dir = "dbfs:/FileStore/jds_faiss_index"

        # retriever = load_retriever(persist_dir) # This step might need to be optimized, otherwise FAISS database need to be loaded for each query

        retriever = jds_retriever
        
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






def web_search (state):
    """
    Search information via searching engine to answer user's question

    Args:
        state (messages): The current state

    Returns:
        str: Message to be sent to the next agent, grade_web_answer. If passed, it will be the final answer. 
    """
    question = state['messages'][0].content


    def search_duckduckgo(question):
        """
        Perform a search query using DuckDuckGo and return the search results.
        """
        wrapper = DuckDuckGoSearchAPIWrapper(
                        region="us-en",  # https://pypi.org/project/duckduckgo-search/#regions
                        safesearch= 'off',
                        time="y", # https://python.langchain.com/api_reference/community/utilities/langchain_community.utilities.duckduckgo_search.DuckDuckGoSearchAPIWrapper.html
                        source = 'news', # can change to 'news' or more 
                        max_results=5
                        )
        search = DuckDuckGoSearchResults(backend="news", output_format='list',api_wrapper=wrapper) # backend can be modified
        retrieved_answer = search.invoke(question)

        return retrieved_answer


    def generate_response_from_search(question, retrieved_texts):
        """
        Generate a response using LLM, incorporating retrieved texts.
        """

        if not retrieved_texts:
            return "I couldn't find relevant information."

        # Format retrieved texts
        formatted_retrieved_texts = ""
        for result in retrieved_texts:
            formatted_retrieved_texts += f"Title: {result['title']}\n"
            formatted_retrieved_texts += f"Source: {result['source']}\n"
            formatted_retrieved_texts += f"Snippet: {result['snippet']}\n"
            formatted_retrieved_texts += f"Link: {result['link']}\n\n" 

        prompt_template = """
        You are an AI assistant. Below is a question and some relevant information retrieved from the web.
        Question: {question}
        Retrieved Information: {retrieved_texts}
        Using the information above, provide a response to the question of less than 200 words. 
        Also, include the article titles, sources, and links in your answer following your answer. If you don't know, simply state "I don't know".
        """
        
        prompt = PromptTemplate(input_variables=["question", "retrieved_texts"], template=prompt_template)
        llm_chain = prompt | llama70b
        response = llm_chain.invoke({"question": question, "retrieved_texts": formatted_retrieved_texts})

        return response
    
    
    retrieved_texts = search_duckduckgo(question)
    response = generate_response_from_search(question, retrieved_texts)
    # return {'messages': [response]}
    state["messages"].append(response)
    
    # Return the updated state
    return {"messages": state["messages"]}






def decision_agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.
    
    Args:
        state (messages): The current state
    
    Returns:
        dict: The updated state with the agent response appended to messages
    """
    # print("---DECIDE WHETHER TO RETRIEVE---")

    message = state["messages"][-1].content

    model = llama70b 
    # use gemini before. But it's too smart to be guided only use jds-retriever. for a demo purpose, it;s not stable. but woudl suggest to switch to GPT for better performance. And Open source, eg. llama does not work at all. <<<<<<<<<<<<<<<<<<<<<<<--------------------------------------------------
    
    jds_rag_tool = Tool(func = jds_rag, name ='jds_retriever', description = 'retrieve insights from journal of dairy science (JDS) and generate response')
    tools = [jds_rag_tool]
    model = model.bind_tools(tools)
    
    response = model.invoke(message)
    state["messages"].append(response)
    
    return {"messages": state['messages']}





def grade_jds_answer(state) -> Literal["end", "web_search"]:
    """
    Determines whether the answer from retrieved documents are relevant to the question.

    Args:
        state (messages): The current state
    
    Returns:
        str: A decision for whether the answer from JDS documents are relevant or not
    """

    # print("---CHECK RELEVANCE FROM JDS---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = gemini # This model can be modified later to be a different one according to users' preference or computation restrictioin  <<<<<<<<<<<<<<<<<<<<<<<--------------------------------------------------

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of the answer based on retrieved documents to a user question. \n 
        Here is the answer from retrieved documents: \n\n {answer} \n\n
        Here is the user question: {question} \n
        If the answer generated from the retrieved documents contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["answer", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    answer = last_message.content

    scored_result = chain.invoke({"question": question, "answer": answer})
    score = scored_result.binary_score

    if score == "yes":
        # print("---DECISION: ANSWER IS RELEVANT---")
        return "end"

    else:
        # print("---DECISION: ANSWER IS NOT RELEVANT---")
        return "web"





def grade_web_answer(state) -> Literal["end", "rewrite"]:
    """
    Determines whether the answer from web search are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the answer from web search are relevant or not
    """

    # print("---CHECK RELEVANCE FROM WEB SEARCH---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = gemini # This model can be modified later to be a different one according to users' preference or computation restrictioin <<<<<<<<<<<<<<<<<<<<<<<--------------------------------------------------

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of the answer based on retrieved documents to a user question. \n 
        Here is the answer from retrieved documents: \n\n {answer} \n\n
        Here is the user question: {question} \n
        If the answer generated from the retrieved documents contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["answer", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    answer = last_message.content

    scored_result = chain.invoke({"question": question, "answer": answer})

    score = scored_result.binary_score

    if score == "yes":
        # print("---DECISION: ANSWER IS RELEVANT---")
        return "end"

    else:
        # print("---DECISION: ANSWER IS NOT RELEVANT---")
        # return "rewrite"
        return "response"



def response(state):
    """
    Generate a response for user queries that are out of the scope of this chatbot.
    Args: state (messages): The current state

    Returns: dict: The updated state with the the response
    """
    final_phase_input = state["messages"][-1].content

    msg = [
        HumanMessage(
            content="""Generate a polite and friendly response within 80 words, informing users that their question is outside the chatbot's scope, which primarily focuses on providing insights related to dairy science. While the chatbot can search the web, this capability is mainly limited to search news related to Ag or dairy field. If users believe their question should be answerable, ask them to provide more context for better assistance. """,
        )
    ]
    # Final agent
    model = llama70b  # use mixtral for now. <<<<<<<<<<<<<<<<<<<<<<<--------------------------------------------------
    response = model.invoke(msg)
    state["messages"].append(response)
    
    return {"messages": state['messages']}



class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

def load_graph():
    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the nodes we will cycle between
    workflow.add_node("decide_tool", decision_agent)  # decision agent to retrieve or end
    # retrieve = ToolNode([jds_rag])
    workflow.add_node("jds_retrieve", jds_rag)  # retrieve JDS insights
    workflow.add_node("web_search", web_search) # Web search the answer
    workflow.add_node("ask_for_more", response)  # Re-write the question

    # Connect agent nodes
    workflow.add_edge(START, "decide_tool")
    workflow.add_conditional_edges("decide_tool", tools_condition,{"tools": "jds_retrieve",  END: END}) # Decide whether to retrieve
    workflow.add_conditional_edges("jds_retrieve", grade_jds_answer, {'web':'web_search', 'end': END}) # Decide whether to websearch
    workflow.add_conditional_edges("web_search", grade_web_answer,{"response": "ask_for_more", "end": END}) # Decide whether to rewrite
    workflow.add_edge("ask_for_more", END)

    # Compile
    graph = workflow.compile()

    return graph



mlflow.models.set_model(load_graph())
