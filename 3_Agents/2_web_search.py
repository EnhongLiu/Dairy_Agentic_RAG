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





