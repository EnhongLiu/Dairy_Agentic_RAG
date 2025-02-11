from typing import Literal
from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_core.messages import HumanMessage


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

    print("---CHECK RELEVANCE FROM JDS---")

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
        print("---DECISION: ANSWER IS RELEVANT---")
        return "end"

    else:
        print("---DECISION: ANSWER IS NOT RELEVANT---")
        # print(score)
        return "web"





def grade_web_answer(state) -> Literal["end", "rewrite"]:
    """
    Determines whether the answer from web search are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the answer from web search are relevant or not
    """

    print("---CHECK RELEVANCE FROM WEB SEARCH---")

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
        print("---DECISION: ANSWER IS RELEVANT---")
        return "end"

    else:
        print("---DECISION: ANSWER IS NOT RELEVANT---")
        # print(score)
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
            content=f""" Generate a polite and friendly response, within 70 words, informing users that their question is outside the chatbot's scope, which primarily focuses on providing insights related to dairy science. While the chatbot can search the web, this capability is mainly limited to search news related to Ag or dairy field. If users believe their question should be answerable, ask them to provide more context for better assistance. """,
        )
    ]
    # Rewrite agent
    model = mixtral  # use mixtral for now. <<<<<<<<<<<<<<<<<<<<<<<--------------------------------------------------
    response = model.invoke(msg)
    state["messages"].append(response)
    
    return {"messages": state['messages']}
