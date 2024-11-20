import logging
from typing import Optional
import os

from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from nemoguardrails import LLMRails

from nemoguardrails.actions import action

from dotenv import load_dotenv
load_dotenv()

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PARENT_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)


def get_documents():
    directory_path = os.path.join(CURRENT_DIRECTORY, 'kb')

    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.md'):
            loader = TextLoader(os.path.join(directory_path, filename))
            docs = loader.load()
            documents.extend(docs)
    return documents


@action(name='ask_questions')
async def ask_questions(input: str):
    print("---------------- Entered Function ----------------")
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
    vector_store = Chroma(
        persist_directory=os.path.join(PARENT_DIRECTORY, 'faults'),
        embedding_function=embedding_model
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={
                                          "k": 3})
    results = retriever.get_relevant_documents(input)

    combined_context = "\n Event Details Finished. \n".join([doc.page_content for doc in results])

    system_prompt = """
    You are a diagnostic assistant whose sole purpose is to ask clarifying questions to determine if a user-described event is similar to any of the past events provided. Your goal is to verify if strategies used in past events can be applied to the current situation.

    Input Format:
    - **User Described Event**: A description of the current issue provided by the user.
    - **Relevant Past Events**: A list of three past events that might share similarities with the current issue.

    Your Task:
    1. Compare the user-described event to each of the past events. 
    2. Ask specific, concise questions to clarify whether key aspects of the user-described event align with any of the past events.
    3. Focus on:
    - Verifying similarities between the described event and the past events.
    - Checking whether strategies or actions from past events are applicable.

    Rules:
    1. Only use the user-described event and the past events as your basis for generating questions.
    2. Do not provide solutions, analysis, or introduce external knowledge.
    3. Limit your response to 3-5 targeted, context-based questions.
    4. Each question should explicitly connect the described event to a specific past event or strategy.

    Examples of Questions:
    - "In the past event where [specific past event detail], [specific strategy] was used. Have you tried this approach?"
    - "Does the current issue involve [specific detail from a past event], similar to [past event]?"
    - "Are there any differences in [specific aspect] between the described event and [past event]?"

    Remember, your role is solely to gather information about similarities and the potential reuse of past strategies.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User Described Event: {input}\nRelevant Past Events:\n {context}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm
    response = chain.invoke(
        {
            "input": input,
            "context": combined_context
        }
    )
    return response.content


def init(app: LLMRails):
    if not os.path.exists(os.path.join(PARENT_DIRECTORY, 'faults')):
        documents = get_documents()
        embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
        vector_store = Chroma.from_documents(
            documents, embedding_model, persist_directory=os.path.join(PARENT_DIRECTORY, 'faults'))

    app.register_action(ask_questions, "ask_questions")
