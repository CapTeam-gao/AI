import os
import json
from dotenv import load_dotenv
load_dotenv()
from trash.fewshot import examples

from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder , FewShotChatMessagePromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser


def get_llm(model = "solar-pro2"):
    return ChatUpstage(model=model) #model이라고 지정해도 ChatUpstage 객체를 return하고, 그걸 llm 변수에 담아서 llm으로 불러와야함.


def get_prompt():
    prompt = """
    당신은 학생들이 쓴 일지를 선생님들이 보기쉽도록 요약하는 AI입니다.
    학생들이 쓴 일지에서 중요한 부분들은 남겨두고 의미를 바꾸지 말고 잘 요약해주시길 바람
    
    
    context : {context}"""

    s_prompt = ChatPromptTemplate.from_message(
        [
            ("system",prompt),
            ("human","{input}")
        ]
    )
    return s_prompt


def get_ai_message():
    llm = get_llm()
    s_prompt = get_prompt()
    chain = s_prompt | llm
    response = chain.invoke()

    return print(response.content)

