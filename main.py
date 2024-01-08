import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(
    openai_api_key=api_key,
    temperature=0.1,
)
# ------
from operator import itemgetter

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory(llm=model)

conversation = ConversationChain(llm=model, memory=memory)

print(conversation.predict(input="안녕, 나는 토니 스타크라고 해."))
print(conversation.predict(input="내 이름이 뭐였지?"))

print(memory.load_memory_variables({}))
