from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from RAG_chatbot.config import Config

class RAGChainBuilder:
    def __init__(self,vector_store):
        self.vector_store=vector_store
        if not Config.NVIDIA_API_KEY:
            raise ValueError("❌ NVIDIA_KEY not found. Please check your .env file.")
        self.model=ChatNVIDIA(model=Config.RAGMODEL,temperature=0.3)
        self.history_store={}
    
    def get_history(self,session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id]=ChatMessageHistory()
        return self.history_store[session_id]
    

    def build_chain(self):
        retriever=self.vector_store.as_retriever(search_kwargs={"k":3})
        contextualize_q_system_prompt=("Given a chat history and the latest user question,"
                                       "which might reference context in the chat history,"
                                       "formulate a standalone question which can be understood,"
                                       "without the chat history. Do NOT answer the question,"
                                       "just reformulate it if needed and otherwise return it as is.")
        
        contextualize_prompt=ChatPromptTemplate.from_messages([("system",contextualize_q_system_prompt),
                                                       MessagesPlaceholder("chat_history"),
                                                       ("human","{input}"),])
        
        system_prompt=("You are an assistant for question-answering tasks."
                       "Use the following pieces of retrieved context to answer"
                       "the question. If you don't know the answer, say that you"
                       "don't know. Use three sentences maximum and keep the"
                       "answer concise."
                       "\n\n""{context}")
        
        qa_prompt=ChatPromptTemplate.from_messages([("system",system_prompt),
                                                       MessagesPlaceholder("chat_history"),
                                                       ("human","{input}"),])
        
        history_aware_retriever=create_history_aware_retriever(self.model,retriever,contextualize_prompt)
        quest_ans_chain=create_stuff_documents_chain(self.model,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,quest_ans_chain)


    
    
        return RunnableWithMessageHistory(
            rag_chain,self.get_history,input_messages_key="input",
            history_messages_key="chat_history",output_messages_key="answer",)
    
