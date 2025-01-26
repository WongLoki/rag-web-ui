import json
import base64
from typing import List, AsyncGenerator
from sqlalchemy.orm import Session
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from app.core.config import settings, ModelConfig
from app.models.chat import Message
from app.models.knowledge import KnowledgeBase, Document
from langchain.globals import set_verbose, set_debug
from app.services.vector_store import VectorStoreFactory
import logging
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from app.services.llm_factory import LLMFactory
import traceback
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

set_verbose(True)
set_debug(True)

# Define prompt templates
OPENAI_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the question.\n\nContext: {context}"),
    ("human", "{input}")
])

OLLAMA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the context below:\n\nContext: {context}"),
    ("human", "{question}")
])

def create_openai_chain(llm, retriever, chat_history):
    """Create a retrieval chain for OpenAI"""
    chain = (
        {
            "context": retriever,
            "chat_history": RunnablePassthrough(),
            "input": RunnablePassthrough()
        }
        | OPENAI_PROMPT
        | llm
    )
    return chain

def format_docs(docs) -> str:
    """Format the document list as a string"""
    return "\n\n".join(doc.page_content for doc in docs)

def create_ollama_chain(llm, retriever, chat_history):
    """Create a retrieval chain for Ollama"""

    # Create a simple chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | OLLAMA_PROMPT
        | llm
        | {"output": RunnablePassthrough()}
    )
    return chain

async def generate_response(
    query: str,
    messages: dict,
    knowledge_base_ids: List[int],
    chat_id: int,
    model: str,
    db: Session
) -> AsyncGenerator[str, None]:
    try:
        logger.info(f"Starting to process query: {query}")
        logger.info(f"Knowledge base IDs: {knowledge_base_ids}")
        

        message = Message(
            chat_id=chat_id,
            role="user",
            content=query
        )
        db.add(message)
        db.commit()

      # Create bot message placeholder
        bot_message = Message(
            content="",
            role="assistant",
            chat_id=chat_id
        )
        db.add(bot_message)
        db.commit()
        logger.info("Bot message placeholder created")

       # Obtain knowledge base and documents
        knowledge_bases = (
            db.query(KnowledgeBase)
            .filter(KnowledgeBase.id.in_(knowledge_base_ids))
            .all()
        )
        
        # Create embeddings and Large Language Models (LLMs).
        embeddings = LLMFactory.create_embeddings()
        llm = LLMFactory.create_llm(model)

        # Create vector storage
        vector_stores = []
        for kb in knowledge_bases:
            logger.info(f"Processing knowledge base: {kb.id}")
            documents = db.query(Document).filter(Document.knowledge_base_id == kb.id).all()
            if documents and len(documents) > 0:
                # Extract text content from chunks
                texts = []
                metadatas = []
                for doc in documents:
                    for chunk in doc.chunks:
                        texts.append(chunk.chunk_metadata["page_content"])
                        metadatas.append({
                            "source": doc.file_name,
                            "file_path": doc.file_path
                        })
                
                if texts:  # Only create vector storage when there is text content.
                    vector_store = VectorStoreFactory.create(
                        store_type=settings.VECTOR_STORE_TYPE,
                        collection_name=f"kb_{kb.id}",
                        embedding_function=embeddings,
                        texts=texts,
                        metadatas=metadatas
                    )
                    vector_stores.append(vector_store)

        if not vector_stores:
            error_msg = "No documents found in knowledge base."
            if settings.DEFAULT_LLM_PROVIDER == "openai":
                yield f'0:"{error_msg}"\n'
            else:
                yield f"data: {error_msg}\n\n"
            return

        # Build a search engine
        retriever = vector_stores[0].as_retriever()

        # Handle Chat History
        chat_history = []
        for msg in messages["messages"]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

        # According to the provider, different processing methods are chosen.
        if settings.DEFAULT_LLM_PROVIDER == "openai":
            chain = create_openai_chain(llm, retriever, chat_history)
        else:  # ollama
            chain = create_ollama_chain(llm, retriever, chat_history)

        # Stream
        full_response = ""
        async for chunk in chain.astream(query):
            if settings.DEFAULT_LLM_PROVIDER == "openai":
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if content:
                    escaped_chunk = content.replace('"', '\\"').replace('\n', '\\n')
                    yield f'0:"{escaped_chunk}"\n'
                    full_response = content
            else:
                # Ollama
                if isinstance(chunk, dict) and "output" in chunk:
                    content = chunk["output"].content if hasattr(chunk["output"], "content") else str(chunk["output"])
                    if content:
                        yield f"data: {content}\n\n"
                        full_response += content 

        # Update message content
        bot_message.content = full_response
        db.commit()
        logger.info("Bot message updated")

        if settings.DEFAULT_LLM_PROVIDER != "openai":
            yield "data: [DONE]\n\n"

    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Stack trace: {traceback.format_exc()}")
        if settings.DEFAULT_LLM_PROVIDER == "openai":
            yield f'3:"{error_msg}"\n'
        else:
            yield f"data: {error_msg}\n\n"
    finally:
        db.close()