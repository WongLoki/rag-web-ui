import json
import base64
from typing import List, AsyncGenerator
from sqlalchemy.orm import Session
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from app.core.config import settings, ModelConfig
from app.models.chat import Message
from app.models.knowledge import KnowledgeBase, Document
from langchain.globals import set_verbose, set_debug
from app.services.vector_store import VectorStoreFactory
import logging
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

set_verbose(True)
set_debug(True)

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
        
        # Create user message
        user_message = Message(
            content=query,
            role="user",
            chat_id=chat_id
        )
        db.add(user_message)
        db.commit()
        logger.info("User message created")
        
        # Create bot message placeholder
        bot_message = Message(
            content="",
            role="assistant",
            chat_id=chat_id
        )
        db.add(bot_message)
        db.commit()
        
        # Get knowledge bases and their documents
        knowledge_bases = (
            db.query(KnowledgeBase)
            .filter(KnowledgeBase.id.in_(knowledge_base_ids))
            .all()
        )
        
        # Initialize embeddings
        logger.info("Initializing embeddings with Ollama")
        embeddings = OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,  # 使用 base_url 而不是 api_base
            model=settings.OLLAMA_EMBEDDING_MODEL
        )
        
        # Create vector stores
        vector_stores = []
        for kb in knowledge_bases:
            logger.info(f"Processing knowledge base: {kb.id}")
            documents = db.query(Document).filter(Document.knowledge_base_id == kb.id).all()
            logger.info(f"Found {len(documents)} documents for KB {kb.id}")
            
            if documents:
                vector_store = VectorStoreFactory.create(
                    store_type=settings.VECTOR_STORE_TYPE,
                    collection_name=f"kb_{kb.id}",
                    embedding_function=embeddings,
                )
                logger.info(f"Vector store created for KB {kb.id}")
                vector_stores.append(vector_store)
        
        if not vector_stores:
            error_msg = "I don't have any knowledge base to help answer your question."
            yield f'0:"{error_msg}"\n'
            yield 'd:{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0}}\n'
            bot_message.content = error_msg
            db.commit()
            return
        
        # Use first vector store for now
        retriever = vector_stores[0].as_retriever()
        
        # Initialize LLM
        logger.info(f"Initializing LLM with model: {model}")
        model_config = next(
            (m for m in settings.MODEL_CONFIGS if m.id == model),
            settings.MODEL_CONFIGS[0]
        )

        if model_config.type == "openai":
            llm = ChatOpenAI(
                temperature=0,
                streaming=True,
                model=model,
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_API_BASE
            )
        else:  # ollama
            llm = ChatOpenAI(
                temperature=0,
                streaming=True,
                model=model,
                openai_api_key="EMPTY",
                openai_api_base=settings.OLLAMA_BASE_URL
            )

        # Create contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        # Create history aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm, 
            retriever,
            contextualize_q_prompt
        )

        # Create QA prompt
        qa_system_prompt = (
            "You are given a user question, and please write clean, concise and accurate answer to the question. "
            "You will be given a set of related contexts to the question, which are numbered sequentially starting from 1. "
            "Each context has an implicit reference number based on its position in the array (first context is 1, second is 2, etc.). "
            "Please use these contexts and cite them using the format [citation:x] at the end of each sentence where applicable. "
            "Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. "
            "Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. "
            "Say 'information is missing on' followed by the related topic, if the given context do not provide sufficient information. "
            "If a sentence draws from multiple contexts, please list all applicable citations, like [citation:1][citation:2]. "
            "Other than code and specific names and citations, your answer must be written in the same language as the question. "
            "Be concise.\n\nContext: {context}\n\n"
            "Remember: Cite contexts by their position number (1 for first context, 2 for second, etc.) and don't blindly "
            "repeat the contexts verbatim."
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # 修改 create_stuff_documents_chain 来自定义 context 格式
        document_prompt = PromptTemplate.from_template("\n\n- {page_content}\n\n")

        # Create QA chain
        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt,
            document_variable_name="context",
            document_prompt=document_prompt
        )

        # Create retrieval chain
        retrieval_chain = (
            {
                "context": retriever,
                "input": RunnablePassthrough(),
                "chat_history": lambda x: x.get("chat_history", [])
            }
            | question_answer_chain
        )

        # 在检索链创建之前添加日志
        logger.info("Creating retrieval chain with retriever: %s", retriever)
        logger.info("Question answer chain: %s", question_answer_chain)

        # Generate response
        chat_history = []
        for message in messages["messages"]:
            if message["role"] == "user":
                chat_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                # if include __LLM_RESPONSE__, only use the last part
                if "__LLM_RESPONSE__" in message["content"]:
                    message["content"] = message["content"].split("__LLM_RESPONSE__")[-1]
                chat_history.append(AIMessage(content=message["content"]))

        full_response = ""
        # 在 astream 之前添加日志
        logger.info("Streaming response with input: %s", {
            "input": query,
            "chat_history": chat_history
        })
        async for chunk in retrieval_chain.astream({
            "input": query,
            "chat_history": chat_history
        }):
            # 检查 chunk 的类型
            if isinstance(chunk, str):
                # 如果是字符串，直接输出
                escaped_chunk = chunk.replace('"', '\\"').replace('\n', '\\n')
                yield f'0:"{escaped_chunk}"\n'
                full_response += chunk
            elif isinstance(chunk, dict):
                # 如果是字典，按原来的逻辑处理
                if "context" in chunk:
                    serializable_context = []
                    for context in chunk["context"]:
                        serializable_doc = {
                            "page_content": context.page_content.replace('"', '\\"'),
                            "metadata": context.metadata,
                        }
                        serializable_context.append(serializable_doc)
                    
                    escaped_context = json.dumps({
                        "context": serializable_context
                    })
                    base64_context = base64.b64encode(escaped_context.encode()).decode()
                    separator = "__LLM_RESPONSE__"
                    
                    yield f'0:"{base64_context}{separator}"\n'
                    full_response += base64_context + separator

        # Update bot message content
        bot_message.content = full_response
        db.commit()
            
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        error_message = f"Error generating response: {str(e)}"
        yield '3:{text}\n'.format(text=error_message)
        
        # Update bot message with error
        if 'bot_message' in locals():
            bot_message.content = error_message
            db.commit()
    finally:
        db.close()