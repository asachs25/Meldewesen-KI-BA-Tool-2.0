# Complete rag_cag.py with deserialization fixes and error handling

import streamlit as st
import os
from pathlib import Path
import tempfile
import pandas as pd

# LangChain Imports - Updated to use langchain_ollama (latest)
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    # Fallback to old import if new package not available
    from langchain_community.llms import Ollama as OllamaLLM
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from utils.ui import custom_divider

# Get LLM only when needed to avoid initialization issues
def get_llm():
    return OllamaLLM(model="llama3")

# Initialize embeddings model
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

# Load and process documents
def process_document(file_path):
    """Load and process a document based on its file extension"""
    file_extension = Path(file_path).suffix.lower()
    
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.csv':
            loader = CSVLoader(file_path)
        elif file_extension == '.xlsx':
            loader = UnstructuredExcelLoader(file_path)
        else:  # Default to text loader for .txt, .md and other text files
            loader = TextLoader(file_path)
            
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return []

# Split documents into chunks
def split_documents(documents):
    """Split documents into chunks for better retrieval"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Build or update the vector database - FIXED with deserialization parameter
def build_vectordb(chunks, directory_name):
    """Create or update the vector database for the documents"""
    embeddings = get_embeddings()
    
    # Check if vector store already exists
    vectorstore_path = f"{directory_name}_vectorstore"
    if os.path.exists(vectorstore_path):
        try:
            # Load existing vectorstore and add new documents - FIXED
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_documents(chunks)
        except Exception as e:
            st.warning(f"Error loading existing vector store: {str(e)}. Creating new one.")
            # Create new vectorstore if loading fails
            vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        # Create new vectorstore
        vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save the updated vectorstore
    vectorstore.save_local(vectorstore_path)
    return vectorstore

# Create a retriever from vectorstore - FIXED with deserialization parameter
def get_retriever(directory_name):
    """Get a retriever for the specified directory"""
    embeddings = get_embeddings()
    vectorstore_path = f"{directory_name}_vectorstore"
    
    # Check if vector store exists
    if os.path.exists(vectorstore_path):
        try:
            # FIXED - Added allow_dangerous_deserialization=True
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        except Exception as e:
            st.error(f"Error loading vector store from {vectorstore_path}: {str(e)}")
            st.info("You may need to rebuild your vector database. Use the 'Rebuild Vector Database' button in the Settings tab.")
            return None
    return None

# Enhanced function to clear and rebuild vector stores
def clear_and_rebuild_vectorstores():
    """Clear existing vector stores and rebuild from documents"""
    import shutil
    
    # Clear existing vector stores
    for vectorstore_dir in ["rag_docs_vectorstore", "cag_docs_vectorstore"]:
        if os.path.exists(vectorstore_dir):
            shutil.rmtree(vectorstore_dir)
            st.info(f"Cleared {vectorstore_dir}")
    
    # Rebuild from existing documents
    for doc_type in ["rag_docs", "cag_docs"]:
        doc_dir = Path(doc_type)
        if doc_dir.exists():
            all_chunks = []
            for file_path in doc_dir.glob("*"):
                if file_path.is_file():
                    try:
                        documents = process_document(str(file_path))
                        if documents:
                            chunks = split_documents(documents)
                            all_chunks.extend(chunks)
                    except Exception as e:
                        st.error(f"Error processing {file_path}: {e}")
            
            if all_chunks:
                build_vectordb(all_chunks, doc_type)
                st.success(f"Rebuilt {doc_type} vectorstore with {len(all_chunks)} chunks")

# Create a combined retriever from both RAG and CAG vectorstores - Enhanced error handling
def get_combined_retriever():
    """Get a combined retriever from both RAG and CAG vectorstores"""
    try:
        rag_retriever = get_retriever("rag_docs")
        cag_retriever = get_retriever("cag_docs")
        
        if rag_retriever and cag_retriever:
            # Return a function that combines results from both retrievers
            def combined_retrieve(query):
                try:
                    rag_docs = rag_retriever.get_relevant_documents(query)
                    cag_docs = cag_retriever.get_relevant_documents(query)
                    return rag_docs + cag_docs
                except Exception as e:
                    st.error(f"Error retrieving documents: {str(e)}")
                    return []
            
            # Create a retriever-like object with the combined retrieve method
            class CombinedRetriever:
                def get_relevant_documents(self, query):
                    return combined_retrieve(query)
                    
            return CombinedRetriever()
        elif rag_retriever:
            return rag_retriever
        elif cag_retriever:
            return cag_retriever
        return None
    except Exception as e:
        st.error(f"Error creating combined retriever: {str(e)}")
        return None

# Create a QA chain
def create_qa_chain(retriever):
    """Create a question-answering chain with the retriever"""
    if not retriever:
        return None
        
    llm = get_llm()
    
    # Create memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create a conversational retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    
    return qa_chain

def show_rag_cag():
    # Create directories if they don't exist
    os.makedirs("rag_docs", exist_ok=True)
    os.makedirs("cag_docs", exist_ok=True)
    os.makedirs("rag_docs_vectorstore", exist_ok=True)
    os.makedirs("cag_docs_vectorstore", exist_ok=True)

    # --- Document Management Section ---
    st.subheader("\U0001F4CA RAG/CAG Document Manager")
    
    # Tabs for upload, view, and settings
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "📚 View Documents", "⚙️ Settings"])
    
    # --- UPLOAD DOCUMENTS TAB ---
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RAG Documents")
            st.write("Upload documents to augment AI responses with relevant context.")
            rag_files = st.file_uploader("Upload RAG documents", 
                                        type=["txt", "pdf", "md", "csv", "xlsx"], 
                                        accept_multiple_files=True, 
                                        key="rag")
            if rag_files:
                with st.spinner("Processing documents..."):
                    all_chunks = []
                    for file in rag_files:
                        # Save the file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as temp_file:
                            temp_file.write(file.read())
                            temp_path = temp_file.name
                        
                        # Save a permanent copy
                        path = os.path.join("rag_docs", file.name)
                        with open(path, "wb") as f:
                            # Need to seek to beginning since we already read the file
                            file.seek(0)
                            f.write(file.read())
                        
                        # Process the document
                        st.info(f"Processing {file.name}...")
                        documents = process_document(temp_path)
                        if documents:
                            chunks = split_documents(documents)
                            all_chunks.extend(chunks)
                            st.success(f"Processed {file.name}: {len(chunks)} chunks extracted")
                        
                        # Clean up temp file
                        os.unlink(temp_path)
                    
                    if all_chunks:
                        # Build or update vector database
                        with st.spinner("Building vector database..."):
                            build_vectordb(all_chunks, "rag_docs")
                            st.success(f"Vector database updated with {len(all_chunks)} chunks from {len(rag_files)} documents")
        
        with col2:
            st.subheader("CAG Documents")
            st.write("Upload documents to create custom AI answers.")
            cag_files = st.file_uploader("Upload CAG documents", 
                                        type=["txt", "pdf", "md", "csv", "xlsx"], 
                                        accept_multiple_files=True, 
                                        key="cag")
            if cag_files:
                with st.spinner("Processing documents..."):
                    all_chunks = []
                    for file in cag_files:
                        # Save the file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as temp_file:
                            temp_file.write(file.read())
                            temp_path = temp_file.name
                        
                        # Save a permanent copy
                        path = os.path.join("cag_docs", file.name)
                        with open(path, "wb") as f:
                            # Need to seek to beginning since we already read the file
                            file.seek(0)
                            f.write(file.read())
                        
                        # Process the document
                        st.info(f"Processing {file.name}...")
                        documents = process_document(temp_path)
                        if documents:
                            chunks = split_documents(documents)
                            all_chunks.extend(chunks)
                            st.success(f"Processed {file.name}: {len(chunks)} chunks extracted")
                        
                        # Clean up temp file
                        os.unlink(temp_path)
                    
                    if all_chunks:
                        # Build or update vector database
                        with st.spinner("Building vector database..."):
                            build_vectordb(all_chunks, "cag_docs")
                            st.success(f"Vector database updated with {len(all_chunks)} chunks from {len(cag_files)} documents")
    
    # --- VIEW DOCUMENTS TAB ---
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RAG Documents Library")
            rag_docs = list(Path("rag_docs").glob("*"))
            if rag_docs:
                for i, doc in enumerate(rag_docs):
                    col_doc, col_action = st.columns([3, 1])
                    with col_doc:
                        st.write(f"📄 {doc.name}")
                    with col_action:
                        if st.button("Delete", key=f"del_rag_{i}"):
                            # Remove the document
                            os.remove(doc)
                            # Rebuild the vector database
                            if os.path.exists("rag_docs_vectorstore"):
                                st.warning("Document deleted. You should rebuild the vector database.")
                            st.rerun()
            else:
                st.info("No RAG documents uploaded yet.")
                
            # Add a rebuild button
            if rag_docs and os.path.exists("rag_docs"):
                if st.button("Rebuild RAG Vector Database"):
                    with st.spinner("Rebuilding vector database..."):
                        all_chunks = []
                        for doc in rag_docs:
                            documents = process_document(str(doc))
                            if documents:
                                chunks = split_documents(documents)
                                all_chunks.extend(chunks)
                        
                        # Rebuild vector database
                        if all_chunks:
                            if os.path.exists("rag_docs_vectorstore"):
                                import shutil
                                shutil.rmtree("rag_docs_vectorstore")
                            build_vectordb(all_chunks, "rag_docs")
                            st.success(f"Vector database rebuilt with {len(all_chunks)} chunks")
                        else:
                            st.error("No valid documents to build vector database")
                
        with col2:
            st.subheader("CAG Documents Library")
            cag_docs = list(Path("cag_docs").glob("*"))
            if cag_docs:
                for i, doc in enumerate(cag_docs):
                    col_doc, col_action = st.columns([3, 1])
                    with col_doc:
                        st.write(f"📄 {doc.name}")
                    with col_action:
                        if st.button("Delete", key=f"del_cag_{i}"):
                            # Remove the document
                            os.remove(doc)
                            # Rebuild the vector database
                            if os.path.exists("cag_docs_vectorstore"):
                                st.warning("Document deleted. You should rebuild the vector database.")
                            st.rerun()
            else:
                st.info("No CAG documents uploaded yet.")
            
            # Add a rebuild button
            if cag_docs and os.path.exists("cag_docs"):
                if st.button("Rebuild CAG Vector Database"):
                    with st.spinner("Rebuilding vector database..."):
                        all_chunks = []
                        for doc in cag_docs:
                            documents = process_document(str(doc))
                            if documents:
                                chunks = split_documents(documents)
                                all_chunks.extend(chunks)
                        
                        # Rebuild vector database
                        if all_chunks:
                            if os.path.exists("cag_docs_vectorstore"):
                                import shutil
                                shutil.rmtree("cag_docs_vectorstore")
                            build_vectordb(all_chunks, "cag_docs")
                            st.success(f"Vector database rebuilt with {len(all_chunks)} chunks")
                        else:
                            st.error("No valid documents to build vector database")
    
    # --- SETTINGS TAB ---
    with tab3:
        st.subheader("RAG/CAG Settings")
        
        st.write("Configure your retrieval settings:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=1000, step=100,
                                  help="Size of document chunks for retrieval. Smaller chunks are more precise, larger chunks provide more context.")
            
            chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50,
                                     help="Overlap between chunks to ensure context continuity across chunk boundaries.")
        
        with col2:
            num_chunks = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=4, step=1,
                                  help="Number of document chunks to retrieve for each query.")
            
            model_options = {
                "all-MiniLM-L6-v2": "all-MiniLM-L6-v2 (Default, Fast)",
                "all-mpnet-base-v2": "all-mpnet-base-v2 (Better quality, Slower)"
            }
            embedding_model = st.selectbox("Embedding Model", options=list(model_options.keys()),
                                         format_func=lambda x: model_options[x],
                                         help="Model used to create embeddings for document chunks")
        
        if st.button("Save Settings"):
            st.session_state.rag_settings = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "num_chunks": num_chunks,
                "embedding_model": embedding_model
            }
            st.success("Settings saved! Rebuild your vector databases for settings to take effect.")
        
        # ADDED: Maintenance section for vector database issues
        st.markdown("---")
        st.subheader("🔧 Maintenance")
        st.write("Use these tools if you encounter vector database issues:")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Rebuild All Vector Databases", type="secondary"):
                with st.spinner("Rebuilding all vector databases..."):
                    clear_and_rebuild_vectorstores()
                    st.success("All vector databases have been rebuilt!")

        with col2:
            if st.button("🗑️ Clear All Vector Databases", type="secondary"):
                import shutil
                cleared = []
                for vectorstore_dir in ["rag_docs_vectorstore", "cag_docs_vectorstore"]:
                    if os.path.exists(vectorstore_dir):
                        shutil.rmtree(vectorstore_dir)
                        cleared.append(vectorstore_dir)
                if cleared:
                    st.success(f"Cleared: {', '.join(cleared)}")
                else:
                    st.info("No vector databases to clear.")

        # ADDED: System status information
        st.markdown("---")
        st.subheader("📊 System Status")
        
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            rag_status = "✅ Active" if os.path.exists("rag_docs_vectorstore") else "❌ Not Ready"
            st.metric("RAG System", rag_status)
        
        with status_col2:
            cag_status = "✅ Active" if os.path.exists("cag_docs_vectorstore") else "❌ Not Ready"
            st.metric("CAG System", cag_status)
        
        with status_col3:
            retriever = get_combined_retriever()
            retriever_status = "✅ Ready" if retriever else "❌ Not Available"
            st.metric("Retrieval System", retriever_status)
    
    # --- AI Chat Section ---
    st.markdown(custom_divider("AI Chat with RAG/CAG Context"), unsafe_allow_html=True)
    st.subheader("💬 Chat with your Documents")
    
    # Initialize chat history
    if "rag_cag_messages" not in st.session_state:
        st.session_state.rag_cag_messages = []
    
    # Initialize conversation memory
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    # Get retriever for documents
    retriever = get_combined_retriever()
    
    # Display system status
    col1, col2, col3 = st.columns(3)
    with col1:
        if os.path.exists("rag_docs_vectorstore"):
            st.success("✅ RAG System: Active")
        else:
            st.warning("⚠️ RAG System: No vector database")
    
    with col2:
        if os.path.exists("cag_docs_vectorstore"):
            st.success("✅ CAG System: Active")
        else:
            st.warning("⚠️ CAG System: No vector database")
    
    with col3:
        if retriever:
            st.success("✅ Retrieval System: Ready")
        else:
            st.warning("⚠️ Retrieval System: Not available")
    
    # Display chat messages
    for message in st.session_state.rag_cag_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.write(f"**Source {i+1}:** {source}")
    
    # Get user input
    user_query = st.chat_input("Ask something based on your documents...")
    
    # Handle user input
    if user_query:
        # Add user message to chat history
        st.session_state.rag_cag_messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Generate AI response using the retrieval-based approach
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if retriever:
                    try:
                        # Create QA chain if not already created
                        if "qa_chain" not in st.session_state:
                            st.session_state.qa_chain = create_qa_chain(retriever)
                        
                        # Run the chain to get a response
                        response = st.session_state.qa_chain({"question": user_query})
                        answer = response["answer"]
                        
                        # Extract sources for display
                        sources = []
                        if "source_documents" in response:
                            for doc in response["source_documents"]:
                                if hasattr(doc, "metadata") and "source" in doc.metadata:
                                    source = f"{doc.metadata['source']}"
                                    if source not in sources:
                                        sources.append(source)
                        
                        st.write(answer)
                        
                        # Save the response with sources
                        response_data = {
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        }
                        
                        # Show sources
                        if sources:
                            with st.expander("View Sources"):
                                for i, source in enumerate(sources):
                                    st.write(f"**Source {i+1}:** {source}")
                    
                    except Exception as e:
                        st.error(f"Error processing your query: {str(e)}")
                        st.info("Try rebuilding your vector databases in the Settings tab if this error persists.")
                        
                        response_data = {
                            "role": "assistant", 
                            "content": f"I encountered an error processing your query: {str(e)}. Please try rebuilding the vector databases."
                        }
                else:
                    # Fallback to basic LLM if no retriever is available
                    try:
                        llm = get_llm()
                        
                        # Load RAG and CAG documents as context (fallback method)
                        context = ""
                        for file in Path("rag_docs").glob("*"):
                            try:
                                context += f"\n--- RAG Document: {file.name} ---\n"
                                context += file.read_text(errors="ignore")
                            except Exception as e:
                                context += f"Error reading file {file.name}: {str(e)}\n"
                                
                        for file in Path("cag_docs").glob("*"):
                            try:
                                context += f"\n--- CAG Document: {file.name} ---\n"
                                context += file.read_text(errors="ignore")
                            except Exception as e:
                                context += f"Error reading file {file.name}: {str(e)}\n"
                        
                        # Create prompt with context
                        prompt = f"""
You are an AI assistant with access to the following document context:

{context}

Use this context to inform your answer. If the context doesn't contain relevant information, 
just answer based on your general knowledge. If you don't know, say so.

Question: {user_query}
"""
                        response = llm(prompt)
                        st.write(response)
                        
                        # Save the response without sources
                        response_data = {
                            "role": "assistant", 
                            "content": response
                        }
                    
                    except Exception as e:
                        st.error(f"Error with fallback processing: {str(e)}")
                        response_data = {
                            "role": "assistant", 
                            "content": "I'm sorry, I encountered an error processing your request. Please check the system settings and try again."
                        }
                
                # Add assistant response to chat history
                st.session_state.rag_cag_messages.append(response_data)