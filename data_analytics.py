import streamlit as st
import pandas as pd
import sqlite3
import os
import shutil
from pathlib import Path
from langchain_community.llms import Ollama
from utils.ui import custom_divider

# Import the RAG/CAG retrieval functions from rag_cag.py
try:
    from pages.rag_cag import get_retriever, get_combined_retriever
except ImportError:
    # If import fails due to circular dependency, define dummy functions
    def get_retriever(dir_name):
        return None
    def get_combined_retriever():
        return None

# LLM Setup - Note: Only initialize if needed
def get_llm():
    return Ollama(model="llama3")

# Database structure function - MOVED OUTSIDE
def get_database_structure(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view');")
    tables = cursor.fetchall()

    structure = {}
    for table in tables:
        table_name, table_type = table
        structure[table_name] = {"type": table_type, "columns": []}
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        for column in columns:
            structure[table_name]["columns"].append({"name": column[1], "type": column[2]})
    conn.close()
    return structure

# SQL generation function - MOVED OUTSIDE
def frage_ki(natürliche_frage, db_structure, modifizieren=False, use_rag=True, use_cag=True):
    # Initialize the LLM when needed
    llm = get_llm()
    
    # Load the context for the AI using our new vector-based retrieval if possible
    context = get_context_from_retriever(natürliche_frage, use_rag, use_cag)
    
    if modifizieren:
        prompt = f"""
Du bist ein SQL-Experte. Verwende den folgenden Befehl, um Änderungen an einer SQLite-Datenbank vorzunehmen. 
Beachte dabei die Struktur der Datenbank. Achte darauf, alle Spaltennamen in doppelte Anführungszeichen zu setzen. 
Gebe nur den SQL-Befehl und sonst nichts zurück.

Datenbankstruktur:
{db_structure}

Zusätzlicher Kontext aus Dokumenten:
{context}

Frage: "{natürliche_frage}"
SQL (Änderung):
"""
    else:
        prompt = f"""
Du bist ein SQL-Experte. Übersetze die folgende natürliche Sprache in einen SQL-Befehl für eine SQLite-Datenbank 
mit folgender Struktur. Achte darauf, alle Spaltennamen in doppelte Anführungszeichen zu setzen. 
Gebe nur den SQL-Befehl und sonst nichts zurück.

Datenbankstruktur:
{db_structure}

Zusätzlicher Kontext aus Dokumenten:
{context}

Frage: "{natürliche_frage}"
SQL:
"""
    response = llm(prompt)
    return response.strip()

# Load RAG and CAG context using the vector retriever
def get_context_from_retriever(query, use_rag=True, use_cag=True):
    context_text = ""
    
    # Try to use the vector retriever if available
    if use_rag or use_cag:
        try:
            # Get the appropriate retriever
            if use_rag and use_cag:
                retriever = get_combined_retriever()
            elif use_rag:
                retriever = get_retriever("rag_docs")
            elif use_cag:
                retriever = get_retriever("cag_docs")
            else:
                retriever = None
                
            # Get documents from retriever
            if retriever:
                docs = retriever.get_relevant_documents(query)
                
                # Add documents to context
                for i, doc in enumerate(docs):
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        source = doc.metadata["source"]
                        if "rag_docs" in source:
                            context_text += f"\n---\n[RAG] {Path(source).name}\n"
                        elif "cag_docs" in source:
                            context_text += f"\n---\n[CAG] {Path(source).name}\n"
                        else:
                            context_text += f"\n---\nDocument {i+1}\n"
                    else:
                        context_text += f"\n---\nDocument {i+1}\n"
                    
                    context_text += doc.page_content + "\n"
                    
                return context_text
        except Exception as e:
            st.warning(f"Error using vector retriever: {str(e)}. Falling back to direct file loading.")
    
    # Fallback: Load context directly from files
    return load_context_docs(use_rag, use_cag)

# Load RAG and CAG context directly from files (fallback method)
def load_context_docs(use_rag=True, use_cag=True):
    context_text = ""

    if use_rag:
        for file in Path("rag_docs").glob("*"):
            try:
                context_text += f"\n---\n[RAG] {file.name}\n"
                context_text += file.read_text(errors="ignore")
            except:
                context_text += f"Error reading {file.name}\n"

    if use_cag:
        for file in Path("cag_docs").glob("*"):
            try:
                context_text += f"\n---\n[CAG] {file.name}\n"
                context_text += file.read_text(errors="ignore")
            except:
                context_text += f"Error reading {file.name}\n"

    return context_text

# Main function for Streamlit page
def show_data_analytics():
    # --- Neue Rollen-Auswahl ---
    rolle = st.radio("Modus auswählen", ["SQL Abfrage", "Mapping Abgleich"], horizontal=True)
    custom_divider(rolle)

    # --- SQL Abfrage Modus ---
    if rolle == "SQL Abfrage":
        # Upload database
        st.subheader("Datenbank Upload")
        uploaded_db = st.file_uploader("SQLite Datenbank hochladen", type=["db", "sqlite", "sqlite3"])
        
        if uploaded_db is not None:
            # Save uploaded database
            db_path = "uploaded_database.db"
            with open(db_path, "wb") as f:
                f.write(uploaded_db.getbuffer())
            
            # Get database structure
            db_structure = get_database_structure(db_path)
            
            # Display database structure
            st.subheader("Datenbankstruktur")
            for table_name, table_info in db_structure.items():
                with st.expander(f"Tabelle: {table_name}"):
                    columns_df = pd.DataFrame(table_info["columns"])
                    st.dataframe(columns_df)
            
            # Natural language query input
            st.subheader("Natürliche Sprache Abfrage")
            natürliche_frage = st.text_area(
                "Geben Sie Ihre Frage in natürlicher Sprache ein:",
                placeholder="z.B. 'Zeige alle Kunden, die in den letzten 30 Tagen bestellt haben'"
            )
            
            # Options
            col1, col2, col3 = st.columns(3)
            with col1:
                use_rag = st.checkbox("RAG Kontext verwenden", value=True)
            with col2:
                use_cag = st.checkbox("CAG Kontext verwenden", value=True)
            with col3:
                modifizieren = st.checkbox("Datenbank ändern", value=False)
            
            if st.button("SQL generieren"):
                if natürliche_frage:
                    with st.spinner("Generiere SQL..."):
                        try:
                            sql_query = frage_ki(natürliche_frage, db_structure, modifizieren, use_rag, use_cag)
                            
                            st.subheader("Generierter SQL-Befehl")
                            st.code(sql_query, language="sql")
                            
                            # Execute query option
                            if st.button("SQL ausführen"):
                                try:
                                    conn = sqlite3.connect(db_path)
                                    cursor = conn.cursor()
                                    cursor.execute(sql_query)
                                    
                                    if sql_query.strip().upper().startswith("SELECT"):
                                        results = cursor.fetchall()
                                        columns = [description[0] for description in cursor.description]
                                        df = pd.DataFrame(results, columns=columns)
                                        st.subheader("Ergebnisse")
                                        st.dataframe(df)
                                    else:
                                        conn.commit()
                                        st.success(f"Befehl erfolgreich ausgeführt. {cursor.rowcount} Zeilen betroffen.")
                                    
                                    conn.close()
                                except Exception as e:
                                    st.error(f"Fehler beim Ausführen: {str(e)}")
                        except Exception as e:
                            st.error(f"Fehler beim Generieren des SQL: {str(e)}")
                else:
                    st.warning("Bitte geben Sie eine Frage ein.")
        else:
            st.info("Bitte laden Sie eine SQLite-Datenbank hoch, um fortzufahren.")
    
    # --- Mapping Abgleich Modus ---
    else:
        st.info("Mapping-Abgleich-Funktionalität wird hier implementiert...")