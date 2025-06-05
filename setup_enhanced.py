#!/usr/bin/env python3
"""
Quick Setup Script for Enhanced Business Analyst Tool
This script sets up the enhanced version with landing page and two main functions
"""

import os
import sys
import shutil

def create_directory_structure():
    """Create all necessary directories"""
    directories = [
        "rag_docs",
        "cag_docs",
        "mapping_docs",
        "rag_docs_vectorstore",
        "cag_docs_vectorstore",
        "mapping_docs_vectorstore",
        "static"
    ]
    
    print("📁 Creating directory structure...")
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"  ✓ Created {dir_name}/")

def check_files():
    """Check if all required files exist"""
    print("\n📄 Checking required files...")
    
    required_files = {
        "main.py": "FastAPI backend",
        "landing.html": "Landing page",
        "index.html": "Main application"
    }
    
    missing_files = []
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"  ✓ {filename} ({description}) - Found")
        else:
            print(f"  ✗ {filename} ({description}) - Missing!")
            missing_files.append(filename)
    
    if missing_files:
        print("\n⚠️  Missing files detected!")
        print("Please ensure you have saved:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nRefer to the implementation guide for the correct file contents.")
        return False
    
    return True

def create_demo_content():
    """Create demo content for testing"""
    print("\n📝 Creating demo content...")
    
    # Create a demo RAG document
    demo_rag = """# Meldewesen Regulatory Reporting Guidelines

## Overview
This document outlines the key requirements for regulatory reporting in the financial sector.

## Key Requirements
1. Data accuracy and completeness
2. Timely submission of reports
3. Compliance with regulatory standards
4. Proper data validation and quality checks

## Common Issues
- Data format inconsistencies between systems
- Missing mandatory fields
- Incorrect mapping rules
- Synchronization delays between DWH and Abacus360
"""
    
    with open("rag_docs/demo_guidelines.md", "w", encoding="utf-8") as f:
        f.write(demo_rag)
    print("  ✓ Created demo RAG document")
    
    # Create a demo mapping rules file
    demo_mapping = """# Mapping Rules - Customer Data

## Field Mappings
| Source Field (DWH) | Target Field (Abacus360) | Transformation |
|-------------------|-------------------------|----------------|
| customer_id       | CUST_ID                | Uppercase      |
| customer_name     | CUST_NAME              | Trim spaces    |
| registration_date | REG_DATE               | Format: YYYY-MM-DD |

## Validation Rules
- customer_id must be unique
- customer_name cannot be empty
- registration_date must be valid date
"""
    
    with open("mapping_docs/demo_mapping_rules.md", "w", encoding="utf-8") as f:
        f.write(demo_mapping)
    print("  ✓ Created demo mapping rules")

def display_instructions():
    """Display setup completion instructions"""
    print("\n" + "="*60)
    print("🎉 Setup Complete!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("\n1. Install dependencies (if not already done):")
    print("   pip install fastapi uvicorn python-multipart langchain langchain-community \\")
    print("   faiss-cpu sentence-transformers pypdf unstructured openpyxl pandas numpy aiofiles httpx")
    
    print("\n2. Start Ollama (in a separate terminal):")
    print("   ollama serve")
    
    print("\n3. Make sure llama3 is installed:")
    print("   ollama pull llama3")
    
    print("\n4. Start the application:")
    print("   python main.py")
    
    print("\n5. Access the application:")
    print("   🏠 Landing Page: http://localhost:8000")
    print("   📊 Dashboard: http://localhost:8000/app")
    
    print("\n✨ Features:")
    print("   - Knowledge Assistant: Upload documents and ask questions")
    print("   - Business Mapping Analyzer: Analyze data mappings and errors")
    
    print("\n📚 Demo Content:")
    print("   - Check rag_docs/ for demo guidelines")
    print("   - Check mapping_docs/ for demo mapping rules")
    
    print("\n" + "="*60)

def main():
    print("="*60)
    print("Enhanced Business Analyst Tool - Setup Script")
    print("="*60)
    
    # Create directories
    create_directory_structure()
    
    # Check files
    if not check_files():
        print("\n❌ Setup incomplete due to missing files.")
        print("Please add the missing files and run this script again.")
        sys.exit(1)
    
    # Create demo content
    create_demo_content()
    
    # Display instructions
    display_instructions()
    
    # Offer to install dependencies
    print("\nWould you like to install Python dependencies now? (y/n): ", end="")
    response = input().lower()
    
    if response == 'y':
        print("\n📦 Installing dependencies...")
        import subprocess
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "fastapi", "uvicorn", "python-multipart", "langchain", "langchain-community",
                "faiss-cpu", "sentence-transformers", "pypdf", "unstructured", "openpyxl",
                "pandas", "numpy", "aiofiles", "httpx"
            ], check=True)
            print("\n✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("\n❌ Error installing dependencies. Please install manually.")
    
    print("\n✅ Setup script completed!")
    print("Run 'python main.py' to start the application.")

if __name__ == "__main__":
    main()