#!/usr/bin/env python3
"""
Automated Installation and Setup Script for Business Analysis Tool
Run this script to install all dependencies and start the application
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """Run a command and display status"""
    print(f"\n📦 {description}...")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} - Success!")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {description} - Failed!")
        return False

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Your version: {sys.version}")
        return False
    print(f"✅ Python version {version.major}.{version.minor} - OK!")
    return True

def main():
    print("=" * 60)
    print("Business Analysis Tool - Automated Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install packages
    packages = [
        "fastapi",
        "uvicorn",
        "python-multipart",
        "langchain",
        "langchain-community",
        "faiss-cpu",
        "sentence-transformers",
        "pypdf",
        "unstructured",
        "openpyxl",
        "pandas",
        "numpy",
        "aiofiles",
        "httpx"
    ]
    
    print("\n📦 Installing required packages...")
    package_string = " ".join(packages)
    
    if run_command(f"{sys.executable} -m pip install {package_string}", 
                   "Installing all packages"):
        print("\n✅ All packages installed successfully!")
    else:
        print("\n⚠️ Some packages failed to install. Try running:")
        print(f"pip install {package_string}")
    
    # Create required directories
    print("\n📁 Creating required directories...")
    directories = ["rag_docs", "cag_docs", "static", "rag_docs_vectorstore", "cag_docs_vectorstore"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}/")
    
    # Check for required files
    print("\n📄 Checking required files...")
    required_files = {
        "main.py": "FastAPI backend",
        "index.html": "Web frontend"
    }
    
    all_files_present = True
    for file, description in required_files.items():
        if os.path.exists(file):
            print(f"✅ {file} ({description}) - Found!")
        else:
            print(f"❌ {file} ({description}) - Missing!")
            all_files_present = False
    
    if not all_files_present:
        print("\n⚠️ Some required files are missing!")
        print("Make sure you have:")
        print("1. main.py - The FastAPI backend")
        print("2. index.html - The web frontend")
        sys.exit(1)
    
    # Check Ollama
    print("\n🤖 Checking Ollama installation...")
    try:
        subprocess.run("ollama --version", shell=True, check=True, 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Ollama is installed!")
        
        # Check if llama3 model exists
        print("🔍 Checking for llama3 model...")
        result = subprocess.run("ollama list", shell=True, capture_output=True, text=True)
        if "llama3" in result.stdout:
            print("✅ llama3 model is available!")
        else:
            print("⚠️ llama3 model not found. Please run: ollama pull llama3")
    except:
        print("❌ Ollama is not installed!")
        print("Please install from: https://ollama.com/")
        print("Then run: ollama pull llama3")
    
    # Final instructions
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print("\nTo start the application:")
    print("1. Make sure Ollama is running: ollama serve")
    print("2. In another terminal, run: python main.py")
    print("3. Open your browser to: http://localhost:8000")
    print("\nThe app will automatically match your system's dark/light mode!")
    
    # Offer to start the server
    print("\n" + "=" * 60)
    response = input("Would you like to start the server now? (y/n): ").lower()
    if response == 'y':
        print("\n🚀 Starting server...")
        print("Press Ctrl+C to stop the server")
        time.sleep(2)
        try:
            subprocess.run(f"{sys.executable} main.py", shell=True)
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped!")

if __name__ == "__main__":
    main()