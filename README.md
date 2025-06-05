# Meldewesen-KI-BA-Tool-2.0
# 🏢 Business Intelligence Platform - Setup Guide

## 📋 Overview

This is a professional AI-powered Business Intelligence Platform designed for regulatory mapping error analysis. The tool helps business analysts identify and resolve data mapping issues using advanced language models with contextual document analysis.

### ✨ Key Features
- **AI-Powered Analysis**: Multiple LLM models (Llama 3, Mistral, Gemma 2, CodeLlama)
- **Document Context**: RAG (Regulatory Guidelines) & CAG (Company Procedures) 
- **SQL Analytics**: Natural language to SQL conversion with database analysis
- **Corporate Branding**: Fully customizable enterprise interface
- **Dark/Light Mode**: Professional UI with theme switching
- **Audit Logging**: Enterprise-grade compliance tracking

---

## 🚀 Quick Start (Automated Installation)

### Option 1: One-Click Setup Script
```bash
# Download and run the automated installer
python install_and_run.py
```

### Option 2: Corporate Setup Script (Recommended for Enterprise)
```bash
# Make the setup script executable
chmod +x setup_corporate_env.sh

# Run the corporate setup (includes branding configuration)
./setup_corporate_env.sh
```

---

## 📋 Manual Installation Guide

### Step 1: System Requirements

**Minimum Requirements:**
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **OS**: Windows 10/11, macOS 10.15+, or Linux

**Check Python Version:**
```bash
python3 --version
# Should show Python 3.8.x or higher
```

### Step 2: Install Ollama (AI Model Server)

Ollama is required to run the AI models locally.

**🐧 Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**🍎 macOS:**
```bash
# Using Homebrew
brew install ollama

# Or download from https://ollama.com
```

**🪟 Windows:**
```bash
# Download installer from https://ollama.com
# Run the installer and follow instructions
```

**Start Ollama Service:**
```bash
ollama serve
```

### Step 3: Download AI Models

Download the required AI models (this may take 10-30 minutes):

```bash
# Required models
ollama pull llama3      # Primary analysis model (4.7GB)
ollama pull mistral     # Fast error detection (4.1GB)

# Optional models
ollama pull gemma2      # Structured data analysis (5.4GB)
ollama pull codellama   # Technical transformation analysis (3.8GB)
ollama pull llama3.1    # Advanced reasoning (8.0GB)
```

**Verify models are installed:**
```bash
ollama list
```

### Step 4: Install Python Dependencies

**Create Virtual Environment (Recommended):**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

**Install Required Packages:**
```bash
# Install all dependencies
pip install fastapi uvicorn python-multipart python-dotenv
pip install langchain langchain-community ollama
pip install faiss-cpu sentence-transformers
pip install pypdf unstructured openpyxl python-docx
pip install pandas numpy aiofiles httpx requests pydantic

# Or install from requirements.txt (if available)
pip install -r requirements.txt
```

### Step 5: Project Setup

**Create Directory Structure:**
```bash
# Create required directories
mkdir -p rag_docs cag_docs mapping_docs static
mkdir -p rag_docs_vectorstore cag_docs_vectorstore
mkdir -p logs backups
```

**Create Environment Configuration (.env file):**
```bash
# Create .env file with corporate settings
cat > .env << EOF
# Corporate Branding Configuration
CORPORATE_APP_NAME="Professional Business Intelligence Platform"
CORPORATE_ORGANIZATION="Your Organization Name"
CORPORATE_LOGO_URL="/static/your-logo.svg"
CORPORATE_PRIMARY_COLOR="#86bc25"
CORPORATE_SECONDARY_COLOR="#0d2818"
CORPORATE_SUPPORT_EMAIL="support@yourcompany.com"

# Technical Configuration
OLLAMA_HOST=http://localhost:11434
DEFAULT_MODEL=llama3
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=True

# Document Processing
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CONTEXT_DOCS=6
MAX_FILE_SIZE=10485760
EOF
```

---

## 🖥️ Running the Application

### Start the Application

**Method 1: Direct Python**
```bash
# Ensure Ollama is running in another terminal
ollama serve

# Start the application
python main.py
```

**Method 2: Using the Startup Script**
```bash
# If you used the corporate setup
./start_corporate_analyzer.sh
```

### Access the Application

Once started, you can access:

- **🏠 Landing Page**: http://localhost:8000
- **📊 Main Analyzer**: http://localhost:8000/analyzer  
- **📚 API Documentation**: http://localhost:8000/docs
- **🔧 Health Check**: http://localhost:8000/api/health

---

## 📁 Project Structure

```
business-intelligence-platform/
├── 🌐 Frontend
│   ├── landing.html              # Corporate landing page
│   ├── index.html                # Main analyzer interface
│   └── static/                   # Assets and logos
├── 🔧 Backend
│   ├── main.py                   # FastAPI server
│   ├── .env                      # Configuration file
│   └── pages/
│       ├── rag_cag.py           # Document processing
│       └── data_analytics.py     # SQL analytics
├── 📚 Document Storage
│   ├── rag_docs/                # Regulatory guidelines
│   ├── cag_docs/                # Company procedures  
│   ├── mapping_docs/            # Technical mappings
│   ├── rag_docs_vectorstore/    # AI vector database
│   └── cag_docs_vectorstore/    # AI vector database
├── 🛠️ Operations
│   ├── logs/                    # Application logs
│   ├── backups/                 # Backup storage
│   └── requirements.txt         # Python dependencies
└── 📖 Setup Scripts
    ├── setup_corporate_env.sh    # Corporate setup
    ├── install_and_run.py        # Automated installer
    └── setup_enhanced.py         # Python setup script
```

---

## 🎯 How to Use the Platform

### 1. Upload Documents
- **RAG Documents**: Upload regulatory guidelines, compliance documents
- **CAG Documents**: Upload company procedures, internal guidelines
- **Supported formats**: PDF, TXT, MD, CSV, XLSX, DOCX

### 2. Select AI Model
- **Llama 3**: Balanced analysis for general use
- **Mistral**: Fast error detection and analysis
- **Gemma 2**: Structured data validation specialist
- **CodeLlama**: Technical transformation analysis

### 3. Configure Context
- **RAG + CAG**: Full contextual analysis (recommended)
- **RAG Only**: Regulatory focus
- **CAG Only**: Company procedure focus
- **No Context**: Direct AI analysis

### 4. Run Analysis
- Describe your mapping problem in natural language
- Get detailed analysis with actionable recommendations
- Export results for documentation

### 5. SQL Analytics (Bonus Feature)
- Upload SQLite databases
- Ask questions in natural language
- Get generated SQL queries and results

---

## 🎨 Corporate Branding

### Customize Your Platform

**Update Logo:**
```bash
# Replace with your corporate logo
cp your-company-logo.svg static/your-logo.svg
```

**Update Colors:**
Edit the `.env` file:
```bash
CORPORATE_PRIMARY_COLOR="#YOUR_HEX_COLOR"
CORPORATE_SECONDARY_COLOR="#YOUR_HEX_COLOR"
```

**Update Organization Info:**
```bash
CORPORATE_APP_NAME="Your Platform Name"
CORPORATE_ORGANIZATION="Your Company Name"
CORPORATE_SUPPORT_EMAIL="support@yourcompany.com"
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

**❌ "Ollama not found" Error**
```bash
# Check if Ollama is installed
ollama --version

# If not installed, follow Step 2 above
# If installed but not running:
ollama serve
```

**❌ "Model not available" Error**
```bash
# Check available models
ollama list

# Download missing models
ollama pull llama3
ollama pull mistral
```

**❌ "Port already in use" Error**
```bash
# Check what's using port 8000
lsof -i :8000

# Kill the process or change port in .env
APP_PORT=8001
```

**❌ Python Import Errors**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check virtual environment is activated
which python  # Should show venv path
```

**❌ Vector Database Issues**
```bash
# Clear and rebuild vector databases
rm -rf rag_docs_vectorstore cag_docs_vectorstore

# Restart the application to rebuild
python main.py
```

### Log Files

Check these files for detailed error information:
- `analyzer.log` - Application logs
- `logs/startup.log` - Startup logs  
- `logs/ollama.log` - AI model logs

---

## 📊 System Monitoring

### Health Check
```bash
# Check system status
curl http://localhost:8000/api/health

# Check corporate configuration
curl http://localhost:8000/api/corporate/config
```

### Performance Optimization

**For Better Performance:**
- Use SSD storage for faster document processing
- Allocate more RAM to improve AI model performance
- Use GPU-enabled Ollama for faster inference (if available)

**Memory Usage:**
- Each AI model uses 4-8GB RAM
- Vector databases use 100-500MB per 1000 documents
- Keep 2-4GB free for document processing

---

## 🔒 Security & Compliance

### Enterprise Considerations

**Audit Logging:**
- All analysis activities are logged
- User actions are tracked for compliance
- Logs available at `/api/audit/logs`

**Data Privacy:**
- All processing happens locally
- No data sent to external APIs
- Documents stored locally only

**Network Security:**
- Application runs on localhost by default
- Configure firewall rules for network access
- Use HTTPS in production environments

---

## 🆘 Support & Resources

### Getting Help

**Documentation:**
- API Documentation: http://localhost:8000/docs
- Interactive API testing available in browser

**Common Commands:**
```bash
# Check system status
python -c "import fastapi, ollama, langchain; print('✅ All packages installed')"

# Test Ollama connection
curl http://localhost:11434/api/version

# Restart everything
pkill -f "ollama serve"
pkill -f "main.py"
ollama serve &
python main.py
```

**Community Resources:**
- [Ollama Documentation](https://ollama.com/docs)
- [LangChain Documentation](https://docs.langchain.com)
- [FastAPI Documentation](https://fastapi.tiangolo.com)

---

## 📈 Next Steps

### Advanced Configuration

1. **Database Integration**: Connect to PostgreSQL/MySQL for production data
2. **SSO Integration**: Add corporate authentication
3. **Monitoring**: Set up Prometheus/Grafana dashboards
4. **Scaling**: Deploy with Docker/Kubernetes for multiple users

### Feature Extensions

1. **Additional Models**: Add specialized models for your industry
2. **Custom Templates**: Create analysis templates for common scenarios  
3. **Reporting**: Build automated reporting dashboards
4. **Integration**: Connect to existing BI tools and databases

---

## ✅ Quick Verification Checklist

Before using the platform, verify:

- [ ] Python 3.8+ installed and working
- [ ] Ollama service running (`ollama serve`)
- [ ] At least one AI model downloaded (`ollama list`)
- [ ] All Python dependencies installed (`pip list`)
- [ ] Application starts without errors (`python main.py`)
- [ ] Web interface accessible (http://localhost:8000)
- [ ] Document upload works (test with sample file)
- [ ] AI analysis returns results (test with simple query)

---

**🎉 Congratulations! Your Business Intelligence Platform is ready for professional regulatory mapping analysis.**

For technical support or customization requests, contact your system administrator or refer to the troubleshooting section above.
