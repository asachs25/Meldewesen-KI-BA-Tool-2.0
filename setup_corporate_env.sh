#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Corporate branding variables - CUSTOMIZE THESE
CORPORATE_NAME="Your Organization"
CORPORATE_APP_NAME="Professional Business Intelligence Platform"
CORPORATE_DOMAIN="yourcompany.com"
CORPORATE_LOGO_URL="/static/your-logo.svg"
CORPORATE_PRIMARY_COLOR="#86bc25"
CORPORATE_SECONDARY_COLOR="#0d2818"

print_corporate_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}   ${CORPORATE_APP_NAME}${NC}"
    echo -e "${BLUE}   ${CORPORATE_NAME} - Enterprise Setup${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
}

print_section() {
    echo -e "${CYAN}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

collect_corporate_info() {
    print_section "Corporate Branding Configuration"
    echo ""
    
    read -p "Organization Name [$CORPORATE_NAME]: " input_org
    CORPORATE_NAME=${input_org:-$CORPORATE_NAME}
    
    read -p "Application Name [$CORPORATE_APP_NAME]: " input_app
    CORPORATE_APP_NAME=${input_app:-$CORPORATE_APP_NAME}
    
    read -p "Corporate Domain [$CORPORATE_DOMAIN]: " input_domain
    CORPORATE_DOMAIN=${input_domain:-$CORPORATE_DOMAIN}
    
    read -p "Primary Brand Color [$CORPORATE_PRIMARY_COLOR]: " input_primary
    CORPORATE_PRIMARY_COLOR=${input_primary:-$CORPORATE_PRIMARY_COLOR}
    
    read -p "Secondary Brand Color [$CORPORATE_SECONDARY_COLOR]: " input_secondary
    CORPORATE_SECONDARY_COLOR=${input_secondary:-$CORPORATE_SECONDARY_COLOR}
    
    echo ""
    print_success "Corporate information collected"
}

check_python() {
    print_section "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_success "Python $PYTHON_VERSION found"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Python version meets requirements (>= 3.8)"
        else
            print_error "Python version $PYTHON_VERSION is too old. Minimum required: 3.8"
            exit 1
        fi
    else
        print_error "Python3 not found. Please install Python 3.8+ first."
        exit 1
    fi
}

create_corporate_venv() {
    print_section "Setting up corporate Python environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Corporate virtual environment created"
    else
        print_info "Virtual environment already exists"
    fi
    
    source venv/bin/activate
    print_success "Corporate environment activated"
    
    pip install --upgrade pip
    print_success "pip upgraded"
}

install_enterprise_deps() {
    print_section "Installing enterprise Python dependencies..."
    
    # Core enterprise packages
    echo "Installing core enterprise packages..."
    pip install fastapi uvicorn python-multipart python-dotenv
    
    # AI/ML enterprise packages
    echo "Installing AI/ML packages..."
    pip install langchain langchain-community ollama
    
    # Enterprise vector database
    echo "Installing vector database packages..."
    pip install faiss-cpu sentence-transformers
    
    # Enterprise document processing
    echo "Installing document processing packages..."
    pip install pypdf unstructured openpyxl python-docx
    
    # Enterprise data processing
    echo "Installing data processing packages..."
    pip install pandas numpy
    
    # Enterprise utilities
    echo "Installing enterprise utilities..."
    pip install aiofiles httpx requests pydantic
    
    print_success "All enterprise dependencies installed"
}

check_ollama() {
    print_section "Checking Ollama installation..."
    
    if command -v ollama &> /dev/null; then
        OLLAMA_VERSION=$(ollama --version | head -n1)
        print_success "Ollama found: $OLLAMA_VERSION"
        return 0
    else
        print_warning "Ollama not found"
        return 1
    fi
}

install_ollama() {
    print_section "Installing Ollama for enterprise deployment..."
    
    OS="$(uname -s)"
    case "${OS}" in
        Linux*)
            print_info "Installing Ollama on Linux (Enterprise)..."
            curl -fsSL https://ollama.com/install.sh | sh
            ;;
        Darwin*)
            print_info "Installing Ollama on macOS (Enterprise)..."
            if command -v brew &> /dev/null; then
                brew install ollama
            else
                print_error "Homebrew required for macOS installation"
                print_info "Please install Ollama manually from https://ollama.com"
                exit 1
            fi
            ;;
        *)
            print_error "Please install Ollama manually from https://ollama.com"
            exit 1
            ;;
    esac
    
    print_success "Ollama installation completed"
}

start_ollama() {
    print_section "Starting Ollama enterprise service..."
    
    if pgrep -f "ollama serve" > /dev/null; then
        print_info "Ollama service already running"
    else
        print_info "Starting Ollama service..."
        nohup ollama serve > ollama.log 2>&1 &
        sleep 3
        
        if pgrep -f "ollama serve" > /dev/null; then
            print_success "Ollama service started successfully"
        else
            print_error "Failed to start Ollama service"
            exit 1
        fi
    fi
}

pull_enterprise_models() {
    print_section "Downloading enterprise-grade models..."
    
    # Required enterprise models
    REQUIRED_MODELS=("llama3" "mistral")
    
    # Optional enterprise models
    OPTIONAL_MODELS=("llama3.1" "gemma2" "codellama")
    
    for model in "${REQUIRED_MODELS[@]}"; do
        print_info "Downloading $model (required for enterprise use)..."
        if ollama pull "$model"; then
            print_success "$model downloaded successfully"
        else
            print_error "Failed to download required model: $model"
            exit 1
        fi
    done
    
    echo ""
    print_info "Optional enterprise models available:"
    for model in "${OPTIONAL_MODELS[@]}"; do
        echo "  - $model"
    done
    
    read -p "Download optional enterprise models? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for model in "${OPTIONAL_MODELS[@]}"; do
            print_info "Downloading $model..."
            if ollama pull "$model"; then
                print_success "$model downloaded successfully"
            else
                print_warning "Failed to download $model (skipping)"
            fi
        done
    fi
}

create_corporate_directories() {
    print_section "Creating corporate directory structure..."
    
    directories=(
        "rag_docs" 
        "cag_docs" 
        "mapping_docs" 
        "static"
        "static/brand-assets"
        "rag_docs_vectorstore" 
        "cag_docs_vectorstore"
        "logs"
        "backups"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_success "Created $dir/"
    done
}

create_corporate_config() {
    print_section "Creating corporate configuration..."
    
    # Create corporate environment configuration
    cat > .env << EOF
# Corporate Branding Configuration
CORPORATE_APP_NAME="$CORPORATE_APP_NAME"
CORPORATE_ORGANIZATION="$CORPORATE_NAME"
CORPORATE_LOGO_URL="$CORPORATE_LOGO_URL"
CORPORATE_PRIMARY_COLOR="$CORPORATE_PRIMARY_COLOR"
CORPORATE_SECONDARY_COLOR="$CORPORATE_SECONDARY_COLOR"
CORPORATE_SUPPORT_EMAIL="support@$CORPORATE_DOMAIN"
CORPORATE_DOCS_URL="/docs"

# Enterprise Technical Configuration
OLLAMA_HOST=http://localhost:11434
DEFAULT_MODEL=llama3
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=True

# Enterprise Document Processing
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CONTEXT_DOCS=6
MAX_FILE_SIZE=10485760

# Enterprise Paths
RAG_DOCS_PATH=./rag_docs
CAG_DOCS_PATH=./cag_docs
MAPPING_DOCS_PATH=./mapping_docs
LOG_PATH=./logs

# Enterprise Security (Configure for production)
# JWT_SECRET_KEY=your-secret-key
# DATABASE_URL=postgresql://user:pass@localhost/dbname
# REDIS_URL=redis://localhost:6379
EOF
    
    print_success "Corporate configuration created (.env)"
    
    # Create corporate requirements
    cat > requirements.txt << EOF
# Enterprise Core Framework
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
python-dotenv>=1.0.0

# Enterprise AI/ML Framework
langchain>=0.1.0
langchain-community>=0.0.12
ollama>=0.1.8

# Enterprise Vector Database
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2

# Enterprise Document Processing
pypdf>=3.17.0
unstructured>=0.10.0
openpyxl>=3.1.0
python-docx>=0.8.11

# Enterprise Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Enterprise HTTP & Utilities
aiofiles>=23.2.0
httpx>=0.25.0
requests>=2.31.0
pydantic>=2.0.0

# Enterprise Security & Monitoring
# psycopg2>=2.9.0  # Uncomment for PostgreSQL
# redis>=4.5.0     # Uncomment for Redis
# prometheus-client>=0.17.0  # Uncomment for monitoring
EOF
    
    print_success "Enterprise requirements created (requirements.txt)"
}

create_corporate_demo_data() {
    print_section "Creating corporate demo content..."
    
    # Corporate RAG demo document
    cat > rag_docs/corporate_regulatory_guidelines.md << EOF
# $CORPORATE_NAME - Regulatory Reporting Guidelines

## Enterprise Overview
This document outlines the regulatory reporting requirements and data governance standards for $CORPORATE_NAME's enterprise data platform.

## Enterprise Data Quality Standards
1. **Data Completeness**: All mandatory regulatory fields must be 100% populated
2. **Data Accuracy**: Enterprise data must maintain 99.9% accuracy standards
3. **Data Timeliness**: Regulatory reports must be submitted within prescribed deadlines
4. **Data Consistency**: Enterprise-wide consistency across all reporting periods

## Enterprise Mapping Requirements
- Field mappings must comply with $CORPORATE_NAME data governance policies
- All transformations require approval through enterprise change management
- Data lineage must be documented for regulatory audit purposes
- Enterprise validation rules must be implemented at each processing stage

## $CORPORATE_NAME Compliance Framework
- Regulatory compliance is monitored by enterprise risk management
- Data quality metrics are tracked through enterprise dashboards
- All mapping errors require immediate escalation to enterprise data office
- Regular compliance audits are conducted by internal enterprise teams
EOF
    
    # Corporate CAG demo document
    cat > cag_docs/corporate_mapping_procedures.md << EOF
# $CORPORATE_NAME - Enterprise Mapping Procedures

## Corporate Data Transformation Standards
All data transformations within $CORPORATE_NAME must follow these enterprise procedures:

### Customer Data Processing
- Source: Enterprise DWH.customers.id → Target: $CORPORATE_NAME.REPORTING.customer_reference
- Enterprise Transformation: Apply corporate prefix "${CORPORATE_NAME^^}_CUST_" to all IDs
- Corporate Validation: Must comply with $CORPORATE_NAME naming conventions

### Enterprise Transaction Processing
- Source: Enterprise DWH.transactions.amount → Target: $CORPORATE_NAME.REPORTING.transaction_value
- Enterprise Rule: Convert all amounts to corporate base currency (EUR)
- $CORPORATE_NAME Validation: Amounts exceeding €50,000 require additional enterprise approval

### Corporate Date Standardization
- All enterprise timestamps must be in UTC format
- $CORPORATE_NAME Standard: YYYY-MM-DD HH:MM:SS UTC
- Enterprise Source Timezone: Europe/Frankfurt (Corporate HQ)

## $CORPORATE_NAME Governance
- All mapping changes require enterprise architecture review
- Data stewards must approve all transformation logic
- Corporate audit trail must be maintained for all modifications
EOF
    
    print_success "Corporate demo content created"
}

create_corporate_startup_script() {
    print_section "Creating corporate startup script..."
    
    cat > start_corporate_analyzer.sh << 'EOF'
#!/bin/bash

echo "🏢 Starting Corporate Business Intelligence Platform..."
echo "🔹 Initializing enterprise environment..."

# Load corporate configuration
if [ -f .env ]; then
    source .env
    echo "✅ Corporate configuration loaded"
else
    echo "❌ Corporate configuration not found (.env file missing)"
    exit 1
fi

# Activate corporate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Corporate virtual environment activated"
else
    echo "❌ Corporate virtual environment not found"
    exit 1
fi

# Verify corporate Ollama service
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "🤖 Starting corporate Ollama service..."
    nohup ollama serve > logs/ollama.log 2>&1 &
    sleep 3
fi

# Verify required enterprise models
echo "🔍 Verifying enterprise models..."
if ! ollama list | grep -q "llama3"; then
    echo "📥 Downloading required enterprise model (llama3)..."
    ollama pull llama3
fi

# Create logs directory if not exists
mkdir -p logs

# Start corporate application
echo "🌐 Starting corporate web application..."
echo "📱 Corporate Platform: http://localhost:${APP_PORT:-8000}"
echo "📊 Enterprise API: http://localhost:${APP_PORT:-8000}/docs"
echo "🏢 Organization: ${CORPORATE_ORGANIZATION}"
echo ""
echo "Press Ctrl+C to stop the corporate platform"

# Log startup event
echo "$(date): Corporate platform started" >> logs/startup.log

python main.py
EOF
    
    chmod +x start_corporate_analyzer.sh
    print_success "Corporate startup script created (start_corporate_analyzer.sh)"
}

create_logo_placeholder() {
    print_section "Creating logo placeholder..."
    
    # Create SVG logo placeholder
    cat > static/your-logo.svg << EOF
<svg width="200" height="60" xmlns="http://www.w3.org/2000/svg">
  <rect width="200" height="60" fill="$CORPORATE_PRIMARY_COLOR" rx="5"/>
  <text x="100" y="35" font-family="Arial, sans-serif" font-size="14" font-weight="bold" 
        text-anchor="middle" fill="white">$CORPORATE_NAME</text>
</svg>
EOF
    
    print_success "Logo placeholder created (replace with your actual logo)"
}

verify_corporate_installation() {
    print_section "Verifying corporate installation..."
    
    # Check Python packages
    if python3 -c "import fastapi, ollama, langchain, dotenv" 2>/dev/null; then
        print_success "Enterprise Python packages verified"
    else
        print_error "Enterprise package verification failed"
        return 1
    fi
    
    # Check Ollama
    if ollama list > /dev/null 2>&1; then
        print_success "Enterprise Ollama service verified"
    else
        print_error "Enterprise Ollama verification failed"
        return 1
    fi
    
    # Check corporate configuration
    if [ -f .env ]; then
        print_success "Corporate configuration verified"
    else
        print_error "Corporate configuration missing"
        return 1
    fi
    
    return 0
}

show_corporate_completion() {
    echo ""
    print_corporate_header
    echo -e "${GREEN}🎉 Corporate Installation Completed Successfully!${NC}"
    echo ""
    echo -e "${CYAN}🏢 Corporate Configuration:${NC}"
    echo "• Organization: ${YELLOW}$CORPORATE_NAME${NC}"
    echo "• Application: ${YELLOW}$CORPORATE_APP_NAME${NC}"
    echo "• Primary Color: ${YELLOW}$CORPORATE_PRIMARY_COLOR${NC}"
    echo "• Domain: ${YELLOW}$CORPORATE_DOMAIN${NC}"
    echo ""
    echo -e "${CYAN}🚀 Next Steps:${NC}"
    echo "1. Start corporate platform: ${YELLOW}./start_corporate_analyzer.sh${NC}"
    echo "2. Access corporate interface: ${BLUE}http://localhost:8000${NC}"
    echo "3. Replace logo: ${YELLOW}static/your-logo.svg${NC}"
    echo "4. Customize branding: ${YELLOW}Follow branding_instructions.md${NC}"
    echo ""
    echo -e "${CYAN}📁 Corporate Project Structure:${NC}"
    echo "├── 🌐 Corporate Frontend"
    echo "│   ├── landing.html              # Corporate landing page"
    echo "│   ├── index.html                # Corporate analyzer interface"
    echo "│   └── static/your-logo.svg      # Corporate logo (placeholder)"
    echo "├── 🔧 Corporate Backend"
    echo "│   ├── main.py                   # Corporate FastAPI server"
    echo "│   └── .env                      # Corporate configuration"
    echo "├── 📚 Corporate Documents"
    echo "│   ├── rag_docs/                 # Corporate regulatory docs"
    echo "│   ├── cag_docs/                 # Corporate procedures"
    echo "│   └── mapping_docs/             # Corporate technical docs"
    echo "└── 🔧 Corporate Operations"
    echo "    ├── logs/                     # Corporate logs"
    echo "    ├── backups/                  # Corporate backups"
    echo "    └── start_corporate_analyzer.sh"
    echo ""
    echo -e "${CYAN}🔧 Corporate Commands:${NC}"
    echo "• Check enterprise status: ${YELLOW}ollama list${NC}"
    echo "• View corporate logs: ${YELLOW}tail -f logs/startup.log${NC}"
    echo "• Stop Ollama service: ${YELLOW}pkill -f 'ollama serve'${NC}"
    echo "• Update enterprise models: ${YELLOW}ollama pull <model-name>${NC}"
    echo ""
    echo -e "${GREEN}🏢 Corporate Business Intelligence Platform Ready! 🚀${NC}"
    echo -e "${BLUE}Powered by $CORPORATE_NAME Enterprise Technology${NC}"
}

# Main corporate installation flow
main() {
    print_corporate_header
    
    # Collect corporate branding information
    collect_corporate_info
    
    # Technical setup
    check_python
    create_corporate_venv
    install_enterprise_deps
    
    # Ollama setup
    if ! check_ollama; then
        install_ollama
    fi
    
    start_ollama
    pull_enterprise_models
    
    # Corporate project setup
    create_corporate_directories
    create_corporate_config
    create_corporate_demo_data
    create_corporate_startup_script
    create_logo_placeholder
    
    # Verification and completion
    if verify_corporate_installation; then
        show_corporate_completion
    else
        print_error "Corporate installation verification failed. Please check the logs."
        exit 1
    fi
}

# Run corporate installation if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi