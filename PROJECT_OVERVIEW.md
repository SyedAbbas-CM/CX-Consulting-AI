# CX Consulting AI - Project Overview

## 🎯 **Project Summary**
**CX Consulting AI** is a sophisticated RAG-based (Retrieval Augmented Generation) AI assistant specifically designed for customer experience consulting tasks. The system combines document intelligence, conversational AI, and project management capabilities to help consultants analyze documents, generate deliverables, and manage client projects.

## 🏗️ **System Architecture**

### **Core Technology Stack**
- **Backend**: FastAPI (Python async web framework)
- **LLM Engine**: llama.cpp with Gemma 12B model (Apple Silicon optimized)
- **Vector Database**: ChromaDB for semantic document search
- **Memory Layer**: Redis for chat history and caching
- **Frontend**: React.js application
- **Document Processing**: Multi-format support (PDF, DOCX, XLSX, etc.)
- **Authentication**: JWT-based with SQLite user database

### **Architecture Diagram**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React.js      │    │   FastAPI       │    │   llama.cpp     │
│   Frontend      │◄──►│   Backend       │◄──►│   Gemma 12B     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                      ┌────────┼────────┐
                      ▼        ▼        ▼
              ┌─────────────┐ ┌─────┐ ┌─────────────┐
              │  ChromaDB   │ │Redis│ │   SQLite    │
              │  Vectors    │ │Cache│ │   Users     │
              └─────────────┘ └─────┘ └─────────────┘
```

## 📁 **Project Structure**

```
CX Consulting Agent/
├── app/                          # Main application
│   ├── agents/                   # AI agent implementations
│   ├── api/                      # FastAPI routes and models
│   ├── core/                     # Core services (LLM, config)
│   ├── services/                 # Business logic services
│   ├── utils/                    # Utility functions
│   ├── frontend/                 # React.js frontend
│   ├── data/                     # Data storage
│   └── template_wrappers/        # Prompt templates
├── data/                         # Document storage
├── models/                       # LLM model files
├── requirements.txt              # Python dependencies
├── package.json                  # Node.js dependencies
└── README.md                     # Project documentation
```

## 🔧 **Core Components**

### **1. RAG Engine** (`app/services/rag_engine.py`)
- **Purpose**: Retrieval Augmented Generation for document-based Q&A
- **Features**: Semantic search, context optimization, source attribution
- **Technology**: ChromaDB + BGE embeddings + BM25 hybrid search

### **2. Document Service** (`app/services/document_service.py`)
- **Purpose**: Document ingestion, processing, and management
- **Supported Formats**: PDF, DOCX, XLSX, TXT, MD
- **Features**: Chunking, vectorization, metadata extraction

### **3. Chat Service** (`app/services/chat_service.py`)
- **Purpose**: Conversational interface with memory
- **Features**: Multi-turn conversations, context preservation, Redis persistence
- **TTL**: 60-day expiration with access refresh

### **4. Project Manager** (`app/services/project_manager.py`)
- **Purpose**: Client project organization and management
- **Features**: Project creation, document organization, deliverable tracking
- **Storage**: File-based with atomic operations

### **5. Deliverable Service** (`app/services/deliverable_service.py`)
- **Purpose**: Generate consulting deliverables (reports, presentations)
- **Features**: Template-based generation, multi-format output
- **Templates**: CX strategy, analysis reports, recommendations

### **6. LLM Service** (`app/core/llm_service.py`)
- **Purpose**: Language model abstraction layer
- **Backends**: llama.cpp (primary), Azure OpenAI, Ollama
- **Features**: Async generation, timeout handling, token management

## 🤖 **AI Capabilities**

### **Document Intelligence**
- **Semantic Search**: Find relevant content across document collections
- **Multi-Modal**: Text, tables, structured data extraction
- **Source Attribution**: Track and cite document sources

### **Conversational AI**
- **Context Awareness**: Maintains conversation history and context
- **Domain Expertise**: Specialized for CX consulting terminology
- **Memory Management**: Long-term project memory with Redis

### **Content Generation**
- **Deliverable Templates**: Pre-built CX consulting frameworks
- **Custom Reports**: Generate analysis based on uploaded documents
- **Multi-Format Output**: Markdown, HTML, structured documents

## 🔐 **Security & Authentication**

### **User Management**
- **JWT Tokens**: Secure authentication with expiration
- **Role-Based Access**: Admin and user permissions
- **SQLite Database**: Local user storage with password hashing

### **Data Security**
- **Local Processing**: All LLM processing stays on-device
- **File Isolation**: Project-based document separation
- **Access Controls**: User-specific project access

## 📊 **Performance Characteristics**

### **Current Metrics** (from error analysis)
- **Document Processing**: ~2-5 seconds per document
- **Query Response**: ~94 seconds total (37s prompt + 47s generation)
- **Vector Search**: ~1-2 seconds for retrieval
- **Memory Usage**: ~8GB for Gemma 12B model

### **Optimization Opportunities**
- **Model Switching**: Use lighter models for faster responses
- **Caching**: Implement query result caching
- **Parallel Processing**: Async document processing
- **Index Optimization**: BM25 precomputed indices

## 🛠️ **Development Environment**

### **System Requirements**
- **OS**: macOS (Apple Silicon optimized)
- **Python**: 3.9+ with conda/venv
- **Memory**: 16GB+ RAM recommended
- **Storage**: 20GB+ for models and data

### **Key Dependencies**
```
FastAPI 0.115.4          # Web framework
llama-cpp-python 0.3.1   # LLM inference
chromadb 0.5.18          # Vector database
redis 5.2.0              # Memory/cache
transformers 4.46.3      # HF models
sentence-transformers 3.3.0  # Embeddings
```

### **Development Setup**
```bash
# 1. Clone and setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your model paths

# 3. Start services
python start.py  # Starts Redis and FastAPI
```

## 🔄 **Recent Fixes Applied**
1. **500 Error Fix**: Made conversation_id optional in document mode
2. **Chat List Fix**: Fixed dictionary unpacking in chat summaries
3. **Project Corruption**: Added atomic file operations
4. **Template Search**: Enhanced deliverable template retrieval
5. **TTL Refresh**: Prevent active chat expiration
6. **vLLM Removal**: Cleaned unused dependencies

## 🎯 **Use Cases**

### **Primary Workflows**
1. **Document Analysis**: Upload client documents → Ask questions → Get insights
2. **Project Management**: Create projects → Organize documents → Track progress
3. **Deliverable Generation**: Select templates → Generate reports → Export results
4. **Knowledge Search**: Query across all documents → Find relevant information

### **Target Users**
- **CX Consultants**: Primary users for client work
- **Business Analysts**: Document analysis and reporting
- **Project Managers**: Project organization and tracking
- **Knowledge Workers**: Information retrieval and synthesis

## 📈 **Future Roadmap**
- **Multi-Modal Support**: Image and video document processing
- **Cloud Deployment**: Azure/AWS deployment options
- **API Integrations**: CRM and project management tool connections
- **Advanced Analytics**: Usage metrics and performance dashboards
- **Team Collaboration**: Multi-user project sharing

## 🔧 **Troubleshooting**

### **Common Issues**
1. **Redis Connection**: Ensure Redis is running (`start.py` handles this)
2. **Model Loading**: Check MODEL_PATH in .env file
3. **Memory Issues**: Use smaller models or increase system RAM
4. **Slow Responses**: Consider switching to lighter cross-encoder models

### **Configuration Files**
- **.env**: Environment variables and model paths
- **app/core/config.py**: Application settings and defaults
- **requirements.txt**: Python package dependencies
- **package.json**: Frontend dependencies

---

**Last Updated**: January 2025
**Project Status**: Active Development
**Version**: 1.0.0
