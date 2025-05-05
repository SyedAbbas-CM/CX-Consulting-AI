# CX Consulting AI

An AI-powered consulting assistant that leverages RAG to provide customer experience expertise.

## Features

- Advanced RAG system for accurate CX consulting responses
- Document management for knowledge base maintenance
- User authentication and project management
- Multi-LLM support (local models and Azure OpenAI)
- Conversation memory with Redis integration
- Modern React frontend with Next.js

## Running Locally

### Prerequisites

- Python 3.10+
- Node.js 18+
- Redis server (optional, can use in-memory buffer instead)

### Setup Environment

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/cx-consulting-agent.git
   cd cx-consulting-agent
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install frontend dependencies:
   ```
   cd app/frontend/cx-consulting-ai-3
   npm install
   cd ../../..
   ```

5. Initialize the database:
   ```
   python app/scripts/init_db.py
   ```

### Running the Application

1. Start Redis (if using):
   ```
   redis-server
   ```

2. Start the backend:
   ```
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. Start the frontend:
   ```
   cd app/frontend/cx-consulting-ai-3
   npm run dev
   ```

4. Access the application at:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Running System Tests

To verify all components are working correctly:

```
python app/scripts/test_system.py
```

## Deploying to Azure

This application can be deployed to Azure with minimal configuration changes. The application is designed to work with both local and Azure-based services.

### Prerequisites

- Azure subscription
- Azure CLI installed and configured
- Access to Azure OpenAI service

### Azure Deployment

1. Run the Azure deployment setup script:
   ```
   python deploy_azure.py --action setup --subscription YOUR_SUBSCRIPTION_ID --resource-group YOUR_RESOURCE_GROUP --app-name YOUR_APP_NAME --location eastus
   ```

2. Create Azure OpenAI resource manually in the Azure portal and note the endpoint and key.

3. Update the `.env.azure` file with your Azure OpenAI endpoint, key, and deployment name.

4. Deploy the application to Azure:
   ```
   python deploy_azure.py --action deploy --subscription YOUR_SUBSCRIPTION_ID --resource-group YOUR_RESOURCE_GROUP --app-name YOUR_APP_NAME
   ```

5. Follow the instructions to deploy the frontend to Azure Static Web Apps.

### Environment Configuration

The application uses a flexible configuration system that allows switching between local and Azure deployments by setting the `DEPLOYMENT_MODE` environment variable:

- `DEPLOYMENT_MODE=local`: Uses local services (default)
- `DEPLOYMENT_MODE=azure`: Uses Azure services

The following Azure services are supported:

- LLM: Azure OpenAI Service
- Vector Database: Azure AI Search
- Memory: Azure Redis Cache
- Authentication: SQLite (can be extended to Azure SQL)

## Architecture

The application consists of the following components:

- **Frontend**: Next.js React application
- **Backend API**: FastAPI server
- **LLM Service**: Interface to language models (local or Azure)
- **Document Service**: Manages document processing and retrieval
- **RAG Engine**: Orchestrates the RAG pipeline
- **Memory Manager**: Handles conversation memory (Redis/Azure Redis)
- **Context Optimizer**: Refines retrieved context for improved answers

## License

[MIT License](LICENSE) 