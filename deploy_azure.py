#!/usr/bin/env python
"""
Azure Deployment Script for CX Consulting AI

This script helps deploy the CX Consulting AI application to Azure.
"""
import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deploy CX Consulting AI to Azure')
    parser.add_argument('--subscription', '-s', type=str, help='Azure subscription ID')
    parser.add_argument('--resource-group', '-g', type=str, help='Azure resource group name')
    parser.add_argument('--location', '-l', type=str, default='eastus', help='Azure region (default: eastus)')
    parser.add_argument('--app-name', '-n', type=str, help='App name prefix for Azure resources')
    parser.add_argument('--action', '-a', type=str, choices=['setup', 'deploy', 'clean'], 
                        default='deploy', help='Action to perform')
    return parser.parse_args()

def check_prerequisites():
    """Check if prerequisites are installed."""
    print("Checking prerequisites...")
    
    # Check if Azure CLI is installed
    try:
        result = subprocess.run(['az', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Error: Azure CLI is not installed. Please install it first.")
            print("Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli")
            return False
    except FileNotFoundError:
        print("Error: Azure CLI is not installed. Please install it first.")
        print("Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli")
        return False
    
    # Check if user is logged in to Azure CLI
    result = subprocess.run(['az', 'account', 'show'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error: You are not logged in to Azure CLI. Please run 'az login' first.")
        return False
    
    return True

def generate_env_file(subscription_id, resource_group, app_name, location):
    """Generate a .env file for Azure deployment."""
    print("Generating .env file for Azure deployment...")
    
    env_content = f"""# CX Consulting AI Azure Deployment Settings
DEPLOYMENT_MODE=azure
SUBSCRIPTION_ID={subscription_id}
RESOURCE_GROUP={resource_group}
APP_NAME_PREFIX={app_name}
LOCATION={location}

# LLM settings
LLM_BACKEND=azure

# Azure OpenAI settings
AZURE_OPENAI_ENDPOINT=https://{app_name}-openai.openai.azure.com/
AZURE_OPENAI_KEY=__TO_BE_FILLED__
AZURE_OPENAI_DEPLOYMENT={app_name}-deployment
AZURE_OPENAI_API_VERSION=2023-12-01-preview

# Vector DB settings
VECTOR_DB_TYPE=azure_ai_search
AZURE_SEARCH_ENDPOINT=https://{app_name}-search.search.windows.net
AZURE_SEARCH_KEY=__TO_BE_FILLED__
AZURE_SEARCH_INDEX_NAME=cx-documents

# Memory settings
MEMORY_TYPE=azure_redis
AZURE_REDIS_HOST={app_name}-redis.redis.cache.windows.net
AZURE_REDIS_KEY=__TO_BE_FILLED__
AZURE_REDIS_PORT=6380
AZURE_REDIS_SSL=true

# Database settings
DB_TYPE=sqlite
# For production, consider upgrading to Azure SQL:
# DB_TYPE=azure_sql
# AZURE_SQL_CONNECTION_STRING="Driver={{ODBC Driver 18 for SQL Server}};Server=tcp:{app_name}-sql.database.windows.net,1433;Database=cxConsultingDB;Uid=cxadmin;Pwd=__TO_BE_FILLED__;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
"""
    
    # Write to .env.azure file
    with open('.env.azure', 'w') as f:
        f.write(env_content)
    
    print("Generated .env.azure file. Please fill in the missing values after resources are created.")

def setup_azure_resources(subscription_id, resource_group, location, app_name):
    """Set up Azure resources."""
    print(f"Setting up Azure resources for {app_name}...")
    
    # Set subscription
    subprocess.run(['az', 'account', 'set', '--subscription', subscription_id], check=True)
    
    # Create resource group if it doesn't exist
    result = subprocess.run(
        ['az', 'group', 'exists', '--name', resource_group], 
        capture_output=True, text=True, check=True
    )
    if result.stdout.strip() == 'false':
        print(f"Creating resource group: {resource_group}")
        subprocess.run(
            ['az', 'group', 'create', '--name', resource_group, '--location', location],
            check=True
        )
    
    # Create Azure App Service Plan
    print("Creating App Service Plan...")
    subprocess.run([
        'az', 'appservice', 'plan', 'create',
        '--name', f"{app_name}-plan",
        '--resource-group', resource_group,
        '--sku', 'B1',
        '--is-linux'
    ], check=True)
    
    # Create Azure App Service
    print("Creating App Service...")
    subprocess.run([
        'az', 'webapp', 'create',
        '--name', f"{app_name}-api",
        '--resource-group', resource_group,
        '--plan', f"{app_name}-plan",
        '--runtime', 'PYTHON:3.10'
    ], check=True)
    
    # Create Azure Redis Cache
    print("Creating Azure Redis Cache (this may take a few minutes)...")
    subprocess.run([
        'az', 'redis', 'create',
        '--name', f"{app_name}-redis",
        '--resource-group', resource_group,
        '--location', location,
        '--sku', 'Basic',
        '--vm-size', 'C0'
    ], check=True)
    
    # Create Azure AI Search
    print("Creating Azure AI Search...")
    subprocess.run([
        'az', 'search', 'service', 'create',
        '--name', f"{app_name}-search",
        '--resource-group', resource_group,
        '--sku', 'Basic'
    ], check=True)
    
    # Create Azure Static Web App for frontend
    print("Creating Azure Static Web App for frontend...")
    subprocess.run([
        'az', 'staticwebapp', 'create',
        '--name', f"{app_name}-frontend",
        '--resource-group', resource_group,
        '--location', location,
        '--sku', 'Free'
    ], check=True)
    
    # Note: OpenAI must be created through the Azure portal and approved first
    print("\nNOTE: Azure OpenAI Service must be created manually from the Azure portal.")
    print("Please follow these steps:")
    print("1. Go to https://portal.azure.com")
    print("2. Search for 'Azure OpenAI'")
    print("3. Create a new Azure OpenAI resource")
    print("4. Use the naming convention: {app_name}-openai")
    print("5. After creation, deploy a model and note the deployment name")
    print("6. Update the AZURE_OPENAI_DEPLOYMENT value in .env.azure")
    
    # Get Redis key
    print("\nRetrieving Redis access key...")
    redis_keys = subprocess.run([
        'az', 'redis', 'list-keys',
        '--name', f"{app_name}-redis",
        '--resource-group', resource_group
    ], capture_output=True, text=True, check=True)
    
    redis_keys_json = json.loads(redis_keys.stdout)
    redis_key = redis_keys_json.get('primaryKey', '')
    
    # Get AI Search key
    print("Retrieving AI Search key...")
    search_keys = subprocess.run([
        'az', 'search', 'admin-key', 'show',
        '--service-name', f"{app_name}-search",
        '--resource-group', resource_group
    ], capture_output=True, text=True, check=True)
    
    search_keys_json = json.loads(search_keys.stdout)
    search_key = search_keys_json.get('primaryKey', '')
    
    # Update .env.azure file with keys
    print("Updating .env.azure with keys...")
    env_file = Path('.env.azure')
    if env_file.exists():
        content = env_file.read_text()
        content = content.replace('AZURE_REDIS_KEY=__TO_BE_FILLED__', f'AZURE_REDIS_KEY={redis_key}')
        content = content.replace('AZURE_SEARCH_KEY=__TO_BE_FILLED__', f'AZURE_SEARCH_KEY={search_key}')
        env_file.write_text(content)
    
    print("\nResource setup complete!")
    print("Please create the Azure OpenAI resource manually, then update the .env.azure file.")

def deploy_to_azure(subscription_id, resource_group, app_name):
    """Deploy the application to Azure."""
    print(f"Deploying CX Consulting AI to Azure ({app_name})...")
    
    # Set subscription
    subprocess.run(['az', 'account', 'set', '--subscription', subscription_id], check=True)
    
    # Deploy backend API to Azure App Service
    print("Deploying backend API to Azure App Service...")
    subprocess.run([
        'az', 'webapp', 'up',
        '--name', f"{app_name}-api",
        '--resource-group', resource_group,
        '--sku', 'B1',
        '--location', resource_group,
        '--runtime', 'PYTHON:3.10'
    ], check=True)
    
    # Configure App Service settings from .env.azure
    print("Configuring App Service settings...")
    env_file = Path('.env.azure')
    if env_file.exists():
        env_vars = {}
        for line in env_file.read_text().splitlines():
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key] = value
        
        # Convert env vars to app settings format
        settings_args = []
        for key, value in env_vars.items():
            settings_args.extend(['--settings', f"{key}={value}"])
        
        # Apply settings
        subprocess.run([
            'az', 'webapp', 'config', 'appsettings', 'set',
            '--name', f"{app_name}-api",
            '--resource-group', resource_group,
            *settings_args
        ], check=True)
    
    # Get the API URL
    api_url = f"https://{app_name}-api.azurewebsites.net"
    
    # Create Static Web App configuration for frontend
    print("Preparing frontend deployment...")
    swa_config = {
        "routes": [
            {
                "route": "/api/*",
                "rewrite": f"{api_url}/api/$1"
            },
            {
                "route": "/*",
                "serve": "/index.html",
                "statusCode": 200
            }
        ],
        "responseOverrides": {
            "404": {
                "rewrite": "/index.html",
                "statusCode": 200
            }
        }
    }
    
    # Write to staticwebapp.config.json
    with open('app/frontend/cx-consulting-ai-3/staticwebapp.config.json', 'w') as f:
        json.dump(swa_config, f, indent=2)
    
    print("\nDeployment preparation complete!")
    print(f"API URL: {api_url}")
    print("\nTo deploy the frontend:")
    print(f"1. Go to https://portal.azure.com > Static Web Apps > {app_name}-frontend")
    print("2. Click on 'Deployment' > 'Manage deployment'")
    print("3. Link to your GitHub repository and configure:")
    print("   - Build Preset: 'Next.js'")
    print(f"   - App location: '/app/frontend/cx-consulting-ai-3'")
    print("4. Wait for the deployment to complete")

def clean_azure_resources(subscription_id, resource_group):
    """Clean up Azure resources."""
    print(f"Cleaning up Azure resources in resource group: {resource_group}...")
    
    # Set subscription
    subprocess.run(['az', 'account', 'set', '--subscription', subscription_id], check=True)
    
    # Confirm deletion
    confirm = input(f"This will delete ALL resources in the resource group '{resource_group}'. Are you sure? (y/n): ")
    if confirm.lower() != 'y':
        print("Cleanup cancelled.")
        return
    
    # Delete resource group
    print(f"Deleting resource group: {resource_group}...")
    subprocess.run(['az', 'group', 'delete', '--name', resource_group, '--yes'], check=True)
    
    print("Resources cleaned up successfully.")

def main():
    """Main function."""
    args = parse_args()
    
    if not check_prerequisites():
        sys.exit(1)
    
    # Ensure required arguments are provided
    if not args.subscription:
        args.subscription = input("Enter Azure subscription ID: ")
    
    if not args.resource_group:
        args.resource_group = input("Enter Azure resource group name: ")
    
    if not args.app_name:
        args.app_name = input("Enter app name prefix for Azure resources: ")
    
    # Take action based on command
    if args.action == 'setup':
        generate_env_file(args.subscription, args.resource_group, args.app_name, args.location)
        setup_azure_resources(args.subscription, args.resource_group, args.location, args.app_name)
    elif args.action == 'deploy':
        deploy_to_azure(args.subscription, args.resource_group, args.app_name)
    elif args.action == 'clean':
        clean_azure_resources(args.subscription, args.resource_group)
    
    print("\nAzure deployment script completed successfully!")

if __name__ == "__main__":
    main() 