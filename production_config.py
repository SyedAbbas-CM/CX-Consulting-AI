#!/usr/bin/env python3
"""
Production Configuration Script for CX Consulting AI
Sets optimal environment variables for maximum performance
"""

import os
import sys


def set_production_config():
    """Set production environment variables for optimal performance."""

    production_config = {
        # PRODUCTION OPTIMIZATION FLAGS
        "PRODUCTION_MODE": "true",
        "DISABLE_REQUEST_LOGGING": "true",
        "DISABLE_AUTH_LOGGING": "true",
        "DISABLE_FILE_LOGGING": "true",
        # Logging Configuration
        "LOG_LEVEL": "WARNING",
        "DEBUG": "false",
        # Chat Performance Settings
        "CHAT_MAX_HISTORY_LENGTH": "1000",
        # LLM Performance Settings
        "LLM_TIMEOUT": "300",
        "LLAMA_CPP_VERBOSE": "false",
        # Disable metrics collection in production
        "ENABLE_METRICS": "false",
        # Server Configuration
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "ENABLE_CORS": "true",
        # Deployment Mode
        "DEPLOYMENT_MODE": "azure",
        # Embedding Performance
        "EMBEDDING_DEVICE": "auto",
        # Upload Settings
        "MAX_UPLOAD_SIZE_PER_FILE": "104857600",
        # Context Optimization
        "USE_RERANKING": "true",
        "MAX_CONTEXT_TOKENS_OPTIMIZER": "2048",
    }

    print("🚀 Setting production configuration for maximum performance...")

    for key, value in production_config.items():
        os.environ[key] = value
        print(f"✅ {key}={value}")

    print("\n🎯 Production optimizations applied:")
    print("   • Logging reduced to WARNING level only")
    print("   • Request/Auth logging disabled")
    print("   • File logging disabled")
    print("   • Chat history increased to 1000 messages")
    print("   • Metrics collection disabled")
    print("   • LLM verbose output disabled")
    print("\n🔥 Your app is now optimized for 100% production speed!")


if __name__ == "__main__":
    set_production_config()

    # Optionally start the app
    if len(sys.argv) > 1 and sys.argv[1] == "--start":
        print("\n🚀 Starting production server...")
        os.system("python start.py")
