from typing import Dict, List, Tuple, Optional, Any
import os
import time
import json
import logging
from collections import deque
from dotenv import load_dotenv
import uuid
import asyncio
import redis
from app.core.config import get_settings

# Load environment variables
load_dotenv()

# Get settings
settings = get_settings()

# Configure logger
logger = logging.getLogger("cx_consulting_ai.memory_manager")

class MemoryManager:
    """Memory manager for conversation history."""
    
    def __init__(self, memory_type: str = None, max_items: int = None):
        """
        Initialize the memory manager.
        
        Args:
            memory_type: Type of memory (buffer, redis, azure_redis)
            max_items: Maximum number of interactions to store per conversation
        """
        self.memory_type = memory_type or settings.MEMORY_TYPE
        self.max_items = max_items or int(os.getenv("MAX_MEMORY_ITEMS", "10"))
        
        # Initialize conversations dict if using buffer memory
        if self.memory_type == "buffer":
            self.conversations: Dict[str, List[Dict[str, Any]]] = {}
            # Track project associations
            self.conversation_projects: Dict[str, str] = {}
        elif self.memory_type in ["redis", "azure_redis"]:
            # Initialize Redis connection if using redis memory
            try:
                # Get connection info based on deployment type
                redis_info = settings.get_redis_connection_info()
                
                if self.memory_type == "azure_redis":
                    # Azure Redis requires specific connection parameters
                    self.redis = redis.Redis(
                        host=redis_info["host"],
                        port=redis_info["port"],
                        password=redis_info["password"],
                        ssl=redis_info["ssl"],
                        decode_responses=redis_info.get("decode_responses", True)
                    )
                    logger.info(f"Connected to Azure Redis Cache at {redis_info['host']}:{redis_info['port']}")
                else:
                    # Standard Redis connection
                    self.redis = redis.from_url(redis_info["url"])
                    logger.info(f"Connected to Redis at {redis_info['url']}")
                    
                # Test connection
                self.redis.ping()
            except ImportError:
                logger.warning("Redis package not installed, falling back to buffer memory")
                self.memory_type = "buffer"
                self.conversations = {}
                self.conversation_projects = {}
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                logger.warning("Falling back to buffer memory")
                self.memory_type = "buffer"
                self.conversations = {}
                self.conversation_projects = {}
        
        logger.info(f"Memory manager initialized with type={self.memory_type}, max_items={self.max_items}")
    
    async def add_interaction(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a user-assistant interaction to the conversation history.
        
        Args:
            conversation_id: Conversation ID
            user_message: User's message
            assistant_message: Assistant's response
            metadata: Optional metadata about the interaction
        """
        timestamp = time.time()
        interaction = {
            "user": user_message,
            "assistant": assistant_message,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }
        
        if self.memory_type == "buffer":
            # Initialize conversation if it doesn't exist
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
            
            # Add interaction
            self.conversations[conversation_id].append(interaction)
            
            # Limit conversation length
            if len(self.conversations[conversation_id]) > self.max_items:
                self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_items:]
        
        elif self.memory_type in ["redis", "azure_redis"]:
            try:
                # Get existing conversation
                conversation_key = f"conversation:{conversation_id}"
                conversation_json = self.redis.get(conversation_key)
                
                if conversation_json:
                    conversation = json.loads(conversation_json)
                else:
                    conversation = []
                
                # Add interaction
                conversation.append(interaction)
                
                # Limit conversation length
                if len(conversation) > self.max_items:
                    conversation = conversation[-self.max_items:]
                
                # Save conversation
                self.redis.set(conversation_key, json.dumps(conversation))
                
                # Set expiration (24 hours)
                self.redis.expire(conversation_key, 86400)
            
            except Exception as e:
                logger.error(f"Error adding interaction to Redis: {str(e)}")
                # Fallback to buffer memory for this interaction
                if conversation_id not in self.conversations:
                    self.conversations[conversation_id] = []
                self.conversations[conversation_id].append(interaction)
    
    async def get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get the conversation history for a conversation ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of interactions
        """
        if self.memory_type == "buffer":
            return self.conversations.get(conversation_id, [])
        
        elif self.memory_type in ["redis", "azure_redis"]:
            try:
                conversation_key = f"conversation:{conversation_id}"
                conversation_json = self.redis.get(conversation_key)
                
                if conversation_json:
                    return json.loads(conversation_json)
                else:
                    return []
            
            except Exception as e:
                logger.error(f"Error getting conversation from Redis: {str(e)}")
                # Fallback to buffer memory
                return self.conversations.get(conversation_id, [])
    
    async def format_conversation(self, conversation_id: str) -> str:
        """
        Format the conversation history for inclusion in prompts.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Formatted conversation history
        """
        try:
            conversation = await self.get_conversation(conversation_id)
            
            if not conversation:
                return "No previous conversation."
            
            formatted = []
            for i, interaction in enumerate(conversation):
                try:
                    if isinstance(interaction, dict):
                        if "role" in interaction and "content" in interaction:
                            # New format
                            role = interaction["role"].capitalize()
                            content = interaction.get("content", "")
                            formatted.append(f"{role}: {content}")
                        elif "user" in interaction and "assistant" in interaction:
                            # Old format - for backward compatibility
                            user_message = interaction.get("user", "")
                            assistant_message = interaction.get("assistant", "")
                            formatted.append(f"User: {user_message}")
                            formatted.append(f"Assistant: {assistant_message}")
                except Exception as e:
                    # Skip malformed entries
                    continue
            
            return "\n\n".join(formatted) if formatted else "No previous conversation."
        except Exception as e:
            # Return a safe fallback
            return "No previous conversation."
    
    async def clear_conversation(self, conversation_id: str) -> None:
        """
        Clear the conversation history for a conversation ID.
        
        Args:
            conversation_id: Conversation ID
        """
        if self.memory_type == "buffer":
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
        
        elif self.memory_type in ["redis", "azure_redis"]:
            try:
                conversation_key = f"conversation:{conversation_id}"
                self.redis.delete(conversation_key)
            except Exception as e:
                logger.error(f"Error clearing conversation from Redis: {str(e)}")
                # Also clear from buffer if it exists
                if conversation_id in self.conversations:
                    del self.conversations[conversation_id]
    
    async def get_recent_conversations(self, limit: int = 10) -> List[str]:
        """
        Get a list of recent conversation IDs.
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation IDs
        """
        if self.memory_type == "buffer":
            # Sort conversations by most recent interaction
            sorted_conversations = sorted(
                self.conversations.items(),
                key=lambda x: x[1][-1]["timestamp"] if x[1] else 0,
                reverse=True
            )
            
            # Return conversation IDs
            return [conv_id for conv_id, _ in sorted_conversations[:limit]]
        
        elif self.memory_type in ["redis", "azure_redis"]:
            try:
                # Get all conversation keys
                conversation_keys = self.redis.keys("conversation:*")
                
                # Sort by expiration time (most recently used conversations expire later)
                sorted_keys = sorted(
                    conversation_keys,
                    key=lambda k: self.redis.ttl(k),
                    reverse=True
                )
                
                # Extract conversation IDs
                conversation_ids = [
                    k.decode().split(":", 1)[1] for k in sorted_keys[:limit]
                ]
                
                return conversation_ids
            
            except Exception as e:
                logger.error(f"Error getting recent conversations from Redis: {str(e)}")
                # Fallback to buffer
                sorted_conversations = sorted(
                    self.conversations.items(),
                    key=lambda x: x[1][-1]["timestamp"] if x[1] else 0,
                    reverse=True
                )
                return [conv_id for conv_id, _ in sorted_conversations[:limit]]
    
    async def create_conversation(self, project_id: Optional[str] = None) -> str:
        """
        Create a new conversation and return its ID.
        
        Args:
            project_id: Optional project ID to associate with the conversation
            
        Returns:
            New conversation ID
        """
        conversation_id = str(uuid.uuid4())
        
        if self.memory_type == "buffer":
            self.conversations[conversation_id] = []
            if project_id:
                self.conversation_projects[conversation_id] = project_id
        elif self.memory_type in ["redis", "azure_redis"]:
            try:
                conversation_key = f"conversation:{conversation_id}"
                self.redis.set(conversation_key, json.dumps([]))
                # Set expiration (24 hours)
                self.redis.expire(conversation_key, 86400)
                
                # Store project association if provided
                if project_id:
                    self.redis.hset("conversation_projects", conversation_id, project_id)
            except Exception as e:
                logger.error(f"Error creating conversation in Redis: {str(e)}")
                # Fallback to buffer memory
                self.conversations[conversation_id] = []
                if project_id:
                    self.conversation_projects[conversation_id] = project_id
        
        logger.info(f"Created new conversation with ID {conversation_id}" + 
                   (f" for project {project_id}" if project_id else ""))
        return conversation_id
    
    async def get_conversation_project(self, conversation_id: str) -> Optional[str]:
        """
        Get the project ID associated with a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Project ID or None if not associated
        """
        if self.memory_type == "buffer":
            return self.conversation_projects.get(conversation_id)
        
        elif self.memory_type in ["redis", "azure_redis"]:
            try:
                project_id = self.redis.hget("conversation_projects", conversation_id)
                return project_id.decode() if project_id else None
            except Exception as e:
                logger.error(f"Error getting conversation project from Redis: {str(e)}")
                return None
    
    async def set_conversation_project(self, conversation_id: str, project_id: str) -> bool:
        """
        Set the project ID for a conversation.
        
        Args:
            conversation_id: Conversation ID
            project_id: Project ID
            
        Returns:
            True if successful, False otherwise
        """
        # Check if conversation exists
        if not await self.get_conversation(conversation_id):
            logger.warning(f"Conversation {conversation_id} not found")
            return False
        
        if self.memory_type == "buffer":
            self.conversation_projects[conversation_id] = project_id
        
        elif self.memory_type in ["redis", "azure_redis"]:
            try:
                self.redis.hset("conversation_projects", conversation_id, project_id)
            except Exception as e:
                logger.error(f"Error setting conversation project in Redis: {str(e)}")
                return False
        
        logger.info(f"Set conversation {conversation_id} project to {project_id}")
        return True
    
    async def get_project_conversations(self, project_id: str) -> List[str]:
        """
        Get all conversation IDs for a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            List of conversation IDs
        """
        if self.memory_type == "buffer":
            return [
                conv_id for conv_id, proj_id in self.conversation_projects.items()
                if proj_id == project_id
            ]
        
        elif self.memory_type in ["redis", "azure_redis"]:
            try:
                # Get all conversation-project mappings
                mappings = self.redis.hgetall("conversation_projects")
                
                # Filter by project ID
                conversations = [
                    conv_id.decode() for conv_id, proj_id in mappings.items()
                    if proj_id.decode() == project_id
                ]
                
                return conversations
            except Exception as e:
                logger.error(f"Error getting project conversations from Redis: {str(e)}")
                return []
    
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            conversation_id: Conversation ID
            role: Message role (user or assistant)
            content: Message content
            metadata: Optional metadata about the message
        """
        # Check if conversation exists
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            # Create conversation if it doesn't exist
            if self.memory_type == "buffer":
                self.conversations[conversation_id] = []
            elif self.memory_type in ["redis", "azure_redis"]:
                try:
                    conversation_key = f"conversation:{conversation_id}"
                    self.redis.set(conversation_key, json.dumps([]))
                    # Set expiration (24 hours)
                    self.redis.expire(conversation_key, 86400)
                except Exception as e:
                    logger.error(f"Error creating conversation in Redis: {str(e)}")
                    # Fallback to buffer memory
                    self.conversations[conversation_id] = []
        
        timestamp = time.time()
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }
        
        if self.memory_type == "buffer":
            # Add message
            self.conversations[conversation_id].append(message)
            
            # Limit conversation length
            if len(self.conversations[conversation_id]) > self.max_items:
                self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_items:]
        
        elif self.memory_type in ["redis", "azure_redis"]:
            try:
                # Get existing conversation
                conversation_key = f"conversation:{conversation_id}"
                conversation_json = self.redis.get(conversation_key)
                
                if conversation_json:
                    conversation = json.loads(conversation_json)
                else:
                    conversation = []
                
                # Add message
                conversation.append(message)
                
                # Limit conversation length
                if len(conversation) > self.max_items:
                    conversation = conversation[-self.max_items:]
                
                # Save conversation
                self.redis.set(conversation_key, json.dumps(conversation))
                
                # Reset expiration (24 hours from now)
                self.redis.expire(conversation_key, 86400)
            
            except Exception as e:
                logger.error(f"Error adding message to Redis: {str(e)}")
                # Fallback to buffer memory for this message
                if conversation_id not in self.conversations:
                    self.conversations[conversation_id] = []
                self.conversations[conversation_id].append(message)
    
    async def get_formatted_history(self, conversation_id: str, limit: int = 10) -> str:
        """
        Get the conversation history in a format suitable for inclusion in prompts.
        
        Args:
            conversation_id: The ID of the conversation
            limit: Maximum number of recent messages to include
            
        Returns:
            Formatted conversation history string
        """
        messages = await self.get_conversation(conversation_id)
        if not messages:
            return ""
        
        # Only use the most recent messages up to the limit
        if limit > 0 and len(messages) > limit:
            messages = messages[-limit:]
        
        # Format the conversation history
        formatted_history = []
        for msg in messages:
            if "role" in msg and "content" in msg:
                if msg["role"] == "user":
                    formatted_history.append(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    formatted_history.append(f"Assistant: {msg['content']}")
            elif "user" in msg and "assistant" in msg:
                formatted_history.append(f"User: {msg['user']}")
                formatted_history.append(f"Assistant: {msg['assistant']}")
        
        return "\n\n".join(formatted_history)
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            True if conversation was deleted, False if it didn't exist
        """
        if self.memory_type == "buffer":
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                # Also remove from project mapping
                if conversation_id in self.conversation_projects:
                    del self.conversation_projects[conversation_id]
                logger.info(f"Deleted conversation {conversation_id}")
                return True
            else:
                logger.warning(f"Conversation {conversation_id} not found")
                return False
        
        elif self.memory_type in ["redis", "azure_redis"]:
            try:
                conversation_key = f"conversation:{conversation_id}"
                exists = self.redis.exists(conversation_key)
                
                if exists:
                    self.redis.delete(conversation_key)
                    # Also remove from project mapping
                    self.redis.hdel("conversation_projects", conversation_id)
                    logger.info(f"Deleted conversation {conversation_id}")
                    return True
                else:
                    logger.warning(f"Conversation {conversation_id} not found")
                    return False
            
            except Exception as e:
                logger.error(f"Error deleting conversation from Redis: {str(e)}")
                # Also try to delete from buffer if it exists
                if conversation_id in self.conversations:
                    del self.conversations[conversation_id]
                    if conversation_id in self.conversation_projects:
                        del self.conversation_projects[conversation_id]
                    return True
                return False 