import redis
import uuid
import json
from datetime import datetime, timezone
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("cx_consulting_ai.chat_service")


class ChatService:
    """
    Manages chat sessions and history in Redis.
    ───────────────────────────────────────────
    Key layout (all helper methods are centralised, **use them everywhere**):

        chat:{chat_id}:metadata      – Hash   (project_id, name, timestamps…)
        chat:{chat_id}:messages      – List   (JSON strings, newest → left/0)
        project:{project_id}:chats   – Set    (chat_id, chat_id, …)
    """

    # ---------- construction & helpers --------------------------------------------------

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("ChatService connected to Redis @ %s", redis_url)
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Redis at {redis_url}: {e}")
            raise ConnectionError(f"Could not connect to Redis for ChatService: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during ChatService Redis initialization: {e}", exc_info=True)
            raise


    # helper key builders
    def _meta_key(self, chat_id: str) -> str:
        return f"chat:{chat_id}:metadata"

    def _msg_key(self, chat_id: str) -> str:
        return f"chat:{chat_id}:messages"

    def _project_set_key(self, project_id: str) -> str:
        return f"project:{project_id}:chats"

    # ---------- public API --------------------------------------------------------------

    def create_chat(self, project_id: str, chat_name: Optional[str] = None) -> Dict[str, Any]:
        chat_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        chat_name = chat_name or f"Chat {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"

        metadata = {
            "chat_id": chat_id,
            "project_id": project_id,
            "name": chat_name,
            "created_at": now_iso,
            "last_updated": now_iso,
        }

        try:
            with self.redis_client.pipeline() as pipe:
                pipe.hmset(self._meta_key(chat_id), metadata)
                pipe.sadd(self._project_set_key(project_id), chat_id)
                # ensure an empty list exists for messages
                pipe.rpush(self._msg_key(chat_id), ":placeholder:")
                pipe.lpop(self._msg_key(chat_id))
                pipe.execute()
            logger.info("Created chat '%s' for project '%s'", chat_id, project_id)
            return metadata
        except Exception as e:
            logger.exception("Redis error creating chat: %s", e)
            raise

    def list_chats_for_project(self, project_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        chat_ids = list(self.redis_client.smembers(self._project_set_key(project_id))) or []
        # Manual sorting if needed (e.g., by 'created_at' after fetching all metadata) - currently unordered Set
        paginated = chat_ids[offset: offset + limit] 
        summaries: List[Dict[str, Any]] = []

        if not paginated:
            return summaries

        with self.redis_client.pipeline() as pipe:
            for cid in paginated:
                pipe.hgetall(self._meta_key(cid))
            metas = pipe.execute()

        for cid, meta in zip(paginated, metas):
            if not meta:
                logger.warning("Missing metadata for chat %s in project %s", cid, project_id)
                continue
            # Back-compat: some old chats miss last_updated → fallback to created_at
            meta.setdefault("last_updated", meta.get("created_at"))
            summaries.append({"chat_id": cid, **meta})
        return summaries

    def get_chat_history(self, chat_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        # Check metadata exists first to confirm chat validity
        if not self.redis_client.exists(self._meta_key(chat_id)):
            logger.warning("Request for history of non-existent chat %s", chat_id)
            return [] # Return empty list if chat metadata doesn't exist

        raw_messages = self.redis_client.lrange(self._msg_key(chat_id), offset, offset + limit - 1)
        messages: List[Dict[str, Any]] = []
        for raw in raw_messages:
            try:
                messages.append(json.loads(raw))
            except Exception:
                logger.exception("Corrupt message in chat %s: %s", chat_id, raw)
        return messages # Return the list of message dicts

    def add_message_to_chat(self, chat_id: str, role: str, content: str) -> bool:
        if role not in {"user", "assistant"}:
            logger.error("Invalid role '%s'", role)
            return False
        # Use the correct key check
        if not self.redis_client.exists(self._meta_key(chat_id)):
            logger.warning("Attempt to write to non-existent chat %s", chat_id)
            return False

        now_iso = datetime.utcnow().isoformat()
        msg_json = json.dumps({"role": role, "content": content, "timestamp": now_iso})
        try:
            with self.redis_client.pipeline() as pipe:
                # Use the correct key for messages
                pipe.lpush(self._msg_key(chat_id), msg_json)
                # Use the correct key for metadata
                pipe.hset(self._meta_key(chat_id), "last_updated", now_iso) 
                pipe.execute()
            return True
        except Exception:
            logger.exception("Failed to add message to chat %s", chat_id)
            return False

    async def get_chat_summary(self, chat_id: str) -> Optional[Dict[str, Any]]:
        # Use the correct key
        meta = self.redis_client.hgetall(self._meta_key(chat_id))
        return meta or None

    async def delete_chat(self, chat_id: str) -> bool:
        # Use the correct key
        meta = self.redis_client.hgetall(self._meta_key(chat_id))
        if not meta:
            logger.warning("Delete requested for missing chat %s", chat_id)
            return False
        project_id = meta.get("project_id")
        try:
            with self.redis_client.pipeline() as pipe:
                # Use correct keys
                pipe.delete(self._meta_key(chat_id))
                pipe.delete(self._msg_key(chat_id))
                if project_id:
                    # Use correct key
                    pipe.srem(self._project_set_key(project_id), chat_id)
                pipe.execute()
            logger.info("Deleted chat %s", chat_id)
            return True
        except Exception:
            logger.exception("Failed to delete chat %s", chat_id)
            return False

