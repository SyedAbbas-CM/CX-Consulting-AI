import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis.asyncio as redis

logger = logging.getLogger("cx_consulting_ai.chat_service")

# Define TTL in seconds (e.g., 60 days)
CHAT_TTL_SECONDS = 60 * 60 * 24 * 60


class ChatService:
    """
    Manages chat sessions and history in Redis using asyncio.
    ───────────────────────────────────────────
    Key layout (all helper methods are centralised, **use them everywhere**):

        chat:{chat_id}:metadata      – Hash   (project_id, name, timestamps…)
        chat:{chat_id}:messages      – List   (JSON strings, oldest → left/0, newest → right/-1)
        project:{project_id}:chats   – Set    (chat_id, chat_id, …)
    """

    # ---------- construction & helpers --------------------------------------------------

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        max_history_length: int = 1000,
    ):
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.max_history_length = max_history_length
            # PRODUCTION OPTIMIZED: Reduce logging noise
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    f"ChatService initialized with Redis @ {redis_url} (async client) and max history length {self.max_history_length}"
                )
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during ChatService Redis initialization: {e}",
                exc_info=True,
            )
            raise

    # helper key builders
    def _meta_key(self, chat_id: str) -> str:
        return f"chat:{chat_id}:metadata"

    def _msg_key(self, chat_id: str) -> str:
        return f"chat:{chat_id}:messages"

    def _project_set_key(self, project_id: str) -> str:
        return f"project:{project_id}:chats"

    # ---------- main API methods -------------------------------------------------------

    async def create_chat(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        chat_name: Optional[str] = None,
        journey_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        chat_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        if not chat_name:
            chat_name = f"Chat {chat_id[:8]}"

        metadata = {
            "chat_id": chat_id,
            "name": chat_name,
            "project_id": project_id or "",
            "user_id": user_id,
            "created_at": now_iso,
            "last_updated": now_iso,
        }
        if journey_type:
            metadata["journey_type"] = journey_type

        try:
            async with self.redis_client.pipeline(transaction=True) as pipe:
                meta_key = self._meta_key(chat_id)
                pipe.hset(meta_key, mapping=metadata)
                pipe.expire(meta_key, CHAT_TTL_SECONDS)

                msg_key = self._msg_key(chat_id)
                pipe.expire(msg_key, CHAT_TTL_SECONDS)

                if project_id:
                    project_key = self._project_set_key(project_id)
                    pipe.sadd(project_key, chat_id)
                    pipe.expire(project_key, CHAT_TTL_SECONDS)

                await pipe.execute()

            # PRODUCTION OPTIMIZED: Only log in debug mode
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Created chat {chat_id} for user {user_id} in project {project_id}"
                )

            return metadata
        except Exception as e:
            logger.error(
                f"Failed to create chat for user {user_id}: {e}", exc_info=True
            )
            raise

    async def list_chats_for_project(
        self, project_id: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        chat_ids = (
            list(await self.redis_client.smembers(self._project_set_key(project_id)))
            or []
        )
        paginated = chat_ids[offset : offset + limit]
        summaries: List[Dict[str, Any]] = []

        if not paginated:
            return summaries

        async with self.redis_client.pipeline() as pipe:
            for cid in paginated:
                pipe.hgetall(self._meta_key(cid))
            metas = await pipe.execute()

        for cid, meta in zip(paginated, metas):
            if not meta:
                # PRODUCTION OPTIMIZED: Only log warnings in debug mode
                if logger.isEnabledFor(logging.DEBUG):
                    logger.warning(
                        "Missing metadata for chat %s in project %s", cid, project_id
                    )
                continue
            # Ensure essential fields are present, providing defaults if reasonable
            summary = {
                "chat_id": cid,  # This is the definitive ID from the project's set
                "name": meta.get("name", f"Chat {cid[:8]}"),  # Default name if missing
                "project_id": meta.get("project_id"),  # Can be None if not set in meta
                "user_id": meta.get("user_id"),  # Can be None
                "created_at": meta.get(
                    "created_at", datetime.now(timezone.utc).isoformat()
                ),  # Default to now if missing
                "last_updated": meta.get(
                    "last_updated",
                    meta.get("created_at", datetime.now(timezone.utc).isoformat()),
                ),  # Default to created_at or now
            }
            summaries.append(summary)
        return summaries

    async def get_chat_history(
        self, chat_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        meta_key = self._meta_key(chat_id)
        msg_key = self._msg_key(chat_id)

        # Atomically check existence and refresh TTL if exists
        async with self.redis_client.pipeline(transaction=True) as pipe:
            pipe.exists(meta_key)
            pipe.expire(meta_key, CHAT_TTL_SECONDS)  # Refresh TTL on read
            pipe.expire(msg_key, CHAT_TTL_SECONDS)  # Refresh TTL on read
            results = await pipe.execute()

        if not results[0]:  # results[0] is the output of pipe.exists(meta_key)
            # PRODUCTION OPTIMIZED: Only log warnings in debug mode
            if logger.isEnabledFor(logging.DEBUG):
                logger.warning(
                    "Request for history of non-existent chat %s (or TTL expired)",
                    chat_id,
                )
            return []

        # Fetch project_id from metadata to refresh project set TTL if possible
        # This requires an extra hget call or for get_chat_summary to be used first if we always want to refresh project set TTL
        meta = await self.get_chat_summary(
            chat_id
        )  # This will also refresh meta_key and msg_key TTLs again
        if meta and meta.get("project_id"):
            project_key = self._project_set_key(meta.get("project_id"))
            await self.redis_client.expire(project_key, CHAT_TTL_SECONDS)

        raw_messages = await self.redis_client.lrange(msg_key, -limit, -1)
        messages: List[Dict[str, Any]] = []
        for raw in raw_messages:
            try:
                messages.append(json.loads(raw))
            except Exception:
                # PRODUCTION OPTIMIZED: Only log exceptions in debug mode
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception("Corrupt message in chat %s: %s", chat_id, raw)
        return messages

    async def add_message_to_chat(
        self,
        chat_id: str,
        *,
        role: str,
        content: str,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        if role not in {"user", "assistant"}:
            logger.error("Invalid role '%s'", role)
            return False
        if not await self.redis_client.exists(self._meta_key(chat_id)):
            logger.warning("Attempt to write to non-existent chat %s", chat_id)
            return False

        now_iso = datetime.now(timezone.utc).isoformat()
        message_payload = {"role": role, "content": content, "timestamp": now_iso}
        if user_id:
            message_payload["user_id"] = user_id
        if project_id:
            message_payload["project_id"] = project_id

        msg_json = json.dumps(message_payload)
        try:
            async with self.redis_client.pipeline(transaction=True) as pipe:
                meta_key = self._meta_key(chat_id)
                msg_key = self._msg_key(chat_id)

                pipe.rpush(msg_key, msg_json)
                pipe.ltrim(msg_key, -self.max_history_length, -1)
                pipe.hset(meta_key, "last_updated", now_iso)

                # Refresh TTLs on write
                pipe.expire(meta_key, CHAT_TTL_SECONDS)
                pipe.expire(msg_key, CHAT_TTL_SECONDS)

                # If project_id is available (it is an arg to this function now)
                # We should also refresh the project's set of chats TTL
                if project_id:
                    project_key = self._project_set_key(project_id)
                    pipe.expire(project_key, CHAT_TTL_SECONDS)
                elif not project_id:
                    # Attempt to get project_id from metadata if not passed directly
                    # This adds an extra read, so it's better if project_id is passed consistently
                    current_meta = await self.redis_client.hgetall(meta_key)
                    pid_from_meta = current_meta.get("project_id")
                    if pid_from_meta:
                        project_key = self._project_set_key(pid_from_meta)
                        pipe.expire(project_key, CHAT_TTL_SECONDS)

                await pipe.execute()
            return True
        except Exception as e:
            logger.exception("Failed to add message to chat %s: %s", chat_id, e)
            return False

    async def get_chat_summary(self, chat_id: str) -> Optional[Dict[str, Any]]:
        meta_key = self._meta_key(chat_id)
        msg_key = self._msg_key(
            chat_id
        )  # For consistency, refresh message TTL too if accessing summary

        async with self.redis_client.pipeline(transaction=True) as pipe:
            pipe.hgetall(meta_key)
            pipe.expire(meta_key, CHAT_TTL_SECONDS)  # Refresh TTL on read
            pipe.expire(msg_key, CHAT_TTL_SECONDS)  # Refresh TTL on read
            results = await pipe.execute()

        meta = results[0]

        if meta and meta.get("project_id"):
            project_key = self._project_set_key(meta.get("project_id"))
            await self.redis_client.expire(project_key, CHAT_TTL_SECONDS)

        return meta or None

    async def delete_chat(self, chat_id: str) -> bool:
        meta = await self.redis_client.hgetall(self._meta_key(chat_id))
        if not meta:
            logger.warning("Delete requested for missing chat %s", chat_id)
            return False
        project_id = meta.get("project_id")
        try:
            async with self.redis_client.pipeline(transaction=True) as pipe:
                pipe.delete(self._meta_key(chat_id))
                pipe.delete(self._msg_key(chat_id))
                if project_id:
                    pipe.srem(self._project_set_key(project_id), chat_id)
                await pipe.execute()
            logger.info("Deleted chat %s", chat_id)
            return True
        except Exception as e:
            logger.exception("Failed to delete chat %s: %s", chat_id, e)
            return False
