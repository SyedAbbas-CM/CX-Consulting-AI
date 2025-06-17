from __future__ import annotations

"""Utility functions for manipulating the workspace-root .env file.

Currently only supports simple key=value pairs – comments are preserved.
"""

import logging
from pathlib import Path
from typing import Dict

from dotenv import find_dotenv

logger = logging.getLogger("env_manager")


def _find_env_file(explicit_path: str | Path | None = None) -> Path | None:
    """Locate the .env file.

    Order of preference:
    1. `explicit_path` if provided
    2. `dotenv.find_dotenv()`
    3. Current working directory `.env`
    """
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        return p if p.exists() else None

    found = find_dotenv(usecwd=True)
    if found:
        return Path(found)

    cwd_env = Path.cwd() / ".env"
    return cwd_env if cwd_env.exists() else None


def update_env_vars(
    updates: Dict[str, str], env_path: str | Path | None = None
) -> bool:
    """Patch the .env file with the given key/value pairs.

    Existing keys are overwritten; new keys are appended.
    Returns True on success, False otherwise.
    """
    env_file = _find_env_file(env_path)
    if not env_file:
        # Attempt to create .env in CWD if not found
        try:
            env_file = Path.cwd() / ".env"
            env_file.touch(exist_ok=True)
            logger.info("Created new .env file at %s", env_file)
        except Exception as e:
            logger.error(".env file not found and could not create one: %s", e)
            return False

    try:
        lines: list[str] = env_file.read_text().splitlines(keepends=True)
    except Exception as e:
        logger.error("Failed reading %s: %s", env_file, e)
        return False

    keys_to_update = {k.upper(): v for k, v in updates.items()}
    updated_lines: list[str] = []
    handled = set()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            updated_lines.append(line)
            continue

        k, _ = stripped.split("=", 1)
        key_upper = k.upper()
        if key_upper in keys_to_update:
            new_val = keys_to_update[key_upper]
            updated_lines.append(f"{key_upper}={new_val}\n")
            handled.add(key_upper)
        else:
            updated_lines.append(line)

    # Append any keys not already present
    for k, v in keys_to_update.items():
        if k not in handled:
            updated_lines.append(f"{k}={v}\n")

    try:
        env_file.write_text("".join(updated_lines))
        logger.info("Updated %s with %d variables.", env_file, len(keys_to_update))
        return True
    except Exception as e:
        logger.error("Failed writing %s: %s", env_file, e)
        return False
