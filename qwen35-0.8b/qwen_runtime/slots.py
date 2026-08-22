"""Runtime-owned resident Session slots."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from .binding import Engine, Session


class SlotBusy(RuntimeError):
    pass


@dataclass
class Slot:
    id: int
    session: Session
    owner: Optional[str] = None
    busy: bool = False
    last_used: float = 0.0


class SlotPool:
    """
    Preallocated resident sessions.

    owner is the runtime session id. FREE means owner=None; IDLE means an owner
    exists but busy=False; BUSY means one request currently mutates the Session.
    """

    def __init__(self, engine: Engine, slot_count: int, context_size: int):
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        self._slots = [
            Slot(index, engine.create_session(context_size))
            for index in range(slot_count)
        ]
        self._owners: dict[str, Slot] = {}
        self._lock = asyncio.Lock()

    @property
    def slots(self) -> tuple[Slot, ...]:
        return tuple(self._slots)

    @asynccontextmanager
    async def acquire(self, owner: str, *, persistent: bool = True):
        """Acquire an existing owner or assign a FREE/LRU-IDLE slot."""
        async with self._lock:
            slot = self._owners.get(owner)
            if slot is not None and slot.busy:
                raise SlotBusy(f"session {owner!r} already has a running request")
            if slot is None:
                slot = next((item for item in self._slots if item.owner is None), None)
                if slot is None:
                    idle = [item for item in self._slots if not item.busy]
                    if not idle:
                        raise SlotBusy("all session slots are busy")
                    slot = min(idle, key=lambda item: item.last_used)
                    if slot.owner is not None:
                        self._owners.pop(slot.owner, None)
                    await asyncio.to_thread(slot.session.reset)
                slot.owner = owner
                self._owners[owner] = slot
            slot.busy = True

        failed = False
        try:
            yield slot
        except BaseException:
            failed = True
            raise
        finally:
            async with self._lock:
                slot.busy = False
                slot.last_used = time.monotonic()
                # Ephemeral requests and failed timelines must not remain addressable.
                if not persistent or failed:
                    if slot.owner is not None:
                        self._owners.pop(slot.owner, None)
                    slot.owner = None
                    await asyncio.to_thread(slot.session.reset)

    async def forget(self, owner: str) -> bool:
        async with self._lock:
            slot = self._owners.get(owner)
            if slot is None:
                return False
            if slot.busy:
                raise SlotBusy(f"session {owner!r} is busy")
            self._owners.pop(owner)
            slot.owner = None
            await asyncio.to_thread(slot.session.reset)
            return True

    async def close(self) -> None:
        async with self._lock:
            if any(slot.busy for slot in self._slots):
                raise SlotBusy("cannot close SlotPool while a request is running")
            for slot in self._slots:
                slot.session.close()
            self._owners.clear()
            self._slots.clear()
