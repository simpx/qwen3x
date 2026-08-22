import unittest

from qwen_runtime.slots import SlotBusy, SlotPool


class FakeSession:
    def __init__(self, index):
        self.index = index
        self.resets = 0
        self.closed = False

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


class FakeEngine:
    def __init__(self):
        self.sessions = []

    def create_session(self, _context_size):
        session = FakeSession(len(self.sessions))
        self.sessions.append(session)
        return session


class SlotPoolTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = FakeEngine()
        self.pool = SlotPool(self.engine, slot_count=2, context_size=128)

    async def asyncTearDown(self):
        await self.pool.close()

    async def test_owner_returns_to_same_resident_session(self):
        async with self.pool.acquire("agent-a") as first:
            first_id = first.id
        async with self.pool.acquire("agent-a") as second:
            self.assertEqual(second.id, first_id)
            self.assertEqual(second.session.resets, 0)

    async def test_lru_idle_slot_is_reset_before_new_owner(self):
        async with self.pool.acquire("agent-a") as first:
            first_id = first.id
        async with self.pool.acquire("agent-b"):
            pass
        async with self.pool.acquire("agent-c") as reused:
            self.assertEqual(reused.id, first_id)
            self.assertEqual(reused.session.resets, 1)

    async def test_same_owner_cannot_have_two_writers(self):
        lease = self.pool.acquire("agent-a")
        await lease.__aenter__()
        try:
            with self.assertRaises(SlotBusy):
                async with self.pool.acquire("agent-a"):
                    pass
        finally:
            await lease.__aexit__(None, None, None)

    async def test_ephemeral_and_failed_sessions_are_forgotten(self):
        async with self.pool.acquire("temporary", persistent=False) as slot:
            session = slot.session
        self.assertEqual(session.resets, 1)
        with self.assertRaises(RuntimeError):
            async with self.pool.acquire("broken") as slot:
                broken = slot.session
                raise RuntimeError("boom")
        self.assertGreaterEqual(broken.resets, 1)


if __name__ == "__main__":
    unittest.main()
