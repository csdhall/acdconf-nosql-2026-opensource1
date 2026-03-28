from __future__ import annotations

import json
import unittest
from pathlib import Path


class SeedDataTests(unittest.TestCase):
    def test_default_seed_messages_are_public_and_complete(self) -> None:
        seed_path = Path(__file__).resolve().parents[2] / "scripts" / "default_seed_messages.json"
        payload = json.loads(seed_path.read_text(encoding="utf-8"))
        messages = payload.get("messages")

        self.assertIsInstance(messages, list)
        self.assertEqual(len(messages), 60)
        self.assertTrue(all(isinstance(message, str) and message.strip() for message in messages))

        flattened = "\n".join(messages).lower()
        self.assertNotIn("cazton.com", flattened)
        self.assertNotIn("github.com/cazton", flattened)


if __name__ == "__main__":
    unittest.main()
