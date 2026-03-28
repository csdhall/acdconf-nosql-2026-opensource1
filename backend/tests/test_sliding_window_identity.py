from __future__ import annotations

import unittest

from backend.strategies import sliding_window


class SlidingWindowIdentityTests(unittest.TestCase):
    def test_extract_identity_anchor_with_role_and_company(self) -> None:
        anchor = sliding_window._extract_identity_anchor(
            "Hi, I'm John Doe, lead developer at FakeCompany."
        )
        self.assertIsNotNone(anchor)
        assert anchor is not None
        self.assertIn("John Doe", anchor)
        self.assertIn("lead developer", anchor)
        self.assertIn("FakeCompany", anchor)

    def test_extract_identity_anchor_name_only(self) -> None:
        anchor = sliding_window._extract_identity_anchor("My name is Jordan Park")
        self.assertEqual(anchor, "User identity: Jordan Park.")


if __name__ == "__main__":
    unittest.main()
