from __future__ import annotations

import unittest

from backend.strategies import entity_graph


class EntityGraphHintTests(unittest.TestCase):
    def test_extracts_pilot_customer_hints(self) -> None:
        hints = entity_graph._extract_hint_entities_from_user_message(
            "The pilot customers are: FakeShopOne, FakeShopTwo, and FakeShopThree."
        )
        names = {h.get("name") for h in hints}
        self.assertIn("FakeShopOne", names)
        self.assertIn("FakeShopTwo", names)
        self.assertIn("FakeShopThree", names)

        for name in ["FakeShopOne", "FakeShopTwo", "FakeShopThree"]:
            match = next(h for h in hints if h.get("name") == name)
            facts = [str(x).lower() for x in match.get("facts", [])]
            self.assertIn("pilot customer", facts)

    def test_extracts_identity_role_company_hints(self) -> None:
        hints = entity_graph._extract_hint_entities_from_user_message(
            "Hi, I'm John Doe, lead developer at FakeCompany."
        )
        person = next(h for h in hints if h.get("name") == "John Doe")
        facts = [str(x).lower() for x in person.get("facts", [])]

        self.assertTrue(any("lead developer" in f for f in facts))
        self.assertTrue(any("works at fakecompany" in f for f in facts))


if __name__ == "__main__":
    unittest.main()
