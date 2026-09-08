import sys
import types
import unittest
from unittest.mock import patch


# database.py chỉ cần các symbol này ở import time; test không gọi Supabase SDK.
fake_supabase_module = types.ModuleType("supabase")
fake_supabase_module.create_client = lambda *args, **kwargs: None
fake_supabase_module.Client = object
sys.modules.setdefault("supabase", fake_supabase_module)

from backend.app import database


class FakeQuery:
    def __init__(self, data):
        self.data = data
        self.order_args = None
        self.limit_value = None

    def table(self, _name):
        return self

    def select(self, _fields):
        return self

    def eq(self, _field, _value):
        return self

    def order(self, field, desc=False):
        self.order_args = (field, desc)
        return self

    def limit(self, value):
        self.limit_value = value
        return self

    def execute(self):
        return types.SimpleNamespace(data=self.data)


class RecentMessageTests(unittest.TestCase):
    def test_reads_latest_messages_then_restores_chronological_order(self):
        newest_first = [
            {"role": "assistant", "content": "newest"},
            {"role": "user", "content": "older"},
        ]
        query = FakeQuery(newest_first)

        with patch.object(database, "get_supabase", return_value=query):
            messages = database.get_recent_messages_for_context("session-1", last_n=2)

        self.assertEqual(query.order_args, ("created_at", True))
        self.assertEqual(query.limit_value, 2)
        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "older"},
                {"role": "assistant", "content": "newest"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
