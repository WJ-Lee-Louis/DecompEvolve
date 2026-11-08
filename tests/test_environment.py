import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from decompevolve.environment import load_environment


class EnvironmentTests(unittest.TestCase):
    def test_loads_quoted_key_without_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('OPENAI_API_KEY="fake-${UNCHANGED}-value"\n', encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                load_environment(path)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "fake-${UNCHANGED}-value")

    def test_existing_environment_takes_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OPENAI_API_KEY=fake-file-value\n", encoding="utf-8")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-shell-value"}, clear=True):
                load_environment(path)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "fake-shell-value")

    def test_missing_file_is_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True):
                load_environment(Path(directory) / "absent.env")
                self.assertNotIn("OPENAI_API_KEY", os.environ)
