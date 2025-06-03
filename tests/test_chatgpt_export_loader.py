import os
import tempfile

import pytest

import apiana.batch.chatgpt.chatgpt_export_loader as loader


class TestChatGPTExportLoader:
    def test_load_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("")
            temp_path = f.name
        try:
            with pytest.raises(Exception):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json")
            temp_path = f.name
        try:
            with pytest.raises(Exception):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_non_list_structure(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"title": "Test"}')
            temp_path = f.name
        try:
            with pytest.raises(Exception):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_missing_required_fields(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"title": "Test"}')
            temp_path = f.name
        try:
            with pytest.raises(Exception):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)
