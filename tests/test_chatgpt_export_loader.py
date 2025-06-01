import os
import tempfile

import pytest

from apiana.batch.chatgpt.chatgpt_export_loader import ChatGPTExportLoader


class TestChatGPTExportLoader:
    @pytest.fixture
    def loader(self):
        return ChatGPTExportLoader()

    def test_load_empty_file(self, loader):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("")
            temp_path = f.name
        try:
            with pytest.raises(Exception):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_invalid_json(self, loader):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json")
            temp_path = f.name
        try:
            with pytest.raises(Exception):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_non_list_structure(self, loader):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{\"title\": \"Test\"}")
            temp_path = f.name
        try:
            with pytest.raises(Exception):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_missing_required_fields(self, loader):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{\"title\": \"Test\"}")
            temp_path = f.name
        try:
            with pytest.raises(Exception):
                loader.load(temp_path)
        finally:
            os.unlink(temp_path)