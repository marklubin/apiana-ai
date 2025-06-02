"""
Idempotent operations for processing individual conversations.
Split into two operations: summary generation and enrichment/storage.
"""

import logging
import os.path

from apiana.batch.chatgpt.chatgpt_export_loader import ChatGPTExportLoader


logger = logging.getLogger(__name__)


class ChatGPTExportProcessor:
    def __init__(self, loader: ChatGPTExportLoader):
        self.loader = loader

    def extract_convos_with_persist(self, input_file: str, output_dir: str):
        with open(input_file, "r") as f:
            convos = self.loader.load(input_file)
            os.makedirs(output_dir, exist_ok=True)
            for i, c in enumerate(convos):
                output_file_name = f"{i}_{c.title.replace(' ', '_').lower()}.json"
                output_path = os.path.join(output_dir, output_file_name)
                output = c.to_json(indent=2)
                with open(output_path, "w") as f:
                    f.write(output)
