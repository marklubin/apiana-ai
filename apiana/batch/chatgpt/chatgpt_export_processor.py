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

    def extract_convos_with_persist(self, input_file: str, output_dir: str ):
        with open(input_file, "r") as f:
            convos = self.loader.load(input_file)
            os.makedirs(output_dir, exist_ok=True)
            for c in convos:
                output_path = os.path.join(output_dir, f"{c.title}.json")
                output = c.to_json(indent=2)
                with open(output_path, "w") as f:
                    f.write(output)



