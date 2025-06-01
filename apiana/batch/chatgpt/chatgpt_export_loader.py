import json
import logging

from typing import List


from apiana.types.chatgpt_conversation import ChatGPTConversation
log = logging.getLogger(__name__)


class ChatGPTExportLoader:

    def load(self, file_path: str) -> List[ChatGPTConversation]:
        validated_convos = []
        failed_imports = 0

        try:
            log.info(f"🔍 Starting to process file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                conversations = json.load(file)

            if not isinstance(conversations, list):
                log.error("Invalid file format: 'conversations' is not a list")
                raise Exception("Invalid file format: 'conversations' is not a list")

            print(f"🔍 Found {len(conversations)} conversations to process")

            for i, convo_data in enumerate(conversations):
                try:
                    validated_convo = ChatGPTConversation.from_dict(convo_data)
                    if validated_convo:
                        validated_convos.append(validated_convo)
                        print(f"✅ Processed conversation {i}: '{validated_convo.title}' ({len(validated_convo.messages)} messages)")
                    else:
                        failed_imports += 1
                        print(f"⚠️  Skipped conversation {i}: no valid content")

                except Exception as e:
                    failed_imports += 1
                    log.error(f"Failed to process conversation {i}: {e}")

        except Exception as e:
                error_msg = f"Unexpected error processing file {file_path}: {e}"
                log.error(error_msg)
                raise Exception(error_msg, e)

        return validated_convos