from apiana.llm.openai_client import LLMClient
from apiana.types.chatgpt_conversation import ChatGPTConversation


class SummaryGenerator:
    def generate(self, convo: ChatGPTConversation) -> str:
        llm_client = LLMClient(
            model="hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M",
            base_url="https://ollama.kairix.net",
            api_key="api-key",
            temperature=0.8,
            max_tokens=512,
            system_prompt="Summarize the following conversations:\n ",
        )

        llm_client.generate_text(f"this is a convo:\n {convo.to_json()}")
