from apiana.llm.openai_client import LLMClient


class SummaryGenerator:
    def generate(self, text: str) -> str:
        llm_client = LLMClient(
            model="gpt-4.1-mini",
            api_key="<KEY>",
            temperature=0.8,
            max_tokens=512,
            system_prompt="Summarize the following conversations:\n ",
        )

        return llm_client.generate_text(f"this is a convo:\n {text}")
