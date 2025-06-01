from dataclasses import dataclass


@dataclass
class PromptConfig:
    name: str
    system_prompt_file: str
    user_prompt_template_file: str


@dataclass
class InferenceProviderConfig:
    base_url: str
    api_key: str


@dataclass
class TextGenerationModelConfig:
    model_name: str
    temperature: float
    max_tokens: int

    prompt_config: PromptConfig
    inference_provider_config: InferenceProviderConfig
