import genai_core.clients

from langchain_aws import BedrockLLM
from langchain.prompts.prompt import PromptTemplate

from ..base import ModelAdapter
from .base import get_guardrails
from genai_core.registry import registry


class BedrockCohereCommandAdapter(ModelAdapter):
    def __init__(self, model_id, *args, **kwargs):
        self.model_id = model_id

        super().__init__(*args, **kwargs)

    def get_llm(self, model_kwargs={}):
        bedrock = genai_core.clients.get_bedrock_client()

        params = {}
        if "temperature" in model_kwargs:
            params["temperature"] = model_kwargs["temperature"]
        if "maxTokens" in model_kwargs:
            params["max_tokens"] = model_kwargs["maxTokens"]
        params["return_likelihoods"] = "GENERATION"

        extra = {}
        guardrails = get_guardrails()
        if len(guardrails.keys()) > 0:
            extra = {"guardrails": guardrails}
        return BedrockLLM(
            client=bedrock,
            model_id=self.model_id,
            model_kwargs=params,
            streaming=model_kwargs.get("streaming", False),
            callbacks=[self.callback_handler],
            **extra
        )

    def get_prompt(self):
        template = """

Human: Ce qui suit est une conversation amicale entre un humain et une IA. Si l'IA ne connaît pas la réponse à une question, elle dit sincèrement qu'elle ne sait pas. De plus, l'IA répondra en Français et l'IA donnera toutes les références associées à chaque réponse.

Current conversation:
{chat_history}

Question: {input}

Assistant:"""

        input_variables = ["input", "chat_history"]
        prompt_template_args = {
            "chat_history": "{chat_history}",
            "input_variables": input_variables,
            "template": template,
        }
        prompt_template = PromptTemplate(**prompt_template_args)

        return prompt_template


# Register the adapter
registry.register(
    r"^bedrock\.cohere\.command-(text|light-text).*", BedrockCohereCommandAdapter
)
