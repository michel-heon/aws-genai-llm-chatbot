import json
import os

from langchain_community.llms.sagemaker_endpoint import (
    LLMContentHandler,
    SagemakerEndpoint,
)
from langchain.prompts.prompt import PromptTemplate

from ...base import ModelAdapter
from genai_core.registry import registry


class MistralInstructContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt, model_kwargs) -> bytes:
        input_str = json.dumps(
            {
                "inputs": prompt,
                "parameters": {
                    "do_sample": True,
                    "max_new_tokens": model_kwargs.get("max_new_tokens", 512),
                    "top_p": model_kwargs.get("top_p", 0.9),
                    "temperature": model_kwargs.get("temperature", 0.6),
                    "return_full_text": False,
                    "stop": ["###", "</s>"],
                },
            }
        )
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes):
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"]


content_handler = MistralInstructContentHandler()


class SMMistralInstructAdapter(ModelAdapter):
    def __init__(self, model_id, **kwargs):
        self.model_id = model_id

        super().__init__(**kwargs)

    def get_llm(self, model_kwargs={}):
        params = {}
        if "temperature" in model_kwargs:
            params["temperature"] = model_kwargs["temperature"]
        if "topP" in model_kwargs:
            params["top_p"] = model_kwargs["topP"]
        if "maxTokens" in model_kwargs:
            params["max_new_tokens"] = model_kwargs["maxTokens"]

        return SagemakerEndpoint(
            endpoint_name=self.get_endpoint(self.model_id),
            region_name=os.environ["AWS_REGION"],
            content_handler=content_handler,
            model_kwargs=params,
            callbacks=[self.callback_handler],
        )

    def get_qa_prompt(self):
        template = """<s>[INST] Ce qui suit est une conversation amicale entre un humain et une IA. Si l'IA ne connaît pas la réponse à une question, elle dit sincèrement qu'elle ne sait pas. De plus, l'IA répondra en Français et l'IA donnera toutes les références associées à chaque réponse.[/INST]

{context}
</s>[INST] {question} [/INST]"""  # noqa: E501

        return PromptTemplate.from_template(template)

    def get_prompt(self):
        template = """<s>[INST] Ce qui suit est une conversation amicale entre un humain et une IA. Si l'IA ne connaît pas la réponse à une question, elle dit sincèrement qu'elle ne sait pas. De plus, l'IA répondra en Français et l'IA donnera toutes les références associées à chaque réponse.[/INST]

{chat_history}
<s>[INST] {input} [/INST]"""  # noqa: E501

        return PromptTemplate.from_template(template)

    def get_condense_question_prompt(self):
        template = """<s>[INST] Ce qui suit est une conversation amicale entre un humain et une IA. Si l'IA ne connaît pas la réponse à une question, elle dit sincèrement qu'elle ne sait pas. De plus, l'IA répondra en Français et l'IA donnera toutes les références associées à chaque réponse.[/INST]

{chat_history}
</s>[INST] {question} [/INST]"""  # noqa: E501

        return PromptTemplate.from_template(template)


# Register the adapter
registry.register(r"(?i)sagemaker\.mistralai-Mistral*", SMMistralInstructAdapter)
registry.register(r"(?i)sagemaker\.mistralai/Mistral*", SMMistralInstructAdapter)
