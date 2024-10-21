import os
from typing import Any, List

from ..base import ModelAdapter
from genai_core.registry import registry
import genai_core.clients

from aws_lambda_powertools import Logger

from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_aws import ChatBedrockConverse
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.prompt import PromptTemplate

logger = Logger()


def get_guardrails() -> dict:
    if "BEDROCK_GUARDRAILS_ID" in os.environ:
        return {
            "guardrailIdentifier": os.environ["BEDROCK_GUARDRAILS_ID"],
            "guardrailVersion": os.environ.get("BEDROCK_GUARDRAILS_VERSION", "DRAFT"),
        }
    return {}


class BedrockChatAdapter(ModelAdapter):
    def __init__(self, model_id, *args, **kwargs):
        self.model_id = model_id

        super().__init__(*args, **kwargs)

    def get_qa_prompt(self):
        system_prompt = (
            "Vous êtes un assistant IA utilisant la Génération Augmentée par Récupération (RAG). Répondez aux questions de l'utilisateur uniquement en vous basant sur  les informations contenues dans les documents fournis. N'ajoutez aucune information supplémentaire et ne faites aucune supposition qui ne soit pas directement soutenue par ces documents. Si vous ne trouvez pas la réponse dans les documents, informez l'utilisateur que l'information n'est pas disponible. Si possible, dressez la liste des documents référencés. \n\n{context}"
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def get_prompt(self):
        prompt_template = ChatPromptTemplate(
            [
                (
                    "system",
                    (
                        "The following is a friendly conversation between "
                        "a human and an AI."
                        "If the AI does not know the answer to a question, it "
                        "truthfully says it does not know."
                    ),
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        return prompt_template

    def get_condense_question_prompt(self):
        contextualize_q_system_prompt = (
            "Vous êtes un assistant IA capable de répondre aux questions en fonction de vos connaissances préalables. Répondez aux questions de l'utilisateur uniquement avec des informations que vous connaissez déjà. N'ajoutez aucune information non vérifiée ou spéculative. Si vous ne connaissez pas la réponse à une question, informez l'utilisateur que vous n'avez pas suffisamment d'informations pour répondre. Si possible, dressez la liste des documents référencés."
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def get_llm(self, model_kwargs={}, extra={}):
        bedrock = genai_core.clients.get_bedrock_client()
        params = {}
        if "temperature" in model_kwargs:
            params["temperature"] = model_kwargs["temperature"]
        if "topP" in model_kwargs:
            params["top_p"] = model_kwargs["topP"]
        if "maxTokens" in model_kwargs:
            params["max_tokens"] = model_kwargs["maxTokens"]

        guardrails = get_guardrails()
        if len(guardrails.keys()) > 0:
            params["guardrails"] = guardrails

        return ChatBedrockConverse(
            client=bedrock,
            model=self.model_id,
            disable_streaming=model_kwargs.get("streaming", False) == False
            or self.disable_streaming,
            callbacks=[self.callback_handler],
            **params,
            **extra,
        )


class BedrockChatNoStreamingAdapter(BedrockChatAdapter):
    """Some models do not support system streaming using the converse API"""

    def __init__(self, *args, **kwargs):
        super().__init__(disable_streaming=True, *args, **kwargs)


class BedrockChatNoSystemPromptAdapter(BedrockChatAdapter):
    """Some models do not support system and message history in the conversion API"""

    def get_prompt(self):
        template = """Vous êtes un assistant IA utilisant la Génération Augmentée par Récupération (RAG). Répondez aux questions de l'utilisateur uniquement en vous basant sur  les informations contenues dans les documents fournis. N'ajoutez aucune information supplémentaire et ne faites aucune supposition qui ne soit pas directement soutenue par ces documents. Si vous ne trouvez pas la réponse dans les documents, informez l'utilisateur que l'information n'est pas disponible. Si possible, dressez la liste des documents référencés.


Conversation en cours:
{chat_history}

La question: {input}

Assistant:"""  # noqa: E501
        return PromptTemplateWithHistory(
            template=template, input_variables=["input", "chat_history"]
        )

    def get_condense_question_prompt(self):
        template = """A partir de la conversation suivante et d'une question de suivi, reformulez la question de suivi pour en faire une question indépendante, dans sa langue d'origine.

Historique du chat:
{chat_history}
Suivi des entrées: {input}
Question isolée:"""  # noqa: E501
        return PromptTemplateWithHistory(
            template=template, input_variables=["input", "chat_history"]
        )

    def get_qa_prompt(self):
        template = """Utilisez les éléments de contexte suivants pour répondre à la question finale. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.

{context}

Question: {input}
Réponse utile:"""  # noqa: E501
        return PromptTemplateWithHistory(
            template=template, input_variables=["input", "content"]
        )


class BedrockChatNoStreamingNoSystemPromptAdapter(BedrockChatNoSystemPromptAdapter):
    """Some models do not support system streaming using the converse API"""

    def __init__(self, *args, **kwargs):
        super().__init__(disable_streaming=True, *args, **kwargs)


# Register the adapters
registry.register(r"^bedrock.ai21.jamba*", BedrockChatAdapter)
registry.register(r"^bedrock.ai21.j2*", BedrockChatNoStreamingNoSystemPromptAdapter)
registry.register(
    r"^bedrock\.cohere\.command-(text|light-text).*", BedrockChatNoSystemPromptAdapter
)
registry.register(r"^bedrock\.cohere\.command-r.*", BedrockChatAdapter)
registry.register(r"^bedrock.anthropic.claude*", BedrockChatAdapter)
registry.register(
    r"^bedrock.meta.llama*",
    BedrockChatAdapter,
)
registry.register(
    r"^bedrock.mistral.mistral-large*",
    BedrockChatAdapter,
)
registry.register(
    r"^bedrock.mistral.mistral-small*",
    BedrockChatAdapter,
)
registry.register(
    r"^bedrock.mistral.mistral-7b-*",
    BedrockChatNoSystemPromptAdapter,
)
registry.register(
    r"^bedrock.mistral.mixtral-*",
    BedrockChatNoSystemPromptAdapter,
)
registry.register(r"^bedrock.amazon.titan-t*", BedrockChatNoSystemPromptAdapter)


class PromptTemplateWithHistory(PromptTemplate):
    def format(self, **kwargs: Any) -> str:
        chat_history = kwargs["chat_history"]
        if isinstance(chat_history, List):
            # RunnableWithMessageHistory is provided a list of BaseMessage as a history
            # Since this model does not support history, we format the common prompt to
            # list the history
            chat_history_str = ""
            for message in chat_history:
                if isinstance(message, BaseMessage):
                    prefix = ""
                    if isinstance(message, AIMessage):
                        prefix = "AI: "
                    elif isinstance(message, HumanMessage):
                        prefix = "Human: "
                    chat_history_str += prefix + message.content + "\n"
            kwargs["chat_history"] = chat_history_str
        return super().format(**kwargs)
