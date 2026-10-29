import os
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

from openai import AsyncOpenAI, APIConnectionError, RateLimitError, APIError
from anthropic import AsyncAnthropic, APIConnectionError as AnthropicConnectionError, RateLimitError as AnthropicRateLimitError, APIError as AnthropicAPIError

from schemas import ChatMessage, ModelConfig, ModelResponse


class BaseLLMClient(ABC):
    @abstractmethod
    async def generate(self, messages: list[ChatMessage], config: Optional[ModelConfig] = None) -> ModelResponse:
        pass

    @abstractmethod
    async def generate_stream(self, messages: list[ChatMessage], config: Optional[ModelConfig] = None) -> AsyncGenerator[str, None]:
        pass


class OpenAIClient(BaseLLMClient):
    """
    Cliente universal compatible con OpenAI, Groq, OpenRouter, DeepSeek y cualquier
    proveedor compatible con la API de OpenAI mediante OPENAI_BASE_URL.
    """
    def __init__(self, api_key: str | None = None, base_url: str | None = None, config: Optional[ModelConfig] = None):
        key = api_key or (getattr(config, 'api_key', None) if config else None) or os.getenv("OPENAI_API_KEY")
        url = base_url or os.getenv("OPENAI_BASE_URL")
        self.default_config = config
        
        if not key:
            raise ValueError("OPENAI_API_KEY no encontrada en las variables de entorno.")

        # Si hay un base_url configurado (Groq, OpenRouter, DeepSeek), redirigimos la conexión
        if url:
            self.client = AsyncOpenAI(api_key=key, base_url=url)
        else:
            self.client = AsyncOpenAI(api_key=key)

    async def generate(self, messages: list[ChatMessage], config: Optional[ModelConfig] = None) -> ModelResponse:
        cfg = config or self.default_config
        model_name = getattr(cfg, 'model_name', getattr(cfg, 'model', 'openai/gpt-oss-120b')) if cfg else 'openai/gpt-oss-120b'
        temp = getattr(cfg, 'temperature', 0.7) if cfg else 0.7
        max_toks = getattr(cfg, 'max_tokens', 1000) if cfg else 1000

        try:
            formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=formatted_messages,
                temperature=temp,
                max_tokens=max_toks,
            )
            choice = response.choices[0].message
            usage = response.usage
            
            return ModelResponse(
                content=choice.content or "",
                provider="Groq/OpenAI",
                model=model_name,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
            )
        except RateLimitError:
            return ModelResponse(content="[Error] Límite de tasa excedido.", provider="OpenAI", model=model_name)
        except APIConnectionError:
            return ModelResponse(content="[Error] Falla de conexión con el servidor.", provider="OpenAI", model=model_name)
        except APIError as e:
            return ModelResponse(content=f"[Error API]: {str(e)}", provider="OpenAI", model=model_name)

    async def generate_stream(self, messages: list[ChatMessage], config: Optional[ModelConfig] = None) -> AsyncGenerator[str, None]:
        cfg = config or self.default_config
        model_name = getattr(cfg, 'model_name', getattr(cfg, 'model', 'openai/gpt-oss-120b')) if cfg else 'openai/gpt-oss-120b'
        temp = getattr(cfg, 'temperature', 0.7) if cfg else 0.7
        max_toks = getattr(cfg, 'max_tokens', 1000) if cfg else 1000

        try:
            formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
            stream = await self.client.chat.completions.create(
                model=model_name,
                messages=formatted_messages,
                temperature=temp,
                max_tokens=max_toks,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except (RateLimitError, APIConnectionError, APIError) as e:
            yield f"\n[Error en streaming: {str(e)}]"


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str | None = None, config: Optional[ModelConfig] = None):
        key = api_key or (getattr(config, 'api_key', None) if config else None) or os.getenv("ANTHROPIC_API_KEY")
        self.default_config = config
        if not key:
            raise ValueError("ANTHROPIC_API_KEY no encontrada.")
        self.client = AsyncAnthropic(api_key=key)

    async def generate(self, messages: list[ChatMessage], config: Optional[ModelConfig] = None) -> ModelResponse:
        cfg = config or self.default_config
        model_name = getattr(cfg, 'model_name', getattr(cfg, 'model', 'claude-3-5-haiku-20241022')) if cfg else 'claude-3-5-haiku-20241022'
        max_toks = getattr(cfg, 'max_tokens', 1000) if cfg else 1000

        try:
            system_prompt = next((m.content for m in messages if m.role == "system"), None)
            user_messages = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]

            kwargs = {
                "model": model_name,
                "max_tokens": max_toks,
                "messages": user_messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = await self.client.messages.create(**kwargs)
            text_content = response.content[0].text if response.content else ""
            
            return ModelResponse(
                content=text_content,
                provider="Anthropic",
                model=model_name,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )
        except AnthropicRateLimitError:
            return ModelResponse(content="[Error] Límite de tasa excedido en Anthropic.", provider="Anthropic", model=model_name)
        except AnthropicConnectionError:
            return ModelResponse(content="[Error] Falla de conexión con Anthropic.", provider="Anthropic", model=model_name)
        except AnthropicAPIError as e:
            return ModelResponse(content=f"[Error API Anthropic]: {str(e)}", provider="Anthropic", model=model_name)

    async def generate_stream(self, messages: list[ChatMessage], config: Optional[ModelConfig] = None) -> AsyncGenerator[str, None]:
        cfg = config or self.default_config
        model_name = getattr(cfg, 'model_name', getattr(cfg, 'model', 'claude-3-5-haiku-20241022')) if cfg else 'claude-3-5-haiku-20241022'
        max_toks = getattr(cfg, 'max_tokens', 1000) if cfg else 1000

        try:
            system_prompt = next((m.content for m in messages if m.role == "system"), None)
            user_messages = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]

            kwargs = {
                "model": model_name,
                "max_tokens": max_toks,
                "messages": user_messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except (AnthropicRateLimitError, AnthropicConnectionError, AnthropicAPIError) as e:
            yield f"\n[Error en streaming de Anthropic: {str(e)}]"


class AsyncLLMManager:
    def __init__(self, provider: str = "openai"):
        self.provider = provider.lower()
        if self.provider in ["openai", "groq", "openrouter", "deepseek"]:
            self.client: BaseLLMClient = OpenAIClient()
        elif self.provider == "anthropic":
            self.client = AnthropicClient()
        else:
            raise ValueError(f"Proveedor no soportado: {provider}")

    @classmethod
    def get_client(cls, config) -> BaseLLMClient:
        provider = getattr(config, 'provider', 'openai').lower()
        if provider in ["openai", "groq", "openrouter", "deepseek"]:
            return OpenAIClient(config=config)
        elif provider == "anthropic":
            return AnthropicClient(config=config)
        else:
            raise ValueError(f"Proveedor no soportado: {provider}")

    async def generate(self, messages: list[ChatMessage], config: Optional[ModelConfig] = None) -> ModelResponse:
        return await self.client.generate(messages, config)

    async def generate_stream(self, messages: list[ChatMessage], config: Optional[ModelConfig] = None) -> AsyncGenerator[str, None]:
        async for chunk in self.client.generate_stream(messages, config):
            yield chunk