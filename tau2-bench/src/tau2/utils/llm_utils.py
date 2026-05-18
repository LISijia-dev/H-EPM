import json
import re
import hashlib
import uuid
from typing import Any, Optional
from loguru import logger
from openai import AzureOpenAI, OpenAI
import os

# 思考模型配置
# ALLOW_SONNET_THINKING = False

# if not ALLOW_SONNET_THINKING:
#     logger.info("Sonnet thinking is disabled")

from azure.identity import (
    DefaultAzureCredential,
    ChainedTokenCredential,
    AzureCliCredential,
    get_bearer_token_provider,
)

import base64
import io
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None


def get_azure_model(
    model_name,
    model_version="2025-04-14",
    instance="msra/shared",
    api_version="2025-04-01-preview",
    scope="api://trapi/.default",
):
    if "4o" in model_name:
        model_version = "2024-11-20"
        api_version = "2024-10-21"
    if '5' in model_name:
        model_version="2025-08-07"
        api_version="2024-12-01-preview"
    if '5.1' in model_name:
        model_version="2025-11-13"
        api_version="2024-12-01-preview"
    deployment_name = re.sub(r"[^a-zA-Z0-9\-_.]", "", f"{model_name}_{model_version}")
    logger.info(
        f"Using Azure model: {model_name}, version: {model_version}, instance: {instance}, deployment: {deployment_name}"
    )
    credential = get_bearer_token_provider(
        ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential(
                exclude_cli_credential=True,
                # Exclude other credentials we are not interested in.
                exclude_environment_credential=True,
                exclude_shared_token_cache_credential=True,
                exclude_developer_cli_credential=True,
                exclude_powershell_credential=True,
                exclude_interactive_browser_credential=True,
                exclude_visual_studio_code_credentials=True,
                managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
            ),
        ),
        scope,
    )
    endpoint = f"https://trapi.research.microsoft.com/{instance}"
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=credential,
        api_version=api_version,
    )
    return client, deployment_name



def get_custom_model(model_name, base_url="http://localhost:8000/v1"):
    client = OpenAI(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    return client, model_name



from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# Optional: Redis cache
try:
    import redis
except ImportError:
    redis = None

class LLMCache:
    def __init__(self, use_redis=False, redis_host='localhost', redis_port=6379, redis_db=0, redis_password=None):
        self.use_redis = use_redis and redis is not None
        if self.use_redis:
            self.redis = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db, password=redis_password)
        else:
            self.local_cache = {}

    def _make_key(self, *args, **kwargs):
        key = json.dumps([args, kwargs], sort_keys=True, default=str)
        return hashlib.sha256(key.encode()).hexdigest()

    def get(self, *args, **kwargs):
        key = self._make_key(*args, **kwargs)
        if self.use_redis:
            value = self.redis.get(key)
            if value:
                return json.loads(value)
        else:
            return self.local_cache.get(key)
        return None

    def set(self, value, *args, **kwargs):
        key = self._make_key(*args, **kwargs)
        if self.use_redis:
            self.redis.set(key, json.dumps(value), ex=REDIS_CACHE_TTL)
        else:
            self.local_cache[key] = value

# 全局缓存对象
if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        cache = LLMCache(
            use_redis=True,
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            redis_db=0,
            redis_password=REDIS_PASSWORD,
        )
        logger.info(f"LLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        cache = LLMCache(use_redis=False)
        logger.info("LLM: Using local cache")
    else:
        raise ValueError(f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'")
else:
    cache = None
    logger.info("LLM: Cache is disabled")


def to_openai_messages(messages: list[Message]) -> list[dict]:
    openai_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            openai_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if hasattr(message, "is_tool_call") and message.is_tool_call():
                # 检查tool_calls数组长度，OpenAI API限制为128
                if message.tool_calls and len(message.tool_calls) > 128:
                    logger.warning(f"Tool calls array too long ({len(message.tool_calls)}), truncating to 128")
                    tool_calls = [
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                            "type": "function",
                        }
                        for tc in message.tool_calls[:128]  # 只取前128个
                    ]
                else:
                    tool_calls = [
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                            "type": "function",
                        }
                        for tc in message.tool_calls
                    ]
            openai_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            openai_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            openai_messages.append({"role": "system", "content": message.content})
    return openai_messages


def get_azure_model(model_name, model_version="2025-04-14", instance="msra/shared", api_version="2025-04-01-preview", scope="api://trapi/.default"):
    from azure.identity import (
        DefaultAzureCredential,
        ChainedTokenCredential,
        AzureCliCredential,
        get_bearer_token_provider,
    )
    if "4o" in model_name:
        model_version = "2024-11-20"
        api_version = "2024-10-21"

    if "5" in model_name:
        model_version = "2025-08-07"
        api_version = "2024-12-01-preview"
    if '5.1' in model_name:
        model_version="2025-11-13"
        api_version="2024-12-01-preview"
    deployment_name = re.sub(r"[^a-zA-Z0-9\-_.]", "", f"{model_name}_{model_version}")
    credential = get_bearer_token_provider(
        ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential(
                exclude_cli_credential=True,
                exclude_environment_credential=True,
                exclude_shared_token_cache_credential=True,
                exclude_developer_cli_credential=True,
                exclude_powershell_credential=True,
                exclude_interactive_browser_credential=True,
                exclude_visual_studio_code_credentials=True,
                managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
            ),
        ),
        scope,
    )
    endpoint = f"https://trapi.research.microsoft.com/{instance}"
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=credential,
        api_version=api_version,
    )
    return client, deployment_name

def get_openai_model(model_name, base_url="http://localhost:8000/v1"):
    client = OpenAI(
        base_url=base_url,
        api_key="EMPTY",
    )
    return client, model_name
    # base_url="http://localhost:8000/v1",
    # api_key="EMPTY"



def get_hf_model(model_name, device="cuda"):
    if AutoModelForCausalLM is None or AutoTokenizer is None or pipeline is None:
        raise ImportError("transformers 库未安装，请先安装 transformers")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, force_download=True).to(device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        raise
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
    return pipe


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    use_azure: bool = False,
    cache_obj: Optional[LLMCache] = None,
    use_hf: bool = False,  # 新增参数
    hf_device: str = "cuda",  # 新增参数
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model, with cache support.
    """
    openai_messages = to_openai_messages(messages)
    tools_schema = [tool.openai_schema for tool in tools] if tools else None

    if 'gpt' in model:
        use_azure = True
    cache_args = {
        "model": model,  # 字符串
        "messages": openai_messages,
        "tools": tools_schema,
        "tool_choice": tool_choice,
        **kwargs,
    }
    if cache_obj is None:
        cache_obj = cache
    if cache_obj:
        cached = cache_obj.get(**cache_args)
        if cached:
            logger.info("LLM cache hit")
            return AssistantMessage(**cached)

    # 自动调整 max_tokens，建议不超过 4096
    if "max_tokens" in kwargs and kwargs["max_tokens"] > 4096:
        logger.warning(f"max_tokens={kwargs['max_tokens']} 超过推荐值，自动调整为 4096")
        kwargs["max_tokens"] = 4096

    # 禁用思考模型
    # if not ALLOW_SONNET_THINKING and model.startswith("claude"):
    #     kwargs["thinking"] = {"type": "disabled"}
    #     logger.info(f"Thinking disabled for model: {model}")

    # 新增：Hugging Face本地推理分支

    max_retries = kwargs.pop("max_retries", 8)
    for retry in range(max_retries):
        try:
            if use_azure:

                client, deployment_name = get_azure_model(model)
                # Only include tool_choice if tools are provided
                create_params = {
                    "model": deployment_name,
                    "messages": openai_messages,
                    **kwargs,
                }
                if tools_schema is not None:
                    create_params["tools"] = tools_schema
                    if tool_choice is not None:
                        create_params["tool_choice"] = tool_choice
                
                if '5.1' in deployment_name:
                    create_params["reasoning_effort"] = "high"
                
                response = client.chat.completions.create(**create_params)

            else:
                client, model_name = get_openai_model(model)
                # from qwen_agent.llm import get_chat_model
                # # print(f"[DEBUG] model_name: {tools_schema}")
                # client = get_chat_model({
                #     "model": model,
                #     "base_url": "http://localhost:8000/v1",
                #     "api_key": "EMPTY",
                #     "generate_config": {
                #         "extra_body": {
                #             "chat_template_kwargs": {
                #                 "enable_thinking": False, 
                #                 "enable_tool_use": True,
                #             }
                #         }
                #     }
                # })
                # Only include tool_choice if tools are provided
                create_params = {
                    "model": model,
                    "messages": openai_messages,
                    "extra_body": {
                        "chat_template_kwargs": {
                            "enable_thinking": False  # 关闭思考模式
                        }
                    },
                    **kwargs,
                }
                if tools_schema is not None:
                    create_params["tools"] = tools_schema
                    # Use tool_choice parameter if provided, otherwise default to 'auto' when tools are available
                    if tool_choice is not None:
                        create_params["tool_choice"] = tool_choice
                    else:
                        create_params["tool_choice"] = 'auto'
                
                response = client.chat.completions.create(**create_params)

                # params = {
                #     "model": model_name,
                #     "messages": openai_messages,
                # }
                # if tools_schema:
                #     params["tools"] = tools_schema
                #     if tool_choice is not None:
                #         params["tool_choice"] = tool_choice
                # params.update(kwargs)
                # response = client.chat.completions.create(**params)

            choice = response.choices[0]
            content = choice.message.content
            tool_calls = getattr(choice.message, "tool_calls", None)
            usage = getattr(response, "usage", None)
            if usage is not None and not isinstance(usage, dict):
                if hasattr(usage, "model_dump"):
                    usage = usage.model_dump()
                elif hasattr(usage, "to_dict"):
                    usage = usage.to_dict()
                else:
                    usage = dict(usage)
            cost = None  # 可根据 usage 计算 cost

            tool_calls_obj = None
            if tool_calls:
                # 检查响应中的tool_calls数组长度，OpenAI API限制为128
                if len(tool_calls) > 128:
                    logger.warning(f"Response tool_calls array too long ({len(tool_calls)}), truncating to 128")
                    tool_calls = tool_calls[:128]  # 只取前128个
                
                tool_calls_obj = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                    for tc in tool_calls
                ]

            # Parse inline <tool_call> blocks in content (e.g., from Qwen3),
            # converting them to structured tool_calls.
            # Example:
            # <tool_call> {"name": "get_customer_by_phone", "arguments": {"phone_number": "1234567890"}} </tool_call>
            if (
                (tool_calls_obj is None or len(tool_calls_obj) == 0)
                and isinstance(content, str)
                and "<tool_call>" in content
            ):
                inline_tool_calls: list[ToolCall] = []
                cleaned_content = content
                # Support multiple <tool_call> blocks
                pattern = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", re.IGNORECASE)
                for match in pattern.finditer(content):
                    json_str = match.group(1)
                    try:
                        data = json.loads(json_str)
                        name = data.get("name")
                        arguments = data.get("arguments", {})
                        if not isinstance(arguments, dict):
                            # Coerce non-dict arguments into dict
                            if isinstance(arguments, str):
                                try:
                                    arguments = json.loads(arguments)
                                except Exception:
                                    arguments = {"arg": arguments}
                            else:
                                arguments = {"arg": arguments}
                        if name:
                            inline_tool_calls.append(
                                ToolCall(
                                    id=f"call_{uuid.uuid4().hex[:24]}",
                                    name=name,
                                    arguments=arguments,
                                )
                            )
                    except Exception as e:
                        # If parsing fails, skip this block and keep content as-is
                        logger.warning(f"Failed to parse tool_call block: {e}")
                        pass
                if inline_tool_calls:
                    tool_calls_obj = inline_tool_calls
                    # Remove all tool_call blocks from content
                    cleaned_content = pattern.sub("", cleaned_content).strip()
                    # If nothing else remains, set content to None to indicate tool calls only
                    content = cleaned_content if cleaned_content else None

            message = AssistantMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls_obj,
                cost=cost,
                usage=usage,
                raw_data=response.model_dump() if hasattr(response, "model_dump") else response,
            )

            if cache_obj:
                cache_obj.set(message.dict(), **cache_args)

            return message

        except Exception as e:
            error_str = str(e)
            # 针对 max_tokens 不支持、需改用 max_completion_tokens 的错误（如 o1、5.1 等模型）
            if (
                hasattr(e, "status_code") and e.status_code == 400
                and "max_tokens" in error_str.lower()
                and "max_completion_tokens" in error_str.lower()
                and ("unsupported" in error_str.lower() or "invalid_request_error" in error_str.lower())
            ):
                if "max_tokens" in kwargs:
                    val = kwargs.pop("max_tokens")
                    kwargs["max_completion_tokens"] = val
                    logger.warning(
                        f"Model {model} requires max_completion_tokens instead of max_tokens, "
                        f"retrying with max_completion_tokens={val}"
                    )
                    continue
            # 针对 temperature 参数不支持的错误（如某些模型不支持 temperature=0.0）
            if (
                hasattr(e, "status_code") and e.status_code == 400
                and "temperature" in error_str.lower()
                and ("unsupported" in error_str.lower() or "does not support" in error_str.lower())
            ):
                if "temperature" in kwargs:
                    logger.warning(
                        f"Model {model} does not support temperature={kwargs['temperature']}, "
                        f"removing temperature parameter and retrying"
                    )
                    kwargs.pop("temperature")
                    continue  # 直接重试，不增加等待时间
            # 针对 429 错误动态增加等待时间
            if hasattr(e, "status_code") and e.status_code == 429:
                wait_time = 2 + retry
                logger.warning(f"RateLimitError: waiting {wait_time}s before retry, error: {e}")
                import time
                time.sleep(wait_time)
            else:
                logger.error(f"Error for model {model}, retry {retry + 1}/{max_retries}: {e}")
                if retry == max_retries - 1:
                    logger.error(f"All retries failed for model {model}")
                    return AssistantMessage(role="assistant", content=f"Failed to generate response: {str(e)}")
                import time, random
                status = getattr(e, "status_code", None)
                # 5xx 服务端错误：指数退避 + 抖动，给网关恢复时间
                if status is not None and 500 <= status < 600:
                    wait_time = min(60, 2 ** retry) + random.uniform(0, 2)
                else:
                    wait_time = 1
                time.sleep(wait_time)


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage.get("completion_tokens", 0)
        usage["prompt_tokens"] += message.usage.get("prompt_tokens", 0)
    return usage
