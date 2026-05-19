# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Agent role for any model that conforms to OpenAI tool use API"""

import os
import re
import time
import logging
import json
import numpy as np
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional, Union, cast, Dict

# Optional import for embedding functionality
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_AVAILABLE = True
    print("[TOOL_GRAPH] Embedding libraries available. Will use embedding-based similarity.")
except ImportError:
    SentenceTransformer = None  # type: ignore
    cosine_similarity = None  # type: ignore
    EMBEDDING_AVAILABLE = False
    print("[TOOL_GRAPH] Embedding libraries not available. Will use simple word-based similarity.")

from openai import NOT_GIVEN, NotGiven, OpenAI, AzureOpenAI, RateLimitError
try:
    # Available in newer openai python SDKs
    from openai import APIStatusError, APITimeoutError  # type: ignore
except Exception:  # pragma: no cover - fallback for older SDKs
    APIStatusError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from requests.exceptions import HTTPError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from azure.identity import (
    DefaultAzureCredential,
    ChainedTokenCredential,
    AzureCliCredential,
    get_bearer_token_provider,
)

from tool_sandbox.common.execution_context import RoleType, get_current_context
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
    to_openai_messages,
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from tool_sandbox.common.utils import all_logging_disabled
from tool_sandbox.roles.base_role import BaseRole

try:
    # Prefer importing from repo root script location (when running from this workspace)
    from update_tool_graph import ensure_graph_exists, update_graph_from_messages  # type: ignore
except Exception:
    # Fallback: load update_tool_graph.py by absolute path relative to this file
    try:
        import importlib.util

        _update_tool_graph_path = Path(__file__).resolve().parents[3] / "update_tool_graph.py"
        _spec = importlib.util.spec_from_file_location("update_tool_graph", str(_update_tool_graph_path))
        _mod = importlib.util.module_from_spec(_spec) if _spec else None
        if _spec and _spec.loader and _mod:
            _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
            ensure_graph_exists = getattr(_mod, "ensure_graph_exists", None)
            update_graph_from_messages = getattr(_mod, "update_graph_from_messages", None)
        else:
            ensure_graph_exists = None  # type: ignore
            update_graph_from_messages = None  # type: ignore
    except Exception:
        ensure_graph_exists = None  # type: ignore
        update_graph_from_messages = None  # type: ignore

# Azure配置环境变量
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://trapi.research.microsoft.com")
AZURE_INSTANCE = os.environ.get("AZURE_INSTANCE", "msra/shared")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2025-08-01-preview")
AZURE_MODEL_VERSION = os.environ.get("AZURE_MODEL_VERSION", "2025-04-14")
AZURE_SCOPE = os.environ.get("AZURE_SCOPE", "api://trapi/.default")


def get_azure_model(
    model_name,
    model_version="2025-04-14",
    instance="msra/shared",
    api_version="2025-04-01-preview",
    scope="api://trapi/.default",
):
    if '4o' in model_name:
        model_version = "2024-11-20"
        api_version = "2024-10-21" 
    # 先检查 5.1（更具体），再检查 5（更通用），避免 5.1 被 5 的规则覆盖
    if '5.1' in model_name:
        model_version="2025-11-13"
        api_version="2024-12-01-preview"
    elif '5' in model_name:
        model_version="2025-08-07"
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


class OpenAIAPIAgent(BaseRole):
    """Agent role for any model that conforms to OpenAI tool use API"""

    role_type: RoleType = RoleType.AGENT
    model_name: str

    def __init__(self, tool_graph_path: Optional[str] = None, enable_tool_suggestion: bool = True) -> None:
        # 初始化工具图相关属性
        self.tool_graph_path = tool_graph_path
        # 默认开启工具建议功能，除非显式传入 False
        self.enable_tool_suggestion = enable_tool_suggestion
        self.tool_graph_cache: Optional[Dict[str, Any]] = None
        self.node_rules: Optional[Dict[str, Any]] = None
        self.tool_call_history: List[str] = []
        
        # 初始化embedding模型
        if EMBEDDING_AVAILABLE:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embedding_cache = {}
                logger = logging.getLogger(__name__)
                logger.info("✓ Embedding模型加载成功")
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Embedding模型加载失败: {e}，将使用简单的相似度计算")
                self.model = None
                self.embedding_cache = {}
        else:
            self.model = None
            self.embedding_cache = {}
        
        # We set the `base_url` explicitly here to avoid picking up the
        # `OPENAI_BASE_URL` environment variable that may be set for serving models as
        # OpenAI API compatible servers.
        if hasattr(self, 'model_name') and 'gpt' in self.model_name:
            # Use Azure for gpt-4.1 models
            azure_instance = os.environ.get("AZURE_INSTANCE", "msra/shared")
            azure_api_version = os.environ.get("AZURE_API_VERSION", "2025-04-01-preview")
            azure_scope = os.environ.get("AZURE_SCOPE", "api://trapi/.default")
            self.openai_client, self.deployment_name = get_azure_model(
                self.model_name,
                instance=azure_instance,
                api_version=azure_api_version,
                scope=azure_scope,
            )
        else:
            # Use standard OpenAI for other models
            # Enable client-side retries to better handle 429s; server will hint retry-after
            self.openai_client: OpenAI = OpenAI(
                base_url="https://api.openai.com/v1", 
                max_retries=10,
                timeout=120.0  # Increased from 60 to 120 seconds for better handling of slow connections
            )
            self.deployment_name = None

        # Default tool graph path (create lazily on first update/load).
        if not self.tool_graph_path:
            # Keep graph per-model, in current working directory by default.
            safe_model = re.sub(r"[^a-zA-Z0-9\-_.]", "_", str(getattr(self, "model_name", "model")))
            self.tool_graph_path = os.path.join(os.getcwd(), f"tool_graph_{safe_model}.json")

    def respond(self, ending_index: Optional[int] = None) -> None:
        """Reads a List of messages and attempt to respond with a Message

        Specifically, interprets system, user, execution environment messages and sends out NL response to user, or
        code snippet to execution environment.

        Message comes from current context, the last k messages should be directed to this role type
        Response are written to current context as well. n new messages, addressed to appropriate recipient
        k != n when dealing with parallel function call and responses. Parallel function call are expanded into
        individual messages, parallel function call responses are combined as 1 OpenAI API request

        Args:
            ending_index:   Optional index. Will respond to message located at ending_index instead of most recent one
                            if provided. Utility for processing system message, which could contain multiple entries
                            before each was responded to

        Raises:
            KeyError:   When the last message is not directed to this role
        """
        messages: List[Message] = self.get_messages(ending_index=ending_index)
        response_messages: List[Message] = []
        self.messages_validation(messages=messages)
        # Keeps only relevant messages
        messages = self.filter_messages(messages=messages)
        # Does not respond to System
        if messages[-1].sender == RoleType.SYSTEM:
            return
        
        # FORCE summarize_the_task call for first user message
        current_context = get_current_context()
        available_tools = self.get_available_tools()

        # If caller enabled tool suggestion, keep it; do not override externally provided settings here.
        # (Previous code hard-coded Linux-only graph path and always enabled suggestions.)
        # print(f"[TOOL_SUGGESTION] Tool suggestion enabled with tool graph: {self.tool_graph_path}")
        
        # # Check if this is the first user message and summarize_the_task is available
        # user_messages = [msg for msg in messages if msg.sender == RoleType.USER]
        # if (len(user_messages) == 1 and 
        #     "summarize_the_task" in available_tools and 
        #     "summarize_the_task" in current_context.tool_allow_list):
            
        #     # Force call summarize_the_task first
        #     from tool_sandbox.common.tool_conversion import convert_to_openai_tools
        #     from tool_sandbox.common.message_conversion import openai_tool_call_to_python_code
            
        #     # Create a forced summarize_the_task call
        #     import json
        #     import uuid
            
        #     # Create a mock tool call for summarize_the_task
        #     tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
        #     summary_content = "User has initiated a conversation and needs assistance. Available tools include contact management, messaging, and reminder functions. Current task state requires understanding user needs and providing appropriate tool-based assistance."
            
        #     forced_call = Message(
        #         sender=self.role_type,
        #         recipient=RoleType.EXECUTION_ENVIRONMENT,
        #         content=f"call_{tool_call_id}_parameters = {{'summary': '{summary_content}'}}\ncall_{tool_call_id}_response = summarize_the_task(**call_{tool_call_id}_parameters)\nprint(repr(call_{tool_call_id}_response))",
        #         openai_tool_call_id=tool_call_id,
        #         openai_function_name="summarize_the_task",
        #     )
            
        #     # Add the forced call
        #     self.add_messages([forced_call])
        #     print(f"[FORCED] Added summarize_the_task call: {tool_call_id}")
        #     return
        
        # 在生成消息前提供工具建议（将建议注入到最后一条用户消息中供LLM参考）


        # Get OpenAI tools if most recent message is from user
        available_tool_names = set(available_tools.keys())
        openai_tools = (
            convert_to_openai_tools(available_tools)
            if messages[-1].sender == RoleType.USER
            or messages[-1].sender == RoleType.EXECUTION_ENVIRONMENT
            else NOT_GIVEN
        )
        # We need a cast here since `convert_to_openai_tool` returns a plain dict, but
        # `ChatCompletionToolParam` is a `TypedDict`.
        openai_tools = cast(
            Union[Iterable[ChatCompletionToolParam], NotGiven],
            openai_tools,
        )
        # Convert to OpenAI messages.
        openai_messages, _ = to_openai_messages(messages)
        # Guard: avoid sending an assistant tool_calls message without corresponding tool responses.
        # This triggers OpenAI 400: assistant tool_calls must be followed by tool messages.
        if (
            len(openai_messages) > 0
            and isinstance(openai_messages[-1], dict)
            and openai_messages[-1].get("role") == "assistant"
            and openai_messages[-1].get("tool_calls")
        ):
            openai_messages = openai_messages[:-1]
        # Guard: avoid sending an assistant tool_calls message without corresponding tool responses.
        # This triggers OpenAI 400: assistant tool_calls must be followed by tool messages.
        if (
            len(openai_messages) > 0
            and isinstance(openai_messages[-1], dict)
            and openai_messages[-1].get("role") == "assistant"
            and openai_messages[-1].get("tool_calls")
        ):
            # Drop the trailing assistant tool_calls; execution env will answer and we'll include it next turn.
            openai_messages = openai_messages[:-1]
        # Call model
        response = self.model_inference(
            openai_messages=openai_messages, openai_tools=openai_tools
        )
        # Parse response
        openai_response_message = response.choices[0].message
        # Message contains no tool call, aka addressed to user
        if openai_response_message.tool_calls is None:
            assert openai_response_message.content is not None
            response_messages = [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=openai_response_message.content,
                )
            ]
        else:
            assert openai_tools is not NOT_GIVEN
            if not self.enable_tool_suggestion:
                print("[TOOL_SUGGESTION] disabled (enable_tool_suggestion=False)")
            elif not self.tool_graph_path:
                print("[TOOL_SUGGESTION] disabled (tool_graph_path is empty)")
            for tool_call in openai_response_message.tool_calls:
                # Pre-call: provide next-tool suggestions before executing this tool
                if self.enable_tool_suggestion and self.tool_graph_path:
                    try:
                        conversation_context = self._extract_conversation_context(messages)
                        should_use_summary = self._should_call_summarize_tool(tool_call.function.name, conversation_context)
                        if should_use_summary:
                            # 如果需要使用summarize_the_task，则调用它来获取摘要
                            summary_tool = available_tools.get("summarize_the_task")
                            if summary_tool:
                                summary_callable = getattr(summary_tool, "function", None) or summary_tool
                                summary_response = summary_callable(summary=conversation_context)
                                conversation_context = summary_response
                                print(f"[TOOL_SUGGESTION] 使用 summarize_the_task 获取的摘要作为对话上下文")
                            else:
                                print("[TOOL_SUGGESTION] should_use_summary=True but summarize_the_task is not available")
                        suggestions = self._suggest_next_tools(
                            current_tool=tool_call.function.name,
                            conversation_context=conversation_context,
                            available_tools=list(available_tool_names),
                            should_use_summary=should_use_summary,
                            top_k=2,
                        )
                        if suggestions:
                            # 若预生成阶段未注入建议，则仅日志提示以避免重复对话输出
                            suggestion_text = f"next tool suggestion: {', '.join(suggestions)}"
                            print(f"[TOOL_SUGGESTION] (pre-call) 建议下一个工具: {suggestion_text}")
                        else:
                            print(f"[TOOL_SUGGESTION] no candidate suggestions for current tool: {tool_call.function.name}")
                    except Exception as e:
                        print(f"[TOOL_SUGGESTION] 预调用工具建议失败: {e}")
                # 记录工具调用历史
                self.tool_call_history.append(tool_call.function.name)
                
                # The response contains the agent facing tool name so we need to get
                # the execution facing tool name when creating the Python code.
                execution_facing_tool_name = (
                    current_context.get_execution_facing_tool_name(
                        tool_call.function.name
                    )
                )
                response_messages.append(
                    Message(
                        sender=self.role_type,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=openai_tool_call_to_python_code(
                            tool_call,
                            available_tool_names,
                            execution_facing_tool_name=execution_facing_tool_name,
                        ),
                        openai_tool_call_id=tool_call.id,
                        openai_function_name=tool_call.function.name,
                    )
                )
                # 建议已在工具调用之前给出

            # Incrementally update tool graph from this new trajectory segment (starting from empty graph if needed).
            # We update on every trajectory, not only reward==1, to support online graph building.
            try:
                if self.tool_graph_path and update_graph_from_messages is not None and ensure_graph_exists is not None:
                    ensure_graph_exists(Path(self.tool_graph_path))
                    assistant_tool_calls = []
                    for tc in openai_response_message.tool_calls:
                        assistant_tool_calls.append(
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name},
                            }
                        )
                    traj_for_graph = list(openai_messages) + [{"role": "assistant", "tool_calls": assistant_tool_calls}]
                    update_graph_from_messages(
                        traj_for_graph,
                        graph_path=self.tool_graph_path,
                        backup_out=None,
                        check_reward=False,
                    )
            except Exception as e:
                print(f"[TOOL_GRAPH_UPDATE] Online update failed: {e}")

        self.add_messages(response_messages)

    @retry(
        wait=wait_random_exponential(multiplier=1, max=120),
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type((HTTPError, RateLimitError, APIStatusError, APITimeoutError)),
    )
    def model_inference(
        self,
        openai_messages: list[
            dict[
                Literal["role", "content", "tool_call_id", "name", "tool_calls"],
                Any,
            ]
        ],
        openai_tools: Union[Iterable[ChatCompletionToolParam], NotGiven],
    ) -> ChatCompletion:
        """Run OpenAI model inference

        Args:
            openai_messages:    List of OpenAI API format messages
            openai_tools:       List of OpenAI API format tools definition

        Returns:
            OpenAI API chat completion object
        """
        # with all_logging_disabled():
        #     # Use deployment name for Azure models, otherwise use model name
        #     model_to_use = self.deployment_name if self.deployment_name else self.model_name
        #     try:
        #         return self.openai_client.chat.completions.create(
        #             model=model_to_use,
        #             messages=cast(list[ChatCompletionMessageParam], openai_messages),
        #             tools=openai_tools,
        #         )
        #     except RateLimitError as e:
        #         # Log rate limit error for debugging
        #         print(f"[RATE_LIMIT] Encountered rate limit error: {e}")
        #         print(f"[RATE_LIMIT] Error details: {e.response.json() if hasattr(e, 'response') and e.response else 'No response details'}")
        #         # Re-raise to trigger retry logic
        #         raise

        # pre_suggestion_added = False
        if self.enable_tool_suggestion and self.tool_graph_path and len(self.tool_call_history) > 0:
            last_tool = self.tool_call_history[-1]
            if last_tool:
                try:
                    # 取最近一次调用的工具作为当前工具
                    last_tool = self.tool_call_history[-1]

                    conversation_context = self._extract_conversation_context(openai_messages)
                    should_use_summary = self._should_call_summarize_tool(last_tool, conversation_context)
                    if should_use_summary:
                        summary_tool = self.get_available_tools().get("summarize_the_task")
                        if summary_tool:
                            # summary_response = summary_tool(summary=conversation_context)
                            
                            summary_response = self._summary_conversation_context(conversation_context)
                            # print(f"[TOOL_SUGGESTION] 使用 summarize_the_task 获取的摘要作为对话上下文: {summary_response}")
                            conversation_context = summary_response
                            # print("[TOOL_SUGGESTION] summary",conversation_context)
                    available_tool_names_pre = list(self.get_available_tools().keys())
                    suggestions_pre = self._suggest_next_tools(
                        current_tool=last_tool,
                        conversation_context=conversation_context,
                        available_tools=available_tool_names_pre,
                        should_use_summary=should_use_summary,
                        top_k=2,
                    )
                    if suggestions_pre:
                        suggestion_text = f"next tool suggestion: {', '.join(suggestions_pre)}"
                        openai_messages[-1]["content"] += f"\n\n{suggestion_text}"
                        print(f"[TOOL_SUGGESTION] (pre-generate) 已添加助手消息到对话: {suggestion_text}")
                        # 若 top1 建议来源于 weight_fallback 且相似度 < 0.6，则输出 LAST_TOOL 结点规则，否则输出建议工具名
                        # top1 = suggestions_pre[0]
                        # meta = getattr(self, '_last_tool_suggestion_meta', {}) or {}
                        # item = meta.get(top1, {})
                        # reason = item.get('reason')
                        # similarity = item.get('similarity', 1.0)
                        # if reason == 'similarity_and_weight_with_llm_judgment' and isinstance(similarity, (int, float)) and similarity < 0.6:
                        #     rules_text = self._format_decision_rules(last_tool)
                        #     print("===[TOOL_SUGGESTION] Best tool '{0}' has low similarity score {1}".format(last_tool,similarity),'rules_text')
                        #     if rules_text.strip():
                        #         openai_messages[-1]["content"] += rules_text
                        #         print(f"[TOOL_SUGGESTION] (pre-generate) 已添加LAST_TOOL结点信息到对话 (由于weight_fallback且相似度<{0.6}): {last_tool}")
                        #     else:
                        #         suggestion_text = f"next tool suggestion: {', '.join(suggestions_pre)}"
                        #         openai_messages[-1]["content"] += f"\n\n{suggestion_text}"
                        #         # print(f"[TOOL_SUGGESTION] (pre-generate) 已添加助手消息到对话: {suggestion_text}")
                        # else:
                        #     suggestion_text = f"next tool suggestion: {', '.join(suggestions_pre)}"
                        #     openai_messages[-1]["content"] += f"\n\n{suggestion_text}"
                        #     # print(f"[TOOL_SUGGESTION] (pre-generate) 已添加助手消息到对话: {suggestion_text}")


                except Exception as e:
                    import traceback
                    print(f"[TOOL_SUGGESTION] 预生成阶段添加工具建议失败: {e!r} ({type(e).__name__})")
                    print("[TOOL_SUGGESTION] 预生成阶段失败堆栈:\n" + traceback.format_exc())

        with all_logging_disabled():
            # Use deployment name for Azure models, otherwise use model name
            model_to_use = self.deployment_name if self.deployment_name else self.model_name
            try:
                if '5.1' in model_to_use:
                    # print(f"[MODEL_INFERENCE] Using GPT-5.1 model: {model_to_use}")
                    return self.openai_client.chat.completions.create(
                        model=model_to_use,
                        messages=cast(list[ChatCompletionMessageParam], openai_messages),
                        tools=openai_tools,
                        reasoning_effort="high",
                    )
                elif 'Qwen' in self.model_name:
                    # Qwen models need extra_body for chat_template_kwargs
                    return self.openai_client.chat.completions.create(
                        model=model_to_use,
                        messages=cast(list[ChatCompletionMessageParam], openai_messages),
                        tools=openai_tools,
                        tool_choice='auto',
                        extra_body={
                            "chat_template_kwargs": {
                                "enable_thinking": False  # 关闭思考模式
                            }
                        },
                    )
                else:
                    return self.openai_client.chat.completions.create(
                        model=model_to_use,
                        messages=cast(list[ChatCompletionMessageParam], openai_messages),
                        tools=openai_tools,
                    )
            except APITimeoutError as e:
                # Log timeout error with specific handling
                print(f"[TIMEOUT] API request timed out: {e}")
                print(f"[TIMEOUT] This may be due to network connectivity issues or server load.")
                print(f"[TIMEOUT] The request will be retried with exponential backoff.")
                
                # Re-raise to trigger the tenacity retry logic
                raise
            except RateLimitError as e:
                # Log rate limit error and wait before retrying
                print(f"[RATE_LIMIT] Encountered rate limit error: {e}")
                details = (
                    e.response.json() if hasattr(e, "response") and getattr(e, "response", None) else "No response details"
                )
                print(f"[RATE_LIMIT] Error details: {details}")

                # Try to respect server-provided retry hints
                wait_seconds = 5
                try:
                    # 1) Prefer Retry-After header if available
                    if hasattr(e, "response") and getattr(e, "response", None):
                        retry_after = getattr(e.response, "headers", {}).get("retry-after") if hasattr(e.response, "headers") else None
                        if retry_after:
                            wait_seconds = int(float(retry_after))
                    # 2) Fallback: parse seconds from error message/details like "Try again in X seconds."
                    if isinstance(details, dict) and isinstance(details.get("message"), str):
                        import re as _re
                        m = _re.search(r"(\d+(?:\.\d+)?)\s*seconds?", details["message"], _re.IGNORECASE)
                        if m:
                            wait_seconds = int(float(m.group(1)))
                except Exception:
                    # Keep default wait_seconds on any parsing error
                    pass

                # Bound wait to a reasonable range
                wait_seconds = max(1, min(wait_seconds, 60))
                print(f"[RATE_LIMIT] 等待 {wait_seconds} 秒后继续尝试…")
                time.sleep(wait_seconds)

                # Re-raise to trigger the tenacity retry logic after our explicit wait
                raise

    # ===== 工具图建议相关方法 =====
    def _load_tool_graph(self) -> Dict[str, Any]:
        """加载并缓存工具图数据"""
        if self.tool_graph_cache is not None:
            # 确保 node_rules 也被加载（处理旧代码的情况）
            if self.node_rules is None:
                self.node_rules = self.tool_graph_cache.get("node_rules", {})
            return self.tool_graph_cache
        
        if not self.tool_graph_path:
            self.tool_graph_cache = {"nodes": [], "edges": []}
            self.node_rules = {}
            return self.tool_graph_cache
            
        try:
            with open(self.tool_graph_path, "r", encoding="utf-8") as f:
                self.tool_graph_cache = json.load(f)
                # 从加载的数据中提取 node_rules
                self.node_rules = self.tool_graph_cache.get("node_rules", {})
        except Exception as e:
            print(f"[TOOL_GRAPH] 加载工具图失败: {e}")
            self.tool_graph_cache = {"nodes": [], "edges": []}
            self.node_rules = {}
        
        return self.tool_graph_cache

    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的embedding，带缓存"""
        if not self.model:
            return np.zeros(384)  # 返回零向量作为fallback
        
        if text not in self.embedding_cache:
            try:
                self.embedding_cache[text] = self.model.encode([text])[0]
            except Exception as e:
                print(f"[TOOL_GRAPH] Embedding生成失败: {e}")
                return np.zeros(384)
        
        return self.embedding_cache[text]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的余弦相似度（使用embedding模型）"""
        try:
            # 使用embedding模型计算相似度
            if self.model and EMBEDDING_AVAILABLE:
                emb1 = self._get_embedding(text1)
                emb2 = self._get_embedding(text2)
                similarity = float(cosine_similarity([emb1], [emb2])[0][0])
                # print(f"[TOOL_GRAPH] Embedding相似度: {similarity:.4f}")
                return similarity
            else:
                # 回退到简单的词频相似度计算
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())
                
                if not words1 or not words2:
                    return 0.0
                    
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                
                if not union:
                    return 0.0
                    
                similarity = len(intersection) / len(union)
                # print(f"[TOOL_GRAPH] 简单相似度: {similarity:.4f}")
                return float(similarity)
        except Exception as e:
            print(f"[TOOL_GRAPH] 相似度计算失败: {e}")
            return 0.0

    def _get_connected_tools(self, current_tool: str) -> List[Dict[str, Any]]:
        """获取与当前工具相连的工具"""
        graph = self._load_tool_graph()
        connected_tools = []
        
        # 处理 edges 格式
        for edge in graph.get("edges", []):
            try:
                if edge.get("u") == current_tool and edge.get("v"):
                    connected_tools.append({
                        "tool": edge.get("v"),
                        "weight": float(edge.get("weight", 0)),
                        "count": int(edge.get("count", 0)),
                        "current_information": edge.get("current_information", [])
                    })
            except Exception:
                continue
                
        return connected_tools

    def _should_call_summarize_tool(self, current_tool: str, conversation_context: str) -> bool:
        """让大模型判断是否应该调用 summarize 工具"""
        if not self.enable_tool_suggestion or not self.tool_graph_path:
            return False
            
        connected_tools = self._get_connected_tools(current_tool)
        
        # 检查是否有任何相连的工具包含 current_information
        has_current_info = False
        for tool_info in connected_tools:
            if tool_info.get("current_information") and len(tool_info["current_information"]) > 0:
                has_current_info = True
                break
        
        # 如果没有任何current_information，则不需要调用summarize_the_task
        if not has_current_info:
            return False
            
        # 让大模型判断是否需要调用summarize_the_task
        try:
            return self._llm_judge_should_summarize(current_tool, conversation_context, connected_tools)
        except Exception as e:
            print(f"[LLM_JUDGMENT] LLM judgment failed, falling back to default strategy: {e}")
            # 回退策略：如果有current_information就使用总结
            return True

    def _llm_judge_should_summarize(self, current_tool: str, conversation_context: str, connected_tools: List[Dict[str, Any]]) -> bool:
        """使用大模型判断是否应该调用summarize_the_task"""
        try:
            # 构建提示词
            current_info_examples = []
            for tool_info in connected_tools:
                if tool_info.get("current_information"):
                    current_info_examples.extend(tool_info["current_information"][:2])  # 取前2个例子
            
            prompt = f"""You are an intelligent assistant that needs to determine whether to call the summarize_the_task tool to summarize the conversation context.

Current tool: {current_tool}
Conversation context: {conversation_context}

Available current_information examples:
{chr(10).join(current_info_examples[:3]) if current_info_examples else "None"}

Please determine:
1. Does the conversation context need to be summarized to better match with current_information for similarity comparison?
2. Is the raw conversation context sufficient for tool recommendations?

Please only answer "Yes" or "No", do not explain the reason."""

            # 调用大模型进行判断
            messages = [
                {"role": "system", "content": "You are a professional tool selection assistant who can accurately determine whether conversation content needs to be summarized."},
                {"role": "user", "content": prompt}
            ]
            
            model_to_use = self.deployment_name if self.deployment_name else self.model_name
            if '5.1' in model_to_use:
                response = self.openai_client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    max_completion_tokens=10,
                    temperature=0.1,
                    reasoning_effort="high",
                )
            elif 'Qwen' in self.model_name:
                response = self.openai_client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=10,
                    temperature=0.1,
                    tool_choice='auto',
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": False  # 关闭思考模式
                        }
                    },
                )
            else:
                response = self.openai_client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=10,
                    temperature=0.1
                )
            
            result = response.choices[0].message.content.strip().lower()
            should_summarize = "yes" in result or "true" in result or "y" in result
            
            print(f"[LLM_JUDGMENT] LLM judgment result: {should_summarize} (original response: {result})")
            return should_summarize
            
        except Exception as e:
            print(f"[LLM_JUDGMENT] LLM judgment error: {e}")
            # 回退到简单规则
            return len(conversation_context.strip()) > 100

    def _suggest_next_tools(self, current_tool: str, conversation_context: str, 
                           available_tools: List[str] = None, should_use_summary = None,
                           top_k: int = 2) -> List[str]:
        """基于工具图建议下一个工具"""
        if not self.enable_tool_suggestion or not self.tool_graph_path:
            print("[TOOL_SUGGESTION] skip suggesting because feature disabled or graph path missing")
            return []
            
        connected_tools = self._get_connected_tools(current_tool)
        
        if not connected_tools:
            print(f"[TOOL_SUGGESTION] no outgoing edges from tool: {current_tool}")
            return []
            
        # 如果指定了可用工具，过滤连接的工具
        if available_tools:
            connected_tools = [
                tool for tool in connected_tools 
                if tool['tool'] in available_tools
            ]
            
        if not connected_tools:
            print(f"[TOOL_SUGGESTION] connected tools filtered out by availability for tool: {current_tool}")
            return []
        
        # 让大模型判断是否应该调用summarize_the_task
        # should_use_summary = self._should_call_summarize_tool(current_tool, conversation_context)
        print(f"[TOOL_SUGGESTION] LLM judgment on whether to use summarize_the_task: {should_use_summary}")
            
        # 计算工具分数
        tool_scores = []
        
        for tool_info in connected_tools:
            tool_name = tool_info['tool']
            current_infos = tool_info['current_information']
            
            if should_use_summary and current_infos and len(current_infos) > 0:
                # 如果大模型判断需要调用summarize_the_task且有current_information，则进行相似度比较
                max_similarity = 0.0
                best_match = ""
                
                for info in current_infos:
                    similarity = self._calculate_similarity(conversation_context, info)
                    print(f"[TOOL_SUGGESTION] Comparing context with info for tool {tool_name}: {info}")
                    print(f"[TOOL_SUGGESTION] Calculated similarity between context and info for tool {tool_name}: {similarity}")
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = info
                        
                # 如果相似度为 0，则回退到基于权重打分
                if max_similarity == 0.0:
                    score = tool_info['weight'] / 500.0
                    tool_scores.append({
                        'tool': tool_name,
                        'score': score,
                        'reason': 'weight_fallback_with_llm_judgment',
                        'similarity': max_similarity,
                        'weight': tool_info['weight'],
                        'best_match': best_match
                    })
                else:
                    # 结合相似度和权重计算最终分数
                    score = max_similarity
                    tool_scores.append({
                        'tool': tool_name,
                        'score': score,
                        'reason': 'similarity_and_weight_with_llm_judgment',
                        'similarity': max_similarity,
                        'weight': tool_info['weight'],
                        'best_match': best_match
                    })
            else:
                # 如果大模型判断不需要调用summarize_the_task或没有current_information，只使用权重
                score = tool_info['weight'] / 500.0  # 归一化权重
                tool_scores.append({
                    'tool': tool_name,
                    'score': score,
                    'reason': 'weight_only_with_llm_judgment',
                    'weight': tool_info['weight']
                })
                
        # 按分数排序
        tool_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # 缓存评分元数据以便上层根据理由/相似度进行策略分支
        try:
            self._last_tool_suggestion_meta = {item['tool']: item for item in tool_scores}
        except Exception:
            self._last_tool_suggestion_meta = {}

        # 返回前 top_k 个工具
        suggestions = []
        seen = set()
        for item in tool_scores:
            tool_name = item['tool']
            if tool_name not in seen:
                suggestions.append(tool_name)
                seen.add(tool_name)
            if len(suggestions) >= top_k:
                break
                
        return suggestions

    def _format_decision_rules(self, tool_name: str) -> str:
        # 新增：格式化对应 tool 的 decision_rules，结构化方便 LLM 理解
        # 确保工具图已加载（这样 node_rules 也会被加载）
        if self.enable_tool_suggestion and self.tool_graph_path:
            self._load_tool_graph()
        
        decision_rules = []
        if hasattr(self, 'node_rules') and self.node_rules and tool_name in self.node_rules:
            rules = self.node_rules[tool_name].get('decision_rules', [])
            for idx, r in enumerate(rules, 1):
                when = r.get('when', '')
                choose_tool = r.get('choose_tool', '')
                rationale = r.get('rationale', '')
                decision_rules.append(
                    f"[{idx}] When: {when}\n    Choose Tool: {choose_tool}\n    Rationale: {rationale}"
                )
        if decision_rules:
            return f"\n\n# Tool Decision Rules for '{tool_name}'\n" + '\n'.join(decision_rules) + '\n'
        return ''

    def _extract_conversation_context(self, messages) -> str:
        """从消息列表中提取对话上下文"""
        context_parts = []
        
        for msg in messages:
            # Handle both Message objects and dict objects
            if isinstance(msg, dict):
                # Handle OpenAI format messages (dict)
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user" and content:
                    context_parts.append(f"User: {content}")
                elif role == "assistant" and content:
                    context_parts.append(f"Assistant: {content}")
            else:
                # Handle Message objects
                if msg.sender == RoleType.USER and msg.content:
                    context_parts.append(f"User: {msg.content}")
                elif msg.sender == self.role_type and msg.content:
                    context_parts.append(f"Assistant: {msg.content}")
                
        return " ".join(context_parts)
        
    # def _summary_conversation_context(self, messages: List[Message]) -> str:
    #     """通过调用summarize_the_task工具来获取任务总结作为上下文信息"""
    #     try:
    #         # 构建对话内容的摘要
    #         context_parts = []
    #         for msg in messages:
    #             if msg.sender == RoleType.USER and msg.content:
    #                 context_parts.append(f"User: {msg.content}")
    #             elif msg.sender == self.role_type and msg.content:
    #                 context_parts.append(f"Assistant: {msg.content}")
            
    #         conversation_text = " ".join(context_parts)
            
    #         # 调用summarize_the_task工具
    #         from tool_sandbox.tools.summarize import summarize_the_task
    #         summary = summarize_the_task(conversation_text)
    #         print(f"[TOOL_SUMMARY] Generated task summary using summarize_the_task tool: {summary}")
            
    #         return summary
    #     except Exception as e:
    #         # 如果工具调用失败，回退到原始方法
    #         print(f"[WARNING] Failed to call summarize_the_task: {e}")
    #         context_parts = []
    #         for msg in messages:
    #             if msg.sender == RoleType.USER and msg.content:
    #                 context_parts.append(f"User: {msg.content}")
    #             elif msg.sender == self.role_type and msg.content:
    #                 context_parts.append(f"Assistant: {msg.content}")
    #         return " ".join(context_parts)

    def _summary_conversation_context(self, messages: Union[List[Message], str, List[Dict[str, Any]]]) -> str:
        """基于对话message由模型生成摘要，不直接返回原始内容"""
        # 处理不同类型的输入：字符串、Message对象列表或OpenAI格式的dict列表
        if isinstance(messages, str):
            # 如果已经是字符串，直接使用
            conversation_text = messages
        else:
            # 先拼接对话文本
            context_parts = []
            for msg in messages:
                # 处理Message对象
                if hasattr(msg, 'sender'):
                    if msg.sender == RoleType.USER and msg.content:
                        context_parts.append(f"User: {msg.content}")
                    elif msg.sender == self.role_type and msg.content:
                        context_parts.append(f"Assistant: {msg.content}")
                # 处理OpenAI格式的dict
                elif isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user" and content:
                        context_parts.append(f"User: {content}")
                    elif role == "assistant" and content:
                        context_parts.append(f"Assistant: {content}")
            conversation_text = " ".join(context_parts)
        # 优先尝试使用模型生成摘要
        try:
            system_prompt = "Please write a summary of the current state, including information from the environment and the user. "
            user_prompt = f"Summarize the following conversation into a brief summary focusing on current state:\n\n{conversation_text}"
            model_to_use = self.deployment_name if self.deployment_name else self.model_name
            if '5.1' in model_to_use:
                response = self.openai_client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=500,
                    temperature=0.0,
                    reasoning_effort="high",
                )
            elif 'Qwen' in self.model_name:
                response = self.openai_client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=500,
                    temperature=0.0,
                    tool_choice='auto',
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": False  # 关闭思考模式
                        }
                    },
                )
            else:
                response = self.openai_client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=500,
                    temperature=0.0,
                )
            summary = (response.choices[0].message.content or "").strip()
            if summary:
                # print(f"[TOOL_SUMMARY] Generated task summary via model.{summary}")
                return summary
        except Exception as e:
            print(f"[WARNING] Model summarization failed: {e}")
        # 次级尝试：调用工具（主要用于日志追踪），但依旧不直接返回原文
        try:
            from tool_sandbox.tools.summarize import summarize_the_task
            _ = summarize_the_task(conversation_text)
        except Exception:
            pass
        # 最终兜底：返回占位摘要，避免直接泄露原始长文本
        return "Summary is temporarily unavailable. The conversation involves assisting the user with their stated task and constraints."


class GPT_4_0125_Agent(OpenAIAPIAgent):
    model_name = "gpt-4-0125-preview"
    
    def __init__(self, tool_graph_path: Optional[str] = None, enable_tool_suggestion: bool = True) -> None:
        super().__init__(tool_graph_path, enable_tool_suggestion)


class GPT_3_5_0125_Agent(OpenAIAPIAgent):
    model_name = "gpt-3.5-turbo-0125"
    
    def __init__(self, tool_graph_path: Optional[str] = None, enable_tool_suggestion: bool = True) -> None:
        super().__init__(tool_graph_path, enable_tool_suggestion)



class GPT_4_o_2024_05_13_Agent(OpenAIAPIAgent):
    model_name = "gpt-4o-2024-05-13"
    
    def __init__(self, tool_graph_path: Optional[str] = None, enable_tool_suggestion: bool = True) -> None:
        super().__init__(tool_graph_path, enable_tool_suggestion)

class GPT_4_1_Agent(OpenAIAPIAgent):
    model_name = "gpt-4.1"
    
    def __init__(self, tool_graph_path: Optional[str] = None, enable_tool_suggestion: bool = True) -> None:
        super().__init__(tool_graph_path, enable_tool_suggestion)

class GPT_4_1_mini_Agent(OpenAIAPIAgent):
    model_name = "gpt-4.1-mini"
    
    def __init__(self, tool_graph_path: Optional[str] = None, enable_tool_suggestion: bool = True) -> None:
        super().__init__(tool_graph_path, enable_tool_suggestion)

class GPT_4o_Agent(OpenAIAPIAgent):
    model_name = "gpt-4o"
    
    def __init__(self, tool_graph_path: Optional[str] = None, enable_tool_suggestion: bool = True) -> None:
        super().__init__(tool_graph_path, enable_tool_suggestion)

class GPT_5_Agent(OpenAIAPIAgent):
    model_name = "gpt-5"
    
    def __init__(self, tool_graph_path: Optional[str] = None, enable_tool_suggestion: bool = True) -> None:
        super().__init__(tool_graph_path, enable_tool_suggestion)

class GPT_5_1_Agent(OpenAIAPIAgent):
    model_name = "gpt-5.1"
    
    def __init__(self, tool_graph_path: Optional[str] = None, enable_tool_suggestion: bool = True) -> None:
        super().__init__(tool_graph_path, enable_tool_suggestion)

class Qwen_3_8B_Agent(OpenAIAPIAgent):
    model_name = "Qwen/Qwen3-8B"
    
    def __init__(self, tool_graph_path: Optional[str] = None, enable_tool_suggestion: bool = True) -> None:
        super().__init__(tool_graph_path, enable_tool_suggestion)

class Qwen_3_4B_Agent(OpenAIAPIAgent):
    model_name = "Qwen/Qwen3-4B"
    
    def __init__(self, tool_graph_path: Optional[str] = None, enable_tool_suggestion: bool = True) -> None:
        super().__init__(tool_graph_path, enable_tool_suggestion)