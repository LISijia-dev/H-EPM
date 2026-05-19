from __future__ import annotations
import json
import re
import uuid
import hashlib
from typing import Any, Optional, List, Dict
from loguru import logger
from openai import AzureOpenAI, OpenAI
import os

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

# 可选导入embedding相关库
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    logger.warning("Embedding libraries not available. Tool selection will be disabled.")

# Azure配置环境变量
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://trapi.research.microsoft.com")
AZURE_INSTANCE = os.environ.get("AZURE_INSTANCE", "msra/shared")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2025-08-01-preview")
AZURE_MODEL_VERSION = os.environ.get("AZURE_MODEL_VERSION", "2025-08-01")
AZURE_SCOPE = os.environ.get("AZURE_SCOPE", "api://trapi/.default")


class ToolGraphSelector:
    """基于tool graph的工具选择器，支持current_information对比"""
    
    def __init__(self, tool_graph_path: str, enable_embedding: bool = True):
        """
        初始化工具图选择器
        
        Args:
            tool_graph_path: 工具图数据文件路径
            enable_embedding: 是否启用embedding功能
        """
        self.tool_graph_path = tool_graph_path
        self.tool_graph = self._load_tool_graph()
        self.enable_embedding = enable_embedding
        
        # 如果启用embedding且有相关库，初始化embedding模型
        if enable_embedding and EMBEDDING_AVAILABLE:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embedding_cache = {}
                logger.info("✓ Embedding模型加载成功")
            except Exception as e:
                logger.warning(f"Embedding模型加载失败: {e}，将禁用embedding功能")
                logger.info("🔍🔍🔍🔍🔍🔍🔍🔍🔍 Embedding模型加载失败")
                self.model = None
                self.embedding_cache = {}
        else:
            self.model = None
            self.embedding_cache = {}
            
        logger.info(f"工具图选择器初始化完成，加载了 {len(self.tool_graph.get('nodes', []))} 个工具")
        self._log_graph_statistics()
    
    def _load_tool_graph(self) -> Dict:
        """加载工具图数据"""
        try:
            with open(self.tool_graph_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载工具图失败: {e}")
            return {'nodes': [], 'edges': []}
    
    def _log_graph_statistics(self):
        """记录工具图统计信息"""
        edges = self.tool_graph.get('edges', [])
        edges_with_info = [e for e in edges if e.get('current_information') and len(e['current_information']) > 0]
        logger.info(f"工具图统计: 总边数={len(edges)}, 有current_information的边数={len(edges_with_info)}")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的embedding，带缓存"""
        if not self.model:
            return np.zeros(384)  # 返回零向量作为fallback
        
        if text not in self.embedding_cache:
            try:
                self.embedding_cache[text] = self.model.encode([text])[0]
            except Exception as e:
                logger.error(f"🔍 Embedding - 生成失败: {e}")
                return np.zeros(384)
        
        return self.embedding_cache[text]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        if not self.model:
            return 0.0
            
        try:
            emb1 = self._get_embedding(text1)
            emb2 = self._get_embedding(text2)
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            logger.info(f"🔍 相似度计算: {similarity:.4f} - 文本1: {text1[:50]}... | 文本2: {text2[:50]}...")
            return float(similarity)
        except Exception as e:
            logger.error(f"🔍 相似度计算失败: {e}")
            logger.info(f"🔍 相似度计算失败: {e} ...")
            return 0.0
    
    def get_connected_tools(self, tool_name: str) -> List[Dict]:
        """获取与指定工具相连的所有工具"""
        connected_tools = []
        
        # 支持两种格式：edges (u/v) 和 links (source/target)
        edges = self.tool_graph.get('edges', [])
        links = self.tool_graph.get('links', [])
        
        logger.info(f"🔍 工具图调试 - 工具: {tool_name}")
        logger.info(f"🔍 工具图调试 - edges数量: {len(edges)}, links数量: {len(links)}")
        
        # 处理 edges 格式 (u/v)
        for edge in edges:
            if edge.get('u') == tool_name:
                connected_tools.append({
                    'tool': edge['v'],
                    'weight': edge.get('weight', 0),
                    'count': edge.get('count', 0),
                    'current_information': edge.get('current_information', [])
                })
        
        # 处理 links 格式 (source/target)
        for link in links:
            if link.get('source') == tool_name:
                connected_tools.append({
                    'tool': link['target'],
                    'weight': link.get('weight', 1),  # 默认权重为1
                    'count': link.get('count', 1),    # 默认计数为1
                    'current_information': link.get('current_information', [])
                })
        
        # logger.info(f"🔍 工具图调试 - 找到 {len(connected_tools)} 个相连工具: {[t['tool'] for t in connected_tools]}")
        
        return connected_tools
    
    def has_current_information_edges(self, tool_name: str) -> bool:
        """检查指定工具是否有包含current_information的边"""
        connected_tools = self.get_connected_tools(tool_name)
        for tool_info in connected_tools:
            if tool_info['current_information'] and len(tool_info['current_information']) > 0:
                return True
        return False
    
    def select_next_tool(self, current_tool: str, conversation_context: str, 
                        available_tools: List[str] = None) -> Optional[str]:
        """选择下一个工具"""
        logger.info(f"为工具 {current_tool} 选择下一个工具")
        
        # 获取相连的工具
        connected_tools = self.get_connected_tools(current_tool)
        
        if not connected_tools:
            logger.warning(f"工具 {current_tool} 没有相连的工具")
            return None
        
        # 如果指定了可用工具，过滤连接的工具
        if available_tools:
            connected_tools = [
                tool for tool in connected_tools 
                if tool['tool'] in available_tools
            ]
        
        if not connected_tools:
            logger.warning(f"工具 {current_tool} 的相连工具都不在可用工具列表中")
            return None
        
        # 计算相似度
        tool_scores = []
        
        for tool_info in connected_tools:
            tool_name = tool_info['tool']
            current_infos = tool_info['current_information']
            
            if not current_infos or len(current_infos) == 0:
                # 如果没有current_information，使用权重作为分数
                score = tool_info['weight'] / 500.0  # 归一化权重
                # logger.info(f"🔍 工具 {tool_name} 没有 current_information，使用权重打分: {score:.4f}")
                tool_scores.append({
                    'tool': tool_name,
                    'score': score,
                    'reason': 'weight_based',
                    'weight': tool_info['weight']
                })
                continue
            
            # 计算与所有current_information的最大相似度
            max_similarity = 0.0
            best_match = ""
            all_similarities = []
            
            # logger.info(f"🔍 工具 {tool_name} 有 {len(current_infos)} 条 current_information，开始相似度对比")
            
            for i, info in enumerate(current_infos):
                similarity = self._calculate_similarity(conversation_context, info)
                all_similarities.append(similarity)
                # logger.info(f"🔍 信息 {i+1} 相似度: {similarity:.4f} - {info[:100]}...")
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = info
            
            # logger.info(f"🔍 工具 {tool_name} 最高相似度: {max_similarity:.4f}，所有相似度: {[f'{s:.3f}' for s in all_similarities]}")
            
            # 如果相似度为 0，则回退到基于权重打分
            if max_similarity == 0.0:
                score = tool_info['weight'] / 500.0
                # logger.info(f"🔍 工具 {tool_name} 相似度为0，回退到权重打分: {score:.4f}")
                tool_scores.append({
                    'tool': tool_name,
                    'score': score,
                    'reason': 'weight_fallback',
                    'best_match': best_match,
                    'weight': tool_info['weight'],
                    'all_similarities': all_similarities
                })
            else:
                # logger.info(f"🔍 工具 {tool_name} 使用相似度打分: {max_similarity:.4f}")
                tool_scores.append({
                    'tool': tool_name,
                    'score': max_similarity,
                    'reason': 'similarity_based',
                    'best_match': best_match,
                    'weight': tool_info['weight'],
                    'all_similarities': all_similarities
                })         
        
        # 按分数排序
        tool_scores.sort(key=lambda x: x['score'], reverse=True)          
        
        # 选择分数最高的工具
        best_tool = tool_scores[0]         
        logger.info(f"选择工具: {best_tool['tool']}")
        logger.info(f"选择原因: {best_tool['reason']}")
        logger.info(f"分数: {best_tool['score']:.4f}")         
        if best_tool['reason'] == 'similarity_based':
            logger.info(f"最佳匹配: {best_tool['best_match'][:100]}...")         
        
        return best_tool['tool']

    # ========== 新增：由大模型判断是否需要调用 summarize，并据此给出工具建议 ==========
    def _llm_judge_should_summarize(
        self,
        *,
        model: str,
        current_tool: str,
        conversation_context: str,
        connected_tools: List[Dict],
        **kwargs: Any,
    ) -> bool:
        """让大模型判断是否应该调用 summarize 工具，返回 True/False。"""
        try:
            # 仅提取必要的工具信息，避免 prompt 过大
            brief_tools = []
            for t in connected_tools:
                brief_tools.append({
                    "tool": t.get("tool"),
                    "has_current_information": bool(t.get("current_information")),
                    "weight": t.get("weight", 0),
                })

            system_prompt = (
                "You are deciding whether to call a summarization tool named 'summarize_the_task'. "
                "Answer strictly with 'yes' or 'no'. If connected tools include useful current_information "
                "that would benefit from a short task summary, answer 'yes'; otherwise 'no'."
            )
            user_prompt = (
                f"current_tool: {current_tool}\n"
                f"conversation_context:\n{conversation_context}\n\n"
                f"connected_tools (truncated):\n{json.dumps(brief_tools)[:2000]}\n"
                "Should we call summarize_the_task? (yes/no)"
            )

            if 'gpt' in model.lower():
                client, deployment_name = get_azure_model(model)
                resp = client.chat.completions.create(
                    model=deployment_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=8,
                )
            else:
                client, model_name = get_openai_model(model)
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=8,
                )

            answer = (resp.choices[0].message.content or "").strip().lower()
            return answer.startswith("y")
        except Exception as e:
            logger.warning(f"[LLM_JUDGMENT] LLM judgment failed, fallback to default True: {e}")
            return True

    def _should_call_summarize_tool(
        self,
        *,
        model: str,
        current_tool: str,
        conversation_context: str,
        **kwargs: Any,
    ) -> bool:
        """让大模型判断是否应该调用 summarize 工具。若没有任何 current_information，则直接返回 False。"""
        connected_tools = self.get_connected_tools(current_tool)
        if not connected_tools:
            return False

        has_current_info = any(
            t.get("current_information") and len(t.get("current_information")) > 0
            for t in connected_tools
        )
        if not has_current_info:
            return False

        return self._llm_judge_should_summarize(
            model=model,
            current_tool=current_tool,
            conversation_context=conversation_context,
            connected_tools=connected_tools,
            **kwargs,
        )

    def suggest_next_tools(
        self,
        *,
        model: str,
        current_tool: str,
        conversation_context: str,
        available_tools: Optional[List[str]] = None,
        top_k: int = 2,
        should_use_summary: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[str]:
        """基于工具图建议下一个工具：
        - 先由大模型判断是否需要调用 summarize
        - 若需要且存在 current_information，则使用相似度；
        - 否则使用权重。
        """
        connected_tools = self.get_connected_tools(current_tool)
        if not connected_tools:
            return []

        if available_tools:
            connected_tools = [t for t in connected_tools if t.get('tool') in available_tools]
            if not connected_tools:
                return []

        # should_use_summary = self._should_call_summarize_tool(
        #     model=model,
        #     current_tool=current_tool,
        #     conversation_context=conversation_context,
        #     **kwargs,
        # )
        logger.info(f"[TOOL_SUGGESTION] LLM judgment (use summarize): {should_use_summary}")

        tool_scores: List[Dict[str, Any]] = []
        for tool_info in connected_tools:
            tool_name = tool_info.get('tool')
            current_infos = tool_info.get('current_information') or []
            weight = float(tool_info.get('weight', 0.0))

            if should_use_summary and current_infos:
                max_similarity = 0.0
                best_match = ""
                for info in current_infos:
                    sim = self._calculate_similarity(conversation_context, str(info))
                    logger.info(f"======[TOOL_SUGGESTION] _calculate_similarity): {sim}")
                    if sim > max_similarity:
                        max_similarity = sim
                        best_match = str(info)
                if max_similarity == 0.0:
                    score = weight / 500.0
                    tool_scores.append({
                        'tool': tool_name,
                        'score': score,
                        'reason': 'weight_fallback_with_llm_judgment',
                        'similarity': max_similarity,
                        'weight': weight,
                        'best_match': best_match,
                    })
                else:
                    tool_scores.append({
                        'tool': tool_name,
                        'score': float(max_similarity),
                        'reason': 'similarity_based_with_llm_judgment',
                        'weight': weight,
                        'best_match': best_match,
                    })
            else:
                score = weight / 100.0
                tool_scores.append({
                    'tool': tool_name,
                    'score': score,
                    'reason': 'weight_only_with_llm_judgment',
                    'weight': weight,
                })

        tool_scores.sort(key=lambda x: x['score'], reverse=True)
        suggestions: List[str] = []
        seen = set()
        for item in tool_scores:
            name = item['tool']
            if name and name not in seen:
                suggestions.append(name)
                seen.add(name)
            if len(suggestions) >= top_k:
                break
        return suggestions
    
    def get_tool_selection_explanation(self, current_tool: str, conversation_context: str,
                                     available_tools: List[str] = None) -> Dict:
        """获取工具选择的详细解释"""
        try:
            connected_tools = self.get_connected_tools(current_tool)
            # logger.info(f"🔍 工具选择调试 - 原始相连工具数量: {len(connected_tools)}")       
            
            if available_tools:
                connected_tools = [
                    tool for tool in connected_tools 
                    if tool['tool'] in available_tools
                ]
                # logger.info(f"🔍 工具选择调试 - 过滤后相连工具数量: {len(connected_tools)}")
                # logger.info(f"🔍 工具选择调试 - 可用工具: {available_tools[:5]}...")         
            
            if not connected_tools:
                logger.warning(f"🔍 工具选择调试 - 工具 {current_tool} 没有相连的工具")
                return {
                    'error': f'工具 {current_tool} 没有相连的工具',
                    'current_tool': current_tool,
                    'available_connected_tools': 0,
                    'tool_analysis': []
                }         
            
            # logger.info(f"🔍 工具选择调试 - 上下文摘要长度: {len(conversation_context)}")         
            
            tool_analysis = []
            for tool_info in connected_tools:
                try:
                    tool_name = tool_info['tool']
                    current_infos = tool_info['current_information']              
                    
                    if not current_infos:
                        
                        score = tool_info['weight'] / 500.0
                        tool_analysis.append({
                            'tool': tool_name,
                            'score': score,
                            'reason': 'weight_based',
                            'details': {
                                'weight': tool_info['weight'],
                                'count': tool_info['count']
                            }
                        })
                        # logger.info(f"🔍 工具选择调试 - {tool_name}: 基于权重, 分数={score:.4f}")
                    else:
                        similarities = []
                        # logger.info(f"🔍 工具选择调试 - {tool_name} 有 {len(current_infos)} 条 current_information，开始相似度对比")
                        
                        for i, info in enumerate(current_infos):
                            try:
                                similarity = self._calculate_similarity(conversation_context, info)
                                similarities.append({
                                    'info': info,
                                    'similarity': similarity
                                })
                                logger.info(f"🔍 工具选择调试 - {tool_name} 信息 {i+1} 相似度: {similarity:.4f} - {info[:100]}...")
                            except Exception as e:
                                # logger.error(f"🔍 工具选择调试 - {tool_name}: 计算相似度 {i+1} 时出错: {e}")
                                # 如果相似度计算失败，使用默认值
                                similarities.append({
                                    'info': info,
                                    'similarity': 0.0
                                })
                        
                        if similarities:
                            max_sim = max(similarities, key=lambda x: x['similarity'])
                            all_sim_values = [s['similarity'] for s in similarities]
                            logger.info(f"🔍 工具选择调试 - {tool_name} 最高相似度: {max_sim['similarity']:.4f}，所有相似度: {[f'{s:.3f}' for s in all_sim_values]}")
                            
                            if max_sim['similarity'] == 0.0:
                                # 相似度为0则回退到权重
                                score = tool_info['weight'] / 500.0
                                # logger.info(f"🔍 工具选择调试 - {tool_name} 相似度为0，回退到权重打分: {score:.4f}")
                                tool_analysis.append({
                                    'tool': tool_name,
                                    'score': score,
                                    'reason': 'weight_fallback',
                                    'details': {
                                        'best_match': max_sim['info'],
                                        'all_similarities': similarities,
                                        'weight': tool_info['weight']
                                    }
                                })
                            else:
                                # logger.info(f"🔍 工具选择调试 - {tool_name} 使用相似度打分: {max_sim['similarity']:.4f}")
                                tool_analysis.append({
                                    'tool': tool_name,
                                    'score': max_sim['similarity'],
                                    'reason': 'similarity_based',
                                    'details': {
                                        'best_match': max_sim['info'],
                                        'all_similarities': similarities,
                                        'weight': tool_info['weight']
                                    }
                                })
                        else:
                            logger.warning(f"🔍 工具选择调试 - {tool_name}: 没有有效的相似度计算结果")
                except Exception as e:
                    logger.error(f"🔍 工具选择调试 - 处理工具 {tool_info.get('tool', 'unknown')} 时出错: {e}")
                    continue
                        
            tool_analysis.sort(key=lambda x: x['score'], reverse=True)          
            
            result = {
                'current_tool': current_tool,
                'context_summary': conversation_context,
                'available_connected_tools': len(connected_tools),
                'tool_analysis': tool_analysis,
                'selected_tool': tool_analysis[0]['tool'] if tool_analysis else None,
                'selection_score': tool_analysis[0]['score'] if tool_analysis else 0
            }          
            
            logger.info(f"🔍 工具选择调试 - 最终结果: {len(tool_analysis)} 个工具分析")
            if tool_analysis:
                logger.info(f"🔍 工具选择调试 - 推荐工具: {result['selected_tool']}, 分数: {result['selection_score']:.4f}")        
            
            return result
        except Exception as e:
            logger.error(f"🔍 工具选择调试 - get_tool_selection_explanation 出错: {e}")
            return {'error': f'工具选择分析失败: {str(e)}'}


def _extract_conversation_context(messages: list[Message]) -> str:
    """从消息列表中提取对话上下文"""
    context_parts = []
    
    # 添加最近的对话消息
    recent_messages = messages[-10:]  # 最近10条消息
    for msg in recent_messages:
        if hasattr(msg, 'content') and msg.content:
            role = getattr(msg, 'role', 'unknown')
            content = msg.content[:200] if len(msg.content) > 200 else msg.content
            context_parts.append(f"{role}: {content}")
    
    return "\n".join(context_parts)


# def _should_call_summarize_tool(tool_selector: ToolGraphSelector, last_tool_call: str, messages: list[Message]) -> bool:
#     """检查是否应该调用summarize工具"""
#     if not tool_selector or not last_tool_call:
#         return False
    
#     # 检查是否有current_information的边需要summarize
#     return tool_selector.has_current_information_edges(last_tool_call)


def _force_summarize_tool_call(model: str, messages: list[Message], tools: list[Tool], **kwargs) -> Optional[str]:
    """强制调用summarize工具获取摘要，带重试机制"""
    # 查找summarize工具
    summarize_tool = None
    for tool in tools:
        if getattr(tool, 'name', None) in ['summarize_task_state', 'summarize_the_task']:
            summarize_tool = tool
            break
    
    if not summarize_tool:
        logger.warning("未找到summarize工具")
        return None
    
    # 添加重试机制
    max_retries = kwargs.pop("max_retries", 3)  # 默认重试3次
    temp_tools = [summarize_tool]
    temp_tools_schema = [tool.openai_schema for tool in temp_tools]
    openai_messages = to_openai_messages(messages)
    
    for retry in range(max_retries):
        try:
            # 使用Azure或其他模型
            if 'gpt' in model.lower():
                client, deployment_name = get_azure_model(model)
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=openai_messages,
                    tools=temp_tools_schema,
                    tool_choice="required",
                    **{k: v for k, v in kwargs.items() if k not in ["tools", "tool_choice", "max_retries"]},
                )
            else:
                client, model_name = get_openai_model(model)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=openai_messages,
                    tools=temp_tools_schema,
                    tool_choice="required",
                    **{k: v for k, v in kwargs.items() if k not in ["tools", "tool_choice", "max_retries"]},
                )
            
            # 解析响应
            choice = response.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None)
            
            if tool_calls:
                for tc in tool_calls:
                    if tc.function.name in ['summarize_task_state', 'summarize_the_task']:
                        args = json.loads(tc.function.arguments)
                        if 'summary' in args:
                            return args['summary']
                        elif 'current_information' in args:
                            return args['current_information']
            
            return None
            
        except Exception as e:
            # 使用智能重试机制处理速率限制错误
            if _handle_rate_limit_error(e, retry, max_retries, "(summarize工具)"):
                continue
            else:
                # 其他错误或达到最大重试次数，立即返回
                logger.error(f"强制调用summarize工具失败: {e}")
                return None
    
    return None


def _extract_retry_after_from_error(error: Exception) -> int:
    """从错误中提取重试等待时间"""
    wait_time = 60  # 默认等待时间
    
    try:
        # 尝试从错误响应中提取 retry_after
        if hasattr(error, "response") and hasattr(error.response, "json"):
            error_data = error.response.json()
            if "retry_after" in error_data:
                wait_time = int(error_data["retry_after"])
                return wait_time
        
        # 尝试从错误消息中提取等待时间
        error_str = str(error)
        import re
        patterns = [
            r'retry after (\d+) seconds',
            r'retry after (\d+)s',
            r'wait (\d+) seconds',
            r'wait (\d+)s',
            r'(\d+) seconds',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                wait_time = int(match.group(1))
                break
                
    except Exception:
        pass
    
    return wait_time


def _handle_rate_limit_error(error: Exception, retry: int, max_retries: int, context: str = "") -> bool:
    """处理速率限制错误，返回是否应该继续重试"""
    if not (hasattr(error, "status_code") and error.status_code == 429):
        return False
    
    wait_time = _extract_retry_after_from_error(error)
    
    # 使用指数退避策略，但不超过从错误中提取的时间
    exponential_wait = 10 + retry * 6 + 2 ** (retry + 1)
    wait_time = min(wait_time, exponential_wait, 300)  # 最大等待5分钟
    
    if retry < max_retries - 1:
        logger.warning(f"RateLimitError (429) {context}: 等待 {wait_time}s 后重试 {retry + 1}/{max_retries}")
        import time
        time.sleep(wait_time)
        return True
    else:
        logger.error(f"RateLimitError (429) {context}: 达到最大重试次数，放弃重试")
        return False


def _extract_conversation_context_from_dict(messages: list[dict]) -> str:
    """从字典格式的消息列表中提取对话上下文"""
    context_parts = []
    
    # 添加最近的对话消息
    recent_messages = messages[-10:]  # 最近10条消息
    for msg in recent_messages:
        if isinstance(msg, dict) and 'content' in msg and msg['content']:
            role = msg.get('role', 'unknown')
            content = msg['content'][:200] if len(msg['content']) > 200 else msg['content']
            context_parts.append(f"{role}: {content}")
    
    return "\n".join(context_parts)


def _force_summarize_tool_call_dict(model: str, messages: list[dict], tools: list[dict], **kwargs) -> Optional[str]:
    """在 dict 消息/工具形态下，强制调用 summarize 工具获取摘要，带重试机制"""
    # 查找 summarize 工具（支持两种命名）
    summarize_tool = None
    if tools:
        for tool in tools:
            try:
                name = tool.get("function", {}).get("name") or tool.get("name")
            except Exception:
                name = None
            if name in ["summarize_task_state", "summarize_the_task"]:
                summarize_tool = tool
                break
    if not summarize_tool:
        logger.warning("未找到summarize工具 (dict)")
        return None

    # 添加重试机制
    max_retries = kwargs.pop("max_retries", 3)  # 默认重试3次
    temp_tools = [summarize_tool]
    tool_choice = "required"

    for retry in range(max_retries):
        try:
            if 'gpt' in model.lower():
                client, deployment_name = get_azure_model(model)
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=messages,
                    tools=temp_tools,
                    tool_choice=tool_choice,
                    **{k: v for k, v in kwargs.items() if k not in ["tools", "tool_choice", "max_retries"]},
                )
            else:
                client, model_name = get_openai_model(model)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=temp_tools,
                    tool_choice=tool_choice,
                    **{k: v for k, v in kwargs.items() if k not in ["tools", "tool_choice", "max_retries"]},
                )

            choice = response.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    fn = getattr(tc, "function", None)
                    fn_name = getattr(fn, "name", None) if fn else None
                    if fn_name in ["summarize_task_state", "summarize_the_task"]:
                        args_raw = getattr(fn, "arguments", "{}")
                        try:
                            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        except Exception:
                            args = {}
                        if isinstance(args, dict):
                            if "summary" in args and isinstance(args["summary"], str):
                                return args["summary"]
                            if "current_information" in args and isinstance(args["current_information"], str):
                                return args["current_information"]
            return None
            
        except Exception as e:
            # 使用智能重试机制处理速率限制错误
            if _handle_rate_limit_error(e, retry, max_retries, "(summarize工具-dict)"):
                continue
            else:
                # 其他错误或达到最大重试次数，立即返回
                logger.error(f"强制调用summarize工具失败 (dict): {e}")
                return None
    
    return None


def get_azure_model(
    model_name,
    model_version="2025-04-14",
    # model_version="2024-11-20",
    instance="msra/shared",
    api_version="2025-04-01-preview",
    # api_version="2024-10-21",
    scope="api://trapi/.default",
):
    if "4o" in model_name:
        model_version="2024-11-20"
        api_version="2024-10-21"

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


def get_openai_model(model_name, base_url="http://localhost:8000/v1"):
    client = OpenAI(
        base_url=base_url,
        api_key="EMPTY",
    )
    return client, model_name


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
    use_azure: bool = True,
    cache_obj: Optional[LLMCache] = None,
    use_hf: bool = False,  # 新增参数
    hf_device: str = "cuda",  # 新增参数
    tool_graph_path: Optional[str] = None,  # 新增参数：工具图路径
    enable_tool_selection: bool = False,  # 新增参数：是否启用工具选择
    last_tool_call: Optional[str] = None,  # 新增参数：上次调用的工具
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model, with cache support and tool graph selection.
    """
    openai_messages = to_openai_messages(messages)
    tools_schema = [tool.openai_schema for tool in tools] if tools else None

    # 初始化工具图选择器
    tool_selector = None
    if enable_tool_selection and tool_graph_path:
        try:
            tool_selector = ToolGraphSelector(tool_graph_path)
            logger.info("✓ 工具图选择器初始化成功")
        except Exception as e:
            logger.warning(f"工具图选择器初始化失败: {e}")
            tool_selector = None

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
    if 'gpt' in model:
        use_azure = True
    if 'Qwen' in model:
        use_hf = True

    # 自动调整 max_tokens，建议不超过 4096
    if "max_tokens" in kwargs and kwargs["max_tokens"] > 4096:
        logger.warning(f"max_tokens={kwargs['max_tokens']} 超过推荐值，自动调整为 4096")
        kwargs["max_tokens"] = 4096

    # 新增：Hugging Face本地推理分支

    max_retries = kwargs.pop("max_retries", 5)
    # Azure/TRAPI overrides and fallback controls
    azure_instance = kwargs.pop("azure_instance", os.environ.get("AZURE_INSTANCE", "msra/shared"))
    azure_instance_fallback = kwargs.pop("azure_instance_fallback", os.environ.get("AZURE_INSTANCE_FALLBACK"))
    azure_api_version_override = kwargs.pop("azure_api_version", os.environ.get("AZURE_API_VERSION"))
    azure_scope_override = kwargs.pop("azure_scope", os.environ.get("AZURE_SCOPE"))
    # 预调用：在模型调用前注入工具图建议，便于模型优先选择合适工具
    request_messages = openai_messages
    try:
        if tool_selector and tools and last_tool_call:
            # 先获取便宜的上下文用于判断是否需要summary
            pre_context = _extract_conversation_context_from_dict(messages)
            
            # 检查是否需要调用昂贵的summary工具
            should_use_summary = tool_selector._should_call_summarize_tool(
                model=model,
                current_tool=last_tool_call,
                conversation_context=pre_context,
                **{k: v for k, v in kwargs.items() if k not in ["tools", "tool_choice"]},
            )
            
            # 只有在需要时才调用昂贵的summary工具
            if should_use_summary:
                pre_context = _force_summarize_tool_call_dict(model, messages, tools, **{k: v for k, v in kwargs.items() if k not in ["tools", "tool_choice"]}) or pre_context
            
            available_tools_list = [tool.name for tool in tools] if tools else []
            pre_candidates = tool_selector.suggest_next_tools(
                model=model,
                current_tool=last_tool_call,
                conversation_context=pre_context,
                available_tools=available_tools_list,
                should_use_summary=should_use_summary,
                top_k=2,
                **{k: v for k, v in kwargs.items() if k not in ["tools", "tool_choice"]},
            )
            if pre_candidates:
                suggestion_text = f"tool graph suggestion: {', '.join(pre_candidates)}"
                request_messages = [{"role": "system", "content": suggestion_text}] + openai_messages
    except Exception as _e:
        logger.warning(f"预调用工具建议注入失败，跳过: {_e}")

    for retry in range(max_retries):
        try:
            if use_azure:
                
                if azure_api_version_override or azure_scope_override:
                    client, deployment_name = get_azure_model(
                        model,
                        instance=azure_instance,
                        api_version=azure_api_version_override or "2025-04-01-preview",
                        scope=azure_scope_override or "api://trapi/.default",
                    )
                else:
                    client, deployment_name = get_azure_model(model, instance=azure_instance)
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=request_messages,
                    tools=tools_schema,
                    tool_choice=tool_choice,
                    **kwargs,
                )
            elif use_hf:
                client, model_name = get_openai_model(model)
                response = client.chat.completions.create(
                    model=model,
                    messages=request_messages,
                    tools=tools_schema,
                    tool_choice='auto',
                    extra_body={  # 关键参数
                    "chat_template_kwargs": {
                        "enable_thinking": False  # 关闭思考模式
                        }
                    },
                    **kwargs,
                )
            # if use_hf:
            #     from transformers import AutoModelForCausalLM, AutoTokenizer
            #     import torch
            #     tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            #     if hf_device == "cuda" and torch.cuda.is_available():
            #         model_obj = AutoModelForCausalLM.from_pretrained(
            #             model,
            #             torch_dtype=torch.float16,
            #             device_map=None,
            #             trust_remote_code=True
            #         ).cuda()
            #     else:
            #         model_obj = AutoModelForCausalLM.from_pretrained(
            #             model,
            #             torch_dtype=torch.float32,
            #             device_map=None,
            #             trust_remote_code=True
            #         ).cpu()
            #     chat_messages = []
            #     for m in messages:
            #         if hasattr(m, "role") and hasattr(m, "content"):
            #             chat_messages.append({"role": m.role, "content": m.content})
            #     hf_tools = [tool.openai_schema for tool in tools] if tools else None
            #     inputs = tokenizer.apply_chat_template(
            #         chat_messages,
            #         tools=hf_tools,
            #         add_generation_prompt=True,
            #         return_tensors="pt"
            #     ).to(model_obj.device)
            #     gen_kwargs = dict(max_new_tokens=kwargs.get("max_new_tokens", 512), do_sample=False)
            #     if "eos_token_id" in kwargs:
            #         gen_kwargs["eos_token_id"] = kwargs["eos_token_id"]
            #     else:
            #         gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
            #     outputs = model_obj.generate(inputs, **gen_kwargs)
            #     response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)


            else:
                client, model_name = get_openai_model(model)
                # print(f"[DEBUG] model_name: {model_name}")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=request_messages,
                    tools=tools_schema,
                    tool_choice=tool_choice,
                    **kwargs,
                )

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
                tool_calls_obj = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                    for tc in tool_calls
                ]

            if 'Qwen' in model:
            # Parse inline <tool_call> blocks in content (e.g., from Qwen),
            # converting them to structured tool_calls.
            # Example:
            # <tool_call> {"name": "toggle_airplane_mode", "arguments": {"toggle": "off"}} </tool_call>
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
                                        requestor="assistant",
                                    )
                                )
                        except Exception:
                            # If parsing fails, skip this block and keep content as-is
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

            # 移除事后建议：建议已在调用前注入

            return message
        
        except Exception as e:
            # 使用智能重试机制处理速率限制错误
            if _handle_rate_limit_error(e, retry, max_retries, "(主函数)"):
                continue
            elif (
                (hasattr(e, "status_code") and e.status_code == 503)
                or "Allowed Security Groups Definition unavailable" in str(e)
            ):
                # 针对 TRAPI 503 明确处理：记录可操作信息并尝试回退实例
                trapi_url = f"https://trapi.research.microsoft.com/{azure_instance}"
                logger.error(
                    f"TRAPI 503 from {trapi_url}: {e}. "
                    f"可能是网关 Security Groups Definition 暂不可用。如持续失败，请附上后缀 '{azure_instance}' 邮件联系 trapi@microsoft.com。"
                )
                import time
                wait_time = min(2 ** (retry + 1), 30)
                if azure_instance_fallback and azure_instance != azure_instance_fallback:
                    logger.warning(
                        f"切换 Azure 实例: '{azure_instance}' -> '{azure_instance_fallback}' 后重试"
                    )
                    azure_instance = azure_instance_fallback
                time.sleep(wait_time)
            else:
                logger.error(f"Error for model {model}, retry {retry + 1}/{max_retries}: {e}")
                if retry == max_retries - 1:
                    logger.error(f"All retries failed for model {model}")
                    return AssistantMessage(role="assistant", content=f"Failed to generate response: {str(e)}")
                import time
                time.sleep(1)
    
    # 如果所有重试都失败，返回一个默认的错误消息
    logger.error(f"All retries failed for model {model}")
    return AssistantMessage(role="assistant", content="Failed to generate response after all retries")


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


# ========= Tau-Bench compatible completion API =========
class _TBMessage:
    def __init__(self, content: Optional[str], tool_calls: Optional[list] = None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self) -> dict:
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": self.tool_calls,
        }


class _TBChoice:
    def __init__(self, message: _TBMessage):
        self.message = message


class _TBResponse:
    def __init__(self, message: _TBMessage, response_cost: float = 0.0):
        self.choices = [_TBChoice(message)]
        # tau-bench reads this for cost
        self._hidden_params = {"response_cost": response_cost}


def completion(
    *,
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    temperature: float = 0.0,
    custom_llm_provider: Optional[str] = None,
    tool_graph_path: Optional[str] = None,  # 新增参数：工具图路径
    enable_tool_selection: bool = False,  # 新增参数：是否启用工具选择
    last_tool_call: Optional[str] = None,  # 新增参数：上次调用的工具
    **kwargs: Any,
) -> _TBResponse:
    """
    AzureOpenAI-backed drop-in for litellm.completion with the same call style
    used in tau-bench (model/messages/tools/temperature/custom_llm_provider).

    - Does not change how callers specify the model or parameters
    - Returns an object with .choices[0].message and ._hidden_params['response_cost']
    """
    # Provider gate (keep call style intact, but enforce azure in this wrapper)
    if custom_llm_provider and custom_llm_provider.lower() != "azure":
        raise ValueError(f"Unsupported provider '{custom_llm_provider}'. Use 'azure'.")

    # Reuse Azure auth helpers defined above
    # Prefer the simpler get_azure_model API already present in this module
    client, deployment_name = get_azure_model(model)

    # Default tool_choice behavior similar to litellm when tools are present
    tool_choice = kwargs.get("tool_choice")
    if tool_choice is None and tools:
        tool_choice = "auto"

    # 初始化工具图选择器
    tool_selector = None
    if enable_tool_selection and tool_graph_path:
        try:
            tool_selector = ToolGraphSelector(tool_graph_path)
            logger.info("✓ 工具图选择器初始化成功 (completion)")
        except Exception as e:
            logger.warning(f"工具图选择器初始化失败 (completion): {e}")
            tool_selector = None

    max_retries = kwargs.pop("max_retries", 5)
    # Azure/TRAPI overrides and fallback controls

    message_obj: _TBMessage | None = None
    # 预调用：在模型调用前注入工具图建议
    request_messages = messages
    try:
        if tool_selector and tools and last_tool_call:
            # 先获取便宜的上下文用于判断是否需要summary
            pre_context = _extract_conversation_context_from_dict(messages)
            
            # 检查是否需要调用昂贵的summary工具
            should_use_summary = tool_selector._should_call_summarize_tool(
                model=model,
                current_tool=last_tool_call,
                conversation_context=pre_context,
                **{k: v for k, v in kwargs.items() if k not in ["tools", "tool_choice"]},
            )
            
            # 只有在需要时才调用昂贵的summary工具
            if should_use_summary:
                pre_context = _force_summarize_tool_call_dict(model, messages, tools, **{k: v for k, v in kwargs.items() if k not in ["tools", "tool_choice"]}) or pre_context

            available_tools_list = [t["function"]["name"] for t in tools] if tools else []
            pre_candidates = tool_selector.suggest_next_tools(
                model=model,
                current_tool=last_tool_call,
                conversation_context=pre_context,
                available_tools=available_tools_list,
                should_use_summary=should_use_summary,
                top_k=2,
                **{k: v for k, v in kwargs.items() if k not in ["tools", "tool_choice"]},
            )
            if pre_candidates:
                suggestion_text = f"tool graph suggestion: {', '.join(pre_candidates)}"
                request_messages = [{"role": "system", "content": suggestion_text}] + messages
    except Exception as _e:
        logger.warning(f"预调用工具建议注入失败 (completion)，跳过: {_e}")

    for retry in range(max_retries):
        try:
            # Direct OpenAI-compatible request
            resp = client.chat.completions.create(
                model=deployment_name,
                messages=request_messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                **{k: v for k, v in kwargs.items() if k not in {"custom_llm_provider"}},
            )

            choice = resp.choices[0]
            content = choice.message.content
            tool_calls = getattr(choice.message, "tool_calls", None)

            # Normalize tool_calls to what tau-bench agents expect (OpenAI shape)
            norm_tool_calls = None
            if tool_calls:
                norm_tool_calls = [
                    {
                        "id": tc.id,
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]

            message_obj = _TBMessage(content=content, tool_calls=norm_tool_calls)
            
            # 移除事后建议：建议已在调用前注入

        except Exception as e:
            # 使用智能重试机制处理速率限制错误
            if _handle_rate_limit_error(e, retry, max_retries, "(completion函数)"):
                continue
            else:
                logger.error(f"Error for model {model}, retry {retry + 1}/{max_retries}: {e}")
                import time
                time.sleep(min(2 ** (retry + 1), 30))

    # 若重试后仍失败，返回规范化的 _TBResponse，避免上层 AttributeError
    error_message = _TBMessage(content="Failed to generate response after retries", tool_calls=None)
    return _TBResponse(message=message_obj or error_message, response_cost=0.0)
