import json
import os
from copy import deepcopy
from typing import List, Optional, Dict, Any
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

# 可选导入embedding相关库
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_AVAILABLE = True
    print("Embedding libraries successfully imported.")
except ImportError:
    EMBEDDING_AVAILABLE = False
    logger.warning("Embedding libraries not available. Tool selection will be disabled.")
    print("Embedding libraries not available.")

from tau2.agent.base import (
    LocalAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    UserMessage,
)
from tau2.data_model.tasks import Action, Task
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import generate

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()


AGENT_INSTRUCTION_SUMMARY = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

- You should decide when to summarize the current state and information of the task with the `summarize_the_task` tool and call `summarize_the_task` at least once in the conversation to help you make decisions about action.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

# Please deciside when to summarize the current state of the task with the `summarize_task_state` tool and call `summarize_task_state` at least once in the conversation.

# please deciside when to summarize the current state of the task with the `summarize_task_state` tool and at least once in the conversation.



# In turns where you get new information, please summarize the current state of the task with the `summarize_task_state` tool.


# In turns where you get new information, please summarize the current state of the task form the key elements below based on the information you have and don't make up any information.
# Make a plan based on the current state and compare the current plan with the previous plan and update the plan if needed.

# Key elements:
# - Cellular Service
# - Mobile Data
# - MMS

# In turns where you get new information, please summarize the current state of the task and make a plan based on the current state.

# In turns where you get new information, please summarize the current state of the task form the key elements below based on the information you have and don't make up any information.
# Make a plan based on the current state and compare the current plan with the previous plan and update the plan if needed.

# please summary the current state of the task and make decision based on the state.

# You can follow these sugestions:
# ​After EVERY action or set of actions taken:
# Analyze the effectiveness of your previous steps
# Identify what worked well and what didn't
# Explicitly note any new information gained

# Maintain and update these key elements:
# Current environment state
# Progress toward overall goal
# Remaining obstacles/challenges
# Resources available (time, tools, information)

# Based on your reflection:
# Modify your strategy if needed
# Adjust your next steps to account for new information
# Consider alternative approaches that may be more effective
# Explicitly state your reasoning for any changes

# ​Execution with Meta-Cognition:​​
# Before taking action, briefly explain your intended approach
# After action execution, evaluate actual outcomes vs. expectations


SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()

# 工具选择器类
if EMBEDDING_AVAILABLE:
    class EmbeddingBasedToolSelector:
        """基于embedding的工具选择器"""
        
        def __init__(self, tool_graph_path: str, model_name: str = "all-MiniLM-L6-v2"):
            """
            初始化工具选择器
            
            Args:
                tool_graph_path: 工具图数据文件路径
                model_name: embedding模型名称
            """
            self.tool_graph_path = tool_graph_path
            self.tool_graph = self._load_tool_graph()
            
            try:

                self.model = SentenceTransformer(model_name)
                logger.info(f"✓ Embedding模型加载成功: {model_name}")
                print(f"✓ Embedding模型加载成功: {model_name}")
            except Exception as e:
                logger.warning(f"Embedding模型加载失败: {e}，将禁用工具选择功能")
                self.model = None
                print(f"Embedding模型加载失败: {e}，将禁用工具选择功能")
            
            # 缓存embedding以提高性能
            self.embedding_cache = {}
            
            logger.info(f"工具选择器初始化完成，加载了 {len(self.tool_graph.get('nodes', []))} 个工具")
        
        def _load_tool_graph(self) -> Dict:
            """加载工具图数据"""
            try:
                with open(self.tool_graph_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载工具图失败: {e}")
                return {'nodes': [], 'edges': []}
        
        def _get_embedding(self, text: str) -> np.ndarray:
            """获取文本的embedding，带缓存"""
            if not self.model:
                return np.zeros(384)  # 返回零向量作为fallback
            
            if text not in self.embedding_cache:
                try:
                    self.embedding_cache[text] = self.model.encode([text])[0]
                except Exception as e:
                    logger.error(f"生成embedding失败: {e}")
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
                return float(similarity)
            except Exception as e:
                logger.error(f"计算相似度失败: {e}")
                return 0.0
        
        def _summarize_context(self, context: str) -> str:
            """总结对话上下文（已废弃：现在期望外部传入总结内容）"""
            # 兼容保留：若外部未提供总结，仍进行一次简单截断
            if len(context) > 300:
                return f"用户问题: {context[:150]}... 当前状态: {context[-150:]}"
            return f"用户问题: {context}"
        
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
            
            logger.info(f"🔍 工具图调试 - 找到 {len(connected_tools)} 个相连工具: {[t['tool'] for t in connected_tools]}")
            
            return connected_tools
        
        def select_next_tool(self, current_tool: str, conversation_context: str, 
                            available_tools: List[str] = None) -> Optional[str]:
            """选择下一个工具"""
            if not self.model:
                logger.warning("Embedding模型未加载，无法进行工具选择")
                return None
                
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
            
            # 使用外部提供的 summarize 工具产出的内容作为对比文本
            # 若上游未能提供，则退回到本地简要截断
            context_summary = conversation_context or ""
            if not context_summary:
                context_summary = self._summarize_context("")
            logger.info(f"上下文总结(来自summarize工具或回退): {context_summary}")
            
            # 计算相似度
            tool_scores = []
            
            for tool_info in connected_tools:
                tool_name = tool_info['tool']
                current_infos = tool_info['current_information']
                
                if not current_infos:
                    # 如果没有current_information，使用权重作为分数
                    score = tool_info['weight'] / 1000.0  # 归一化权重
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
                
                for info in current_infos:
                    similarity = self._calculate_similarity(context_summary, info)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = info
                
                tool_scores.append({
                    'tool': tool_name,
                    'score': max_similarity,
                    'reason': 'similarity_based',
                    'best_match': best_match,
                    'weight': tool_info['weight']
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
        def get_tool_selection_explanation(self, current_tool: str, conversation_context: str,
                                         available_tools: List[str] = None, should_summarize=None) -> Dict:
            """获取工具选择的详细解释"""
            if not self.model:
                logger.warning("🔍 工具选择调试 - Embedding模型未加载")
                print("🔍 工具选择调试 - Embedding模型未加载")
                return {'error': 'Embedding模型未加载'}                
            connected_tools = self.get_connected_tools(current_tool)
            logger.info(f"🔍 工具选择调试 - 原始相连工具数量: {len(connected_tools)}")       
            if available_tools:
                connected_tools = [
                    tool for tool in connected_tools 
                    if tool['tool'] in available_tools
                ]
                logger.info(f"🔍 工具选择调试 - 过滤后相连工具数量: {len(connected_tools)}")
                logger.info(f"🔍 工具选择调试 - 可用工具: {available_tools[:5]}...")         
            if not connected_tools:
                logger.warning(f"🔍 工具选择调试 - 工具 {current_tool} 没有相连的工具")
                return {
                    'error': f'工具 {current_tool} 没有相连的工具',
                    'current_tool': current_tool,
                    'available_connected_tools': 0,
                    'tool_analysis': []
                }         
            # conversation_context 期望为 summarize 工具产物
            context_summary = conversation_context or ""
            if not context_summary:
                context_summary = self._summarize_context("")         
            logger.info(f"🔍 工具选择调试 - 上下文摘要长度: {len(context_summary)}")       
            print(f"🔍 工具选择调试 - 上下文摘要内容: {context_summary}...")  
            tool_analysis = []
            for tool_info in connected_tools:
                tool_name = tool_info['tool']
                current_infos = tool_info['current_information']              
                if not should_summarize or not current_infos:
                    score = tool_info['weight'] / 1000.0
                    tool_analysis.append({
                        'tool': tool_name,
                        'score': score,
                        'reason': 'weight_based',
                        'details': {
                            'weight': tool_info['weight'],
                            'count': tool_info['count']
                        }
                    })
                    logger.info(f"🔍 工具选择调试 - {tool_name}: 基于权重, 分数={score:.4f}")
                elif should_summarize and current_infos:
                    similarities = []
                    for info in current_infos:
                        similarity = self._calculate_similarity(context_summary, info)
                        similarities.append({
                            'info': info,
                            'similarity': similarity
                        })
                        print(f"🔍 工具选择调试 - {tool_name}: 计算相似度: {similarity:.4f} for info: {info[:100]}...")                 
                    max_sim = max(similarities, key=lambda x: x['similarity'])
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
                    logger.info(f"🔍 工具选择调试 - {tool_name}: 基于相似度, 分数={max_sim['similarity']:.4f}")         
            tool_analysis.sort(key=lambda x: x['score'], reverse=True)          
            result = {
                'current_tool': current_tool,
                'context_summary': context_summary,
                'available_connected_tools': len(connected_tools),
                'tool_analysis': tool_analysis,
                'selected_tool': tool_analysis[0]['tool'] if tool_analysis else None,
                'selection_score': tool_analysis[0]['score'] if tool_analysis else 0
            }          
            logger.info(f"🔍 工具选择调试 - 最终结果: {len(tool_analysis)} 个工具分析")
            if tool_analysis:
                logger.info(f"🔍 工具选择调试 - 推荐工具: {result['selected_tool']}, 分数: {result['selection_score']:.4f}")        
            return result
else:
    # 如果没有embedding库，创建一个空的工具选择器
    class EmbeddingBasedToolSelector:
        def __init__(self, tool_graph_path: str, model_name: str = "all-MiniLM-L6-v2"):
            logger.warning("Embedding libraries not available. Tool selection disabled.")
            self.model = None 
        def select_next_tool(self, current_tool: str, conversation_context: str, 
                            available_tools: List[str] = None) -> Optional[str]:
            return None
        def get_tool_selection_explanation(self, current_tool: str, conversation_context: str,
                                         available_tools: List[str] = None) -> Dict:
            return {'error': 'Embedding libraries not available'}


# 工具选择器类solo
# if EMBEDDING_AVAILABLE:
#     class EmbeddingBasedToolSelector:
#         """基于embedding的工具选择器"""
        
#         def __init__(self, tool_graph_path: str, model_name: str = "all-MiniLM-L6-v2"):
#             """
#             初始化工具选择器
            
#             Args:
#                 tool_graph_path: 工具图数据文件路径
#                 model_name: embedding模型名称
#             """
#             self.tool_graph_path = tool_graph_path
#             self.tool_graph = self._load_tool_graph()
            
#             try:
#                 self.model = SentenceTransformer(model_name)
#                 logger.info(f"✓ Embedding模型加载成功: {model_name}")
#             except Exception as e:
#                 logger.warning(f"Embedding模型加载失败: {e}，将禁用工具选择功能")
#                 self.model = None
            
#             # 缓存embedding以提高性能
#             self.embedding_cache = {}
            
#             logger.info(f"工具选择器初始化完成，加载了 {len(self.tool_graph.get('nodes', []))} 个工具")
        
#         def _load_tool_graph(self) -> Dict:
#             """加载工具图数据"""
#             try:
#                 with open(self.tool_graph_path, 'r', encoding='utf-8') as f:
#                     return json.load(f)
#             except Exception as e:
#                 logger.error(f"加载工具图失败: {e}")
#                 return {'nodes': [], 'edges': []}
        
#         def _get_embedding(self, text: str) -> np.ndarray:
#             """获取文本的embedding，带缓存"""
#             if not self.model:
#                 return np.zeros(384)  # 返回零向量作为fallback
            
#             if text not in self.embedding_cache:
#                 try:
#                     self.embedding_cache[text] = self.model.encode([text])[0]
#                 except Exception as e:
#                     logger.error(f"生成embedding失败: {e}")
#                     return np.zeros(384)
            
#             return self.embedding_cache[text]
        
#         def _calculate_similarity(self, text1: str, text2: str) -> float:
#             """计算两个文本的相似度"""
#             if not self.model:
#                 return 0.0
                
#             try:
#                 emb1 = self._get_embedding(text1)
#                 emb2 = self._get_embedding(text2)
#                 similarity = cosine_similarity([emb1], [emb2])[0][0]
#                 return float(similarity)
#             except Exception as e:
#                 logger.error(f"计算相似度失败: {e}")
#                 return 0.0
        
#         def _summarize_context(self, context: str) -> str:
#             """总结对话上下文（已废弃：现在期望外部传入总结内容）"""
#             # 兼容保留：若外部未提供总结，仍进行一次简单截断
#             if len(context) > 300:
#                 return f"用户问题: {context[:150]}... 当前状态: {context[-150:]}"
#             return f"用户问题: {context}"
        
#         def get_connected_tools(self, tool_name: str) -> List[Dict]:
#             """获取与指定工具相连的所有工具"""
#             connected_tools = []
            
#             for edge in self.tool_graph.get('edges', []):
#                 if edge['u'] == tool_name:
#                     connected_tools.append({
#                         'tool': edge['v'],
#                         'weight': edge.get('weight', 0),
#                         'count': edge.get('count', 0),
#                         'current_information': edge.get('current_information', [])
#                     })
            
#             return connected_tools
        
#         def select_next_tool(self, current_tool: str, conversation_context: str, 
#                             available_tools: List[str] = None) -> Optional[str]:
#             """选择下一个工具"""
#             if not self.model:
#                 logger.warning("Embedding模型未加载，无法进行工具选择")
#                 return None
                
#             logger.info(f"为工具 {current_tool} 选择下一个工具")
            
#             # 获取相连的工具
#             connected_tools = self.get_connected_tools(current_tool)
            
#             if not connected_tools:
#                 logger.warning(f"工具 {current_tool} 没有相连的工具")
#                 return None
            
#             # 如果指定了可用工具，过滤连接的工具
#             if available_tools:
#                 connected_tools = [
#                     tool for tool in connected_tools 
#                     if tool['tool'] in available_tools
#                 ]
            
#             if not connected_tools:
#                 logger.warning(f"工具 {current_tool} 的相连工具都不在可用工具列表中")
#                 return None
            
#             # 使用外部提供的 summarize 工具产出的内容作为对比文本
#             # 若上游未能提供，则退回到本地简要截断
#             context_summary = conversation_context or ""
#             if not context_summary:
#                 context_summary = self._summarize_context("")
#             logger.info(f"上下文总结(来自summarize工具或回退): {context_summary}")
            
#             # 计算相似度
#             tool_scores = []
            
#             for tool_info in connected_tools:
#                 tool_name = tool_info['tool']
#                 current_infos = tool_info['current_information']
                
#                 if not current_infos:
#                     # 如果没有current_information，使用权重作为分数
#                     score = tool_info['weight'] / 1000.0  # 归一化权重
#                     tool_scores.append({
#                         'tool': tool_name,
#                         'score': score,
#                         'reason': 'weight_based',
#                         'weight': tool_info['weight']
#                     })
#                     continue
                
#                 # 计算与所有current_information的最大相似度
#                 max_similarity = 0.0
#                 best_match = ""
                
#                 for info in current_infos:
#                     similarity = self._calculate_similarity(context_summary, info)
#                     if similarity > max_similarity:
#                         max_similarity = similarity
#                         best_match = info
                
#                 tool_scores.append({
#                     'tool': tool_name,
#                     'score': max_similarity,
#                     'reason': 'similarity_based',
#                     'best_match': best_match,
#                     'weight': tool_info['weight']
#                 })
            
#             # 按分数排序
#             tool_scores.sort(key=lambda x: x['score'], reverse=True)
            
#             # 选择分数最高的工具
#             best_tool = tool_scores[0]
            
#             logger.info(f"选择工具: {best_tool['tool']}")
#             logger.info(f"选择原因: {best_tool['reason']}")
#             logger.info(f"分数: {best_tool['score']:.4f}")
            
#             if best_tool['reason'] == 'similarity_based':
#                 logger.info(f"最佳匹配: {best_tool['best_match'][:100]}...")
            
#             return best_tool['tool']
        
#         def get_tool_selection_explanation(self, current_tool: str, conversation_context: str,
#                                          available_tools: List[str] = None) -> Dict:
#             """获取工具选择的详细解释"""
#             if not self.model:
#                 return {'error': 'Embedding模型未加载'}
                
#             connected_tools = self.get_connected_tools(current_tool)
            
#             if available_tools:
#                 connected_tools = [
#                     tool for tool in connected_tools 
#                     if tool['tool'] in available_tools
#                 ]
            
#             # conversation_context 期望为 summarize 工具产物
#             context_summary = conversation_context or ""
#             if not context_summary:
#                 context_summary = self._summarize_context("")
            
#             tool_analysis = []
#             for tool_info in connected_tools:
#                 tool_name = tool_info['tool']
#                 current_infos = tool_info['current_information']
                
#                 if not current_infos:
#                     tool_analysis.append({
#                         'tool': tool_name,
#                         'score': tool_info['weight'] / 1000.0,
#                         'reason': 'weight_based',
#                         'details': {
#                             'weight': tool_info['weight'],
#                             'count': tool_info['count']
#                         }
#                     })
#                 else:
#                     similarities = []
#                     for info in current_infos:
#                         similarity = self._calculate_similarity(context_summary, info)
#                         similarities.append({
#                             'info': info,
#                             'similarity': similarity
#                         })
                    
#                     max_sim = max(similarities, key=lambda x: x['similarity'])
#                     tool_analysis.append({
#                         'tool': tool_name,
#                         'score': max_sim['similarity'],
#                         'reason': 'similarity_based',
#                         'details': {
#                             'best_match': max_sim['info'],
#                             'all_similarities': similarities,
#                             'weight': tool_info['weight']
#                         }
#                     })
            
#             tool_analysis.sort(key=lambda x: x['score'], reverse=True)
            
#             return {
#                 'current_tool': current_tool,
#                 'context_summary': context_summary,
#                 'available_connected_tools': len(connected_tools),
#                 'tool_analysis': tool_analysis,
#                 'selected_tool': tool_analysis[0]['tool'] if tool_analysis else None,
#                 'selection_score': tool_analysis[0]['score'] if tool_analysis else 0
#             }
# else:
#     # 如果没有embedding库，创建一个空的工具选择器
#     class EmbeddingBasedToolSelector:
#         def __init__(self, tool_graph_path: str, model_name: str = "all-MiniLM-L6-v2"):
#             logger.warning("Embedding libraries not available. Tool selection disabled.")
#             self.model = None
        
#         def select_next_tool(self, current_tool: str, conversation_context: str, 
#                             available_tools: List[str] = None) -> Optional[str]:
#             return None
        
#         def get_tool_selection_explanation(self, current_tool: str, conversation_context: str,
#                                          available_tools: List[str] = None) -> Dict:
#             return {'error': 'Embedding libraries not available'}


# no context
# # 工具选择器类
# if EMBEDDING_AVAILABLE:
#     class EmbeddingBasedToolSelector:
#         """基于embedding的工具选择器"""
        
#         def __init__(self, tool_graph_path: str, model_name: str = "all-MiniLM-L6-v2"):
#             """
#             初始化工具选择器
            
#             Args:
#                 tool_graph_path: 工具图数据文件路径
#                 model_name: embedding模型名称
#             """
#             self.tool_graph_path = tool_graph_path
#             self.tool_graph = self._load_tool_graph()
            
#             try:
#                 self.model = SentenceTransformer(model_name)
#                 logger.info(f"✓ Embedding模型加载成功: {model_name}")
#             except Exception as e:
#                 logger.warning(f"Embedding模型加载失败: {e}，将禁用工具选择功能")
#                 self.model = None
            
#             # 缓存embedding以提高性能
#             self.embedding_cache = {}
            
#             logger.info(f"工具选择器初始化完成，加载了 {len(self.tool_graph.get('nodes', []))} 个工具")
        
#         def _load_tool_graph(self) -> Dict:
#             """加载工具图数据"""
#             try:
#                 with open(self.tool_graph_path, 'r', encoding='utf-8') as f:
#                     return json.load(f)
#             except Exception as e:
#                 logger.error(f"加载工具图失败: {e}")
#                 return {'nodes': [], 'edges': []}
        
#         def _get_embedding(self, text: str) -> np.ndarray:
#             """获取文本的embedding，带缓存"""
#             if not self.model:
#                 return np.zeros(384)  # 返回零向量作为fallback
            
#             if text not in self.embedding_cache:
#                 try:
#                     self.embedding_cache[text] = self.model.encode([text])[0]
#                 except Exception as e:
#                     logger.error(f"生成embedding失败: {e}")
#                     return np.zeros(384)
            
#             return self.embedding_cache[text]
        
#         def _calculate_similarity(self, text1: str, text2: str) -> float:
#             """计算两个文本的相似度"""
#             if not self.model:
#                 return 0.0
                
#             try:
#                 emb1 = self._get_embedding(text1)
#                 emb2 = self._get_embedding(text2)
#                 similarity = cosine_similarity([emb1], [emb2])[0][0]
#                 return float(similarity)
#             except Exception as e:
#                 logger.error(f"计算相似度失败: {e}")
#                 return 0.0
        
#         def _summarize_context(self, context: str) -> str:
#             """总结对话上下文（已废弃：现在期望外部传入总结内容）"""
#             # 兼容保留：若外部未提供总结，仍进行一次简单截断
#             if len(context) > 300:
#                 return f"用户问题: {context[:150]}... 当前状态: {context[-150:]}"
#             return f"用户问题: {context}"
        
#         def get_connected_tools(self, tool_name: str) -> List[Dict]:
#             """获取与指定工具相连的所有工具"""
#             connected_tools = []
            
#             # 支持两种格式：edges (u/v) 和 links (source/target)
#             edges = self.tool_graph.get('edges', [])
#             links = self.tool_graph.get('links', [])
            
#             logger.info(f"🔍 工具图调试 - 工具: {tool_name}")
#             logger.info(f"🔍 工具图调试 - edges数量: {len(edges)}, links数量: {len(links)}")
            
#             # 处理 edges 格式 (u/v)
#             for edge in edges:
#                 if edge.get('u') == tool_name:
#                     connected_tools.append({
#                         'tool': edge['v'],
#                         'weight': edge.get('weight', 0),
#                         'count': edge.get('count', 0),
#                         'current_information': edge.get('current_information', [])
#                     })
            
#             # 处理 links 格式 (source/target)
#             for link in links:
#                 if link.get('source') == tool_name:
#                     connected_tools.append({
#                         'tool': link['target'],
#                         'weight': link.get('weight', 1),  # 默认权重为1
#                         'count': link.get('count', 1),    # 默认计数为1
#                         'current_information': link.get('current_information', [])
#                     })
            
#             logger.info(f"🔍 工具图调试 - 找到 {len(connected_tools)} 个相连工具: {[t['tool'] for t in connected_tools]}")
            
#             return connected_tools
        
#         def select_next_tool(self, current_tool: str, conversation_context: str, 
#                             available_tools: List[str] = None) -> Optional[str]:
#             """选择下一个工具（仅基于图边权重）"""
#             logger.info(f"为工具 {current_tool} 选择下一个工具（基于权重）")

#             # 获取相连的工具
#             connected_tools = self.get_connected_tools(current_tool)

#             if not connected_tools:
#                 logger.warning(f"工具 {current_tool} 没有相连的工具")
#                 return None

#             # 如果指定了可用工具，过滤连接的工具
#             if available_tools:
#                 connected_tools = [
#                     tool for tool in connected_tools 
#                     if tool['tool'] in available_tools
#                 ]

#             if not connected_tools:
#                 logger.warning(f"工具 {current_tool} 的相连工具都不在可用工具列表中")
#                 return None

#             # 仅基于权重打分
#             tool_scores = []
#             for tool_info in connected_tools:
#                 tool_scores.append({
#                     'tool': tool_info['tool'],
#                     'score': float(tool_info.get('weight', 0)),
#                     'reason': 'weight_based',
#                     'weight': tool_info.get('weight', 0)
#                 })

#             tool_scores.sort(key=lambda x: x['score'], reverse=True)
#             best_tool = tool_scores[0]

#             logger.info(f"选择工具: {best_tool['tool']}")
#             logger.info(f"选择原因: {best_tool['reason']}")
#             logger.info(f"权重: {best_tool['weight']}")

#             return best_tool['tool']
        
#         def get_tool_selection_explanation(self, current_tool: str, conversation_context: str,
#                                          available_tools: List[str] = None) -> Dict:
#             """获取工具选择的详细解释（仅基于图边权重）"""
#             connected_tools = self.get_connected_tools(current_tool)
#             logger.info(f"🔍 工具选择调试 - 原始相连工具数量: {len(connected_tools)}")

#             if available_tools:
#                 connected_tools = [
#                     tool for tool in connected_tools 
#                     if tool['tool'] in available_tools
#                 ]
#                 logger.info(f"🔍 工具选择调试 - 过滤后相连工具数量: {len(connected_tools)}")
#                 logger.info(f"🔍 工具选择调试 - 可用工具: {available_tools[:5]}...")

#             if not connected_tools:
#                 logger.warning(f"🔍 工具选择调试 - 工具 {current_tool} 没有相连的工具")
#                 return {
#                     'error': f'工具 {current_tool} 没有相连的工具',
#                     'current_tool': current_tool,
#                     'available_connected_tools': 0,
#                     'tool_analysis': []
#                 }

#             tool_analysis = []
#             for tool_info in connected_tools:
#                 tool_name = tool_info['tool']
#                 score = float(tool_info.get('weight', 0))
#                 tool_analysis.append({
#                     'tool': tool_name,
#                     'score': score,
#                     'reason': 'weight_based',
#                     'details': {
#                         'weight': tool_info.get('weight', 0),
#                         'count': tool_info.get('count', 0)
#                     }
#                 })
#                 logger.info(f"🔍 工具选择调试 - {tool_name}: 基于权重, 权重={score}")

#             tool_analysis.sort(key=lambda x: x['score'], reverse=True)

#             result = {
#                 'current_tool': current_tool,
#                 'context_summary': '',
#                 'available_connected_tools': len(connected_tools),
#                 'tool_analysis': tool_analysis,
#                 'selected_tool': tool_analysis[0]['tool'] if tool_analysis else None,
#                 'selection_score': tool_analysis[0]['score'] if tool_analysis else 0
#             }

#             logger.info(f"🔍 工具选择调试 - 最终结果: {len(tool_analysis)} 个工具分析")
#             if tool_analysis:
#                 logger.info(f"🔍 工具选择调试 - 推荐工具: {result['selected_tool']}, 权重: {result['selection_score']}")

#             return result
# else:
#     # 如果没有embedding库，创建一个空的工具选择器
#     class EmbeddingBasedToolSelector:
#         def __init__(self, tool_graph_path: str, model_name: str = "all-MiniLM-L6-v2"):
#             logger.warning("Embedding libraries not available. Tool selection disabled.")
#             self.model = None
        
#         def select_next_tool(self, current_tool: str, conversation_context: str, 
#                             available_tools: List[str] = None) -> Optional[str]:
#             return None
        
#         def get_tool_selection_explanation(self, current_tool: str, conversation_context: str,
#                                          available_tools: List[str] = None) -> Dict:
#             return {'error': 'Embedding libraries not available'}


class LLMAgentState(BaseModel):
    """The state of the agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    # 增强功能：工具选择相关状态
    tool_call_history: list[Dict[str, Any]] = []
    last_tool_suggestion: Optional[str] = None


class LLMAgent(LocalAgent[LLMAgentState]):
    """
    An LLM agent that can be used to solve a task.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        tool_graph_path: Optional[str] = None,
        enable_tool_selection: bool = False,
    ):
        """
        Initialize the LLMAgent.
        
        Args:
            tools: 工具列表
            domain_policy: 域策略
            llm: LLM模型
            llm_args: LLM参数
            tool_graph_path: 工具图数据文件路径（可选）
            enable_tool_selection: 是否启用工具选择功能（可选）
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        # self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        
        # 初始化工具选择器
        self.enable_tool_selection = enable_tool_selection
        
        if enable_tool_selection and tool_graph_path and EMBEDDING_AVAILABLE:
            self.tool_selector = EmbeddingBasedToolSelector(tool_graph_path)
            logger.info("✓ 工具选择器初始化成功")
        else:
            self.tool_selector = None
            if enable_tool_selection and not EMBEDDING_AVAILABLE:
                logger.warning("工具选择功能已请求但embedding库不可用")
            elif enable_tool_selection and not tool_graph_path:
                logger.warning("工具选择功能已请求但未提供工具图路径")
        
        # Add summary tool from GenericToolKit if summary is enabled
        if self.llm_args.get("summary", False):
            from tau2.environment.toolkit import GenericToolKit
            generic_tools = GenericToolKit()
            summary_tool = generic_tools.get_tools()["summarize_task_state"]
            self.tools.append(summary_tool)

    @property
    def system_prompt(self) -> str:
        if self.llm_args.get("summary", False):
            agent_instruction = AGENT_INSTRUCTION_SUMMARY
        else:
            agent_instruction = AGENT_INSTRUCTION
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=agent_instruction
        )
        # if self.llm_args["summary"]==True:            
        #     return SYSTEM_PROMPT.format(
        #         domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION_SUMMARY
        #     )
        # else:
        #     return SYSTEM_PROMPT.format(
        #         domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        #     )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        # messages = state.system_messages + state.messages
        
        # Filter out summary parameter as it's only used for system prompt generation
        llm_kwargs = {k: v for k, v in self.llm_args.items() if k != "summary"}
        
        # Log available tools for monitoring
        if self.llm_args.get("summary", False):
            tool_names = [tool.name for tool in self.tools]
            logger.info(f"🛠️ Available tools: {tool_names}")
            if "summarize_task_state" in tool_names:
                logger.info("✅ summarize_task_state tool is available")
            else:
                logger.warning("❌ summarize_task_state tool is NOT available")
        
        # 在生成消息前提供工具建议（增强功能）
        tool_suggestion_text = ""
        if self.enable_tool_selection and state.tool_call_history:
            # 获取两个候选工具
            suggested_tools = self._suggest_next_tools(state, top_k=2)
            print(f"[TOOL_SUGGESTION] Tool suggestions: {suggested_tools}")
            
            # 将建议添加到消息中
            if suggested_tools:
                tool_suggestion_text = f"Suggested next tools: {', '.join(suggested_tools)}\n\n"
                # 将建议添加到最后的用户消息中，让LLM看到建议
                if state.messages and isinstance(state.messages[-1], UserMessage):
                    state.messages[-1].content += tool_suggestion_text
                    print(f"[TOOL_SUGGESTION] Added tool suggestions to user message")
        messages = state.system_messages + state.messages
        
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            **llm_kwargs,
        )
        
        
        # Monitor tool calls for summarize_task_state
        if self.llm_args.get("summary", False) and assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                if tool_call.name == "summarize_task_state":
                    logger.info(f"🎯 Agent called summarize_task_state tool!")
                    logger.info(f"📋 Arguments: {tool_call.arguments}")
                    print(f"📋 Arguments: {tool_call.arguments}")
                else:
                    logger.info(f"🔧 Agent called tool: {tool_call.name}")


        
        # 记录工具调用历史（增强功能）
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tool_call_record = {
                    'tool': tool_call.name,
                    'arguments': tool_call.arguments,
                    'timestamp': len(state.tool_call_history)
                }
                state.tool_call_history.append(tool_call_record)

        # 工具调用后的建议逻辑已移动到生成消息前
        
        state.messages.append(assistant_message)
        return assistant_message, state

    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed
    
    # 增强功能：工具选择相关方法
    def _extract_conversation_context(self, state: LLMAgentState) -> str:
        """从状态中提取对话上下文"""
        context_parts = []
        
        # 添加最近的对话消息
        recent_messages = state.messages[-10:]  # 最近10条消息
        for msg in recent_messages:
            if hasattr(msg, 'content') and msg.content:
                role = getattr(msg, 'role', 'unknown')
                content = msg.content[:200] if len(msg.content) > 200 else msg.content
                context_parts.append(f"{role}: {content}")
        
        # 添加工具调用历史
        for call in state.tool_call_history[-5:]:  # 最近5次工具调用
            context_parts.append(f"工具调用 {call['tool']}: {call.get('result', '')[:100]}")
        
        return "\n".join(context_parts)

    def _suggest_next_tool(self, state: LLMAgentState) -> Optional[str]:
        """建议下一个工具"""
        if not self.tool_selector or not state.tool_call_history:
            return None
        
        last_tool = state.tool_call_history[-1]['tool']
        # 优先尝试通过 summarize 工具生成用于对比的摘要
        context = self._get_summary_for_selection(state)
        # print("Context for tool selection:", context)
        available_tools = [tool.name for tool in self.tools]
        
        suggested_tool = self.tool_selector.select_next_tool(
            current_tool=last_tool,
            conversation_context=context,
            available_tools=available_tools
        )
        
        if suggested_tool:
            logger.info(f"🎯 建议下一个工具: {suggested_tool}")
            state.last_tool_suggestion = suggested_tool
        
        return suggested_tool

    def _suggest_next_tools(self, state: LLMAgentState, top_k: int = 2) -> List[str]:
        """基于工具选择器分析，返回下一个候选工具列表（按分数排序，最多 top_k 个）。"""
        if not self.tool_selector or not state.tool_call_history:
            return []
        last_tool = state.tool_call_history[-1]['tool']
        available_tools = [tool.name for tool in self.tools]
        
        try:
            # 获取相连的工具信息
            connected_tools = self.tool_selector.get_connected_tools(last_tool)
            
            # 先获取原始对话上下文用于判断
            raw_context = self._extract_conversation_context(state)
            
            # 使用LLM判断是否需要调用summary tool
            should_summarize = self._llm_judge_should_summarize(last_tool, raw_context, connected_tools)
            
            # 根据判断结果决定使用哪个context
            if should_summarize:
                # 如果需要总结，调用summary tool获得新的context
                context = self._get_summary_for_selection(state)  # 这里会调用summary tool
                print(f"[TOOL_SELECTION] Using summarized context for tool selection")
                # print(f"[TOOL_SELECTION] Summarized context: {context}")
            else:
                # 如果不需要总结，使用原始context
                context = raw_context
                print(f"[TOOL_SELECTION] Using raw context for tool selection")
            
            # 根据context类型选择分析策略
            if should_summarize:
                # 如果需要总结，优先考虑有current_information的边
                current_info_tools = [tool for tool in connected_tools if tool.get('current_information')]
                if current_info_tools:
                    # 使用有current_information的工具进行分析
                    analysis = self.tool_selector.get_tool_selection_explanation(
                        current_tool=last_tool,
                        conversation_context=context,
                        available_tools=available_tools,
                        should_summarize=True,
                    )
                    print(f"===[TOOL_SELECTION] Tool selection analysis using current_information")
                else:
                    # 如果没有current_information，则使用权重进行分析
                    weighted_tools = [tool for tool in connected_tools if tool.get('weight', 0) > 0]
                    if weighted_tools:
                        # 按权重排序
                        weighted_tools.sort(key=lambda x: x.get('weight', 0), reverse=True)
                        # 构造基于权重的分析结果
                        tool_analysis = []
                        for tool_info in weighted_tools[:top_k]:
                            if tool_info['tool'] in available_tools:
                                tool_analysis.append({
                                    'tool': tool_info['tool'],
                                    'score': tool_info.get('weight', 0) / 1000.0,
                                    'reason': 'weight_based'
                                })
                        analysis = {'tool_analysis': tool_analysis}
                    else:
                        analysis = {'tool_analysis': []}
            else:
                # 如果不需要总结，直接使用原始context进行分析
                analysis = self.tool_selector.get_tool_selection_explanation(
                    current_tool=last_tool,
                    conversation_context=context,
                    available_tools=available_tools,
                    should_summarize=False,
                )
            
            tool_analysis = analysis.get('tool_analysis', [])
            candidates = [item.get('tool') for item in tool_analysis if item.get('tool')]
            # 过滤可能不存在的名字，并去重
            unique_candidates = []
            seen = set()
            for t in candidates:
                if t in available_tools and t not in seen:
                    unique_candidates.append(t)
                    seen.add(t)
                if len(unique_candidates) >= top_k:
                    break
            if unique_candidates:
                state.last_tool_suggestion = unique_candidates[0]
            return unique_candidates
        except Exception:
            return []

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
            from tau2.data_model.message import SystemMessage, UserMessage
            
            messages = [
                SystemMessage(
                    role="system",
                    content="You are a professional tool selection assistant who can accurately determine whether conversation content needs to be summarized."
                ),
                UserMessage(
                    role="user",
                    content=prompt
                )
            ]
            
            # 使用LLMAgent的generate函数。max_tokens 不宜过小，否则推理模型可能把 token 用于思考导致 content 为空
            response = generate(
                model=self.llm,
                messages=messages,
                max_tokens=64,
                temperature=0.1
            )
            
            result = (response.content or "").strip().lower()
            if not result:
                # 模型返回空 content（常见于 reasoning 模型 + 低 max_tokens，或 API 异常）
                fallback = len(conversation_context.strip()) > 100
                print(f"[LLM_JUDGMENT] Empty LLM response, using fallback rule: should_summarize={fallback} (context length > 100)")
                return fallback
            
            should_summarize = "yes" in result or "true" in result or "y" in result
            print(f"=====[LLM_JUDGMENT] LLM judgment result: {should_summarize} (original response: {result})")
            return should_summarize
            
        except Exception as e:
            print(f"[LLM_JUDGMENT] LLM judgment error: {e}")
            # 回退到简单规则
            return len(conversation_context.strip()) > 100


    # def _suggest_next_tools(self, state: LLMAgentState, top_k: int = 2) -> List[str]:
    #     """基于工具选择器分析，返回下一个候选工具列表（按分数排序，最多 top_k 个）。"""
    #     print(f"🔍 Debug: tool_selector={self.tool_selector is not None}, tool_call_history={len(state.tool_call_history)}")
        
    #     if not self.tool_selector or not state.tool_call_history:
    #         print("❌ Debug: No tool_selector or empty tool_call_history")
    #         return []
        
    #     last_tool = state.tool_call_history[-1]['tool']
    #     context = self._get_summary_for_selection(state)
    #     available_tools = [tool.name for tool in self.tools]
        
    #     # print(f"🔍 Debug: last_tool={last_tool}, context_length={len(context)}, available_tools={available_tools[:5]}...")
    #     print(f"🔍 Debug: context={context}")
        
    #     try:
    #         analysis = self.tool_selector.get_tool_selection_explanation(
    #             current_tool=last_tool,
    #             conversation_context=context,
    #             available_tools=available_tools,
    #         )
    #         print(f"🔍 Debug: analysis keys={list(analysis.keys())}")
            
    #         tool_analysis = analysis.get('tool_analysis', [])
    #         print(f"🔍 Debug: tool_analysis length={len(tool_analysis)}")
            
    #         candidates = [item.get('tool') for item in tool_analysis if item.get('tool')]
    #         print(f"🔍 Debug: candidates={candidates}")
            
    #         # 过滤可能不存在的名字，并去重
    #         unique_candidates = []
    #         seen = set()
    #         for t in candidates:
    #             if t in available_tools and t not in seen:
    #                 unique_candidates.append(t)
    #                 seen.add(t)
    #             if len(unique_candidates) >= top_k:
    #                 break
            
    #         print(f"🔍 Debug: unique_candidates={unique_candidates}")
            
    #         if unique_candidates:
    #             state.last_tool_suggestion = unique_candidates[0]
    #         return unique_candidates
    #     except Exception as e:
    #         print(f"❌ Debug: Exception in _suggest_next_tools: {e}")
    #         return []

    def _get_summary_for_selection(self, state: LLMAgentState) -> str:
        """为工具选择获取摘要文本，强制让模型立刻调用 summarize_task_state，并使用其生成的参数作为 context。失败则回退到本地摘要。"""
        # 查找 summarize 工具
        summarize_tool = None
        for tool in self.tools:
            if getattr(tool, 'name', None) == 'summarize_task_state':
                summarize_tool = tool
                break
        # 通过一次受限调用强制模型调用 summarize_task_state
        if summarize_tool is not None and self.llm is not None:
            try:
                llm_kwargs = {k: v for k, v in self.llm_args.items() if k != "summary"}
                assistant_message = generate(
                    model=self.llm,
                    tools=[summarize_tool],
                    messages=state.system_messages + state.messages,
                    tool_choice="required",
                    **llm_kwargs,
                )
                if getattr(assistant_message, 'tool_calls', None):
                    for tc in assistant_message.tool_calls:
                        if getattr(tc, 'name', '') == 'summarize_task_state':
                            args = getattr(tc, 'arguments', {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except Exception:
                                    args = {"current_information": args}
                            if isinstance(args, dict):
                                ci = args.get('current_information')
                                if isinstance(ci, str) and ci.strip():
                                    return ci
            except Exception as e:
                logger.warning(f"即时 summarize_task_state 调用失败，回退到本地摘要: {e}")
        # 回退到本地摘要（基于最近消息与调用历史）
        return self._extract_conversation_context(state)
        # 回退到本地摘要（基于最近消息与调用历史）
        return self._extract_conversation_context(state)

    def get_tool_selection_analysis(self, state: LLMAgentState) -> Dict[str, Any]:
        """获取工具选择的详细分析"""
        if not self.tool_selector or not state.tool_call_history:
            return {'error': '工具选择器未启用或无工具调用历史'}
        
        last_tool = state.tool_call_history[-1]['tool']
        context = self._extract_conversation_context(state)
        available_tools = [tool.name for tool in self.tools]
        
        return self.tool_selector.get_tool_selection_explanation(
            current_tool=last_tool,
            conversation_context=context,
            available_tools=available_tools
        )


AGENT_GT_INSTRUCTION = """
You are testing that our user simulator is working correctly.
User simulator will have an issue for you to solve.
You must behave according to the <policy> provided below.
To make following the policy easier, we give you the list of resolution steps you are expected to take.
These steps involve either taking an action or asking the user to take an action.

In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

AGENT_GT_INSTRUCTION_SUMMARY = """
You are testing that our user simulator is working correctly.
User simulator will have an issue for you to solve.
You must behave according to the <policy> provided below.
To make following the policy easier, we give you the list of resolution steps you are expected to take.
These steps involve either taking an action or asking the user to take an action.

In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Please deciside when to summarize the current state of the task with the `summarize_task_state` tool and call `summarize_task_state` at least once in the conversation.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

# Please deciside when to summarize the current state of the task with the `summarize_task_state` tool and AI LEAST ONCE in the conversation.


# In turns where you get new information, please summarize the current state of the task and make a plan based on the current state.

# please summary the current state of the task and make decision based on the state.

# You can follow these sugestions:
# ​After EVERY action or set of actions taken:
# Analyze the effectiveness of your previous steps
# Identify what worked well and what didn't
# Explicitly note any new information gained

# Maintain and update these key elements:
# Current environment state
# Progress toward overall goal
# Remaining obstacles/challenges
# Resources available (time, tools, information)

# Based on your reflection:
# Modify your strategy if needed
# Adjust your next steps to account for new information
# Consider alternative approaches that may be more effective
# Explicitly state your reasoning for any changes

# ​Execution with Meta-Cognition:​​
# Before taking action, briefly explain your intended approach
# After action execution, evaluate actual outcomes vs. expectations

# In each turn you can either:
# - Send a message to the user.
# - Make a tool call.
# You cannot do both at the same time.

# You can follow these sugestions:
# ​After EVERY action or set of actions taken:
# Analyze the effectiveness of your previous steps
# Identify what worked well and what didn't
# Explicitly note any new information gained

# Maintain and update these key elements:
# Current environment state
# Progress toward overall goal
# Remaining obstacles/challenges
# Resources available (time, tools, information)

# Based on your reflection:
# Modify your strategy if needed
# Adjust your next steps to account for new information
# Consider alternative approaches that may be more effective
# Explicitly state your reasoning for any changes

# ​Execution with Meta-Cognition:​​
# Before taking action, briefly explain your intended approach
# After action execution, evaluate actual outcomes vs. expectations

SYSTEM_PROMPT_GT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<resolution_steps>
{resolution_steps}
</resolution_steps>
""".strip()


class LLMGTAgent(LocalAgent[LLMAgentState]):
    """
    An GroundTruth agent that can be used to solve a task.
    This agent will receive the expected actions.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        provide_function_args: bool = True,
    ):
        """
        Initialize the LLMAgent.
        If provide_function_args is True, the resolution steps will include the function arguments.
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        assert self.check_valid_task(task), (
            f"Task {task.id} is not valid. Cannot run GT agent."
        )
        self.task = task
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        self.provide_function_args = provide_function_args
        
        # Add summary tool from GenericToolKit if summary is enabled
        if self.llm_args.get("summary", False):
            from tau2.environment.toolkit import GenericToolKit
            generic_tools = GenericToolKit()
            summary_tool = generic_tools.get_tools()["summarize_task_state"]
            self.tools.append(summary_tool)

    @classmethod
    def check_valid_task(cls, task: Task) -> bool:
        """
        Check if the task is valid.
        Only the tasks that require at least one action are valid.
        """
        if task.evaluation_criteria is None:
            return False
        expected_actions = task.evaluation_criteria.actions or []
        if len(expected_actions) == 0:
            return False
        return True

    @property
    def system_prompt(self) -> str:
        if self.llm_args.get("summary", False):
            agent_instruction = AGENT_GT_INSTRUCTION_SUMMARY
        else:
            agent_instruction = AGENT_GT_INSTRUCTION
        return SYSTEM_PROMPT_GT.format(
            agent_instruction=agent_instruction,
            domain_policy=self.domain_policy,
            resolution_steps=self.make_agent_instructions_from_actions(),
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages

        
        # Filter out summary parameter as it's only used for system prompt generation
        llm_kwargs = {k: v for k, v in self.llm_args.items() if k != "summary"}
        
        # Log available tools for monitoring
        if self.llm_args.get("summary", False):
            tool_names = [tool.name for tool in self.tools]
            logger.info(f"🛠️ Available tools: {tool_names}")
            if "summarize_task_state" in tool_names:
                logger.info("✅ summarize_task_state tool is available")
            else:
                logger.warning("❌ summarize_task_state tool is NOT available")
        
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            **llm_kwargs,
        )
        
        # Monitor tool calls for summarize_task_state
        if self.llm_args.get("summary", False) and assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                if tool_call.name == "summarize_task_state":
                    logger.info(f"🎯 Agent called summarize_task_state tool!")
                    print(f"🎯 Agent called summarize_task_state tool!")
                    logger.info(f"📋 Arguments: {tool_call.arguments}")
                else:
                    logger.info(f"🔧 Agent called tool: {tool_call.name}")
        
        state.messages.append(assistant_message)
        
        return assistant_message, state

    # def save_conversation_to_json(self, state: LLMAgentState, conversation_id: Optional[str] = None) -> str:
    #     """
    #     Save the entire conversation to a JSON file for SFT data collection.
    #     This method should be called at the end of the conversation.
        
    #     Args:
    #         state: The current agent state containing all messages
    #         conversation_id: Optional custom conversation ID, if None will generate one
            
    #     Returns:
    #         str: The filepath where the conversation was saved
    #     """
    #     try:
    #         # Create data directory if it doesn't exist
    #         data_dir = os.path.join(os.getcwd(), "data", "messages", "sft_data")
    #         os.makedirs(data_dir, exist_ok=True)
            
    #         # Generate filename with timestamp
    #         import datetime
    #         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
    #         if conversation_id is None:
    #             conversation_id = f"conv_{timestamp}"
                
    #         filename = f"sft_conversation_{conversation_id}_{timestamp}.json"
    #         filepath = os.path.join(data_dir, filename)
            
    #         # Prepare conversation data for SFT
    #         conversation_data = {
    #             "conversation_id": conversation_id,
    #             "timestamp": timestamp,
    #             "system_prompt": state.system_messages[0].content if state.system_messages else "",
    #             "total_messages": len(state.messages),
    #             "messages": []
    #         }
            
    #         # Add all messages from the conversation
    #         for msg in state.messages:
    #             message_data = {
    #                 "role": msg.role,
    #                 "content": msg.content,
    #                 "timestamp": msg.timestamp,
    #                 "turn_idx": msg.turn_idx
    #             }
                
    #             # Add tool calls if present
    #             if hasattr(msg, 'tool_calls') and msg.tool_calls:
    #                 message_data["tool_calls"] = [
    #                     {
    #                         "id": tc.id,
    #                         "name": tc.name,
    #                         "arguments": tc.arguments,
    #                         "requestor": tc.requestor
    #                     } for tc in msg.tool_calls
    #                 ]
                
    #             # Add tool message specific fields
    #             if hasattr(msg, 'id') and hasattr(msg, 'error') and hasattr(msg, 'requestor'):
    #                 message_data.update({
    #                     "id": msg.id,
    #                     "error": msg.error,
    #                     "requestor": msg.requestor
    #                 })
                
    #             conversation_data["messages"].append(message_data)
            
    #         # Save to JSON file
    #         with open(filepath, 'w', encoding='utf-8') as f:
    #             json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            
    #         logger.info(f"SFT conversation data saved to: {filepath}")
    #         return filepath
            
    #     except Exception as e:
    #         logger.error(f"Failed to save SFT conversation data: {e}")
    #         raise

    # def _save_messages_to_json(self, state: LLMAgentState, input_message: ValidAgentInputMessage, assistant_message: AssistantMessage):
    #     """
    #     Save all messages to a JSON file for SFT data collection.
    #     """
    #     try:
    #         # Create data directory if it doesn't exist
    #         data_dir = os.path.join(os.getcwd(), "data", "messages", "sft_data")
    #         os.makedirs(data_dir, exist_ok=True)
            
    #         # Generate filename with timestamp
    #         import datetime
    #         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #         filename = f"sft_conversation_{timestamp}.json"
    #         filepath = os.path.join(data_dir, filename)
            
    #         # Prepare conversation data for SFT
    #         conversation_data = {
    #             "conversation_id": f"conv_{timestamp}",
    #             "timestamp": timestamp,
    #             "system_prompt": state.system_messages[0].content if state.system_messages else "",
    #             "messages": []
    #         }
            
    #         # Add all messages from the conversation
    #         for msg in state.messages:
    #             message_data = {
    #                 "role": msg.role,
    #                 "content": msg.content,
    #                 "timestamp": msg.timestamp,
    #                 "turn_idx": msg.turn_idx
    #             }
                
    #             # Add tool calls if present
    #             if hasattr(msg, 'tool_calls') and msg.tool_calls:
    #                 message_data["tool_calls"] = [
    #                     {
    #                         "id": tc.id,
    #                         "name": tc.name,
    #                         "arguments": tc.arguments,
    #                         "requestor": tc.requestor
    #                     } for tc in msg.tool_calls
    #                 ]
                
    #             # Add tool message specific fields
    #             if hasattr(msg, 'id') and hasattr(msg, 'error') and hasattr(msg, 'requestor'):
    #                 message_data.update({
    #                     "id": msg.id,
    #                     "error": msg.error,
    #                     "requestor": msg.requestor
    #                 })
                
    #             conversation_data["messages"].append(message_data)
            
    #         # Add the latest assistant message
    #         assistant_data = {
    #             "role": assistant_message.role,
    #             "content": assistant_message.content,
    #             "timestamp": assistant_message.timestamp,
    #             "turn_idx": assistant_message.turn_idx
    #         }
            
    #         if assistant_message.tool_calls:
    #             assistant_data["tool_calls"] = [
    #                 {
    #                     "id": tc.id,
    #                     "name": tc.name,
    #                     "arguments": tc.arguments,
    #                     "requestor": tc.requestor
    #                 } for tc in assistant_message.tool_calls
    #             ]
            
    #         conversation_data["messages"].append(assistant_data)
            
    #         # Save to JSON file
    #         with open(filepath, 'w', encoding='utf-8') as f:
    #             json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            
    #         logger.info(f"SFT data saved to: {filepath}")
            
    #     except Exception as e:
    #         logger.error(f"Failed to save SFT data: {e}")


    def save_conversation_to_json(self, state: LLMAgentState, conversation_id: Optional[str] = None, reward: Optional[float] = None) -> str:
        """
        Save the entire conversation to a JSON file for SFT data collection.
        This method should be called at the end of the conversation.
        
        Args:
            state: The current agent state containing all messages
            conversation_id: Optional custom conversation ID, if None will generate one
            reward: Optional reward value for the conversation
            
        Returns:
            str: The filepath where the conversation was saved
        """
        try:
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.getcwd(), "data", "messages", "sft_data_reward_gt")
            os.makedirs(data_dir, exist_ok=True)
            
            # Generate filename with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if conversation_id is None:
                conversation_id = f"conv_{timestamp}"
                
            filename = f"sft_conversation_{conversation_id}_{timestamp}.json"
            filepath = os.path.join(data_dir, filename)
            
            # Prepare conversation data for SFT
            conversation_data = {
                "conversation_id": conversation_id,
                "timestamp": timestamp,
                "system_prompt": state.system_messages[0].content if state.system_messages else "",
                "total_messages": len(state.messages),
                "reward": reward,  # 新增 reward 字段
                "messages": []
            }
            
            # Add all messages from the conversation
            for msg in state.messages:
                message_data = {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "turn_idx": msg.turn_idx
                }
                
                # Add tool calls if present
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    message_data["tool_calls"] = [
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "requestor": tc.requestor
                        } for tc in msg.tool_calls
                    ]
                
                # Add tool message specific fields
                if hasattr(msg, 'id') and hasattr(msg, 'error') and hasattr(msg, 'requestor'):
                    message_data.update({
                        "id": msg.id,
                        "error": msg.error,
                        "requestor": msg.requestor
                    })
                
                conversation_data["messages"].append(message_data)
            
            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"SFT conversation data saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save SFT conversation data: {e}")
            raise

    def _save_messages_to_json(self, state: LLMAgentState, input_message: ValidAgentInputMessage, assistant_message: AssistantMessage, reward: Optional[float] = None):
        """
        Save all messages to a JSON file for SFT data collection, including reward.
        """
        try:
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.getcwd(), "data", "messages", "sft_data_reward_gt")
            os.makedirs(data_dir, exist_ok=True)
            
            # Generate filename with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sft_conversation_{timestamp}.json"
            filepath = os.path.join(data_dir, filename)
            
            # Prepare conversation data for SFT
            conversation_data = {
                "conversation_id": f"conv_{timestamp}",
                "timestamp": timestamp,
                "system_prompt": state.system_messages[0].content if state.system_messages else "",
                "reward": reward,  # 新增 reward 字段
                "messages": []
            }
            
            # Add all messages from the conversation
            for msg in state.messages:
                message_data = {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "turn_idx": msg.turn_idx
                }
                
                # Add tool calls if present
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    message_data["tool_calls"] = [
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "requestor": tc.requestor
                        } for tc in msg.tool_calls
                    ]
                
                # Add tool message specific fields
                if hasattr(msg, 'id') and hasattr(msg, 'error') and hasattr(msg, 'requestor'):
                    message_data.update({
                        "id": msg.id,
                        "error": msg.error,
                        "requestor": msg.requestor
                    })
                
                conversation_data["messages"].append(message_data)
            
            # Add the latest assistant message
            assistant_data = {
                "role": assistant_message.role,
                "content": assistant_message.content,
                "timestamp": assistant_message.timestamp,
                "turn_idx": assistant_message.turn_idx
            }
            
            if assistant_message.tool_calls:
                assistant_data["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "requestor": tc.requestor
                    } for tc in assistant_message.tool_calls
                ]
            
            conversation_data["messages"].append(assistant_data)
            
            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"SFT data saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save SFT data: {e}")

    # def save_conversation_to_json(
    #     self,
    #     state: LLMAgentState,
    #     input_message: Optional[ValidAgentInputMessage] = None,
    #     assistant_message: Optional[AssistantMessage] = None,
    #     reward: float = None,
    #     ground_truth: str = "",
    #     split: str = "train",
    #     trajectory_index: int = 0,
    #     ability: str = "general",
    #     tools_kwargs: dict = None,
    # ) -> str:
    #     """
    #     Save the entire conversation to a JSON file for RL training data collection.
    #     This method should be called at the end of the trajectory.
        
    #     Args:
    #         state: The current agent state containing all messages
    #         input_message: The initial input message (optional, for RL format)
    #         assistant_message: The last assistant message (optional, for RL format)
    #         reward: The reward for this trajectory
    #         ground_truth: The ground truth answer for this trajectory
    #         split: Dataset split ("train" or "test")
    #         trajectory_index: Index of this trajectory for tracking
    #         ability: The ability/category of this task (e.g., "math", "customer_service")
    #         tools_kwargs: Tool-specific kwargs for reward calculation
            
    #     Returns:
    #         str: The filepath where the RL data was saved
    #     """
    #     try:
    #         # Create data directory if it doesn't exist
    #         data_dir = os.path.join(os.getcwd(), "data", "messages", "rl_data")
    #         os.makedirs(data_dir, exist_ok=True)
            
    #         # Generate filename with timestamp
    #         import datetime
    #         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #         filename = f"rl_rollout_{split}_{timestamp}.json"
    #         filepath = os.path.join(data_dir, filename)
            
    #         # Prepare prompt (multi-turn conversation)
    #         prompt = []
    #         if state.system_messages:
    #             prompt.append({
    #                 "role": "system",
    #                 "content": state.system_messages[0].content
    #             })
    #         for msg in state.messages:
    #             msg_dict = {
    #                 "role": msg.role,
    #                 "content": msg.content
    #             }
    #             if hasattr(msg, "tool_calls") and msg.tool_calls:
    #                 msg_dict["tool_calls"] = [
    #                     {
    #                         "id": tc.id,
    #                         "name": tc.name,
    #                         "arguments": tc.arguments,
    #                         "requestor": tc.requestor
    #                     } for tc in msg.tool_calls
    #                 ]
    #             prompt.append(msg_dict)
    #         # Optionally add the latest assistant message if provided and not already in state.messages
    #         if assistant_message is not None and (not state.messages or assistant_message != state.messages[-1]):
    #             assistant_dict = {
    #                 "role": assistant_message.role,
    #                 "content": assistant_message.content
    #             }
    #             if assistant_message.tool_calls:
    #                 assistant_dict["tool_calls"] = [
    #                     {
    #                         "id": tc.id,
    #                         "name": tc.name,
    #                         "arguments": tc.arguments,
    #                         "requestor": tc.requestor
    #                     } for tc in assistant_message.tool_calls
    #                 ]
    #             prompt.append(assistant_dict)

    #         # RL data structure (matching GSM8K format)
    #         rl_data = {
    #             "data_source": "llm_rollout",
    #             "prompt": prompt,
    #             "ability": ability,
    #             "reward_model": {
    #                 "style": "rule" if reward is not None else "env_reward",
    #                 "ground_truth": ground_truth
    #             },
    #             "extra_info": {
    #                 "split": split,
    #                 "index": trajectory_index,
    #                 "timestamp": timestamp,
    #                 "need_tools_kwargs": True,
    #                 "tools_kwargs": tools_kwargs if tools_kwargs is not None else {},
    #                 "interaction_kwargs": {
    #                     "ground_truth": ground_truth
    #                 },
    #             },
    #         }
            
    #         # Add reward to reward_model if available
    #         if reward is not None:
    #             rl_data["reward_model"]["reward"] = reward
            
    #         # Save as json
    #         with open(filepath, "w", encoding="utf-8") as f:
    #             json.dump(rl_data, f, ensure_ascii=False, indent=2)
            
    #         logger.info(f"RL rollout data saved to: {filepath}")
    #         return filepath
            
    #     except Exception as e:
    #         logger.error(f"Failed to save RL data: {e}")
    #         raise



    # def _save_messages_to_json(
    #     self,
    #     state: LLMAgentState,
    #     input_message: ValidAgentInputMessage,
    #     assistant_message: AssistantMessage,
    #     reward: float = None,
    #     ground_truth: str = "",
    #     split: str = "train",
    #     trajectory_index: int = 0,
    #     ability: str = "general",
    #     tools_kwargs: dict = None,
    # ):
    #     """
    #     Save all messages to a JSON file for RL training data collection.
    #     """
    #     try:
    #         # Create data directory if it doesn't exist
    #         data_dir = os.path.join(os.getcwd(), "data", "messages", "rl_data")
    #         os.makedirs(data_dir, exist_ok=True)
    #         print(f"Saving RL data to: {data_dir}")

    #         # Generate filename with timestamp
    #         import datetime
    #         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #         filename = f"rl_rollout_{split}_{timestamp}.json"
    #         filepath = os.path.join(data_dir, filename)

    #         # Prepare prompt (multi-turn conversation)
    #         prompt = []
    #         if state.system_messages:
    #             prompt.append({
    #                 "role": "system",
    #                 "content": state.system_messages[0].content
    #             })
    #         for msg in state.messages:
    #             msg_dict = {
    #                 "role": msg.role,
    #                 "content": msg.content
    #             }
    #             if hasattr(msg, "tool_calls") and msg.tool_calls:
    #                 msg_dict["tool_calls"] = [
    #                     {
    #                         "id": tc.id,
    #                         "name": tc.name,
    #                         "arguments": tc.arguments,
    #                         "requestor": tc.requestor
    #                     } for tc in msg.tool_calls
    #                 ]
    #             prompt.append(msg_dict)
    #         # Add the latest assistant message
    #         assistant_dict = {
    #             "role": assistant_message.role,
    #             "content": assistant_message.content
    #         }
    #         if assistant_message.tool_calls:
    #             assistant_dict["tool_calls"] = [
    #                 {
    #                     "id": tc.id,
    #                     "name": tc.name,
    #                     "arguments": tc.arguments,
    #                     "requestor": tc.requestor
    #                 } for tc in assistant_message.tool_calls
    #             ]
    #         prompt.append(assistant_dict)

    #         # RL data structure (matching GSM8K format)
    #         rl_data = {
    #             "data_source": "llm_rollout",
    #             "prompt": prompt,
    #             "ability": ability,
    #             "reward_model": {
    #                 "style": "rule" if reward is not None else "env_reward",
    #                 "ground_truth": ground_truth
    #             },
    #             "extra_info": {
    #                 "split": split,
    #                 "index": trajectory_index,
    #                 "timestamp": timestamp,
    #                 "need_tools_kwargs": True,
    #                 "tools_kwargs": tools_kwargs if tools_kwargs is not None else {},
    #                 "interaction_kwargs": {
    #                     "ground_truth": ground_truth
    #                 },
    #             },
    #         }
            
    #         # Add reward to reward_model if available
    #         if reward is not None:
    #             rl_data["reward_model"]["reward"] = reward

    #         # Save as json
    #         with open(filepath, "w", encoding="utf-8") as f:
    #             json.dump(rl_data, f, ensure_ascii=False, indent=2)

    #         logger.info(f"RL rollout data saved to: {filepath}")

    #     except Exception as e:
    #         logger.error(f"Failed to save RL data: {e}")


    

    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed

    def make_agent_instructions_from_actions(self) -> str:
        """
        Make agent instructions from a list of actions
        """
        lines = []
        for i, action in enumerate(self.task.evaluation_criteria.actions):
            lines.append(
                f"[Step {i + 1}] {self.make_agent_instructions_from_action(action=action, include_function_args=self.provide_function_args)}"
            )
        return "\n".join(lines)

    @classmethod
    def make_agent_instructions_from_action(
        cls, action: Action, include_function_args: bool = False
    ) -> str:
        """
        Make agent instructions from an action.
        If the action is a user action, returns instructions for the agent to give to the user.
        If the action is an agent action, returns instructions for the agent to perform the action.
        """
        if action.requestor == "user":
            if include_function_args:
                return f"Instruct the user to perform the following action: {action.get_func_format()}."
            else:
                return f"User action: {action.name}."
        elif action.requestor == "assistant":
            if include_function_args:
                return f"Perform the following action: {action.get_func_format()}."
            else:
                return f"Assistant action: {action.name}."
        else:
            raise ValueError(f"Unknown action requestor: {action.requestor}")


AGENT_SOLO_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
You will be provided with a ticket that contains the user's request.
You will need to plan and call the appropriate tools to solve the ticket.

You cannot communicate with the user, only make tool calls.
Stop when you consider that you have solved the ticket.
To do so, send a message containing a single tool call to the `{stop_function_name}` tool. Do not include any other tool calls in this last message.

Always follow the policy. Always make sure you generate valid JSON only.
""".strip()

AGENT_SOLO_INSTRUCTION_SUMMARY = """
You are a customer service agent that helps the user according to the <policy> provided below.
You will be provided with a ticket that contains the user's request.
You will need to plan and call the appropriate tools to solve the ticket.

You cannot communicate with the user, only make tool calls.
Stop when you consider that you have solved the ticket.
To do so, send a message containing a single tool call to the `{stop_function_name}` tool. Do not include any other tool calls in this last message.

- You should decide when to summarize the current state and information of the task with the `summarize_the_task` tool and call `summarize_the_task` at least once in the conversation to help you make decisions about action.

Always follow the policy. Always make sure you generate valid JSON only.
""".strip()

# please deciside when to summarize the current state of the task with the `summarize_task_state` tool and at least once in the conversation.


# Please summarize the current state with the `summarize_task_state` tool when you need to. and do not forget to call the tool.

# In turns where you get new information, please summarize the current state of the task and make a plan based on the current state, compare the current plan with the previous plan and update the plan if needed.

# If you can't solve the problem, please summarize the current state with the `summarize_task_state` tool to find out the current state and problem of the task.
# 

#  You can follow these sugestions:
# ​After EVERY action or set of actions taken:
# Analyze the effectiveness of your previous steps
# Identify what worked well and what didn't
# Explicitly note any new information gained

# Maintain and update these key elements:
# Current environment state
# Progress toward overall goal
# Remaining obstacles/challenges
# Resources available (time, tools, information)

# Based on your reflection:
# Modify your strategy if needed
# Adjust your next steps to account for new information
# Consider alternative approaches that may be more effective
# Explicitly state your reasoning for any changes

# ​Execution with Meta-Cognition:​​
# Before taking action, briefly explain your intended approach
# After action execution, evaluate actual outcomes vs. expectations

SYSTEM_PROMPT_SOLO = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<ticket>
{ticket}
</ticket>
""".strip()


class LLMSoloAgent(LocalAgent[LLMAgentState]):
    """
    An LLM agent that can be used to solve a task without any interaction with the customer.
    The task need to specify a ticket format.
    """

    STOP_FUNCTION_NAME = "done"
    TRANSFER_TOOL_NAME = "transfer_to_human_agents"
    STOP_TOKEN = "###STOP###"

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        tool_graph_path: Optional[str] = None,
        enable_tool_selection: bool = False,
    ):
        """
        Initialize the LLMSoloAgent.
        
        Args:
            tools: 工具列表
            domain_policy: 域策略
            task: 任务对象
            llm: LLM模型
            llm_args: LLM参数
            tool_graph_path: 工具图数据文件路径（可选）
            enable_tool_selection: 是否启用工具选择功能（可选）
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        assert self.check_valid_task(task), (
            f"Task {task.id} is not valid. Cannot run GT agent."
        )
        self.task = task
        self.llm = llm
        self.llm_args = llm_args if llm_args is not None else {}
        
        # 初始化工具选择器
        self.enable_tool_selection = enable_tool_selection
        if enable_tool_selection and tool_graph_path and EMBEDDING_AVAILABLE:
            self.tool_selector = EmbeddingBasedToolSelector(tool_graph_path)
            logger.info("✓ 工具选择器初始化成功")
        else:
            self.tool_selector = None
            if enable_tool_selection and not EMBEDDING_AVAILABLE:
                logger.warning("工具选择功能已请求但embedding库不可用")
            elif enable_tool_selection and not tool_graph_path:
                logger.warning("工具选择功能已请求但未提供工具图路径")
        
        # Add summary tool from GenericToolKit if summary is enabled
        if self.llm_args.get("summary", False):
            from tau2.environment.toolkit import GenericToolKit
            generic_tools = GenericToolKit()
            summary_tool = generic_tools.get_tools()["summarize_task_state"]
            
            # Debug logging to see what's happening with the tool
            logger.info(f"🔍 Summary tool name: {summary_tool.name}")
            logger.info(f"🔍 Summary tool type: {type(summary_tool)}")
            logger.info(f"🔍 Summary tool __name__: {getattr(summary_tool, '__name__', 'N/A')}")
            
            self.tools.append(summary_tool)
            
            # Debug logging after adding
            logger.info(f"🔍 Tools after adding summary tool: {[tool.name for tool in self.tools]}")
        
        self.add_stop_tool()
        self.validate_tools()

    def add_stop_tool(self) -> None:
        """Add the stop tool to the tools."""

        def done() -> str:
            """Call this function when you are done with the task."""
            return self.STOP_TOKEN

        self.tools.append(as_tool(done))

    def validate_tools(self) -> None:
        """Check if the tools are valid."""
        tool_names = {tool.name for tool in self.tools}
        if self.TRANSFER_TOOL_NAME not in tool_names:
            logger.warning(
                f"Tool {self.TRANSFER_TOOL_NAME} not found in tools. This tool is required for the agent to transfer the user to a human agent."
            )
        if self.STOP_FUNCTION_NAME not in tool_names:
            raise ValueError(f"Tool {self.STOP_FUNCTION_NAME} not found in tools.")

    @classmethod
    def check_valid_task(cls, task: Task) -> bool:
        """
        Check if the task is valid.
        Task should contain a ticket and evaluation criteria.
        If the task contains an initial state, the message history should only contain tool calls and responses.
        """
        if task.initial_state is not None:
            message_history = task.initial_state.message_history or []
            for message in message_history:
                if isinstance(message, UserMessage):
                    return False
                if isinstance(message, AssistantMessage) and not message.is_tool_call():
                    return False
            return True
        if task.ticket is None:
            return False
        if task.evaluation_criteria is None:
            return False
        expected_actions = task.evaluation_criteria.actions or []
        if len(expected_actions) == 0:
            return False
        return True

    @property
    def system_prompt(self) -> str:
        if self.llm_args.get("summary", False):
            agent_instruction = AGENT_SOLO_INSTRUCTION_SUMMARY.format(
                stop_function_name=self.STOP_FUNCTION_NAME,
                stop_token=self.STOP_TOKEN,
            )
        else:
            agent_instruction = AGENT_SOLO_INSTRUCTION.format(
                stop_function_name=self.STOP_FUNCTION_NAME,
                stop_token=self.STOP_TOKEN,
            )
        return SYSTEM_PROMPT_SOLO.format(
            agent_instruction=agent_instruction,
            domain_policy=self.domain_policy,
            ticket=self.task.ticket,
        )

    def _check_if_stop_toolcall(self, message: AssistantMessage) -> AssistantMessage:
        """Check if the message is a stop message.
        If the message contains a tool call with the name STOP_FUNCTION_NAME, then the message is a stop message.
        Also check if the message content contains done signal.
        """
        is_stop = False
        
        # 检查 tool calls 中的停止信号
        for tool_call in (message.tool_calls or []):  # 修正：None时用空列表
            if tool_call.name == self.STOP_FUNCTION_NAME:
                is_stop = True
                break
        
        # 检查 content 中的 done 信号
        if not is_stop and message.content:
            try:
                content_obj = json.loads(message.content)
                if content_obj == {"name": "done", "arguments": {}}:
                    is_stop = True
            except Exception:
                pass  # 不是JSON格式，忽略
        
        if is_stop:
            message.content = self.STOP_TOKEN
            message.tool_calls = None
        return message

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        """Check if the message is a stop message."""
        if message.content is None:
            return False
        return cls.STOP_TOKEN in message.content

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: Optional[ValidAgentInputMessage], state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, UserMessage):
            raise ValueError("LLMSoloAgent does not support user messages.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is None:
            assert len(state.messages) == 0, "Message history should be empty"
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages

        # Filter out summary parameter as it's only used for system prompt generation
        llm_kwargs = {k: v for k, v in self.llm_args.items() if k != "summary"}

        # Try to generate a response with tool calls, with retries
        # max_retries = 3
        # for attempt in range(max_retries):
        #     try:
        #         assistant_message = generate(
        #             model=self.llm,
        #             tools=self.tools,
        #             messages=messages,
        #             tool_choice="required",
        #             **llm_kwargs,
        #         )
        #         print('assistant_message',assistant_message)
        #         if assistant_message.is_tool_call():
        #             break
        #         else:
        #             logger.warning(f"Attempt {attempt + 1}: LLM generated response without tool calls, retrying...")
        #             if attempt == max_retries - 1:
        #                 raise ValueError("LLMSoloAgent only supports tool calls. After multiple attempts, LLM still generated response without tool calls.")
        #     except Exception as e:
        #         if attempt == max_retries - 1:
        #             raise e
        #         logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
        #         continue

        max_retries = 5
        done_content = {"name": "done", "arguments": {}}
        done_content_str = json.dumps(done_content, separators=(",", ":"))
        consecutive_done_count = 0
        for attempt in range(max_retries):
            try:
                assistant_message = generate(
                    model=self.llm,
                    tools=self.tools,
                    messages=messages,
                    tool_choice="required",
                    **llm_kwargs,
                )
                # print(self.tools)
                # print('assistant_message', assistant_message)
                # 判断content是否为 {"name": "done", "arguments": {}} 这种JSON
                is_done_content = False
                try:
                    content_obj = json.loads(assistant_message.content)
                    if content_obj == done_content:
                        consecutive_done_count += 1
                        is_done_content = True
                    else:
                        consecutive_done_count = 0
                except Exception:
                    consecutive_done_count = 0  # 不是JSON，重置计数

                if is_done_content and consecutive_done_count >= 3:
                    logger.warning("LLMSoloAgent: Detected 3 consecutive done content, ending task.")
                    # 创建一个停止消息
                    assistant_message.content = self.STOP_TOKEN
                    assistant_message.tool_calls = None
                    break

                if is_done_content:
                    logger.info(f"LLMSoloAgent: Detected done content, task completed successfully.")
                    # 创建一个停止消息，表示任务完成
                    assistant_message.content = self.STOP_TOKEN
                    assistant_message.tool_calls = None
                    break

                if assistant_message.is_tool_call():
                    break
                else:
                    logger.warning(f"Attempt {attempt + 1}: LLM generated response without tool calls, retrying...")
                    if attempt == max_retries - 1:
                        raise ValueError("LLMSoloAgent only supports tool calls. After multiple attempts, LLM still generated response without tool calls.")
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                continue
        # Monitor tool calls for summarize_task_state
        if self.llm_args.get("summary", False) and assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                if tool_call.name == "summarize_task_state":
                    logger.info(f"🎯 Agent called summarize_task_state tool!")
                    logger.info(f"📋 Arguments: {tool_call.arguments}")
                    print('🎯 Agent called summarize_task_state tool!')
                else:
                    logger.info(f"🔧 Agent called tool: {tool_call.name}")
        # 记录工具调用历史（增强功能，与 LLMAgent 对齐）
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tool_call_record = {
                    'tool': tool_call.name,
                    'arguments': tool_call.arguments,
                    'timestamp': len(state.tool_call_history)
                }
                state.tool_call_history.append(tool_call_record)

        # 在工具调用后建议下一个工具（增强功能，与 LLMAgent 对齐），改为建议两个工具
        if assistant_message.tool_calls and self.enable_tool_selection:
            suggested_tools = self._suggest_next_tools(state, top_k=2)
            # suggested_tools = self._suggest_next_tool(state)
            

            # 将建议添加到消息内容中（可选）
            if suggested_tools:
                suggestion_text = f"next tool suggestion: {', '.join(suggested_tools)}"
                print("Tool suggestions:", suggested_tools)
                if assistant_message.content:
                    assistant_message.content += suggestion_text
                else:
                    assistant_message.content = suggestion_text

        message = self._check_if_stop_toolcall(assistant_message)
        state.messages.append(assistant_message)
        return assistant_message, state

    def _extract_conversation_context(self, state: LLMAgentState) -> str:
        """提取对话上下文用于工具选择"""
        context_parts = []
        for msg in state.messages[-5:]:  # 只取最近5条消息
            if isinstance(msg, UserMessage):
                context_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AssistantMessage):
                context_parts.append(f"Agent: {msg.content}")
        return " ".join(context_parts)

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
            from tau2.data_model.message import SystemMessage, UserMessage
            
            messages = [
                SystemMessage(
                    role="system",
                    content="You are a professional tool selection assistant who can accurately determine whether conversation content needs to be summarized."
                ),
                UserMessage(
                    role="user",
                    content=prompt
                )
            ]
            
            # 使用 generate；不传 tools，确保返回文本。max_tokens 不宜过小，否则推理模型易返回空 content
            response = generate(
                model=self.llm,
                messages=messages,
                tools=None,
                max_tokens=64,
                temperature=0.1
            )
            
            result = (response.content or "").strip().lower()
            if not result:
                fallback = len(conversation_context.strip()) > 100
                logger.warning("[LLM_JUDGMENT] Empty LLM response, using fallback rule (context length > 100)")
                print(f"[LLM_JUDGMENT] Empty LLM response, using fallback rule: should_summarize={fallback} (context length > 100)")
                return fallback
            
            should_summarize = "yes" in result or "true" in result or "y" in result
            print(f"[LLM_JUDGMENT] LLM judgment result: {should_summarize} (original response: {result})")
            return should_summarize
            
        except Exception as e:
            print(f"[LLM_JUDGMENT] LLM judgment error: {e}")
            # 回退到简单规则
            return len(conversation_context.strip()) > 100

    def _suggest_next_tools(self, state: LLMAgentState, top_k: int = 2) -> List[str]:
        """基于工具选择器分析，返回下一个候选工具列表（按分数排序，最多 top_k 个）。"""
        if not self.tool_selector or not state.tool_call_history:
            return []
        last_tool = state.tool_call_history[-1]['tool']
        available_tools = [tool.name for tool in self.tools]
        
        try:
            # 获取相连的工具信息
            connected_tools = self.tool_selector.get_connected_tools(last_tool)
            
            # 先获取原始对话上下文用于判断
            raw_context = self._extract_conversation_context(state)
            
            # 使用LLM判断是否需要调用summary tool
            should_summarize = self._llm_judge_should_summarize(last_tool, raw_context, connected_tools)
            
            # 根据判断结果决定使用哪个context
            if should_summarize:
                # 如果需要总结，调用summary tool获得新的context
                context = self._get_summary_for_selection(state)  # 这里会调用summary tool
                print(f"[TOOL_SELECTION] Using summarized context for tool selection")
                # print(f"[TOOL_SELECTION] Summarized context: {context}")
            else:
                # 如果不需要总结，使用原始context
                context = raw_context
                print(f"[TOOL_SELECTION] Using raw context for tool selection")
            
            # 根据context类型选择分析策略
            if should_summarize:
                # 如果需要总结，优先考虑有current_information的边
                current_info_tools = [tool for tool in connected_tools if tool.get('current_information')]
                if current_info_tools:
                    # 使用有current_information的工具进行分析
                    analysis = self.tool_selector.get_tool_selection_explanation(
                        current_tool=last_tool,
                        conversation_context=context,
                        available_tools=available_tools,
                        should_summarize=True,
                    )
                    print(f"===[TOOL_SELECTION] Tool selection analysis using current_information")
                else:
                    # 如果没有current_information，则使用权重进行分析
                    weighted_tools = [tool for tool in connected_tools if tool.get('weight', 0) > 0]
                    if weighted_tools:
                        # 按权重排序
                        weighted_tools.sort(key=lambda x: x.get('weight', 0), reverse=True)
                        # 构造基于权重的分析结果
                        tool_analysis = []
                        for tool_info in weighted_tools[:top_k]:
                            if tool_info['tool'] in available_tools:
                                tool_analysis.append({
                                    'tool': tool_info['tool'],
                                    'score': tool_info.get('weight', 0) / 1000.0,
                                    'reason': 'weight_based'
                                })
                        analysis = {'tool_analysis': tool_analysis}
                    else:
                        analysis = {'tool_analysis': []}
            else:
                # 如果不需要总结，直接使用原始context进行分析
                analysis = self.tool_selector.get_tool_selection_explanation(
                    current_tool=last_tool,
                    conversation_context=context,
                    available_tools=available_tools,
                    should_summarize=False,
                )
            
            tool_analysis = analysis.get('tool_analysis', [])
            candidates = [item.get('tool') for item in tool_analysis if item.get('tool')]
            # 过滤可能不存在的名字，并去重
            unique_candidates = []
            seen = set()
            for t in candidates:
                if t in available_tools and t not in seen:
                    unique_candidates.append(t)
                    seen.add(t)
                if len(unique_candidates) >= top_k:
                    break
            if unique_candidates:
                state.last_tool_suggestion = unique_candidates[0]
            return unique_candidates
        except Exception:
            return []

    def _suggest_next_tool(self, state: LLMAgentState) -> Optional[str]:
        """建议下一个工具"""
        if not self.tool_selector or not state.tool_call_history:
            return None
        
        last_tool = state.tool_call_history[-1]['tool']
        # 优先尝试通过 summarize 工具生成用于对比的摘要
        context = self._get_summary_for_selection(state)
        # print("Context for tool selection:", context)
        available_tools = [tool.name for tool in self.tools]
        
        suggested_tool = self.tool_selector.select_next_tool(
            current_tool=last_tool,
            conversation_context=context,
            available_tools=available_tools
        )
        
        if suggested_tool:
            logger.info(f"🎯 建议下一个工具: {suggested_tool}")
            state.last_tool_suggestion = suggested_tool
        
        return suggested_tool

    def _get_summary_for_selection(self, state: LLMAgentState) -> str:
        """为工具选择获取摘要文本，强制让模型立刻调用 summarize_task_state，并使用其生成的参数作为 context。失败则回退到本地摘要。"""
        # 查找 summarize 工具
        summarize_tool = None
        for tool in self.tools:
            if getattr(tool, 'name', None) == 'summarize_task_state':
                summarize_tool = tool
                break
        # 通过一次受限调用强制模型调用 summarize_task_state
        if summarize_tool is not None and self.llm is not None:
            try:
                llm_kwargs = {k: v for k, v in self.llm_args.items() if k != "summary"}
                assistant_message = generate(
                    model=self.llm,
                    tools=[summarize_tool],
                    messages=state.system_messages + state.messages,
                    tool_choice="required",
                    **llm_kwargs,
                )
                if getattr(assistant_message, 'tool_calls', None):
                    for tc in assistant_message.tool_calls:
                        if getattr(tc, 'name', '') == 'summarize_task_state':
                            args = getattr(tc, 'arguments', {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except Exception:
                                    args = {"current_information": args}
                            if isinstance(args, dict):
                                ci = args.get('current_information')
                                if isinstance(ci, str) and ci.strip():
                                    return ci
            except Exception as e:
                logger.warning(f"即时 summarize_task_state 调用失败，回退到本地摘要: {e}")
        # 回退到本地摘要（基于最近消息与调用历史）
        return self._extract_conversation_context(state)

    def get_tool_selection_analysis(self, state: LLMAgentState) -> Dict[str, Any]:
        """获取工具选择的详细分析"""
        if not self.tool_selector or not state.tool_call_history:
            return {'error': '工具选择器未启用或无工具调用历史'}
        
        last_tool = state.tool_call_history[-1]['tool']
        context = self._extract_conversation_context(state)
        available_tools = [tool.name for tool in self.tools]
        
        return self.tool_selector.get_tool_selection_explanation(
            current_tool=last_tool,
            conversation_context=context,
            available_tools=available_tools
        )

    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed


class DummyUser:
    def generate_next_message(self, *args, **kwargs):
        # 返回 None 或 raise StopIteration 以终止对话
        return None, None
