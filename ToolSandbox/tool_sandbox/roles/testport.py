#!/usr/bin/env python3
"""
测试文件：验证memory和retriever模块的导入
"""

import sys
import os
from pathlib import Path

# 添加reasoning-bank-slm-main/src到Python路径
current_dir = Path(__file__).parent
reasoning_bank_src = current_dir / "reasoning-bank-slm-main" / "src"
sys.path.insert(0, str(reasoning_bank_src))

print(f"当前工作目录: {os.getcwd()}")
print(f"Python路径中添加: {reasoning_bank_src}")
print(f"reasoning_bank_src 是否存在: {reasoning_bank_src.exists()}")
print()

def test_import_memory():
    """测试导入memory模块"""
    try:
        from memory import MemoryItem, ReasoningBank
        print("✅ 成功导入 memory 模块")
        print(f"   - MemoryItem: {MemoryItem}")
        print(f"   - ReasoningBank: {ReasoningBank}")
        return True
    except ImportError as e:
        print(f"❌ 导入 memory 模块失败: {e}")
        return False

def test_import_retriever():
    """测试导入retriever模块"""
    try:
        from retriever import MemoryRetriever
        print("✅ 成功导入 retriever 模块")
        print(f"   - MemoryRetriever: {MemoryRetriever}")
        return True
    except ImportError as e:
        print(f"❌ 导入 retrieval.retriever 模块失败: {e}")
        return False

def test_create_memory_item():
    """测试创建MemoryItem实例"""
    try:
        from memory import MemoryItem
        from datetime import datetime
        
        memory_item = MemoryItem(
            title="测试记忆",
            description="这是一个测试记忆项",
            content="测试内容",
            source_problem_id="test_001",
            success=True,
            created_at=datetime.now().isoformat()
        )
        print("✅ 成功创建 MemoryItem 实例")
        print(f"   - 标题: {memory_item.title}")
        print(f"   - 描述: {memory_item.description}")
        return True
    except Exception as e:
        print(f"❌ 创建 MemoryItem 实例失败: {e}")
        return False

def test_create_retriever():
    """测试创建MemoryRetriever实例"""
    try:
        from retrieval.retriever import MemoryRetriever
        
        # 注意：这可能需要下载模型，可能会比较慢
        print("正在创建 MemoryRetriever 实例（可能需要下载模型）...")
        retriever = MemoryRetriever()
        print("✅ 成功创建 MemoryRetriever 实例")
        print(f"   - 模型名称: {retriever.model_name}")
        return True
    except Exception as e:
        print(f"❌ 创建 MemoryRetriever 实例失败: {e}")
        print("   这通常是因为缺少 sentence-transformers 依赖")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("测试 memory 和 retriever 模块导入")
    print("=" * 60)
    
    # 测试导入
    memory_import_success = test_import_memory()
    retriever_import_success = test_import_retriever()
    
    print("\n" + "-" * 40)
    print("测试创建实例")
    print("-" * 40)
    
    # 测试创建实例
    if memory_import_success:
        test_create_memory_item()
    
    if retriever_import_success:
        test_create_retriever()
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if memory_import_success and retriever_import_success:
        print("✅ 所有模块导入成功！")
        print("\n解决方案：")
        print("在 openai_api_agent.py 中添加以下代码：")
        print("```python")
        print("import sys")
        print("from pathlib import Path")
        print("")
        print("# 添加reasoning-bank-slm-main/src到Python路径")
        print("current_dir = Path(__file__).parent.parent.parent")
        print("reasoning_bank_src = current_dir / 'reasoning-bank-slm-main' / 'src'")
        print("sys.path.insert(0, str(reasoning_bank_src))")
        print("")
        print("try:")
        print("    from memory import MemoryItem, ReasoningBank")
        print("    from retrieval.retriever import MemoryRetriever")
        print("    MEMORY_AVAILABLE = True")
        print("except ImportError as e:")
        print("    MEMORY_AVAILABLE = False")
        print("    print(f'Warning: Memory retrieval system not available: {e}')")
        print("```")
    else:
        print("❌ 部分模块导入失败")
        print("\n可能的原因：")
        print("1. 缺少依赖包：pip install sentence-transformers")
        print("2. 路径配置问题")
        print("3. 文件不存在或损坏")

if __name__ == "__main__":
    main()
