# src/retrieval/retriever.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Optional
import re
import sys
sys.path.append('..')
from .memory import MemoryItem

class MemoryRetriever:
    """Retrieve relevant memories using semantic search"""
    
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        """
        Initialize with a pre-trained model from Hugging Face Hub
        
        Args:
            embedding_model_name: Name of the model from Hugging Face Hub
                                 Default: 'all-MiniLM-L6-v2' (lightweight, fast)
                                 Alternatives: 'all-mpnet-base-v2' (better quality)
        """
        self.model = SentenceTransformer(embedding_model_name)
        print(f"Loaded embedding model: {embedding_model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_memories(self, memories: List[MemoryItem]):
        """Generate embeddings for all memories"""
        texts = [f"{m.title}. {m.description}" for m in memories]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        for memory, embedding in zip(memories, embeddings):
            memory.embedding = embedding.tolist()
    
    def _has_answer_leak(self, memory: MemoryItem, expected_value: str) -> bool:
        """
        Check if memory contains the expected answer in a result context.
        More precise than checking any number - only filters if the number
        appears near result-indicating words.
        """
        if not expected_value:
            return False
        
        memory_text = f"{memory.title} {memory.description} {memory.content}".lower()
        
        # Extract the expected number
        expected_numbers = set(re.findall(r'\b\d+\.?\d*\b', str(expected_value)))
        
        if not expected_numbers:
            return False
        
        # Check if any expected number appears near result keywords
        result_keywords = [
            'answer', 'result', 'total', 'equals', 'final', 
            '=', 'is', 'solution', 'outcome'
        ]
        
        for num in expected_numbers:
            # Find positions of this number in the text
            num_positions = [m.start() for m in re.finditer(r'\b' + re.escape(num) + r'\b', memory_text)]
            
            for pos in num_positions:
                # Check 50 characters before and after the number
                context_start = max(0, pos - 50)
                context_end = min(len(memory_text), pos + 50)
                context = memory_text[context_start:context_end]
                
                # If any result keyword appears near this number, it's likely a leaked answer
                if any(keyword in context for keyword in result_keywords):
                    return True
        
        return False
    
    def retrieve(self, query: str, memories: List[MemoryItem], top_k: int = 3, 
                 expected_value: Optional[str] = None) -> List[Tuple[MemoryItem, float]]:
        """Retrieve top-k most relevant memories with answer leak protection"""
        if not memories:
            return []
        
        # Ensure all memories have embeddings
        if any(m.embedding is None for m in memories):
            self.embed_memories(memories)
        
        # Embed query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Calculate similarities
        memory_embeddings = np.array([m.embedding for m in memories])
        similarities = np.dot(memory_embeddings, query_embedding)
        
        # Get top candidates (more than needed for filtering)
        top_indices = np.argsort(similarities)[-(top_k * 3):][::-1]
        
        candidates = [(memories[idx], float(similarities[idx])) for idx in top_indices]
        
        # Filter out memories that contain the answer in a result context
        if expected_value:
            filtered = [(m, s) for m, s in candidates if not self._has_answer_leak(m, expected_value)]
            results = filtered[:top_k]
        else:
            results = candidates[:top_k]
        
        return results
    
    def format_memories_for_prompt(self, retrieved: List[Tuple[MemoryItem, float]]) -> str:
        """Format retrieved memories for injection into prompt"""
        if not retrieved:
            return ""
        
        formatted = "## Past Strategy Hints:\n\n"
        formatted += "**Important**: These are STRATEGY hints only. Do NOT copy any numbers from them.\n\n"
        
        for idx, (memory, score) in enumerate(retrieved, 1):
            status = "✓ Success Strategy" if memory.success else "✗ Lesson from Failure"
            formatted += f"### Strategy {idx} ({status}):\n"
            formatted += f"**{memory.title}**\n"
            formatted += f"{memory.content}\n\n"
        
        return formatted