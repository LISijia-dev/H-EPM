# src/memory.py
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class MemoryItem:
    """Single memory item in ReasoningBank"""
    title: str
    description: str
    content: str
    source_problem_id: str
    success: bool
    created_at: str
    embedding: Optional[List[float]] = None
    
    def to_dict(self):
        data = asdict(self)
        return data
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class ReasoningBank:
    """Memory storage and retrieval system"""
    
    def __init__(self, storage_path='memory_bank/reasoning_bank.json'):
        self.storage_path = storage_path
        self.memories: List[MemoryItem] = []
        self.load()
    
    def add_memory(self, memory: MemoryItem):
        """Add new memory item"""
        self.memories.append(memory)
        self.save()
    
    def add_memories(self, memories: List[MemoryItem]):
        """Add multiple memories"""
        self.memories.extend(memories)
        self.save()
    
    def get_all_memories(self) -> List[MemoryItem]:
        """Get all memories"""
        return self.memories
    
    def save(self):
        """Persist to disk"""
        with open(self.storage_path, 'w') as f:
            data = [m.to_dict() for m in self.memories]
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load from disk"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.memories = [MemoryItem.from_dict(m) for m in data]
        except FileNotFoundError:
            self.memories = []
    
    def clear(self):
        """Clear all memories"""
        self.memories = []
        self.save()
    
    def __len__(self):
        return len(self.memories)