# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Utility functions for managing train/test splits of scenarios."""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import random

from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.scenarios import named_scenarios
from tool_sandbox.scenarios.single_tool_call_scenarios import (
    named_single_tool_call_scenarios,
)
from tool_sandbox.scenarios.multiple_tool_call_scenarios import (
    named_multiple_tool_call_scenarios,
)
from tool_sandbox.scenarios.multiple_user_turn_scenarios import (
    named_multiple_user_turn_scenarios,
)
from tool_sandbox.scenarios.insufficient_information_scenarios import (
    named_insufficient_information_scenarios,
)


class TrainTestSplitManager:
    """Manager for train/test splits of scenarios."""
    
    def __init__(self, split_dir: Path = Path("data/splits")):
        """
        Initialize the split manager.
        
        Args:
            split_dir: Directory containing the split JSON files
        """
        self.split_dir = split_dir
        self._combined_split = None
        self._module_splits = {}
        self._metadata = None
    
    def load_metadata(self) -> Dict:
        """Load split metadata."""
        if self._metadata is None:
            metadata_file = self.split_dir / "split_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self._metadata = json.load(f)
            else:
                raise FileNotFoundError(f"Split metadata not found at {metadata_file}")
        return self._metadata
    
    def load_combined_split(self) -> Dict[str, List[str]]:
        """Load the combined train/test split."""
        if self._combined_split is None:
            combined_file = self.split_dir / "combined_split.json"
            if combined_file.exists():
                with open(combined_file, 'r', encoding='utf-8') as f:
                    self._combined_split = json.load(f)
            else:
                raise FileNotFoundError(f"Combined split file not found at {combined_file}")
        return self._combined_split
    
    def load_module_split(self, module_name: str) -> Dict[str, List[str]]:
        """Load train/test split for a specific module."""
        if module_name not in self._module_splits:
            module_file = self.split_dir / f"{module_name}_split.json"
            if module_file.exists():
                with open(module_file, 'r', encoding='utf-8') as f:
                    self._module_splits[module_name] = json.load(f)
            else:
                raise FileNotFoundError(f"Module split file not found at {module_file}")
        return self._module_splits[module_name]
    
    def get_train_scenarios(self, module: Optional[str] = None) -> List[str]:
        """
        Get training scenario names.
        
        Args:
            module: Specific module name, or None for combined split
            
        Returns:
            List of training scenario names
        """
        if module is None:
            split = self.load_combined_split()
            return split.get("train", [])
        else:
            split = self.load_module_split(module)
            return split.get("train", [])
    
    def get_test_scenarios(self, module: Optional[str] = None) -> List[str]:
        """
        Get test scenario names.
        
        Args:
            module: Specific module name, or None for combined split
            
        Returns:
            List of test scenario names
        """
        if module is None:
            split = self.load_combined_split()
            return split.get("test", [])
        else:
            split = self.load_module_split(module)
            return split.get("test", [])
    
    def get_all_scenarios(self, module: Optional[str] = None) -> List[str]:
        """
        Get all scenario names (train + test).
        
        Args:
            module: Specific module name, or None for combined split
            
        Returns:
            List of all scenario names
        """
        train_scenarios = self.get_train_scenarios(module)
        test_scenarios = self.get_test_scenarios(module)
        return train_scenarios + test_scenarios
    
    def get_available_modules(self) -> List[str]:
        """Get list of available module names."""
        metadata = self.load_metadata()
        return metadata.get("modules", [])
    
    def get_scenarios_by_category(self, category: str) -> Tuple[List[str], List[str]]:
        """
        Get train/test scenarios for a specific category.
        
        Args:
            category: Category name (e.g., 'single_tool_call', 'multiple_tool_call', etc.)
            
        Returns:
            Tuple of (train_scenarios, test_scenarios)
        """
        train_scenarios = self.get_train_scenarios(category)
        test_scenarios = self.get_test_scenarios(category)
        return train_scenarios, test_scenarios
    
    def validate_split(self) -> Dict[str, any]:
        """
        Validate the train/test split.
        
        Returns:
            Dictionary with validation results
        """
        try:
            metadata = self.load_metadata()
            combined_split = self.load_combined_split()
            
            total_train = len(combined_split.get("train", []))
            total_test = len(combined_split.get("test", []))
            total_scenarios = total_train + total_test
            
            expected_total = metadata.get("total_scenarios", 0)
            expected_ratio = metadata.get("split_ratio", 0.8)
            expected_train = int(expected_total * expected_ratio)
            expected_test = expected_total - expected_train
            
            validation_results = {
                "is_valid": True,
                "total_scenarios": total_scenarios,
                "train_scenarios": total_train,
                "test_scenarios": total_test,
                "split_ratio": total_train / total_scenarios if total_scenarios > 0 else 0,
                "expected_total": expected_total,
                "expected_train": expected_train,
                "expected_test": expected_test,
                "expected_ratio": expected_ratio,
                "module_validation": {}
            }
            
            # Validate each module
            for module in metadata.get("modules", []):
                try:
                    module_split = self.load_module_split(module)
                    module_train = len(module_split.get("train", []))
                    module_test = len(module_split.get("test", []))
                    module_total = module_train + module_test
                    
                    validation_results["module_validation"][module] = {
                        "total": module_total,
                        "train": module_train,
                        "test": module_test,
                        "ratio": module_train / module_total if module_total > 0 else 0
                    }
                except Exception as e:
                    validation_results["module_validation"][module] = {
                        "error": str(e)
                    }
            
            return validation_results
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": str(e)
            }


def load_train_scenarios(preferred_tool_backend: ToolBackend = ToolBackend.DEFAULT,
                        split_dir: Path = Path("data/splits"),
                        module: Optional[str] = None) -> Dict[str, any]:
    """
    Load training scenarios.
    
    Args:
        preferred_tool_backend: Tool backend to use
        split_dir: Directory containing split files
        module: Specific module name, or None for combined split
        
    Returns:
        Dictionary mapping scenario names to scenario objects
    """
    manager = TrainTestSplitManager(split_dir)
    train_scenario_names = manager.get_train_scenarios(module)
    
    # Load all scenarios
    all_scenarios = named_scenarios(preferred_tool_backend)
    
    # Filter to only training scenarios
    train_scenarios = {name: scenario for name, scenario in all_scenarios.items() 
                      if name in train_scenario_names}
    
    return train_scenarios


def load_test_scenarios(preferred_tool_backend: ToolBackend = ToolBackend.DEFAULT,
                       split_dir: Path = Path("data/splits"),
                       module: Optional[str] = None) -> Dict[str, any]:
    """
    Load test scenarios.
    
    Args:
        preferred_tool_backend: Tool backend to use
        split_dir: Directory containing split files
        module: Specific module name, or None for combined split
        
    Returns:
        Dictionary mapping scenario names to scenario objects
    """
    manager = TrainTestSplitManager(split_dir)
    test_scenario_names = manager.get_test_scenarios(module)
    
    # Load all scenarios
    all_scenarios = named_scenarios(preferred_tool_backend)
    
    # Filter to only test scenarios
    test_scenarios = {name: scenario for name, scenario in all_scenarios.items() 
                     if name in test_scenario_names}
    
    return test_scenarios


def get_split_info(split_dir: Path = Path("data/splits")) -> Dict[str, any]:
    """
    Get information about the train/test split.
    
    Args:
        split_dir: Directory containing split files
        
    Returns:
        Dictionary with split information
    """
    manager = TrainTestSplitManager(split_dir)
    
    try:
        metadata = manager.load_metadata()
        combined_split = manager.load_combined_split()
        
        info = {
            "metadata": metadata,
            "total_scenarios": len(combined_split.get("train", [])) + len(combined_split.get("test", [])),
            "train_scenarios": len(combined_split.get("train", [])),
            "test_scenarios": len(combined_split.get("test", [])),
            "modules": manager.get_available_modules(),
            "module_details": {}
        }
        
        # Get details for each module
        for module in info["modules"]:
            try:
                train_scenarios = manager.get_train_scenarios(module)
                test_scenarios = manager.get_test_scenarios(module)
                info["module_details"][module] = {
                    "train_count": len(train_scenarios),
                    "test_count": len(test_scenarios),
                    "total_count": len(train_scenarios) + len(test_scenarios)
                }
            except Exception as e:
                info["module_details"][module] = {"error": str(e)}
        
        return info
        
    except Exception as e:
        return {"error": str(e)}


# Convenience functions for common use cases
def get_train_scenario_names(split_dir: Path = Path("data/splits"), 
                           module: Optional[str] = None) -> List[str]:
    """Get list of training scenario names."""
    manager = TrainTestSplitManager(split_dir)
    return manager.get_train_scenarios(module)


def get_test_scenario_names(split_dir: Path = Path("data/splits"), 
                          module: Optional[str] = None) -> List[str]:
    """Get list of test scenario names."""
    manager = TrainTestSplitManager(split_dir)
    return manager.get_test_scenarios(module)


def get_all_scenario_names(split_dir: Path = Path("data/splits"), 
                         module: Optional[str] = None) -> List[str]:
    """Get list of all scenario names."""
    manager = TrainTestSplitManager(split_dir)
    return manager.get_all_scenarios(module)


def _split_names(names: List[str], *, seed: int, ratio: float) -> Dict[str, List[str]]:
    """Deterministically split names into train/test lists."""
    assert 0.0 < ratio < 1.0
    shuffled = list(sorted(names))
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    split_index = int(len(shuffled) * ratio)
    return {"train": shuffled[:split_index], "test": shuffled[split_index:]}


def generate_and_save_splits(
    *,
    preferred_tool_backend: ToolBackend,
    split_dir: Path = Path("data/splits"),
    split_seed: int = 42,
    split_ratio: float = 0.8,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Generate 0.8/0.2 train/test splits for each scenario module and combined, and save JSON.

    Returns a dict keyed by module name (including "combined") to a dict with keys
    "train" and "test" listing scenario names.
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    # Build per-module name sets from their respective factories
    module_to_named_funcs = {
        "single_tool_call": named_single_tool_call_scenarios,
        "multiple_tool_call": named_multiple_tool_call_scenarios,
        "multiple_user_turn": named_multiple_user_turn_scenarios,
        "insufficient_information": named_insufficient_information_scenarios,
    }

    module_splits: Dict[str, Dict[str, List[str]]] = {}
    total_names: List[str] = []
    for module, factory in module_to_named_funcs.items():
        name_dict = factory(preferred_tool_backend=preferred_tool_backend)
        names = list(name_dict.keys())
        total_names.extend(names)
        module_splits[module] = _split_names(names, seed=split_seed, ratio=split_ratio)
        # Write per-module split
        with open(split_dir / f"{module}_split.json", "w", encoding="utf-8") as f:
            json.dump(module_splits[module], f, indent=4, ensure_ascii=False)

    # Combined split across all scenarios (including augmented ones)
    combined_names = list(named_scenarios(preferred_tool_backend).keys())
    combined_split = _split_names(combined_names, seed=split_seed, ratio=split_ratio)
    with open(split_dir / "combined_split.json", "w", encoding="utf-8") as f:
        json.dump(combined_split, f, indent=4, ensure_ascii=False)

    # Metadata
    metadata = {
        "modules": list(module_to_named_funcs.keys()),
        "split_ratio": split_ratio,
        "split_seed": split_seed,
        "total_scenarios": len(combined_names),
    }
    with open(split_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    # Return all
    module_splits["combined"] = combined_split
    return module_splits
