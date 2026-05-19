# H-EPM: Experience-Evolving Multi-Turn Tool-Use Agent with Hybrid Episodic-Procedural Memory

Official implementation of **H-EPM**, a hybrid episodic–procedural memory framework for multi-turn tool-use agents.

> H-EPM enables agents to evolve from partially overlapping experiences by jointly leveraging:
> - **Episodic memory** for context-aware retrieval
> - **Procedural memory** for reusable tool-transition patterns
>
> The framework improves both:
> - **Inference-time tool selection**
> - **Reinforcement learning exploration** in long-horizon multi-turn environments


---

## 🏗️ Architecture

```text
Historical Trajectories
        │
        ▼
State Summarization Tool
        │
        ▼
State-Annotated Tool Graph
        │
 ┌──────┴──────┐
 ▼             ▼
Procedural   Episodic
 Memory       Memory
(tool graph) (state summaries)
        │
        ▼
Adaptive Tool Selection
```

---



---

## 📊 Supported Benchmarks

H-EPM is evaluated on multiple multi-turn tool-use benchmarks:

- τ-Bench (Telecom)
- τ²-Bench (Retail)
- ToolSandbox

---

## 🧠 Memory Construction

H-EPM constructs a graph from successful historical trajectories.

Each edge stores:

```python
(tool_i, tool_j, weight, state_summary)
```

where:

- `weight` captures procedural transition quality
- `state_summary` stores compact contextual information

---

## 🔍 Adaptive Retrieval Strategy

At each decision step:

1. Locate the previously used tool in the graph
2. Decide whether state summarization is needed
3. If yes:
   - Retrieve episodic memories via state similarity
4. Otherwise:
   - Follow procedural transition weights
5. Return Top-k candidate tools

---

## 📚 Citation



---


