# ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities

This software project accompanies the research paper, [ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities](https://arxiv.org/abs/2408.04682).

Recent large language models (LLMs) advancements sparked a growing research interest in tool assisted LLMs solving real-world challenges, which calls for comprehensive evaluation of tool-use capabilities. While previous works focused on either evaluating over stateless web services (RESTful API), based on a single turn user prompt, or an off-policy dialog trajectory, _ToolSandbox_ includes stateful tool execution, implicit state dependencies between tools, a built-in user simulator supporting on-policy conversational evaluation and a dynamic evaluation strategy for intermediate and final milestones over an arbitrary trajectory. We show that open source and proprietary models have a significant performance gap, and complex tasks like State Dependency, Canonicalization and Insufficient Information defined in _ToolSandbox_ are challenging even the most capable SOTA LLMs, providing brand-new insights into tool-use LLM capabilities.

## Our Extension: Tool Graph + Summarization

On top of the original ToolSandbox, we add two optional capabilities to the
CLI agent:

1. **Summarization** — both the agent and the user simulator can be asked
   to first summarize the current conversation state before producing the
   next action (`--summary on`).
2. **Tool Graph (Tool Suggestion)** — at each turn, instead of exposing the
   full tool inventory, the agent uses the current context to retrieve a
   small candidate set from a precomputed tool-to-tool transition graph
   mined from successful trajectories
   (`--enable_tool_suggestion --tool_graph_path <graph.json>`).

These are wired through the existing `tool_sandbox.cli` entry point and the
convenience launcher [run.sh](run.sh).

### Running

`run.sh` resolves a Python interpreter (env `PYTHON`, then active conda env,
then system `python`, then a local `.venv`) and invokes
`tool_sandbox.cli:main` with the CLI flags you pass.

```bash
# Vanilla baseline
./run.sh --agent GPT_4_o_2024_05_13 --user GPT_4_o_2024_05_13 -t

# With conversation summarization on both sides
./run.sh \
  --agent GPT_4_1 --user GPT_4_1 \
  --summary on \
  --split test

# With summarization AND tool-graph–based tool suggestion
./run.sh \
  --agent GPT_4_1 --user GPT_4_1 \
  --summary on \
  --split test \
  --enable_tool_suggestion \
  --tool_graph_path tool_graph_gpt-5.1.json
```

A prebuilt graph is shipped at [tool_graph_gpt-5.1.json](tool_graph_gpt-5.1.json).

### Building a Tool Graph from Trajectories

Builder: [build_graph.py](build_graph.py). It mines a directed, weighted
tool-call graph from a folder of ToolSandbox conversation logs, filtered by
per-scenario quality.

###### Inputs

- `trajectories_dir/` — output folder of a previous ToolSandbox run; it
  contains one `<scenario_name>/conversation.json` per scenario.
- `result_summary.json` — searched at
  `trajectories_dir/../result_summary.json` first, otherwise at
  `trajectories_dir/result_summary.json`. It must contain a
  `per_scenario_results` array with `scenario_name` and
  `milestone_similarity` fields.

###### Pipeline

1. `load_result_summary` builds `scenario_name -> milestone_similarity`.
2. `rglob("conversation.json")` collects all trajectories. Each trajectory
   is **kept only if `milestone_similarity > 0.55`** (others are dropped
   and counted).
3. `extract_sequences_from_traj` recovers the ordered tool-call sequence
   from assistant `tool_calls`, and (for `summarize_the_task` calls) maps
   `tool_call_id -> summary content` from the corresponding `tool`-role
   replies.
4. `build_graph_from_data` aggregates edges over all kept trajectories:
   - **Nodes**: all distinct tool names excluding `summarize_the_task`.
   - **Normal edges**: every consecutive `(seq[i], seq[i+1])` where neither
     side is `summarize_the_task` contributes `count += 1` and a
     `1 / len(seq)` weight bonus (down-weights long trajectories).
   - **Summarize-bridged edges**: when `seq[i] == summarize_the_task`, the
     nearest non-summary tools before/after it (`prev`, `next`) are bridged
     into an edge `prev -> next` with the same `count += 1` and
     `1 / len(seq)` bonus; the summary text (looked up by `tool_call_id`)
     is appended to that edge's `current_information` list.
5. The graph is written as JSON:

   ```json
   {
     "nodes": ["tool_a", "tool_b", ...],
     "edges": [
       {
         "u": "tool_a",
         "v": "tool_b",
         "weight": 3.27,
         "count": 3,
         "current_information": ["...summary text from bridged edges..."]
       }
     ]
   }
   ```

###### Usage

```bash
# Default paths are hardcoded at the top of main(); override via positional args:
python build_graph.py <trajectories_dir> <output_tool_graph.json>

# Example
python build_graph.py \
  data/suggqwen_Qwen_3_8B_user_gpt-4.1_summary_on_split_train_graph_False_.../ \
  tool_graph_qwen_8b_train.json
```

###### Typical workflow

1. Run `./run.sh ... --summary on --split train` to collect training-split
   trajectories (this also produces `result_summary.json` next to the
   trajectories folder).
2. Run `python build_graph.py <traj_dir> <out.json>` to mine the graph
   from scenarios with `milestone_similarity > 0.55`.
3. Re-run on the test split with the new graph plugged in:
   ```bash
   ./run.sh --agent GPT_4_1 --user GPT_4_1 --summary on \
            --split test --enable_tool_suggestion \
            --tool_graph_path <out.json>
   ```

## Getting started
### Installation
1. Create a virtual environment:
```bash
conda create -n ToolSandbox python=3.9
conda activate ToolSandbox
```

3. Install the dependencies with:
```bash
pip install '.[dev]'
```

    "transformers==4.41.2",

### Running our method

Run the benchmark using our method (with summary enabled and tool suggestion driven by a precomputed tool graph):

```bash
./run.sh --user GPT_4_1 --agent GPT_4_1 --summary on --split test --enable_tool_suggestion --tool_graph_path tool_graph_gpt-4.1.json
```

