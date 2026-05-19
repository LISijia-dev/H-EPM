# τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains


## Setup

1. Clone this repository:

```bash
git clone https://github.com/sierra-research/tau-bench && cd ./tau-bench
```

2. Install from source (which also installs required packages):

```bash
pip install -e .
```

## Run

Run a tool-calling agent on the τ-retail environment:

```bash
python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --user-model gpt-4o --user-model-provider openai --user-strategy llm --max-concurrency 10
```

Set max concurrency according to your API limit(s).

To run specific tasks, use the `--task-ids` flag. For example:

```bash
python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --user-model gpt-4o --user-model-provider openai --user-strategy llm --max-concurrency 10 --task-ids 2 4 6
```

This command will run only the tasks with IDs 2, 4, and 6.

## User simulators

By default, we use `gpt-4o` as the user simulator with strategy `llm`. You can use other models by setting the `--user-model` flag, or other strategies by setting the `--user-strategy` flag. For example, run a tool-calling agent with a claude user simulator:

```bash
python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --max-concurrency 10 --user-model claude-3-5-sonnet-20240620 --user-model-provider anthropic --user-strategy llm
```

## Our Extension: Tool Graph (Tool Suggestion)

On top of the original τ-bench, we add an optional *tool graph* mechanism that
augments the tool-calling agent with a learned graph of tool-to-tool
transitions mined from successful trajectories. At each step, instead of
exposing the agent to the full tool inventory, we use the current
conversation context to retrieve a small set of candidate tools from the
graph. This reduces tool-selection noise on environments with many tools.

### Files added

- Tool-graph–aware agents:
  - [tau_bench/agents/tool_calling_agent.py](tau_bench/agents/tool_calling_agent.py)
  - [tau_bench/agents/tool_calling_agent_graph.py](tau_bench/agents/tool_calling_agent_graph.py)

  Both accept `enable_tool_selection: bool` and `tool_graph_path: Optional[str]`;
  when enabled, they call `_load_tool_graph()` to read the JSON graph and use
  it to filter / re-rank tools per turn.
- Embedding-based retriever helper:
  [tau_bench/llm/azure_completion.py](tau_bench/llm/azure_completion.py)
  (loads the graph and embeds nodes for similarity-based tool retrieval).
- Graph builder: [tool_graph_builder.py](tool_graph_builder.py)
- Pre-built graph: [tool_graph_gpt-4.1.json](tool_graph_gpt-4.1.json)

### Running with a Tool Graph

The toggle is currently wired in code rather than via CLI: see
[tau_bench/run.py](tau_bench/run.py) (around `enable_tool_selection=True,
tool_graph_path=...`). Point `tool_graph_path` at the JSON graph you want to
use, then launch as usual:

```bash
python run.py \
  --agent-strategy tool-calling \
  --env retail \
  --model gpt-4o --model-provider openai \
  --user-model gpt-4.1 --user-model-provider openai \
  --user-strategy llm \
  --max-concurrency 10
```

### Building a Tool Graph from Trajectories

The tool graph JSON is mined from successful trajectories collected from a
previous τ-bench run. Each entry in the results JSON is expected to carry a
top-level `reward` field and a `traj` (or `messages`) chat list; only
entries with `reward == 1.0` are used.

Builder script: [tool_graph_builder.py](tool_graph_builder.py)

#### Pipeline

1. **Load** a τ-bench results JSON (a top-level list of run records).
2. **Filter** to records with `reward == 1.0`.
3. **Extract sequences** via `extract_sequences_from_traj`:
   - First pass: walk `assistant` messages and collect every `tool_calls[*]`
     in order as `{"name", "id"}` entries.
   - Second pass: walk `tool`-role messages and, when
     `name == "summarize_the_task"`, store the reply `content` indexed by
     `tool_call_id` so it can later be attached to the bridged edge.
4. **Aggregate edges** (`build_graph_from_data`):
   - **Nodes** are all distinct tool names except `summarize_the_task`
     itself (summarize is treated as a meta tool, not a graph node).
   - **Normal edges**: for each consecutive `(seq[i], seq[i+1])` where
     neither side is `summarize_the_task`, increment `count` by 1 and add a
     bonus of `1 / len(seq)` to the weight. This downweights edges that
     only show up in very long trajectories.
   - **Summarize-bridged edges**: whenever `seq[i] == summarize_the_task`,
     find the nearest non-summarize tool before it (`prev`) and after it
     (`next`); add an edge `prev -> next` with `count += 1` and a
     **`10 * 1/len(seq)` bonus** (treating a summary step as a strong
     signal that the agent re-planned between `prev` and `next`). The
     summary text stored in step 3 is appended to that edge's
     `current_information` list, providing human-readable notes about why
     this transition happened.
5. **Write** the resulting graph as JSON:

   ```json
   {
     "nodes": ["tool_a", "tool_b", ...],
     "edges": [
       {
         "u": "tool_a",
         "v": "tool_b",
         "weight": 7.83,            // count + sum of (10×)(1/len) bonuses
         "count": 7,
         "current_information": ["...summary text from bridged edges..."]
       }
     ]
   }
   ```

#### Usage

```bash
# Default paths are hardcoded at the top of main(); override via positional args:
python tool_graph_builder.py \
  <path/to/tau_bench_results.json> \
  <output_tool_graph.json>

# Example
python tool_graph_builder.py \
  results/train-retail-train-tool-calling-gpt-4.1-0.0_..._user-gpt-4.1-llm_*.json \
  tool_graph_gpt-4.1.json
```

#### Typical workflow

1. Run `python run.py ...` to produce a results JSON under `results/`.
2. Run `python tool_graph_builder.py <results.json> <out.json>` to mine
   the graph from the successful trajectories.
3. Set `tool_graph_path` in [tau_bench/run.py](tau_bench/run.py) to the new
   graph (and keep `enable_tool_selection=True`), then re-run `python
   run.py ...` to evaluate with tool suggestion enabled.

