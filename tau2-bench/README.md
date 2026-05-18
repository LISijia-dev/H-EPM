# $\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment


## Overview

$\tau^2$-bench implements a simulation framework for evaluating customer service agents across various domains.

Each domain specifies:
- a policy that the agent must follow
- a set of tools that the agent can use
- a set of tasks to evaluate the agent's performance
- Optionally: A set of tools that the user simulator can use

Domains are:
- `mock`
- `airline`
- `retail`
- `telecom`

All the information that an agent developer needs to build an agent for a domain can be accessed through the domain's API docs. See [View domain documentation](#view-domain-documentation) for more details.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench
```

2. Create a new environment (optional)

$\tau^2$-bench requires Python 3.10 or higher. You may create and activate a new environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install tau2

```bash
pip install -e .
```

This will enable you to run the `tau2` command.

**Note:** If you use `pip install .` (without `-e`), you'll need to set the `TAU2_DATA_DIR` environment variable to point to your data directory:

```bash
export TAU2_DATA_DIR=/path/to/your/tau2-bench/data
```

**Check your data directory setup:**

After installation, you can verify that your data directory is correctly configured by running:

```bash
tau2 check-data
```

This command will check if the data directory exists and print instructions if it is missing.

To remove all the generated files and the virtual environment, run:
```bash
make clean
```

## Quick Start

### Setup LLM API keys

We use [LiteLLM](https://github.com/BerriAI/litellm) to manage LLM APIs, so you can use any LLM provider supported by LiteLLM.

To provide your API keys, copy `.env.example` as `.env` and edit it to include your API keys.

### Run agent evaluation

To run a test evaluation on only 5 tasks with 1 trial per task, run:

```bash
tau2 run \ 
--domain airline \
--agent-llm gpt-4.1 \
--user-llm gpt-4.1 \
--num-trials 1 \
--num-tasks 5
```

Results will be saved in `data/tau2/simulations/`.

## Command Line Interface

The `tau2` command provides a unified interface for all functionality:

### Running Benchmark 
```bash
tau2 run \
  --domain <domain> \
  --agent-llm <llm_name> \
  --user-llm <llm_name> \
  --num-trials <trial_count> \
  --task-ids <task_ids> \
  --max-concurrency <concurrent_sims> \
  ...
```

### Viewing Results
```bash
tau2 view
```
This tool allows you to:
- Browse simulation files (in `data/tau2/simulations/`)
- View agent performance metrics
- View a particular simulation
- View task details

### View domain documentation
```bash
tau2 domain <domain>
```
Visit http://127.0.0.1:8004/redoc to see the domain policy and API documentation.

![domain_viewer1](figs/domain_viewer.png)

### Check data configuration
```bash
tau2 check-data
```
This command checks if your data directory is properly configured and all required files are present.

## Experiments

### Running Ablation Studies (No User, or Agent with Oracle Plan)
`telecom` domain enables running ablation studies.

1. Running an LLM in `no-user` mode. In this mode, the LLM is given all the tools and the information upfront.
Just choose `llm_agent_solo` as the agent and `dummy_user` as the user.

```bash
tau2 run \
  --domain telecom \
  --agent llm_agent_solo \
  --agent-llm gpt-4.1 \
  --user dummy_user \
  ...
```

2. Running an LLM in `oracle-plan` mode. In this mode, the LLM is given an oracle plan ahead of time alleviating the need for action planning.
Just choose `llm_agent_gt` as the agent.

```bash
tau2 run \
  --domain telecom \
  --agent llm_agent_gt \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  ...
```

### Running Telecom Domain with Workflow Policy
To test the impact of policy format, we provide an additional "workflow" policy for the telecom domain.
To run using this policy, use the `telecom-workflow` domain.

```bash
tau2 run \
  --domain telecom-workflow \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  ...
```

### Running with Conversation Summarization

We extend the base benchmark with an optional conversation summarization mode.
When enabled (`--summary`), both the agent and the user simulator receive an
additional instruction in their system prompt that asks them to first
summarize the current state of the conversation and then make a decision
based on that summary. This is useful for studying how explicit state
tracking impacts long-horizon, multi-turn tool-using behavior.

- Implementation entry points:
  - Agent prompt: `AGENT_INSTRUCTION_SUMMARY` in [src/tau2/agent/llm_agent.py](src/tau2/agent/llm_agent.py)
  - User prompt: `SYSTEM_PROMPT_SUMMARY` in [src/tau2/user/user_simulator.py](src/tau2/user/user_simulator.py)
- The flag is wired through `RunConfig.summary` and the run name will get a
  `_summary` suffix so summarized runs are easy to distinguish in
  `data/tau2/simulations/`.

Example:

```bash
tau2 run \
  --domain telecom \
  --agent-llm gpt-4o \
  --user-llm gpt-4.1 \
  --summary
```

### Running with a Tool Graph (Tool Suggestion)

We also add a *tool graph* mode that augments the agent with a learned graph
of tool-to-tool transitions and per-tool usage notes. At each step, the agent
uses the current conversation context to retrieve a small set of candidate
tools from the graph instead of being exposed to the full tool inventory. This
reduces tool-selection noise on domains with many tools.

- Tool selector / graph loader: see `ToolSelector` in
  [src/tau2/agent/llm_agent.py](src/tau2/agent/llm_agent.py).
- Graph building / saving utilities:
  [src/tau2/utils/tool_graph_utils.py](src/tau2/utils/tool_graph_utils.py)
  (`build_tool_graph_from_trajectory`, `save_tool_graph`,
  `save_combined_tool_graphs`, `print_tool_graph`).
- The tool graph is a JSON file with `nodes` (tools + descriptions/usage
  info) and `edges`/`links` (observed transitions between tools). Reference
  graphs we built are shipped at the repo root, e.g.
  `tool_graph_gpt-4.1.json`.

The graph path is consumed by `run_tasks(..., tool_graph_path=...)` in
[src/tau2/run.py](src/tau2/run.py). Set it to the JSON graph you want to use
(either one of the provided graphs, or one produced from a previous run).

Example (combine summarization with tool-graph guidance on telecom):

```bash
tau2 run \
  --domain telecom \
  --agent-llm gpt-4o \
  --user-llm gpt-4.1 \
  --summary
# uses the tool_graph_path configured in src/tau2/run.py
```

#### Building a Tool Graph from Trajectories

The tool graph JSON is produced by mining successful trajectories (those with
`reward == 1.0`) collected from a previous run. Each saved trajectory file is
expected to contain a `traj` list of chat messages (with assistant
`tool_calls`) and a top-level `reward` field.

Builder script:
[apitool_graph_builder.py](apitool_graph_builder.py)

What it does:
- Scans an input folder of trajectory JSON files and keeps only the ones with
  `reward == 1.0` (`load_files_from_folder`).
- For each kept trajectory, extracts the ordered sequence of assistant tool
  calls via `extract_sequences_from_traj`, and additionally captures the
  argument content of summarization tools (`summarize_the_task`,
  `summarize_task_state`) as per-edge `current_information`.
- Aggregates consecutive `(u, v)` tool transitions across all trajectories
  into a weighted directed graph (`build_graph_from_data`) and writes it to
  JSON with the schema:

  ```json
  {
    "nodes": ["tool_a", "tool_b", ...],
    "edges": [
      {"u": "tool_a", "v": "tool_b", "weight": 0.42, "count": 7,
       "current_information": ["...summary text..."]}
    ]
  }
  ```

Usage:

```bash
# Default paths are hardcoded at the top of main(); override via positional args:
python apitool_graph_builder.py \
  <input_trajectory_folder> \
  <output_tool_graph.json>

# Example: build a graph from a folder of SFT-style trajectories
python apitool_graph_builder.py \
  /path/to/data/messages/sft_data_2 \
  tool_graph_gpt-4.1_new.json
```

Typical workflow:
1. Run `tau2 run ...` to collect simulations under `data/tau2/simulations/`
   (or any folder of per-task JSON files that include `reward` and `traj`).
2. Run `apitool_graph_builder.py` on that folder to produce a
   `tool_graph_*.json`.
3. Point `tool_graph_path` in [src/tau2/run.py](src/tau2/run.py) at the new
   graph and re-run `tau2 run ... --summary` to evaluate with tool
   suggestion enabled.

## Domains

For all the details see the domains [README](src/tau2/domains/README.md).

### Basics

- Code is located in `src/tau2/domains/`
- Data is located in `data/tau2/domains/`
- Each domain has its own configuration and task definitions

#### View domain-specific policy and API docs:
Run the following command to see the domain policy and API documentation.
```bash
tau2 env <domain>
```

Then visit http://127.0.0.1:8004/redoc

### Environment CLI (beta)

An interactive command-line interface for directly querying and testing domain environments. Features:
- Interactive query interface with domain-specific tools
- Support for multiple domains (airline, mock, etc.)
- Session management with history

To use:
```bash
make env-cli
```

Available commands:
- `:q` - quit the program
- `:d` - change domain
- `:n` - start new session (clears history)

Example usage:
```bash
$ make env-cli

Welcome to the Environment CLI!
Connected to airline domain.

Query (:n new session, :d change domain, :q quit)> What flights are available from SF to LA tomorrow?
Assistant: Let me check the flight availability for you...
[Flight details will appear here]
```

The Environment CLI is useful for:
- Testing domain tools and queries
- Debugging environment responses
- Exploring available domain functionality
- Quick domain interaction without starting the full server stack


## Run tests
To run the test suite use the command

```sh
make test
```

## Config

To configure the framework, see the [config](src/tau2/config.py) file.

### LLM Calls caching
LLM call caching is disabled by default.

To enable LLM calls caching:
    - Make sure `redis` is running.
    - Update the redis config in `config.py` if necessary.
    - Set `LLM_CACHE_ENABLED` to `True` in `config.py`


```# apitau2-bench
