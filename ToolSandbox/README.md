# ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities



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
