# CAGE Challenge 2 - RLLM Repository

This repository contains implementations for the CAGE Challenge 2 cybersecurity simulation, featuring both Reinforcement Learning (RL) and Large Language Model (LLM) based agents.

## PPO Baseline

This repository contains a minimal implementation of a vanilla PPO agent (updated from the original recurrent version). Two agent variants are provided:

* **`Agents/PPOAgent.py`** – a basic wrapper around the lightweight PPO
  algorithm located in `PPO/PPO.py` (adapted from the `PPO_discrete.py`
  example).  It uses the network defined in `PPO/ActorCritic.py`.

* **`Agents/PPOICMAgent.py`** – extends the base agent with an Intrinsic
  Curiosity Module (ICM) to encourage exploration.  This demonstrates how the
  template curiosity module can be integrated into the training loop.

## Intrinsic Curiosity Module

An implementation of the Intrinsic Curiosity Module (ICM) is provided in
`Agents/ICM_template.py`.  The module contains both the forward and inverse
prediction heads described in the original paper.  To integrate curiosity
driven exploration:

1. Instantiate the `ICM` module in your agent and add its parameters to the
   optimizer.
2. Call the module with the previous state, next state and one-hot encoded
   action to obtain an intrinsic reward and auxiliary predictions.
3. Combine the curiosity reward with the environment reward and include the
   forward and inverse losses when updating the policy.

These steps allow you to extend the base agent with curiosity while keeping its
core training loop simple.

## LLM Blue Agent System

The `LLM` package contains a comprehensive LLM-based Blue Team agent system for autonomous cyber defense. The system uses **LangGraph** for state management and prompt orchestration, supporting multiple LLM backends.

### Key Features

- **Modular LLM Backends**: Support for OpenAI, Anthropic Claude, Google Gemini, and local HuggingFace models
- **LangGraph Integration**: Structured workflow with state management and memory
- **Multiple Prompt Templates**: Pre-built templates for different defense strategies
- **Action Validation**: Robust parsing and validation of LLM outputs
- **RL Compatibility**: Drop-in replacement for existing RL agents

### Quick Start

```python
from LLM import LLMBlueAgent

# Create an agent with OpenAI backend
agent = LLMBlueAgent(
    backend_type="openai",
    backend_config={
        "model_name": "gpt-4",
        "temperature": 0.0,
        "max_tokens": 50
    }
)

# Use the agent
observation = "Alert: Suspicious activity detected on User0"
action = agent.get_action(observation)
```

### Available Backends

- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude-3 models
- **Google Gemini**: Gemini Pro
- **Local**: HuggingFace models (e.g., DialoGPT)

### Prompt Templates

The system includes several pre-built templates:
- **base**: Comprehensive template with detailed instructions
- **aggressive**: Prioritizes immediate threat response
- **conservative**: Emphasizes careful analysis before action
- **balanced**: Adaptive strategy based on situation
- **expert**: Detailed reasoning with expert-level instructions
- **simple**: Minimal template for smaller models

### Integration with CAGE Environment

The LLM Blue agent is designed as a drop-in replacement for RL agents:

```python
from CybORG import CybORG
from LLM import LLMBlueAgent

# Create the agent
blue_agent = LLMBlueAgent(backend_type="openai")

# Use in CybORG environment
env = CybORG(agents={'Blue': blue_agent, 'Red': red_agent})
```

For detailed documentation, see [LLM/README.md](LLM/README.md).

## Experimental LLM Agents

The `LLM` package also contains experimental agents that make decisions with large
language models. The agents are orchestrated using the `langgraph` library to
manage the prompt generation flow. Four strategies are provided:

1. **OneShotAgent** – prompts an LLM for an action given the current state.
2. **RAGAgent** – augments the prompt with recently observed transitions.
3. **FinetunedAgent** – assumes the underlying model has been fine-tuned on
   relevant data.
4. **RAGFinetunedAgent** – combines retrieval augmented prompts with a
   fine-tuned model.

All agents log the observed state, chosen action and reward to a small SQLite
database via `TrajectoryDatabase`.
The `langgraph` package is an optional dependency used to construct the
generation graphs for these agents.

## Installation

1. Install dependencies:
```bash
pip install -r Requirements.txt
```

2. Set up API keys (for LLM agents):
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

## Testing

Run the test suites:

```bash
# Test PPO agents
python -m pytest tests/test_ppo.py -v

# Test LLM Blue agent
python -m pytest tests/test_llm_blue_agent.py -v

# Test all agents
python -m pytest tests/test_agents.py -v
```

## Examples

- **LLM Blue Agent**: See `examples/llm_blue_agent_example.py` for comprehensive examples
- **PPO Agents**: Use the agents directly in CybORG environment

## Architecture

The repository follows a modular design:

```
├── Agents/           # Agent implementations (PPO, LLM)
├── PPO/             # PPO algorithm and actor-critic networks
├── LLM/             # LLM agent system with backends and prompts
├── tests/           # Test suites
├── examples/        # Usage examples
└── Requirements.txt # Dependencies
```

## Contributing

To extend the system:

1. **New LLM Backends**: Implement the `LLMBackend` interface
2. **Prompt Templates**: Add to `LLM/prompts.py`
3. **Agent Variants**: Create new agent classes
4. **Tests**: Add corresponding test cases

## References

- [CAGE Challenge 2](https://github.com/cage-challenge/cage-challenge-2)
- [CybORG Framework](https://github.com/cage-challenge/CybORG)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
