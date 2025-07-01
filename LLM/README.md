# LLM Blue Agent System for CAGE Challenge 2

This module provides a comprehensive LLM-based Blue Team agent system for the CAGE Challenge 2 cybersecurity simulation. The system uses **LangGraph** for state management and prompt orchestration, supporting multiple LLM backends for flexible deployment.

## Features

- **Modular LLM Backends**: Support for OpenAI, Anthropic Claude, Google Gemini, and local HuggingFace models
- **LangGraph Integration**: Structured workflow with state management and memory
- **Multiple Prompt Templates**: Pre-built templates for different defense strategies
- **Action Validation**: Robust parsing and validation of LLM outputs
- **RL Compatibility**: Drop-in replacement for existing RL agents
- **Comprehensive Testing**: Full test suite with mock backends

## Architecture

The system consists of several key components:

1. **LLMBlueAgent**: Main agent class that implements the CybORG BaseAgent interface
2. **Backend System**: Modular interface for different LLM providers
3. **LangGraph Workflow**: State management and prompt orchestration
4. **Prompt Templates**: Configurable prompts for different strategies
5. **Action Mapping**: Translation between LLM outputs and environment actions

## Quick Start

### Installation

1. Install the required dependencies:
```bash
pip install -r Requirements.txt
```

2. Set up API keys for your chosen LLM provider:
```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"

# For Google Gemini
export GOOGLE_API_KEY="your-google-key"
```

### Basic Usage

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
print(f"Agent chose action: {action}")
```

## LLM Backends

### OpenAI Backend

```python
from LLM import create_backend

backend = create_backend("openai", model_name="gpt-4", temperature=0.0)
agent = LLMBlueAgent(backend_type="openai", backend_config={
    "model_name": "gpt-4",
    "temperature": 0.0,
    "max_tokens": 50
})
```

### Anthropic Claude Backend

```python
agent = LLMBlueAgent(backend_type="anthropic", backend_config={
    "model_name": "claude-3-sonnet-20240229",
    "temperature": 0.0,
    "max_tokens": 50
})
```

### Google Gemini Backend

```python
agent = LLMBlueAgent(backend_type="gemini", backend_config={
    "model_name": "gemini-pro",
    "temperature": 0.0,
    "max_tokens": 50
})
```

### Local HuggingFace Backend

```python
agent = LLMBlueAgent(backend_type="local", backend_config={
    "model_name": "microsoft/DialoGPT-medium",
    "temperature": 0.0,
    "max_tokens": 50
})
```

## Prompt Templates

The system includes several pre-built prompt templates for different defense strategies:

### Available Templates

- **base**: Comprehensive template with detailed instructions
- **aggressive**: Prioritizes immediate threat response
- **conservative**: Emphasizes careful analysis before action
- **balanced**: Adaptive strategy based on situation
- **expert**: Detailed reasoning with expert-level instructions
- **simple**: Minimal template for smaller models

### Using Different Templates

```python
from LLM.prompts import get_prompt_template

# Get a specific template
aggressive_prompt = get_prompt_template("aggressive")

# Create agent with custom template
agent = LLMBlueAgent(
    backend_type="openai",
    prompt_template=aggressive_prompt
)
```

### Custom Templates

You can also create your own prompt templates:

```python
custom_prompt = """You are a cybersecurity defender.

Available actions:
- Analyze <Host>: Check for threats
- Remove <Host>: Remove malware
- Restore <Host>: Restore system
- Decoy <Host>: Deploy decoy

Current situation: {observation}
Previous actions: {summary}

Choose action:"""

agent = LLMBlueAgent(
    backend_type="openai",
    prompt_template=custom_prompt
)
```

## LangGraph Workflow

The agent uses LangGraph for structured decision-making:

1. **Observation Ingestion**: Converts environment observations to text
2. **Prompt Formatting**: Builds the complete prompt with context
3. **LLM Decision**: Calls the configured LLM backend
4. **Action Parsing**: Validates and maps LLM output to environment actions
5. **State Update**: Maintains history and context for future decisions

### State Management

The agent maintains state across episodes:

```python
# State includes:
state = BlueAgentState(
    current_observation="Alert on User0",
    history=["Step 1: Analyze User0", "Step 2: Remove User0"],
    summary="Analyzed and removed malware from User0",
    episode_step=3
)
```

## Action Mapping

The system automatically maps LLM outputs to environment actions:

```python
# LLM outputs like "Analyze User0" are mapped to action indices
action_mapping = {
    "Analyze User0": 0,
    "Remove User0": 1,
    "Restore User0": 2,
    "Decoy User0": 3,
    # ... more actions
}
```

## Integration with CAGE Environment

The LLM Blue agent is designed as a drop-in replacement for RL agents:

```python
from CybORG import CybORG
from LLM import LLMBlueAgent

# Create the agent
blue_agent = LLMBlueAgent(backend_type="openai")

# Use in CybORG environment
env = CybORG(agents={'Blue': blue_agent, 'Red': red_agent})

# The agent will be called automatically during simulation
```

## Configuration Options

### Agent Configuration

```python
agent = LLMBlueAgent(
    backend_type="openai",           # LLM provider
    backend_config={                 # Backend-specific settings
        "model_name": "gpt-4",
        "temperature": 0.0,
        "max_tokens": 50
    },
    prompt_template=custom_prompt,   # Custom prompt template
    max_history_length=10,          # Number of steps to remember
)
```

### Backend Configuration

Each backend supports different configuration options:

**OpenAI:**
- `model_name`: Model to use (e.g., "gpt-4", "gpt-3.5-turbo")
- `temperature`: Randomness (0.0 = deterministic)
- `max_tokens`: Maximum response length

**Anthropic:**
- `model_name`: Model to use (e.g., "claude-3-sonnet-20240229")
- `temperature`: Randomness
- `max_tokens`: Maximum response length

**Local:**
- `model_name`: HuggingFace model name
- `temperature`: Randomness
- `max_tokens`: Maximum response length

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/test_llm_blue_agent.py -v

# Run specific test categories
python -m pytest tests/test_llm_blue_agent.py::TestLLMBlueAgent -v
python -m pytest tests/test_llm_blue_agent.py::TestBackendSystem -v
```

## Examples

See `examples/llm_blue_agent_example.py` for comprehensive examples including:

- Different backend configurations
- Custom prompt templates
- Environment simulation
- Error handling

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your API keys are set in environment variables
2. **Import Errors**: Install required packages with `pip install -r Requirements.txt`
3. **LangGraph Not Available**: The system falls back to simple mode if LangGraph is not installed
4. **Action Parsing Failures**: Check that your prompt template produces expected output format

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed information about:
# - Prompt formatting
# - LLM calls
# - Action parsing
# - State updates
```

## Performance Considerations

1. **API Costs**: Monitor usage when using cloud LLM providers
2. **Response Time**: Local models may be slower but more cost-effective
3. **Context Length**: Be mindful of token limits for long episodes
4. **Memory Usage**: Large history can increase prompt size

## Contributing

To extend the system:

1. **Add New Backends**: Implement the `LLMBackend` interface
2. **New Prompt Templates**: Add to `LLM/prompts.py`
3. **Custom Actions**: Update action mapping in the agent
4. **Tests**: Add corresponding test cases

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [CAGE Challenge 2](https://github.com/cage-challenge/cage-challenge-2)
- [CybORG Framework](https://github.com/cage-challenge/CybORG) 