from typing import Dict, Any, List, Optional, TypedDict
import logging
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage

from CybORG.Agents import BaseAgent
from CybORG.Shared.Results import Results
from .base import BaseLLMAgent
from .backends import LLMBackend, create_backend

logger = logging.getLogger(__name__)


@dataclass
class BlueAgentState:
    """State for the Blue agent LangGraph workflow."""
    current_observation: str = ""
    history: List[str] = field(default_factory=list)
    summary: str = ""
    raw_llm_output: str = ""
    selected_action: Any = None
    episode_step: int = 0


class LLMBlueAgent(BaseLLMAgent):
    """
    LLM-based Blue Team agent for CAGE Challenge 2.
    
    Uses LangGraph for state management and prompt orchestration.
    Supports multiple LLM backends (OpenAI, Anthropic, Gemini, local models).
    """
    
    def __init__(
        self,
        backend_type: str = "openai",
        backend_config: Dict[str, Any] = None,
        prompt_template: str = None,
        max_history_length: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Initialize LLM backend
        backend_config = backend_config or {}
        self.backend = create_backend(backend_type, **backend_config)
        
        # Agent configuration
        self.max_history_length = max_history_length
        self.prompt_template = prompt_template or self._get_default_prompt_template()
        
        self.graph = self._build_workflow_graph()
        self.state = BlueAgentState()
        
        self.action_mapping = self._build_action_mapping()
    
    def _get_default_prompt_template(self) -> str:
        return """You are a Blue Team cyber defense agent in a network security simulation. You can perform actions to detect and respond to a Red Team attacker. Your objective is to keep the critical systems running and eliminate the attacker as quickly as possible.
                  Environment Details:
                  - The network has multiple subnets (User, Enterprise, Operational) with various hosts.
                  - You receive alerts from monitoring systems about suspicious activity.
                  - You have the following actions available:
                      * Analyze <Host>: Investigate a host for malware or signs of compromise.
                      * Remove <Host>: Remove malicious software from a host (if malware is present and not too entrenched).
                      * Restore <Host>: Revert a host to a clean backup (use if a host is heavily compromised).
                      * Decoy <Host>: Deploy a decoy service on a host to mislead or trap the attacker.
                  - The Red agent is constantly trying to penetrate and move laterally; prioritize actions that minimize their presence.

                  Previous State: {summary}
                  Current Observation: {observation}

                  Based on the above, choose **one** best action from [Analyze, Remove, Restore, Decoy] and format it as '<Action> <HostName>'. Do not add any extra explanation.

                  Examples:
                  - Observation: Alert - HostA has unknown process running → Action: Analyze HostA
                  - Observation: Malware confirmed on HostA → Action: Remove HostA
                  - Observation: HostA heavily compromised → Action: Restore HostA

                  Your action:"""

    def _build_action_mapping(self) -> Dict[str, int]:
        # TODO: customize based on CAGE 2
        actions = []
        hosts = ["User0", "User1", "User2", "Enterprise0", "Enterprise1", "Enterprise2", "Operational0"]
        
        for host in hosts:
            actions.extend([
                f"Analyze {host}",
                f"Remove {host}", 
                f"Restore {host}",
                f"Decoy {host}"
            ])
        
        return {action: idx for idx, action in enumerate(actions)}
    
    def _build_workflow_graph(self):
        if StateGraph is None:
            return None
        
        # state schema
        workflow = StateGraph(BlueAgentState)
        
        # nodes
        workflow.add_node("format_prompt", self._format_prompt_node)
        workflow.add_node("call_llm", self._call_llm_node)
        workflow.add_node("parse_action", self._parse_action_node)
        workflow.add_node("update_state", self._update_state_node)
        
        # flow
        workflow.set_entry_point("format_prompt")
        workflow.add_edge("format_prompt", "call_llm")
        workflow.add_edge("call_llm", "parse_action")
        workflow.add_edge("parse_action", "update_state")
        workflow.add_edge("update_state", END)
        
        return workflow.compile()
    
    def _format_prompt_node(self, state: BlueAgentState) -> Dict[str, Any]:
        prompt = self.prompt_template.format(
            summary=state.summary or "No previous actions",
            observation=state.current_observation
        )
        return {"prompt": prompt}
    
    def _call_llm_node(self, state: BlueAgentState, prompt: str) -> Dict[str, Any]:
        try:
            response = self.backend.generate(prompt)
            return {"raw_llm_output": response}
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return {"raw_llm_output": "Analyze User0"}  # Fallback action
    
    def _parse_action_node(self, state: BlueAgentState, raw_llm_output: str) -> Dict[str, Any]:
        try:
            action_text = raw_llm_output.strip()
            
            # extract action from the text
            for action_name in self.action_mapping.keys():
                if action_name.lower() in action_text.lower():
                    action_idx = self.action_mapping[action_name]
                    return {"selected_action": action_idx, "parsed_action_text": action_name}
            
            # If no exact match, try to parse common patterns
            words = action_text.split()
            if len(words) >= 2:
                action_type = words[0].capitalize()
                host_name = words[1]
                
                # Try to find a matching action
                for action_name in self.action_mapping.keys():
                    if action_name.startswith(action_type) and host_name in action_name:
                        action_idx = self.action_mapping[action_name]
                        return {"selected_action": action_idx, "parsed_action_text": action_name}
            
            # TODO: Fallback to a safe action, sleep?
            logger.warning(f"Could not parse action from: {action_text}")
            return {"selected_action": 0, "parsed_action_text": "Analyze User0"}
            
        except Exception as e:
            logger.error(f"Action parsing error: {e}")
            return {"selected_action": 0, "parsed_action_text": "Analyze User0"}
    
    def _update_state_node(self, state: BlueAgentState, parsed_action_text: str) -> Dict[str, Any]:
        """Update the agent state with the selected action."""
        history_entry = f"Step {state.episode_step}: {parsed_action_text}"
        state.history.append(history_entry)
        
        # Keep history within limits
        if len(state.history) > self.max_history_length:
            state.history = state.history[-self.max_history_length:]
        
        # Update summary
        recent_actions = state.history[-3:] if len(state.history) >= 3 else state.history
        state.summary = "; ".join(recent_actions)
        
        state.episode_step += 1
        
        return {}
    
    def _observation_to_text(self, observation) -> str:
        """Convert environment observation to human-readable text."""
        if isinstance(observation, str):
            return observation
        
        if isinstance(observation, (list, tuple)):
            # Assume it's a vector representation
            return self._vector_observation_to_text(observation)
        
        if isinstance(observation, dict):
            return self._dict_observation_to_text(observation)
        
        # Fallback
        return str(observation)
    
    def _vector_observation_to_text(self, obs_vector) -> str:
        """Convert vector observation to text (placeholder implementation)."""
        # TODO: This should be customized based on the actual CAGE observation format
        return f"Network observation vector with {len(obs_vector)} features"
    
    def _dict_observation_to_text(self, obs_dict) -> str:
        alerts = []
        for key, value in obs_dict.items():
            if value and key.lower() in ['alert', 'compromise', 'malware', 'suspicious']:
                alerts.append(f"{key}: {value}")
        
        if alerts:
            return "; ".join(alerts)
        else:
            return "No alerts detected"
    
    def get_action(self, observation, action_space=None, hidden=None):
        # Convert observation to text
        obs_text = self._observation_to_text(observation)
        self.state.current_observation = obs_text
        
        if self.graph is not None:
            # Use LangGraph workflow
            try:
                result = self.graph.invoke(self.state)
                action = result.selected_action
            except Exception as e:
                logger.error(f"LangGraph execution error: {e}")
                action = 0  # Fallback
        else:
            # Fallback mode without LangGraph
            prompt = self.prompt_template.format(
                summary=self.state.summary or "No previous actions",
                observation=obs_text
            )
            
            try:
                response = self.backend.generate(prompt)
                # Simple parsing without full workflow
                action = self._parse_action_simple(response)
            except Exception as e:
                logger.error(f"Fallback mode error: {e}")
                action = 0
        
        return action
    
    def _parse_action_simple(self, response: str) -> int:
        """Simple action parsing for fallback mode."""
        response = response.strip()
        
        for action_name in self.action_mapping.keys():
            if action_name.lower() in response.lower():
                return self.action_mapping[action_name]
        
        return 0  # TODO: Default to sleep
    
    def train(self, results: Results):
        self._log_transition(results.observation, results.action, results.reward)
    
    def end_episode(self):
        """Reset agent state at episode end."""
        self.state = BlueAgentState()
        super().end_episode()
    
    def set_initial_values(self, action_space, observation):
        pass 