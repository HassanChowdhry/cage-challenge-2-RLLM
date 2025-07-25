import logging
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import statistics

logger = logging.getLogger(__name__)

@dataclass
class EpisodeResult:
    episode_id: int
    total_reward: float
    steps: int
    actions_taken: List[str]
    final_state: str
    duration: float
    red_agent_type: str

@dataclass
class EvaluationResults:
    config: Dict[str, Any]
    episodes: List[EpisodeResult]
    summary: Dict[str, Any]
    
def save_results(config, results: EvaluationResults):
    try:
        output_file = config.get('output_file', 'llm_evaluation_results.json')
        data = {
            "config": {
                "backend_type": results.config.get('backend_type'),
                "prompt_name": results.config.get('prompt_name'),
                "episodes": results.config.get('episodes'),
                "max_steps": results.config.get('max_steps'),
                "red_agent": results.config.get('red_agent')
            },
            "summary": results.summary,
            "episodes": [
                {
                    "episode_id": ep.episode_id,
                    "total_reward": ep.total_reward,
                    "steps": ep.steps,
                    "duration": ep.duration,
                    "red_agent_type": ep.red_agent_type
                }
                for ep in results.episodes
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        
def calculate_summary(episode_results: List[EpisodeResult]) -> Dict[str, Any]:
    if not episode_results:
        return {"error": "No episodes completed"}
    
    rewards = [ep.total_reward for ep in episode_results]
    steps = [ep.steps for ep in episode_results]
    durations = [ep.duration for ep in episode_results]
    
    red_agent_results = {}
    for ep in episode_results:
        agent_type = ep.red_agent_type
        if agent_type not in red_agent_results:
            red_agent_results[agent_type] = []
        red_agent_results[agent_type].append(ep.total_reward)
    
    red_agent_stats = {}
    for agent_type, agent_rewards in red_agent_results.items():
        red_agent_stats[agent_type] = {
            "count": len(agent_rewards),
            "avg_reward": statistics.mean(agent_rewards),
            "success_rate": sum(1 for r in agent_rewards if r > 0) / len(agent_rewards)
        }
    
    summary = {
        "total_episodes": len(episode_results),
        "avg_reward": statistics.mean(rewards),
        "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0,
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "avg_steps": statistics.mean(steps),
        "avg_duration": statistics.mean(durations),
        "total_duration": sum(durations),
        "red_agent_breakdown": red_agent_stats
    }
    
    return summary
        
def print_results(results: EvaluationResults):
    print("\n" + "="*60)
    print("LLM BLUE AGENT EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  Backend: {results.config.get('backend_type', 'gemini')}")
    print(f"  Prompt: {results.config.get('prompt_name', 'zero_shot')}")
    print(f"  Episodes: {results.config.get('episodes', 10)}")
    print(f"  Red Agent: {results.config.get('red_agent', 'random')}")
    
    print(f"\nSummary:")
    print(f"  Total Episodes: {results.summary['total_episodes']}")
    print(f"  Average Reward: {results.summary['avg_reward']:.2f}")
    print(f"  Reward Std Dev: {results.summary['std_reward']:.2f}")
    print(f"  Min/Max Reward: {results.summary['min_reward']:.2f} / {results.summary['max_reward']:.2f}")
    print(f"  Average Steps: {results.summary['avg_steps']:.1f}")
    print(f"  Average Duration: {results.summary['avg_duration']:.2f}s")
    print(f"  Total Duration: {results.summary['total_duration']:.2f}s")
    
    if 'red_agent_breakdown' in results.summary:
        print(f"\nRed Agent Breakdown:")
        for agent_type, stats in results.summary['red_agent_breakdown'].items():
            print(f"  {agent_type}: {stats['count']} episodes, "
                  f"avg reward: {stats['avg_reward']:.2f}")
    
    print("\n" + "="*60)