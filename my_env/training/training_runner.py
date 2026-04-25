# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Training Runner

"""
Runs episodes with baseline agents or tests the environment.
"""

import argparse
import time
from typing import Dict

try:
    from my_env.client import CloudArenaClient
    from my_env.agents import FrontendAgent, MLAgent, DataAgent, DevOpsAgent, RogueAgent
except (ImportError, ModuleNotFoundError):
    try:
        from client import CloudArenaClient
        from agents import FrontendAgent, MLAgent, DataAgent, DevOpsAgent, RogueAgent
    except (ImportError, ModuleNotFoundError):
        from my_env.client import CloudArenaClient
        from my_env.agents import FrontendAgent, MLAgent, DataAgent, DevOpsAgent, RogueAgent


def main():
    parser = argparse.ArgumentParser(description="Cloud Arena Runner")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="FastAPI Server URL")
    parser.add_argument("--rogue", action="store_true", help="Include rogue agent")
    args = parser.parse_args()

    # Initialize baseline agents
    agents = {
        "frontend": FrontendAgent(agent_id="frontend", team="frontend"),
        "ml_pipeline": MLAgent(agent_id="ml_pipeline", team="ml_pipeline"),
        "data_warehouse": DataAgent(agent_id="data_warehouse", team="data_warehouse"),
    }
    
    if args.rogue:
        agents["devops"] = RogueAgent(agent_id="devops", team="devops")
    else:
        agents["devops"] = DevOpsAgent(agent_id="devops", team="devops")

    print(f"Connecting to {args.url}...")
    
    # We use sync client for the runner
    client = CloudArenaClient(base_url=args.url).sync()
    
    total_steps = 0
    start_time = time.time()
    
    with client:
        for episode in range(1, args.episodes + 1):
            print(f"--- Starting Episode {episode} ---")
            result = client.reset()
            obs = result.observation
            
            while not result.done:
                # The environment Observation tells us which agent's turn it is
                current_agent = obs.observing_agent
                
                # Get the agent strategy
                agent = agents.get(current_agent)
                if agent:
                    action = agent.act(obs)
                else:
                    # Fallback to pass
                    try:
                        from my_env.models import ArenaAction, ActionType
                    except ImportError:
                        from models import ArenaAction, ActionType
                    action = ArenaAction(action_type=ActionType.PASS, agent_id=current_agent)
                    
                # Submit action
                result = client.step(action)
                obs = result.observation
                total_steps += 1
                
            print(f"Episode {episode} completed in {obs.current_round} rounds.")
            
    elapsed = time.time() - start_time
    print(f"Finished {args.episodes} episodes ({total_steps} steps) in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
