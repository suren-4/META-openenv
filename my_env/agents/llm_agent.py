# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — LLM Agent

"""
LLM-based agent using transformers.
Submits actions outputted by the LLM logic based on `obs.to_prompt()`.
"""

import json
from transformers import PreTrainedModel, PreTrainedTokenizer
try:
    from my_env.models import ArenaObservation, ArenaAction, ActionType
except ImportError:
    try:
        from ..models import ArenaObservation, ArenaAction, ActionType
    except (ImportError, ValueError):
        from models import ArenaObservation, ArenaAction, ActionType
from .base_agent import BaseAgent


class LLMAgent(BaseAgent):
    """An agent that acts purely via LLM generation."""

    def __init__(self, agent_id: str, team: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        super().__init__(agent_id, team)
        self.model = model
        self.tokenizer = tokenizer

    def act(self, observation: ArenaObservation) -> ArenaAction:
        prompt = observation.to_prompt()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        try:
            # We assume the LLM outputs a JSON payload matching the Action schema
            action_data = json.loads(response_text.strip())
            
            # Ensure agent_id is correctly set
            action_data["agent_id"] = self.agent_id
            
            return ArenaAction(**action_data)
        except Exception:
            # Fallback to PASS if parsing fails or invalid JSON
            return ArenaAction(action_type=ActionType.PASS, agent_id=self.agent_id)
