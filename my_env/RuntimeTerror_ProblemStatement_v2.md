# RuntimeTerror: Cloud Resource Negotiation Arena

## Meta PyTorch OpenEnv Hackathon - Grand Finale Problem Statement
### REVISED for Judging Impact

**Team:** RuntimeTerror  
**Team Leader:** Vikram S  
**Theme:** Multi-Agent Interactions  
**Application Domain:** Cloud Resource Allocation & Negotiation  
**Date:** April 2026

---

## Executive Summary

Four autonomous AI agents manage departmental infrastructure demands in a shared Kubernetes cluster. With limited resources (CPU, memory, GPU time), agents must **negotiate, bid, and form coalitions** to maximize value creation—both for themselves and the company. Through 48 hours of self-play, agents learn to:
- Bid strategically on resource blocks
- Negotiate fair coalition splits
- Adapt to task priority shifts
- Improve long-term partner trust

**Key Deliverable:** A visual dashboard showing agents negotiating resource allocation in real-time, with before/after learning curves.

---

## 1. Problem Statement

### The Real-World Scenario

Your company runs a shared Kubernetes cluster (500 CPU cores, 2TB RAM, 8 GPUs, 100 Mbps bandwidth). Four teams need to schedule work:

1. **Frontend Team** — Deploys web services, needs fast turnaround
2. **ML Pipeline Team** — Runs model training and hyperparameter sweeps
3. **Data Warehouse Team** — Manages ETL jobs and analytics queries
4. **DevOps Team** — Runs monitoring, backups, and infrastructure tests

Every day, new work arrives with:
- **Time sensitivity** — Some tasks are urgent, others can wait
- **Resource profile** — Different tasks need different resource mixes (CPU-heavy vs. GPU-heavy vs. I/O-heavy)
- **Value** — Some tasks contribute more business value than others

**The Challenge:** How can autonomous agents negotiate fair, efficient resource allocation?

### The Core Questions
1. How does an agent **value** a task given its resource cost and deadline?
2. How does it **negotiate** with peers when resources are scarce?
3. How does it **form coalitions** (e.g., "ML + Data team up on a joint pipeline")?
4. How does it **improve** by learning from past negotiation outcomes?

### Success Definition
- **Efficiency:** Maximize total value created across all tasks
- **Fairness:** No team is systematically starved of resources
- **Stability:** Agents develop trust and long-term working relationships
- **Resilience:** When a team cancels or a resource fails, agents adapt

---

## 2. Environment: Kubernetes Resource Marketplace

### Task Types & Examples

| Task Type | Team | Resources | Deadline | Base Value | Example |
|-----------|------|-----------|----------|------------|---------|
| API Deployment | Frontend | 4 CPU, 2GB RAM | 2 hours | 100 | Deploy new checkout flow |
| Model Training | ML | 8 CPU, 4GB RAM, 2 GPU | 8 hours | 300 | Train XGBoost on Q2 data |
| ETL Pipeline | Data | 12 CPU, 8GB RAM | 4 hours | 250 | Ingest daily vendor data |
| Monitoring Stack | DevOps | 2 CPU, 1GB RAM | 1 hour | 50 | Deploy Prometheus scraper |
| **Ad-hoc Analysis** | Data | 4 CPU, 2GB RAM | 6 hours | 150 | Customer segmentation query |
| **Emergency Patch** | Frontend | 2 CPU, 1GB RAM | 30 min | 200 | Security hotfix deployment |

### Marketplace Dynamics

**Task Arrival:**
- Poisson process (default λ=1.5 tasks/10-min window)
- Mix of predictable (scheduled) and unexpected (urgent) work
- Resource profiles vary; some tasks can partially parallelize (coalition-friendly)

**Cluster State:**
- **Total resources:** 500 CPU, 2TB RAM, 8 GPU, 100 Mbps bandwidth (per 10-minute window)
- **Usage visibility:** Agents see current allocation + forecast for next 3 windows
- **Task queue:** Public list of unassigned tasks; agents can propose assignments

**Environmental Stress:**
- **Scenario 1 (Normal):** ~80% cluster utilization, plenty of room for negotiation
- **Scenario 2 (Peak):** ~95% utilization, resource competition is intense
- **Scenario 3 (Failure):** Randomly kill an agent at step T (test resilience & renegotiation)

---

## 3. Agent Capabilities & Behavior

### Perception (What each agent observes)

**Public state:**
- Current queue of unassigned tasks (ID, resources, deadline, value, team affinity)
- Cluster utilization (current CPU%, RAM%, GPU%, bandwidth%)
- Agent roster (which teams are online, their resource reservations)
- History of past task completions (accuracy, time-to-completion, coalition outcomes)

**Private state:**
- Own tasks in flight and their progress
- Own resource budget remaining (can be exceeded if coalition partners contribute)
- Negotiation history with other agents (trust scores)
- Learned bidding model (estimated cost for different task types)

### Action Space (One per 10-minute time-step)

**1. BID on a task (Solo)**
```
Action: BID(task_id, resource_request, price_offered, eta)
  - task_id: Which task to bid on
  - resource_request: Specify exact CPU/RAM/GPU/bandwidth needed
  - price_offered: How much value you'll claim (0–100% of task value)
  - eta: Estimated completion time
Winner: Highest bidder gets task (with tie-breaking by past reliability)
```

**2. PROPOSE COALITION**
```
Action: PROPOSE_COALITION(task_id, peer_agents, subtask_split, reward_split)
  - task_id: Which task
  - peer_agents: List of agents to invite
  - subtask_split: How to decompose work (e.g., "Frontend: API, DevOps: monitoring, Data: logging")
  - reward_split: Reward distribution (e.g., "40% Frontend, 30% DevOps, 30% Data")
Outcome: Peers vote (accept/reject). If unanimous yes, task is split and executed.
```

**3. ACCEPT / REJECT**
```
Action: RESPOND_TO_PROPOSAL(proposal_id, acceptance)
  - Vote on another agent's coalition proposal
```

**4. RENEGOTIATE**
```
Action: RENEGOTIATE(task_id, new_deadline, new_resource_split)
  - Propose revised terms to current coalition members
  - Reason: learned new information about completion time, or another task became higher priority
Outcome: Coalition members vote to adjust or stick with original terms
```

**5. PASS**
```
Action: PASS
  - Skip this round (useful if overloaded or no good bids available)
```

### Communication Model

- **Protocol:** JSON message broadcast to all agents (instant delivery, no latency)
- **Honesty constraint:** Agents cannot lie about past task completion times (verifiable) or skill claims (agents have reputation scores that decay if they misrepresent capability)
- **Message types:**
  ```json
  {
    "type": "BID",
    "sender": "frontend_agent",
    "task_id": "T_api_deploy_001",
    "resource_request": {"cpu": 4, "ram_gb": 2, "gpu": 0},
    "price": 75,
    "eta_minutes": 120,
    "confidence": 0.92  // agent's self-assessed confidence
  }
  ```

---

## 4. Tasks & Subtasks

### Task Structure

Each task has:
- **Primary skill** — Primary team best suited for it
- **Subtasks** — Decomposable work that can be delegated
- **Resource profile** — CPU-heavy, GPU-heavy, memory-intensive, I/O-intensive
- **Deadline** — Hard deadline; penalties for lateness
- **Base value** — Fixed reward for on-time completion

### Example: ML Model Training Task

```
Task ID: T_ml_train_q2
Primary team: ML Pipeline
Resources: 8 CPU, 4GB RAM, 2 GPU
Duration estimate: 6 hours
Deadline: 8 hours from now
Base value: 300 units

Decomposable subtasks:
  - Data preprocessing (Data Warehouse team, 2 CPU, 2GB RAM, 1 hour, 50 value)
  - Model training (ML team, 8 CPU, 4GB RAM, 2 GPU, 4 hours, 200 value)
  - Validation & logging (DevOps team, 2 CPU, 1GB RAM, 1 hour, 50 value)

Potential coalitions:
  - Solo ML team (risky; tight on data prep time)
  - ML + Data (recommended; Data handles preprocessing)
  - ML + Data + DevOps (safest; offload logging/monitoring overhead)
```

### Task Completion Logic

- Agent(s) start execution at time T
- Resource consumption is tracked in real-time
- If resources exceed cluster capacity, task is queued
- Task succeeds if **all** subtasks finish before deadline
- If any subtask fails, entire task fails (coalition breakup penalty applies)

---

## 5. Reward Model & Evaluation

### Individual Agent Reward (Per Task)

```
agent_reward = (task_value × completion_multiplier) - (resource_cost × overhead_factor) + relationship_bonus
```

**Components:**
- **task_value × completion_multiplier:**
  - On-time: 1.0x value claimed in bid
  - 1–30 min late: 0.8x
  - 30–60 min late: 0.5x
  - >60 min late: 0.0x (penalty: -50 units)
  
- **resource_cost × overhead_factor:**
  - Base cost = sum of resources used × cluster price-per-unit
  - Overhead factor = 1.0 if solo, 0.85 if coalition (shared infrastructure gains)
  
- **relationship_bonus:**
  - +10 units per successful coalition with agent X (up to 3 prior coalitions max)
  - -25 units if agent abandons commitment to a coalition partner
  - Trust decays if agent defaults too often (reputation < 0.7)

### Collective Reward (Entire Marketplace, Per Episode)

```
collective_reward = 
  Σ(completed_task_values) 
  - Σ(resource_costs)
  - (failed_tasks × 100)
  + (on_time_bonus × tasks_completed_early)
  + (cluster_efficiency_bonus × (utilized_resources / total_resources))
```

**Phases:**
- **Phase 1 (Episode 1–50):** Agents optimize individual reward
- **Phase 2 (Episode 51–100):** Reward function shifts to emphasize collective reward; agents incentivized to cooperate
- **Final score:** Blend of Phase 1 + Phase 2 results

### Key Metrics for Learning Curves

| Metric | Interpretation |
|--------|-----------------|
| **Bidding accuracy** | (Predicted cost - Actual cost) / Actual cost. Lower = better model of resource consumption. Shows learning. |
| **Win rate** | % of bids that win. Should improve as agent learns to bid competitively but not wastefully. |
| **Coalition success rate** | % of coalitions that complete on-time. Shows agents learning who to trust. |
| **Avg reward per task** | Task value earned per unit time. Should increase as agents optimize. |
| **Reputation score** | Aggregate trust metric based on past behavior (0–1 scale). Agents with high rep win more bids. |

---

## 6. Self-Improvement Strategy

### A. Bidding Strategy Refinement (Thompson Sampling)

Each agent learns a **cost-estimation model** for different task types:

```
Learned model: cost_estimate(task_type, resources) → estimated_cpu_hours
Updates after each task:
  - Compare predicted cost vs. actual cost
  - Update posterior distribution (e.g., Beta for success probability)
  - Adjust next bid to be more competitive
```

**Learning signal:**
- Did the task complete on-time? (binary signal)
- What was the actual resource consumption? (numerical signal)
- Did the bid beat competitors? (feedback on strategy quality)

**Visualization:** Plot bidding accuracy over 100 episodes; expect convergence (S-curve learning)

### B. Coalition Partner Selection (Reputation-Based)

Each agent maintains a **trust graph**:
```
Trust(agent_A → agent_B) ∈ [0, 1]
  - Increases by 0.1 per successful coalition
  - Decays by 0.05 per failed coalition or missed deadline
  - Half-life = 20 episodes
```

When proposing coalitions, agents prioritize high-trust partners.

**Learning signal:**
- Did partners deliver on their subtask? (on-time completion)
- Did they communicate honestly? (actual effort vs. promised effort)
- Did the coalition revenue split make sense? (fair allocation)

**Visualization:** Show trust graph animation over time (agents forming clusters of trusted partners)

### C. Task Valuation Learning (Value Estimation)

Agents learn which **task types** generate the highest profit per resource:

```
Profitability = task_value / (predicted_resource_cost × time_to_complete)
Over time, agents prefer high-profitability tasks even if they're lower absolute value
```

**Learning signal:**
- After completing 10 instances of "API deployment" tasks, agent learns: "These are fast, low-value. Not worth fighting for."
- After completing 5 "Model training" tasks, agent learns: "Slow, high-value, risky if deadlines slip."

---

## 7. Demonstration & Storytelling (For Judges)

### The 3-Minute Pitch Structure

**0:00–0:30 (Intro & Problem)**
> "Every company with microservices faces this: Teams need resources, cluster is limited, who decides? We built an AI solution."

**0:30–1:30 (Live Demo)**
Show **real-time dashboard**:
- Left pane: Task queue (incoming tasks, status)
- Center pane: Agent bids & negotiations (animated message log)
- Right pane: Resource utilization (pie chart updating in real-time)
- Bottom: Reward curves for each agent

**Demo scenario:** 3-minute compressed episode (~50 tasks, 4 agents)

**1:30–2:30 (Learning Curves)**
Show **before vs. after**:
- Chart 1: Bidding accuracy over 100 episodes (S-curve improvement)
- Chart 2: Avg reward per agent (upward trend)
- Chart 3: Coalition success rate (improves over time)

**2:30–3:00 (Wrap-up & Impact)**
> "Our agents learned to negotiate like humans—fairly, efficiently, and with long-term thinking. This scales to real Kubernetes clusters."

### Demo Artifacts

You'll need:
1. **Live environment simulator** (Python, runs in 3 min)
2. **Real-time dashboard** (HTML/React, updates every frame)
3. **Pre-computed learning curves** (JSON data + Matplotlib graphs)

---

## 8. Implementation Roadmap (On-Site, 48-hour)

### Phase 1: Core Environment (Hours 0–10)
- [ ] Task generator (Poisson arrival, diverse resource profiles)
- [ ] Cluster simulator (track CPU/RAM/GPU allocation)
- [ ] Agent state management
- [ ] Basic bid/coalition logic

### Phase 2: Agent Baselines (Hours 10–20)
- [ ] Greedy agent (always bids; wins via volume)
- [ ] Conservative agent (careful bidding; low risk)
- [ ] Coalition-focused agent (favors partnerships)
- [ ] Baseline comparison (which strategy wins?)

### Phase 3: Learning & Adaptation (Hours 20–35)
- [ ] Thompson sampling for cost estimation
- [ ] Trust graph & partner selection
- [ ] Reward computation & tracking
- [ ] Learning loop integration

### Phase 4: Visualization & Dashboard (Hours 35–42)
- [ ] Real-time task/bid/coalition animation
- [ ] Resource utilization pie charts
- [ ] Reward curves (per-agent, collective)
- [ ] Reputation heatmap (who trusts whom)

### Phase 5: Evaluation & Polish (Hours 42–48)
- [ ] Run 100-episode learning experiment
- [ ] Generate final metrics dashboard
- [ ] Create demo scenario (3-min compressed episode)
- [ ] Polish presentation & slides

---

## 9. Evaluation Against Judging Criteria

### Environment Innovation (40% weight)
✅ **Multi-agent negotiation in infrastructure allocation is novel** — tests LLM theory-of-mind, strategic bidding, coalition formation  
✅ **Realistic but tractable** — inspired by real Kubernetes scheduling problems, but simplified enough for 48-hour build  
✅ **Meaningful complexity** — not a toy problem; involves resource constraints, deadlines, fairness  

### Storytelling & Demo (30% weight)
✅ **Concrete scenario** — every software engineer knows Kubernetes resource contention  
✅ **Visual demo is engaging** — live dashboard, real-time negotiations, clear narrative arc  
✅ **Judges can follow instantly** — no jargon needed; clear from problem statement alone  

### Training Progress (20% weight)
✅ **Clear learning signal** — bidding accuracy, coalition success rate, reward improvement all measurable  
✅ **Visualizable curves** — show S-curve convergence, trust graph clustering, reward growth  
✅ **Self-improvement is explicit** — agents genuinely learn across episodes  

### Reward Pipeline (10% weight)
✅ **Multi-layer reward** — individual + collective incentives, reputation bonuses, relationship multipliers  
✅ **Coherent training loop** — reward computation → gradient signal → policy update → next episode  
✅ **Integrates with HF TRL** — reward model easily plugs into standard RL training pipeline  

---

## 10. Technical Requirements

### OpenEnv Integration
Your environment will inherit from `OpenEnv.Env`:
```python
class CloudResourceNegotiationArena(OpenEnv.Env):
    def __init__(self, num_agents=4, cluster_capacity={...}, task_arrival_rate=1.5):
        ...
    
    def step(self, actions: Dict[str, Action]) -> Tuple[Observations, Rewards, Dones, Info]:
        # Process agent bids/proposals
        # Update cluster state
        # Assign tasks & compute rewards
        # Return observations for next step
        ...
    
    def reset(self):
        # Initialize cluster, agents, empty task queue
        ...
```

### HuggingFace TRL Integration
Use Unsloth or HF TRL to train a policy:
```python
from trl import PPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
trainer = PPOTrainer(
    model=model,
    args=training_args,
    reward_fn=reward_model,
    environment=CloudResourceNegotiationArena()
)

trainer.train()  # Agents learn via RL
```

---

## 11. Success Criteria (What You'll Show Judges)

By end of hackathon:

✅ **Working environment** — Tasks arrive, agents bid, coalitions form, tasks complete  
✅ **Live demo scenario** — 3-minute compressed episode showing negotiation in action  
✅ **Learning curves** — Graphs showing agent improvement over 100 episodes  
✅ **Real-time dashboard** — Animated resource allocation view during demo  
✅ **HF TRL integration** — Minimal training script showing agents learning (even if just 10 episodes)  
✅ **Blog post or video** — <2 min explainer on HuggingFace Hub or YouTube  

---

## 12. Extensions & Advanced Features (If Time Allows)

- **Multi-episode memory:** Agents remember past coalition outcomes across episodes
- **Skill specialization:** Agents develop expertise (e.g., "ML agent gets better at training tasks over time")
- **Dynamic pricing:** Cluster price-per-CPU fluctuates based on demand (adds strategic depth)
- **Agent failures:** Simulate agent crash; watch others renegotiate mid-task
- **Realistic workload patterns:** Use real Kubernetes trace data (Google Cluster Traces)

---

## References

1. Lowe et al. (2017). "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." *ICML*.
2. Rashid et al. (2018). "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning." *ICML*.
3. Kumar et al. (2020). "A Laplacian Framework for Option Discovery in Reinforcement Learning." *ICML*.
4. Shoham & Leyton-Brown (2008). *Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations.*
5. Google Cluster Traces: https://github.com/google/cluster-data

---

**End of Problem Statement**

*Next: Share with team, divide implementation tasks, sketch architecture.*
