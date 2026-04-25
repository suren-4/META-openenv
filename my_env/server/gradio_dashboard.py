# Copyright (c) RuntimeTerror Team. All rights reserved.
# Cloud Resource Negotiation Arena — Gradio Dashboard

import gradio as gr
import plotly.graph_objects as go
import json
from datetime import datetime
try:
    import pandas as pd
except ImportError:
    # Dummy pd if not installed
    class DummyPD:
        def DataFrame(self, *args, **kwargs): return []
    pd = DummyPD()

def get_utilization_plot(env):
    """Generate Plotly figure for Cluster Utilization."""
    if env is None or not hasattr(env, 'cluster'):
        return go.Figure()
    
    try:
        util = env.cluster.utilization()
        fig = go.Figure(data=[
            go.Bar(name='CPU', x=['Utilization %'], y=[util.get('cpu', 0)], marker_color='#3b82f6'),
            go.Bar(name='RAM', x=['Utilization %'], y=[util.get('ram', 0)], marker_color='#10b981'),
            go.Bar(name='GPU', x=['Utilization %'], y=[util.get('gpu', 0)], marker_color='#8b5cf6'),
        ])
        fig.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title="Cluster Resource Utilization",
            margin=dict(t=40, b=20, l=20, r=20),
            height=300
        )
        return fig
    except Exception as e:
        print(f"Error in get_utilization_plot: {e}")
        return go.Figure()

def get_trust_matrix(env):
    """Generate data for Trust Matrix (returns list of lists)."""
    if env is None or not hasattr(env, 'trust'):
        return []
    
    try:
        matrix = env.trust.get_trust_matrix()
        if not matrix:
            return []
            
        # Standard roles in order
        agents = ["frontend", "ml_pipeline", "data_warehouse", "devops"]
        rows = []
        for r_agent in agents:
            row = [r_agent]
            for col_agent in agents:
                val = 0.0
                if r_agent in matrix and col_agent in matrix[r_agent]:
                    val = matrix[r_agent][col_agent]
                row.append(f"{val:.2f}")
            rows.append(row)
            
        return rows
    except Exception as e:
        print(f"Error in get_trust_matrix: {e}")
        return []

def trigger_adversarial(env, event_type: str):
    """Judge control to manually trigger adversarial events."""
    if not hasattr(env, 'scenario'):
        return "❌ Error: Environment not initialized correctly."
    
    if event_type == "rogue":
        if hasattr(env, 'agents') and env.agents:
            import random
            target = random.choice(list(env.agents.keys()))
            env.agents[target].is_rogue = True
            return f"🚨 CORRUPTION ALERT: Agent '{target}' has been compromised!"
        return "❌ No agents available to target."
    
    elif event_type == "failure":
        if hasattr(env, 'cluster'):
            env.cluster.apply_failure("cpu", 100)
            env.cluster.apply_failure("ram_gb", 256)
            return "⚠️ CRITICAL FAILURE: Hardware nodes reported offline. Capacity reduced."
            
    elif event_type == "surge":
        if hasattr(env, '_market_multiplier'):
            env._market_multiplier = 3.0
            return "📈 MARKET SURGE: Resource demand has spiked 300%!"
            
    return "✅ Adversarial event injected successfully."

def get_oversight_report(env):
    """Retrieve latest report from Oversight Agent."""
    if hasattr(env, 'oversight'):
        return env.oversight.get_latest_report()
    return "Fleet AI: Scanning cluster... no anomalies detected."

def _escape_md(text: str) -> str:
    import re
    """Escape Markdown special characters in user-controlled content."""
    return re.sub(r"([\\`*_\{\}\[\]()#+\-.!|~>])", r"\\\1", str(text))

def _format_observation(data: dict) -> str:
    """Format reset/step response for Markdown display."""
    lines = []
    obs = data.get("observation", {})
    if isinstance(obs, dict):
        if obs.get("prompt"):
            lines.append(f"**Prompt:**\n\n{_escape_md(obs['prompt'])}\n")
    reward = data.get("reward")
    done = data.get("done")
    if reward is not None:
        lines.append(f"**Reward:** `{reward}`")
    if done is not None:
        lines.append(f"**Done:** `{done}`")
    return "\n".join(lines) if lines else "*No observation data*"

def extract_tables(data):
    obs = data.get("observation", {})
    
    tasks_data = []
    for t in obs.get("unassigned_tasks", []):
        res = t.get("resources_required", {})
        tasks_data.append([
            t.get("id", ""),
            t.get("task_type", ""),
            t.get("primary_team", ""),
            res.get("cpu", 0),
            res.get("ram_gb", 0),
            t.get("base_value", 0),
            t.get("deadline_minutes", 0)
        ])
        
    agents_data = []
    for a in obs.get("agent_roster", []):
        agents_data.append([
            a.get("agent_id", ""),
            f"{a.get('cumulative_reward', 0):.1f}",
            a.get("reputation", 1.0),
            a.get("tasks_in_flight_count", 0)
        ])
        
    return tasks_data, agents_data

def build_arena_dashboard(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    """
    Builds the custom Arena Dashboard Gradio tab with improved UI elements.
    """
    env = web_manager.env
    
    # Pre-defined choices from models.py Enums
    action_types = ["bid", "propose_coalition", "respond", "renegotiate", "pass"]
    agent_roles = ["frontend", "ml_pipeline", "data_warehouse", "devops"]
    
    with gr.Blocks() as dashboard:
        gr.Markdown(
            """
            <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 15px; border: 1px solid #334155; margin-bottom: 20px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);'>
                <h1 style='color: #4ade80; font-size: 2.5rem; margin-bottom: 5px; text-shadow: 0 0 15px rgba(74, 222, 128, 0.3);'>🌌 Cloud Resource Arena</h1>
                <p style='color: #94a3b8; font-size: 1.1rem;'>Autonomous Multi-Agent Governance & Strategic Negotiation</p>
                <div style='display: flex; justify-content: center; gap: 20px; margin-top: 15px;'>
                    <span style='background: #1e293b; padding: 5px 15px; border-radius: 20px; color: #4ade80; border: 1px solid #4ade80; font-size: 0.8rem;'>LIVE TELEMETRY</span>
                    <span style='background: #1e293b; padding: 5px 15px; border-radius: 20px; color: #f472b6; border: 1px solid #f472b6; font-size: 0.8rem;'>ADVERSARIAL EVAL</span>
                    <span style='background: #1e293b; padding: 5px 15px; border-radius: 20px; color: #60a5fa; border: 1px solid #60a5fa; font-size: 0.8rem;'>FLEET AI OVERSIGHT</span>
                </div>
            </div>
            """
        )
        
        with gr.Tabs():
            with gr.Tab("Arena Control"):
                with gr.Row():
                    # Left Column: Charts and Trust
                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 📊 Live Cluster Intelligence")
                            plot_util = gr.Plot(label="Utilization")
                            refresh_plot_btn = gr.Button("🔄 Refresh Telemetry")
                            
                        with gr.Group():
                            gr.Markdown("### 🤝 Agent Trust Heatmap")
                            trust_df = gr.Dataframe(headers=["Agent", "frontend", "ml_pipeline", "data_warehouse", "devops"])
                            refresh_trust_btn = gr.Button("🔄 Refresh Trust Graph")
                    
                    # Right Column: Controls and Reports
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### ⚖️ Judge / Director Controls")
                            with gr.Row():
                                trigger_rogue_btn = gr.Button("💥 Corrupt Agent", variant="stop")
                                trigger_fail_btn = gr.Button("⚠️ Hardware Fail", variant="secondary")
                                trigger_surge_btn = gr.Button("📈 Market Surge", variant="secondary")
                            status_text = gr.Textbox(label="System Response", interactive=False, placeholder="Awaiting judge command...")
                            
                        with gr.Group():
                            gr.Markdown("### 🤖 Fleet AI (Oversight Agent)")
                            oversight_text = gr.Textbox(label="Oversight Narrative", lines=4, value="Fleet AI initializing...", interactive=False)
                            refresh_oversight_btn = gr.Button("🔍 Interrogate Fleet AI", variant="primary")

            with gr.Tab("Playground (Interactive Step)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎮 Step Control")
                        with gr.Group():
                            agent_id = gr.Dropdown(choices=agent_roles, label="Acting Agent", value="frontend")
                            action_type = gr.Dropdown(choices=action_types, label="Action Type", value="pass")
                            
                            # Conditional fields based on action_type
                            with gr.Column(visible=False) as bid_fields:
                                task_id = gr.Textbox(label="Task ID", placeholder="Enter task ID...")
                                price = gr.Number(label="Price Offered", value=10.0)
                                eta = gr.Slider(minimum=1, maximum=120, step=1, label="ETA (minutes)", value=30)
                            
                            with gr.Column(visible=False) as coalition_fields:
                                coalition_task_id = gr.Textbox(label="Task ID", placeholder="Enter task ID...")
                                peer_agents = gr.CheckboxGroup(choices=agent_roles, label="Peer Agents")
                                
                            with gr.Column(visible=False) as respond_fields:
                                proposal_id = gr.Textbox(label="Proposal ID", placeholder="Enter proposal ID...")
                                accept = gr.Radio(choices=["Accept", "Reject"], label="Decision", value="Accept")

                            step_btn = gr.Button("Execute Step", variant="primary")
                            reset_btn = gr.Button("Reset Environment", variant="secondary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Active Environment State")
                        tasks_table = gr.Dataframe(headers=["Task ID", "Type", "Team", "CPU", "RAM", "Value", "Deadline"], label="Available Tasks (Copy Task ID from here)")
                        agents_table = gr.Dataframe(headers=["Agent", "Reward", "Reputation", "In Flight"], label="Agent Roster & Rewards")
                        with gr.Accordion("Raw Logs (Advanced)", open=False):
                            obs_display = gr.Markdown(value="Click **Reset** or **Step** to see results.")
                            raw_json = gr.Code(label="Raw JSON Response", language="json")

        # UI Logic for showing/hiding fields
        def update_fields(a_type):
            return {
                bid_fields: gr.update(visible=(a_type == "bid")),
                coalition_fields: gr.update(visible=(a_type == "propose_coalition")),
                respond_fields: gr.update(visible=(a_type == "respond"))
            }
        
        action_type.change(fn=update_fields, inputs=action_type, outputs=[bid_fields, coalition_fields, respond_fields])

        # Step and Reset actions
        async def execute_step(agent, a_type, t_id, p, e, c_t_id, peers, prop_id, decision):
            action_data = {
                "agent_id": agent,
                "action_type": a_type
            }
            if a_type == "bid":
                action_data["task_id"] = t_id
                action_data["price_offered"] = p
                action_data["eta_minutes"] = e
            elif a_type == "propose_coalition":
                action_data["task_id"] = c_t_id
                action_data["peer_agents"] = peers
            elif a_type == "respond":
                action_data["proposal_id"] = prop_id
                action_data["accept"] = (decision == "Accept")
            
            try:
                data = await web_manager.step_environment(action_data)
                t_data, a_data = extract_tables(data)
                return _format_observation(data), t_data, a_data, json.dumps(data, indent=2)
            except Exception as exc:
                return f"Error: {exc}", [], [], ""

        async def reset_env():
            try:
                data = await web_manager.reset_environment()
                t_data, a_data = extract_tables(data)
                return _format_observation(data), t_data, a_data, json.dumps(data, indent=2)
            except Exception as exc:
                return f"Error: {exc}", [], [], ""

        # Callbacks
        refresh_plot_btn.click(fn=lambda: get_utilization_plot(env), outputs=plot_util)
        refresh_trust_btn.click(fn=lambda: get_trust_matrix(env), outputs=trust_df)
        
        # Adversarial controls
        trigger_rogue_btn.click(fn=lambda: trigger_adversarial(env, "rogue"), outputs=status_text)
        trigger_fail_btn.click(fn=lambda: trigger_adversarial(env, "failure"), outputs=status_text)
        trigger_surge_btn.click(fn=lambda: trigger_adversarial(env, "surge"), outputs=status_text)
        
        # Oversight
        refresh_oversight_btn.click(fn=lambda: get_oversight_report(env), outputs=oversight_text)
        
        step_btn.click(
            fn=execute_step,
            inputs=[agent_id, action_type, task_id, price, eta, coalition_task_id, peer_agents, proposal_id, accept],
            outputs=[obs_display, tasks_table, agents_table, raw_json]
        )
        reset_btn.click(fn=reset_env, outputs=[obs_display, tasks_table, agents_table, raw_json])
        
        # Callbacks are sufficient
        pass
        
    return dashboard
