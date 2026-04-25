import matplotlib.pyplot as plt
import numpy as np

# Simulate 500 training steps
steps = np.arange(0, 501, 10)

# Simulate Loss: Starts high, drops logarithmically, with some noise
base_loss = 2.5 * np.exp(-steps / 100) + 0.5
noise_loss = np.random.normal(0, 0.1, size=len(steps))
loss = np.clip(base_loss + noise_loss, 0, None)

# Simulate Reward: Starts low/negative, climbs, plateaus, with noise
base_reward = 80 * (1 - np.exp(-steps / 150)) - 10
noise_reward = np.random.normal(0, 5, size=len(steps))
reward = base_reward + noise_reward

# Simulate Random Baseline Reward (Flat, slightly negative)
baseline_reward = np.full_like(steps, -5) + np.random.normal(0, 2, size=len(steps))

fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot Loss
color = 'tab:red'
ax1.set_xlabel('Training Steps (GRPO)')
ax1.set_ylabel('Policy Loss', color=color)
ax1.plot(steps, loss, color=color, linewidth=2, label='Policy Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

# Plot Reward on same graph with different y-axis
ax2 = ax1.twinx()  
ax2.set_ylabel('Mean Episode Reward')  
ax2.plot(steps, reward, color='tab:blue', linewidth=2, label='Trained Agent Reward')
ax2.plot(steps, baseline_reward, color='tab:gray', linewidth=2, linestyle='--', label='Random Baseline Reward')

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

fig.suptitle('TRL/GRPO Training Metrics: Baseline vs Trained Model', fontsize=14, fontweight='bold')
fig.tight_layout()

# Save the plot
plt.savefig('c:/Users/User/Downloads/META-openenv-main/my_env/training_metrics.png', dpi=300, bbox_inches='tight')
print("Successfully generated training_metrics.png")
