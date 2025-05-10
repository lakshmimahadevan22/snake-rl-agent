import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from IPython import display
import datetime
import os

plt.ion()

def display_metrics(scores, mean_scores, window_size=10):
    display.clear_output(wait=True)
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('Snake RL Training Progress', fontsize=16)
    ax1.set_xlabel('Episodes', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    
    
    ax1.plot(scores, 'b-', alpha=0.3, label='Episode Score')
    
    ax1.plot(mean_scores, 'r-', linewidth=2, label='Average Score')
    
    if len(scores) > window_size:
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(scores)), moving_avg, 'g-', 
                linewidth=2, label=f'{window_size}-Episode Moving Avg')
    
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')
    
    if len(scores) > 0:
        ax1.text(len(scores)-1, scores[-1], f"{scores[-1]}", 
                fontsize=10, ha='center', va='bottom')
    if len(mean_scores) > 0:
        ax1.text(len(mean_scores)-1, mean_scores[-1], f"{mean_scores[-1]:.1f}", 
                fontsize=10, ha='center', va='bottom', color='r')
    
    
    if len(scores) > 0:
        max_score = max(max(scores), 10)  
        ax1.set_ylim(0, max_score * 1.2)
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('Performance Metrics', fontsize=14)
    
    recent_avg = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
    
    best_score = max(scores) if scores else 0
    

    if len(scores) >= 20:
        prev_avg = np.mean(scores[-20:-10])
        improvement = ((recent_avg - prev_avg) / prev_avg) * 100 if prev_avg > 0 else 0
    else:
        improvement = 0
    
    metrics = ['Recent Avg', 'Best Score', 'Total Games']
    values = [recent_avg, best_score, len(scores)]
    colors = ['green', 'blue', 'orange']
    
    ax2.bar(metrics, values, color=colors, alpha=0.7)
    ax2.set_ylabel('Value')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(values):
        ax2.text(i, v, f"{v:.1f}" if i < 2 else f"{v}", 
                ha='center', va='bottom', fontsize=10)
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('Learning Progress', fontsize=14)
    
    if len(scores) > 0:
        segments = min(5, len(scores) // 10)
        if segments > 0:
            segment_size = len(scores) // segments
            segment_avgs = []
            
            for i in range(segments):
                start_idx = i * segment_size
                end_idx = start_idx + segment_size
                segment_avgs.append(np.mean(scores[start_idx:end_idx]))
            
            ax3.plot(range(1, segments+1), segment_avgs, 'o-', color='purple', linewidth=2)
            ax3.set_xlabel('Training Segment')
            ax3.set_ylabel('Average Score')
            ax3.set_xticks(range(1, segments+1))
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            if len(segment_avgs) >= 2:
                total_improvement = ((segment_avgs[-1] - segment_avgs[0]) / segment_avgs[0]) * 100 if segment_avgs[0] > 0 else 0
                ax3.set_title(f'Learning Progress: {total_improvement:.1f}% Improvement', fontsize=14)
        else:
            ax3.text(0.5, 0.5, 'More training data needed\nto display learning progress', 
                    ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'No training data yet', ha='center', va='center', transform=ax3.transAxes)
    
    plt.figtext(0.02, 0.02, f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
               fontsize=8)
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    
    
    if len(scores) % 50 == 0 and len(scores) > 0:
        save_training_plot(fig, scores, mean_scores)
    
    return fig

def save_training_plot(fig, scores, mean_scores):
    """Save the current training plot to file"""
    
    plots_dir = './training_plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_progress_{len(scores)}ep_{timestamp}.png"
    filepath = os.path.join(plots_dir, filename)
    
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filepath}")