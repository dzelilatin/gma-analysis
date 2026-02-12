import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- FINAL DATA ---
inference_times = {
    'Mac (Stress Test)': 222.44, 
    'HPC (Standard)': 35.48,   
    'HPC (Scaled)': 38.63       
}

training_times = {
    'HPC Training': 266.77
}

# --- PLOT 1: INFERENCE COMPARISON ---
def plot_inference():
    plt.figure(figsize=(10, 6))
    names = list(inference_times.keys())
    times = list(inference_times.values())
    
    colors = ['#e74c3c', '#3498db', '#2ecc71'] # Red, Blue, Green
    bars = plt.bar(names, times, color=colors)
    
    plt.title('Inference Performance: Mac vs. HPC (2,147 Images)', fontsize=14, fontweight='bold')
    plt.ylabel('Execution Time (Seconds)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Label the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}s', ha='center', fontweight='bold')

    plt.savefig('results/hpc_vs_mac_inference.png')
    print("✅ Created hpc_vs_mac_inference.png")

# --- PLOT 2: TRAINING CURVES ---
def plot_history():
    try:
        df = pd.read_csv('results/history_full.csv')
        plt.figure(figsize=(12, 5))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(df['accuracy'], label='Training', color='#3498db', linewidth=2)
        plt.plot(df['val_accuracy'], label='Validation', color='#e67e22', linewidth=2)
        plt.title('Model Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(df['loss'], label='Training', color='#3498db', linewidth=2)
        plt.plot(df['val_loss'], label='Validation', color='#e67e22', linewidth=2)
        plt.title('Model Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/training_curves.png')
        print("✅ Created training_curves.png")
    except:
        print("⚠️ history_full.csv not found. Move it to this folder to generate curves!")

if __name__ == "__main__":
    plot_inference()
    plot_history()