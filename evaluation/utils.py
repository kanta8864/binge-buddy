import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def plot_and_save_trust_scores(df, save_path):
    """
    Plots and saves three visualizations of user trust scores:
    - Histogram
    - Boxplot by group
    - Violin plot by group
    
    Parameters:
    - df (pd.DataFrame): Dataframe with 'mean_trust_score' and 'group' columns.
    - save_path (Path): Directory to save the plots.
    
    Saves:
    - trust_histogram.png
    - trust_boxplot.png
    - trust_violinplot.png
    """
    save_path = Path(save_path)  # Ensure it's a Path object
    save_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    colors = {"Semantic Memory": "skyblue", "Episodic Memory": "salmon"}

    # # --- Histogram ---
    # plt.figure(figsize=(8, 5))
    # sns.histplot(df, x='mean_trust_score', bins=10, kde=True, hue="group", 
    #              palette=colors, alpha=0.7)  
    # plt.xlabel("Mean Trust Score")
    # plt.ylabel("Number of Users")
    # plt.title("Distribution of User Trust Scores")
    # plt.legend(title="Group")
    # plt.grid(True, linestyle="--", alpha=0.5)
    # plt.savefig(save_path / "trust_histogram.png", dpi=300)
    # plt.show()
    # plt.close()
    
    # --- Boxplot by Group ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="group", y="mean_trust_score", data=df, hue="group",
                palette=colors, fliersize=4, linewidth=1.2, width=0.6, boxprops=dict(alpha=0.7))
    plt.xlabel("Group")
    plt.ylabel("Mean Trust Score")
    plt.title("User Trust Scores by Group")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(save_path / "trust_boxplot.png", dpi=300)
    plt.show()
    plt.close()
    
    # --- Violin Plot by Group ---
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="group", y="mean_trust_score", data=df, hue="group",
                   palette=colors, inner="quartile", alpha=0.7)
    plt.xlabel("Group")
    plt.ylabel("Mean Trust Score")
    plt.title("Trust Score Distribution by Group")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(save_path / "trust_violinplot.png", dpi=300)
    plt.show()
    plt.close()
    
    print(f"Plots saved successfully in: {save_path}")
