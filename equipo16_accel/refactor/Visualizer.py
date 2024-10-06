# src/visualization/visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self):
        pass

    def plot_histograms(self, df, columns):
        df[columns].hist(bins=20)
        plt.tight_layout()
        plt.show()

    def plot_kde(self, df, columns):
        fig, axes = plt.subplots(len(columns), 1, figsize=(10, 15))
        for idx, col in enumerate(columns):
            sns.kdeplot(df[col], ax=axes[idx], fill=True)
            axes[idx].set_title(f'KDE of {col}')
        plt.tight_layout()
        plt.show()

    def plot_time_series(self, df, x_col, y_cols, hue_col=None):
        for y_col in y_cols:
            sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col)
            plt.title(f'Time Series of {y_col}')
            plt.show()
