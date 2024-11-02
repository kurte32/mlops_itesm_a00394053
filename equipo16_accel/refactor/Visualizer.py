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

    def plot_boxplots(self, df, x_col, y_cols, hue_col=None):
            for y_col in y_cols:
                sns.boxplot(x=x_col, y=y_col, hue=hue_col, data=df)
                plt.title(f'Boxplot of {y_col} by {x_col}')
                plt.tight_layout()
                plt.show()

    def plot_correlation_matrix(self, df, columns):
        correlation_matrix = df[columns].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='magma', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

    def plot_vibration_vs_rpm(self, df, x_col, y_col, hue_col):
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=x_col, y=y_col, hue=hue_col, palette='magma', data=df)
        plt.title('Vibration Magnitude vs RPM by Configuration')
        plt.xlabel('% de RPM, 100 indica 100%')
        plt.ylabel('Vibración Total')
        plt.tight_layout()
        plt.show()


    def display_summary_statistics(self, df, group_by_cols, target_cols):
        summary_stats = df.groupby(group_by_cols)[target_cols].describe()
        print(summary_stats)
        # Optionally, save to a file
        summary_stats.to_csv("summary_statistics.csv")
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title('Heatmap of Correlation Matrix')
        plt.tight_layout()
        plt.show()
