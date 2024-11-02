# Visualizer.py

import os
import matplotlib
# Use a non-interactive backend to prevent display warnings
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List


class Visualizer:
    def __init__(self):
        # Define the output directory
        self.output_dir = "visualizations"
        # Create the directory if it does not exist
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_histograms(self, df: pd.DataFrame, columns: List[str]) -> None:
        """
        Plots histograms for the specified columns and saves the plot as an image.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            columns (List[str]): The list of columns to plot histograms for.
        """
        try:
            df[columns].hist(bins=20, figsize=(10, 8))
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, "histograms.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Histograms saved to {save_path}")
        except Exception as e:
            print(f"Error in plot_histograms: {e}")

    def plot_kde(self, df: pd.DataFrame, columns: List[str]) -> None:
        """
        Plots Kernel Density Estimation (KDE) plots for the specified columns and saves the plot as an image.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            columns (List[str]): The list of columns to plot KDEs for.
        """
        try:
            fig, axes = plt.subplots(len(columns), 1, figsize=(10, 15))
            if len(columns) == 1:
                axes = [axes]
            for idx, col in enumerate(columns):
                sns.kdeplot(df[col], ax=axes[idx], fill=True)
                axes[idx].set_title(f'KDE of {col}')
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, "kde_plots.png")
            plt.savefig(save_path)
            plt.close()
            print(f"KDE plots saved to {save_path}")
        except Exception as e:
            print(f"Error in plot_kde: {e}")

    def plot_boxplots(self, df: pd.DataFrame, x_col: str, y_cols: List[str], hue_col: str = None) -> None:
        """
        Plots boxplots for the specified y-columns grouped by the x-column and saves each plot as an image.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            x_col (str): The column name to group by on the x-axis.
            y_cols (List[str]): The list of columns to plot boxplots for.
            hue_col (str, optional): The column name to use for color encoding. Defaults to None.
        """
        try:
            for y_col in y_cols:
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=x_col, y=y_col, hue=hue_col, data=df)
                plt.title(f'Boxplot of {y_col} by {x_col}')
                plt.tight_layout()
                filename = f'boxplot_{y_col}.png'
                save_path = os.path.join(self.output_dir, filename)
                plt.savefig(save_path)
                plt.close()
                print(f"Boxplot for {y_col} saved to {save_path}")
        except Exception as e:
            print(f"Error in plot_boxplots: {e}")

    def plot_correlation_matrix(self, df: pd.DataFrame, columns: List[str]) -> None:
        """
        Plots a correlation matrix heatmap for the specified numeric columns and saves the plot as an image.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            columns (List[str]): The list of numeric columns to include in the correlation matrix.
        """
        try:
            correlation_matrix = df[columns].corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, "correlation_matrix.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Correlation matrix saved to {save_path}")
        except Exception as e:
            print(f"Error in plot_correlation_matrix: {e}")

    def plot_vibration_vs_rpm(self, df: pd.DataFrame, x_col: str, y_col: str, hue_col: str) -> None:
        """
        Plots a scatter plot of vibration magnitude vs RPM, colored by configuration, and saves the plot as an image.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            x_col (str): The column name for the x-axis (e.g., 'pctid').
            y_col (str): The column name for the y-axis (e.g., 'vibration_magnitude').
            hue_col (str): The column name for color encoding (e.g., 'configuración').
        """
        try:
            plt.figure(figsize=(12, 6))
            sns.scatterplot(x=x_col, y=y_col, hue=hue_col, palette='magma', data=df)
            plt.title('Vibration Magnitude vs RPM by Configuration')
            plt.xlabel('% de RPM, 100 indica 100%')
            plt.ylabel('Vibración Total')
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, "vibration_vs_rpm.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Vibration vs RPM plot saved to {save_path}")
        except Exception as e:
            print(f"Error in plot_vibration_vs_rpm: {e}")

    def display_summary_statistics(self, df: pd.DataFrame, group_by_cols: List[str], target_cols: List[str]) -> None:
        """
        Computes and displays summary statistics grouped by specified columns.
        Also plots and saves the correlation matrix for the target columns.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            group_by_cols (List[str]): The list of columns to group by.
            target_cols (List[str]): The list of numeric columns to compute statistics for.
        """
        try:
            # Compute summary statistics grouped by specified columns
            summary_stats = df.groupby(group_by_cols)[target_cols].describe()
            print(summary_stats)
            # Save summary statistics to a CSV file
            csv_filename = "summary_statistics.csv"
            csv_path = os.path.join(self.output_dir, csv_filename)
            summary_stats.to_csv(csv_path)
            print(f"Summary statistics saved to {csv_path}")

            # Compute and plot the correlation matrix for numeric columns only
            numeric_cols = df[target_cols].select_dtypes(include=['float64', 'int64']).columns.tolist()
            self.plot_correlation_matrix(df, numeric_cols)
        except Exception as e:
            print(f"Error in display_summary_statistics: {e}")
