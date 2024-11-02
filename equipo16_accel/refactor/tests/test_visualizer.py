# tests/test_visualizer.py

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from refactor.Visualizer import Visualizer



class TestVisualizer(unittest.TestCase):
    @patch('src.visualization.Visualizer.os.makedirs')
    def test_init_creates_output_dir(self, mock_makedirs):
        # Initialize Visualizer
        visualizer = Visualizer()
        
        # Assert that os.makedirs was called with the correct directory
        mock_makedirs.assert_called_once_with(visualizer.output_dir, exist_ok=True)
        
        # Assert that output_dir is set correctly
        self.assertEqual(visualizer.output_dir, "visualizations")

    @patch('src.visualization.Visualizer.plt')
    @patch('src.visualization.Visualizer.pd.DataFrame.hist')
    @patch('src.visualization.Visualizer.os.path.join', return_value='visualizations/histograms.png')
    def test_plot_histograms_success(self, mock_path_join, mock_hist, mock_plt):
        # Create a sample DataFrame
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1]
        })
        columns = ['A', 'B']
        
        # Initialize Visualizer
        visualizer = Visualizer()
        
        # Call plot_histograms
        visualizer.plot_histograms(df, columns)
        
        # Assert that hist was called with correct parameters
        mock_hist.assert_called_once_with(bins=20, figsize=(10, 8))
        
        # Assert that plt.tight_layout was called
        mock_plt.tight_layout.assert_called_once()
        
        # Assert that plt.savefig was called with the correct path
        mock_plt.savefig.assert_called_once_with('visualizations/histograms.png')
        
        # Assert that plt.close was called
        mock_plt.close.assert_called_once()

    @patch('src.visualization.Visualizer.plt')
    @patch('src.visualization.Visualizer.sns.kdeplot')
    @patch('src.visualization.Visualizer.os.path.join', return_value='visualizations/kde_plots.png')
    def test_plot_kde_success_multiple_columns(self, mock_path_join, mock_kdeplot, mock_plt):
        # Create a sample DataFrame
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1]
        })
        columns = ['A', 'B']
        
        # Initialize Visualizer
        visualizer = Visualizer()
        
        # Mock subplot creation
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        # Call plot_kde
        visualizer.plot_kde(df, columns)
        
        # Assert that subplots was called with correct parameters
        mock_plt.subplots.assert_called_once_with(len(columns), 1, figsize=(10, 15))
        
        # Assert that sns.kdeplot was called for each column
        calls = [
            unittest.mock.call(df['A'], ax=mock_axes[0], fill=True),
            unittest.mock.call(df['B'], ax=mock_axes[1], fill=True)
        ]
        mock_kdeplot.assert_has_calls(calls, any_order=False)
        
        # Assert that titles were set correctly
        mock_axes[0].set_title.assert_called_once_with('KDE of A')
        mock_axes[1].set_title.assert_called_once_with('KDE of B')
        
        # Assert that plt.tight_layout was called
        mock_plt.tight_layout.assert_called_once()
        
        # Assert that plt.savefig was called with the correct path
        mock_plt.savefig.assert_called_once_with('visualizations/kde_plots.png')
        
        # Assert that plt.close was called
        mock_plt.close.assert_called_once()
    
    @patch('src.visualization.Visualizer.plt')
    @patch('src.visualization.Visualizer.sns.kdeplot')
    @patch('src.visualization.Visualizer.os.path.join', return_value='visualizations/kde_plots.png')
    def test_plot_kde_success_single_column(self, mock_path_join, mock_kdeplot, mock_plt):
        # Create a sample DataFrame
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5]
        })
        columns = ['A']
        
        # Initialize Visualizer
        visualizer = Visualizer()
        
        # Mock subplot creation
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        # Call plot_kde
        visualizer.plot_kde(df, columns)
        
        # Assert that subplots was called with correct parameters
        mock_plt.subplots.assert_called_once_with(len(columns), 1, figsize=(10, 15))
        
        # Assert that sns.kdeplot was called for the single column
        mock_kdeplot.assert_called_once_with(df['A'], ax=mock_ax, fill=True)
        
        # Assert that title was set correctly
        mock_ax.set_title.assert_called_once_with('KDE of A')
        
        # Assert that plt.tight_layout was called
        mock_plt.tight_layout.assert_called_once()
        
        # Assert that plt.savefig was called with the correct path
        mock_plt.savefig.assert_called_once_with('visualizations/kde_plots.png')
        
        # Assert that plt.close was called
        mock_plt.close.assert_called_once()

    @patch('src.visualization.Visualizer.plt')
    @patch('src.visualization.Visualizer.sns.boxplot')
    @patch('src.visualization.Visualizer.os.path.join')
    def test_plot_boxplots_success_no_hue(self, mock_path_join, mock_boxplot, mock_plt):
        # Create a sample DataFrame
        df = pd.DataFrame({
            'Category': ['A', 'B', 'A', 'B'],
            'Value1': [10, 20, 15, 25],
            'Value2': [5, 3, 6, 2]
        })
        x_col = 'Category'
        y_cols = ['Value1', 'Value2']
        
        # Mock os.path.join to return a predictable path
        mock_path_join.side_effect = lambda x, y: f'{x}/{y}'
        
        # Initialize Visualizer
        visualizer = Visualizer()
        
        # Call plot_boxplots without hue
        visualizer.plot_boxplots(df, x_col, y_cols)
        
        # Assert that boxplot was called for each y_col
        expected_calls = [
            unittest.mock.call(x=x_col, y='Value1', hue=None, data=df),
            unittest.mock.call(x=x_col, y='Value2', hue=None, data=df)
        ]
        mock_boxplot.assert_has_calls(expected_calls, any_order=False)
        
        # Assert that plt.figure was called twice
        self.assertEqual(mock_plt.figure.call_count, 2)
        
        # Assert that plt.tight_layout and plt.savefig were called correctly
        expected_save_calls = [
            unittest.mock.call('/visualizations/boxplot_Value1.png'),
            unittest.mock.call('/visualizations/boxplot_Value2.png')
        ]
        mock_path_join.assert_any_call('visualizations', 'boxplot_Value1.png')
        mock_path_join.assert_any_call('visualizations', 'boxplot_Value2.png')
        mock_plt.tight_layout.assert_called()
        self.assertEqual(mock_plt.savefig.call_count, 2)
        
        # Assert that plt.close was called twice
        self.assertEqual(mock_plt.close.call_count, 2)

    @patch('src.visualization.Visualizer.plt')
    @patch('src.visualization.Visualizer.sns.boxplot')
    @patch('src.visualization.Visualizer.os.path.join')
    def test_plot_boxplots_success_with_hue(self, mock_path_join, mock_boxplot, mock_plt):
        # Create a sample DataFrame
        df = pd.DataFrame({
            'Category': ['A', 'B', 'A', 'B'],
            'Subcategory': ['X', 'X', 'Y', 'Y'],
            'Value1': [10, 20, 15, 25],
            'Value2': [5, 3, 6, 2]
        })
        x_col = 'Category'
        y_cols = ['Value1', 'Value2']
        hue_col = 'Subcategory'
        
        # Mock os.path.join to return a predictable path
        mock_path_join.side_effect = lambda x, y: f'{x}/{y}'
        
        # Initialize Visualizer
        visualizer = Visualizer()
        
        # Call plot_boxplots with hue
        visualizer.plot_boxplots(df, x_col, y_cols, hue_col=hue_col)
        
        # Assert that boxplot was called for each y_col with hue
        expected_calls = [
            unittest.mock.call(x=x_col, y='Value1', hue=hue_col, data=df),
            unittest.mock.call(x=x_col, y='Value2', hue=hue_col, data=df)
        ]
        mock_boxplot.assert_has_calls(expected_calls, any_order=False)
        
        # Assert that plt.figure was called twice
        self.assertEqual(mock_plt.figure.call_count, 2)
        
        # Assert that plt.tight_layout and plt.savefig were called correctly
        expected_save_calls = [
            unittest.mock.call('/visualizations/boxplot_Value1.png'),
            unittest.mock.call('/visualizations/boxplot_Value2.png')
        ]
        mock_path_join.assert_any_call('visualizations', 'boxplot_Value1.png')
        mock_path_join.assert_any_call('visualizations', 'boxplot_Value2.png')
        mock_plt.tight_layout.assert_called()
        self.assertEqual(mock_plt.savefig.call_count, 2)
        
        # Assert that plt.close was called twice
        self.assertEqual(mock_plt.close.call_count, 2)

    @patch('src.visualization.Visualizer.plt')
    @patch('src.visualization.Visualizer.sns.heatmap')
    @patch('src.visualization.Visualizer.os.path.join', return_value='visualizations/correlation_matrix.png')
    def test_plot_correlation_matrix_success(self, mock_path_join, mock_heatmap, mock_plt):
        # Create a sample DataFrame
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': [2, 3, 2, 3, 2]
        })
        columns = ['A', 'B', 'C']
        
        # Initialize Visualizer
        visualizer = Visualizer()
        
        # Call plot_correlation_matrix
        visualizer.plot_correlation_matrix(df, columns)
        
        # Assert that df.corr() was called correctly
        expected_corr = df[columns].corr()
        
        # Assert that sns.heatmap was called with the correlation matrix
        mock_heatmap.assert_called_once_with(expected_corr, annot=True, cmap='coolwarm', linewidths=0.5)
        
        # Assert that plt.title was set correctly
        mock_plt.title.assert_called_once_with('Correlation Matrix')
        
        # Assert that plt.tight_layout was called
        mock_plt.tight_layout.assert_called_once()
        
        # Assert that plt.savefig was called with the correct path
        mock_plt.savefig.assert_called_once_with('visualizations/correlation_matrix.png')
        
        # Assert that plt.close was called
        mock_plt.close.assert_called_once()

    @patch('src.visualization.Visualizer.plt')
    @patch('src.visualization.Visualizer.sns.scatterplot')
    @patch('src.visualization.Visualizer.os.path.join', return_value='visualizations/vibration_vs_rpm.png')
    def test_plot_vibration_vs_rpm_success(self, mock_path_join, mock_scatterplot, mock_plt):
        # Create a sample DataFrame
        df = pd.DataFrame({
            'pctid': [100, 80, 60, 40, 20],
            'vibration_magnitude': [10, 15, 20, 25, 30],
            'configuración': ['Normal', 'Perpendicular', 'Opuesto', 'Normal', 'Perpendicular']
        })
        x_col = 'pctid'
        y_col = 'vibration_magnitude'
        hue_col = 'configuración'
        
        # Initialize Visualizer
        visualizer = Visualizer()
        
        # Call plot_vibration_vs_rpm
        visualizer.plot_vibration_vs_rpm(df, x_col, y_col, hue_col)
        
        # Assert that sns.scatterplot was called with correct parameters
        mock_scatterplot.assert_called_once_with(x=x_col, y=y_col, hue=hue_col, palette='magma', data=df)
        
        # Assert that plt.title, plt.xlabel, plt.ylabel were set correctly
        mock_plt.title.assert_called_once_with('Vibration Magnitude vs RPM by Configuration')
        mock_plt.xlabel.assert_called_once_with('% de RPM, 100 indica 100%')
        mock_plt.ylabel.assert_called_once_with('Vibración Total')
        
        # Assert that plt.tight_layout was called
        mock_plt.tight_layout.assert_called_once()
        
        # Assert that plt.savefig was called with the correct path
        mock_plt.savefig.assert_called_once_with('visualizations/vibration_vs_rpm.png')
        
        # Assert that plt.close was called
        mock_plt.close.assert_called_once()

    @patch.object(Visualizer, 'plot_correlation_matrix')
    @patch('src.visualization.Visualizer.pd.DataFrame.groupby')
    @patch('src.visualization.Visualizer.pd.DataFrame.describe', return_value=pd.DataFrame({
        'Metric1': {'count': 2, 'mean': 15.0, 'std': 7.07},
        'Metric2': {'count': 2, 'mean': 10.0, 'std': 7.07}
    }))
    @patch('src.visualization.Visualizer.os.path.join', return_value='visualizations/summary_statistics.csv')
    @patch('src.visualization.Visualizer.pd.DataFrame.to_csv')
    @patch('builtins.print')
    def test_display_summary_statistics_success(
        self,
        mock_print,
        mock_to_csv,
        mock_path_join,
        mock_describe,
        mock_groupby,
        mock_plot_corr
    ):
        # Create a sample DataFrame
        df = pd.DataFrame({
            'Group': ['A', 'A', 'B', 'B'],
            'Metric1': [10, 20, 30, 40],
            'Metric2': [5, 15, 25, 35]
        })
        group_by_cols = ['Group']
        target_cols = ['Metric1', 'Metric2']
        
        # Mock groupby
        mock_group = MagicMock()
        mock_groupby.return_value = mock_group
        
        # Initialize Visualizer
        visualizer = Visualizer()
        
        # Call display_summary_statistics
        visualizer.display_summary_statistics(df, group_by_cols, target_cols)
        
        # Assert that groupby was called with correct columns
        mock_groupby.assert_called_once_with(group_by_cols)
        
        # Assert that describe was called on the groupby object
        mock_group.describe.assert_called_once()
        
        # Assert that to_csv was called with correct path
        mock_to_csv.assert_called_once_with('visualizations/summary_statistics.csv')
        
        # Assert that print was called with summary and save message
        expected_summary = mock_describe.return_value
        mock_print.assert_any_call(expected_summary)
        mock_print.assert_any_call("Summary statistics saved to summary_statistics.csv")
        
        # Assert that plot_correlation_matrix was called with numeric columns
        mock_plot_corr.assert_called_once_with(df, target_cols)


if __name__ == '__main__':
    unittest.main()
