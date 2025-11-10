"""
Unit tests for visualization components.
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import tempfile
import os

from src.visualization.chart_generator import ChartGenerator
from src.visualization.heatmap_generator import HeatmapGenerator
from src.visualization.confidence_interval_plotter import ConfidenceIntervalPlotter
from src.visualization.dashboard_builder import DashboardBuilder
from src.interfaces import VisualizationError, ReportData, AnalysisResult


class TestChartGenerator:
    """Test cases for ChartGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ChartGenerator()
        
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'occupation': ['Manager', 'Engineer', 'Nurse', 'Teacher', 'Analyst'],
            'value': [75.2, 68.5, 82.1, 71.3, 79.8],
            'error': [3.2, 2.8, 4.1, 3.5, 2.9],
            'category': ['Management', 'Technical', 'Healthcare', 'Education', 'Technical'],
            'size': [100, 85, 120, 95, 88]
        })
    
    def test_create_bar_chart_basic(self):
        """Test basic bar chart creation."""
        fig = self.generator.create_bar_chart(
            self.sample_data, 'occupation', 'value', 'Test Bar Chart'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == 'bar'
        assert len(fig.data[0].x) == 5
        assert fig.layout.title.text == 'Test Bar Chart'
    
    def test_create_bar_chart_with_colors(self):
        """Test bar chart with color coding."""
        fig = self.generator.create_bar_chart(
            self.sample_data, 'occupation', 'value', 'Colored Bar Chart',
            color_col='category'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # Should have multiple traces for different categories
    
    def test_create_bar_chart_with_errors(self):
        """Test bar chart with error bars."""
        fig = self.generator.create_bar_chart(
            self.sample_data, 'occupation', 'value', 'Bar Chart with Errors',
            error_col='error'
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.data[0].error_y is not None
        assert fig.data[0].error_y.visible == True
    
    def test_create_bar_chart_horizontal(self):
        """Test horizontal bar chart creation."""
        fig = self.generator.create_bar_chart(
            self.sample_data, 'occupation', 'value', 'Horizontal Bar Chart',
            orientation='h'
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.data[0].orientation == 'h'
    
    def test_create_pie_chart(self):
        """Test pie chart creation."""
        fig = self.generator.create_pie_chart(
            self.sample_data, 'value', 'occupation', 'Test Pie Chart'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == 'pie'
        assert len(fig.data[0].labels) == 5
        assert fig.layout.title.text == 'Test Pie Chart'
    
    def test_create_scatter_plot_basic(self):
        """Test basic scatter plot creation."""
        fig = self.generator.create_scatter_plot(
            self.sample_data, 'value', 'error', 'Test Scatter Plot'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].mode == 'markers'
        assert len(fig.data[0].x) == 5
    
    def test_create_scatter_plot_with_colors(self):
        """Test scatter plot with color coding."""
        fig = self.generator.create_scatter_plot(
            self.sample_data, 'value', 'error', 'Colored Scatter Plot',
            color_col='category'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # Multiple traces for different categories
    
    def test_create_scatter_plot_with_size(self):
        """Test scatter plot with size coding."""
        fig = self.generator.create_scatter_plot(
            self.sample_data, 'value', 'error', 'Sized Scatter Plot',
            size_col='size'
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.data[0].marker.size is not None
    
    def test_create_scatter_plot_with_trendline(self):
        """Test scatter plot with trendline."""
        fig = self.generator.create_scatter_plot(
            self.sample_data, 'value', 'error', 'Scatter with Trendline',
            trendline=True
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Data points + trendline
        assert fig.data[1].name == 'Trendline'
    
    def test_create_line_chart(self):
        """Test line chart creation."""
        fig = self.generator.create_line_chart(
            self.sample_data, 'occupation', 'value', 'Test Line Chart'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert 'lines' in fig.data[0].mode
        assert len(fig.data[0].x) == 5
    
    def test_create_line_chart_with_groups(self):
        """Test line chart with grouping."""
        fig = self.generator.create_line_chart(
            self.sample_data, 'occupation', 'value', 'Grouped Line Chart',
            group_col='category'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # Multiple lines for different categories
    
    def test_create_multi_chart(self):
        """Test multi-chart creation."""
        data_list = [self.sample_data, self.sample_data]
        chart_configs = [
            {'type': 'bar', 'x_col': 'occupation', 'y_col': 'value', 'name': 'Chart 1'},
            {'type': 'scatter', 'x_col': 'value', 'y_col': 'error', 'name': 'Chart 2'}
        ]
        subplot_titles = ['Bar Chart', 'Scatter Plot']
        
        fig = self.generator.create_multi_chart(
            data_list, chart_configs, 'Multi Chart Test', subplot_titles, rows=1, cols=2
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        assert fig.data[0].type == 'bar'
        assert fig.data[1].type == 'scatter'
    
    def test_add_confidence_intervals(self):
        """Test adding confidence intervals to existing figure."""
        fig = self.generator.create_line_chart(
            self.sample_data, 'occupation', 'value', 'Line with CI'
        )
        
        fig_with_ci = self.generator.add_confidence_intervals(
            fig, self.sample_data, 'occupation', 'value', 'error'
        )
        
        assert isinstance(fig_with_ci, go.Figure)
        assert len(fig_with_ci.data) >= 3  # Original line + confidence interval traces
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(VisualizationError):
            self.generator.create_bar_chart(empty_data, 'x', 'y', 'Test')
    
    def test_missing_columns_handling(self):
        """Test handling of missing columns."""
        with pytest.raises(VisualizationError):
            self.generator.create_bar_chart(
                self.sample_data, 'nonexistent_col', 'value', 'Test'
            )


class TestHeatmapGenerator:
    """Test cases for HeatmapGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = HeatmapGenerator()
        
        # Sample correlation matrix
        self.correlation_data = pd.DataFrame({
            'Physical': [1.0, 0.3, -0.2, 0.5],
            'Cognitive': [0.3, 1.0, 0.7, -0.1],
            'Environmental': [-0.2, 0.7, 1.0, 0.4],
            'Social': [0.5, -0.1, 0.4, 1.0]
        }, index=['Physical', 'Cognitive', 'Environmental', 'Social'])
        
        # Sample risk data
        self.risk_data = pd.DataFrame({
            'occupation': ['Manager', 'Engineer', 'Nurse'] * 3,
            'risk_factor': ['Heat', 'Heat', 'Heat', 'Chemical', 'Chemical', 'Chemical', 'Height', 'Height', 'Height'],
            'risk_value': [0.2, 0.1, 0.8, 0.3, 0.6, 0.9, 0.1, 0.4, 0.3]
        })
        
        # Sample precision data
        self.precision_data = pd.DataFrame({
            'occupation': ['Manager', 'Engineer', 'Nurse'] * 2,
            'requirement': ['Physical', 'Physical', 'Physical', 'Cognitive', 'Cognitive', 'Cognitive'],
            'precision': [0.8, 0.9, 0.6, 0.7, 0.85, 0.75]
        })
    
    def test_create_correlation_heatmap(self):
        """Test correlation heatmap creation."""
        fig = self.generator.create_correlation_heatmap(
            self.correlation_data, 'Test Correlation Heatmap'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == 'heatmap'
        assert fig.data[0].z.shape == (4, 4)
        assert fig.layout.title.text == 'Test Correlation Heatmap'
    
    def test_create_correlation_heatmap_with_significance(self):
        """Test correlation heatmap with significance indicators."""
        # Create mock significance matrix
        significance_matrix = pd.DataFrame({
            'Physical': [0.0, 0.01, 0.1, 0.001],
            'Cognitive': [0.01, 0.0, 0.001, 0.2],
            'Environmental': [0.1, 0.001, 0.0, 0.05],
            'Social': [0.001, 0.2, 0.05, 0.0]
        }, index=['Physical', 'Cognitive', 'Environmental', 'Social'])
        
        fig = self.generator.create_correlation_heatmap(
            self.correlation_data, 'Correlation with Significance',
            significance_matrix=significance_matrix
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) > 0  # Should have significance annotations
    
    def test_create_risk_heatmap(self):
        """Test risk heatmap creation."""
        fig = self.generator.create_risk_heatmap(
            self.risk_data, 'occupation', 'risk_factor', 'risk_value',
            'Test Risk Heatmap'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == 'heatmap'
        assert fig.layout.title.text == 'Test Risk Heatmap'
    
    def test_create_precision_heatmap(self):
        """Test precision heatmap creation."""
        fig = self.generator.create_precision_heatmap(
            self.precision_data, 'occupation', 'requirement', 'precision',
            'Test Precision Heatmap'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == 'heatmap'
        assert len(fig.layout.annotations) > 0  # Should have precision annotations
    
    def test_create_interactive_heatmap(self):
        """Test interactive heatmap creation."""
        fig = self.generator.create_interactive_heatmap(
            self.risk_data, 'Interactive Risk Heatmap',
            'risk_factor', 'occupation', 'risk_value'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == 'heatmap'
    
    def test_clustered_heatmap_fallback(self):
        """Test clustered heatmap with fallback when scipy unavailable."""
        # This should fall back to regular heatmap if scipy is not available
        fig = self.generator.create_clustered_heatmap(
            self.correlation_data, 'ward', 'Clustered Heatmap'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == 'heatmap'
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(VisualizationError):
            self.generator.create_risk_heatmap(empty_data, 'x', 'y', 'z', 'Test')


class TestConfidenceIntervalPlotter:
    """Test cases for ConfidenceIntervalPlotter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plotter = ConfidenceIntervalPlotter()
        
        # Sample data with estimates and errors
        self.sample_data = pd.DataFrame({
            'occupation': ['Manager', 'Engineer', 'Nurse', 'Teacher', 'Analyst'],
            'estimate': [75.2, 68.5, 82.1, 71.3, 79.8],
            'std_error': [3.2, 2.8, 4.1, 3.5, 2.9],
            'category': ['Management', 'Technical', 'Healthcare', 'Education', 'Technical']
        })
        
        # Sample data with confidence intervals
        self.ci_data = pd.DataFrame({
            'x_value': [1, 2, 3, 4, 5],
            'estimate': [10.5, 12.3, 15.1, 13.8, 16.2],
            'lower_ci': [8.2, 10.1, 12.8, 11.5, 14.0],
            'upper_ci': [12.8, 14.5, 17.4, 16.1, 18.4]
        })
    
    def test_create_error_bar_plot(self):
        """Test error bar plot creation."""
        fig = self.plotter.create_error_bar_plot(
            self.sample_data, 'occupation', 'estimate', 'std_error',
            'Test Error Bar Plot'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert fig.data[0].error_y is not None
        assert fig.data[0].error_y.visible == True
        assert fig.layout.title.text == 'Test Error Bar Plot'
    
    def test_create_error_bar_plot_with_precision_colors(self):
        """Test error bar plot with precision-based coloring."""
        fig = self.plotter.create_error_bar_plot(
            self.sample_data, 'occupation', 'estimate', 'std_error',
            'Precision Colored Plot', color_by_precision=True
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 4  # Main trace + precision legend traces
    
    def test_create_confidence_band_plot(self):
        """Test confidence band plot creation."""
        fig = self.plotter.create_confidence_band_plot(
            self.ci_data, 'x_value', 'estimate', 'lower_ci', 'upper_ci',
            'Test Confidence Band Plot'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # Upper bound + lower bound + main line (+ optional points)
        assert fig.layout.title.text == 'Test Confidence Band Plot'
    
    def test_create_confidence_band_plot_with_points(self):
        """Test confidence band plot with individual points."""
        fig = self.plotter.create_confidence_band_plot(
            self.ci_data, 'x_value', 'estimate', 'lower_ci', 'upper_ci',
            'Band Plot with Points', show_points=True
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4  # Band + line + points
    
    def test_create_precision_dashboard(self):
        """Test precision dashboard creation."""
        fig = self.plotter.create_precision_dashboard(
            self.sample_data, 'estimate', 'std_error', 'category',
            'Test Precision Dashboard'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4  # Four subplots
        assert fig.layout.title.text == 'Test Precision Dashboard'
    
    def test_create_uncertainty_comparison(self):
        """Test uncertainty comparison plot."""
        data_list = [self.sample_data, self.sample_data.copy()]
        labels = ['Dataset 1', 'Dataset 2']
        
        fig = self.plotter.create_uncertainty_comparison(
            data_list, labels, 'estimate', 'std_error',
            'Uncertainty Comparison'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Two datasets
        assert fig.layout.title.text == 'Uncertainty Comparison'
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(VisualizationError):
            self.plotter.create_error_bar_plot(
                empty_data, 'x', 'y', 'error', 'Test'
            )


class TestDashboardBuilder:
    """Test cases for DashboardBuilder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = DashboardBuilder()
        
        # Sample data for dashboard panels
        self.sample_data = pd.DataFrame({
            'occupation': ['Manager', 'Engineer', 'Nurse', 'Teacher', 'Analyst'],
            'value': [75.2, 68.5, 82.1, 71.3, 79.8],
            'error': [3.2, 2.8, 4.1, 3.5, 2.9],
            'category': ['Management', 'Technical', 'Healthcare', 'Education', 'Technical'],
            'reliability': [0.85, 0.92, 0.78, 0.88, 0.91]
        })
        
        # Sample ReportData for interactive dashboard
        self.report_data = ReportData(
            title="Test Report",
            description="Test report description",
            analysis_results=[
                AnalysisResult(
                    occupation_category="Management",
                    metric_name="Physical Demands",
                    value=75.2,
                    confidence_interval=(72.0, 78.4),
                    reliability_score=0.85,
                    footnote_context=["Test footnote"]
                ),
                AnalysisResult(
                    occupation_category="Technical",
                    metric_name="Cognitive Requirements",
                    value=68.5,
                    confidence_interval=(65.7, 71.3),
                    reliability_score=0.92,
                    footnote_context=[]
                )
            ],
            visualizations=[],
            metadata={"test": "data"},
            generation_timestamp=datetime.now()
        )
    
    def test_create_multi_panel_dashboard_grid(self):
        """Test multi-panel dashboard with grid layout."""
        panels = [
            {
                'type': 'bar',
                'title': 'Bar Chart',
                'data': self.sample_data,
                'config': {'x_col': 'occupation', 'y_col': 'value'}
            },
            {
                'type': 'scatter',
                'title': 'Scatter Plot',
                'data': self.sample_data,
                'config': {'x_col': 'value', 'y_col': 'error'}
            }
        ]
        
        fig = self.builder.create_multi_panel_dashboard(
            panels, 'Test Multi-Panel Dashboard', layout='grid', cols=2
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        assert fig.layout.title.text == 'Test Multi-Panel Dashboard'
    
    def test_create_multi_panel_dashboard_vertical(self):
        """Test multi-panel dashboard with vertical layout."""
        panels = [
            {
                'type': 'bar',
                'title': 'Bar Chart',
                'data': self.sample_data,
                'config': {'x_col': 'occupation', 'y_col': 'value'}
            },
            {
                'type': 'line',
                'title': 'Line Chart',
                'data': self.sample_data,
                'config': {'x_col': 'occupation', 'y_col': 'value'}
            }
        ]
        
        fig = self.builder.create_multi_panel_dashboard(
            panels, 'Vertical Dashboard', layout='vertical'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        assert fig.layout.height == 800  # 2 rows * 400 height
    
    def test_create_interactive_dashboard(self):
        """Test interactive dashboard creation from ReportData."""
        fig = self.builder.create_interactive_dashboard(self.report_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Should have multiple panels
        assert "Interactive Dashboard" in fig.layout.title.text
    
    def test_create_drill_down_dashboard(self):
        """Test drill-down dashboard creation."""
        hierarchy_data = pd.DataFrame({
            'level1': ['A', 'A', 'B', 'B', 'C'],
            'level2': ['A1', 'A2', 'B1', 'B2', 'C1'],
            'value': [10, 15, 20, 25, 30]
        })
        
        fig = self.builder.create_drill_down_dashboard(
            hierarchy_data, ['level1', 'level2'], 'value',
            'Drill-Down Dashboard'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # Top level view
        assert fig.layout.title.text == 'Drill-Down Dashboard'
    
    def test_create_responsive_dashboard(self):
        """Test responsive dashboard creation."""
        panels = [
            {
                'type': 'bar',
                'title': 'Responsive Chart',
                'data': self.sample_data,
                'config': {'x_col': 'occupation', 'y_col': 'value'}
            }
        ]
        
        fig = self.builder.create_responsive_dashboard(
            panels, 'Responsive Dashboard'
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.autosize == True
        assert 'responsive_config' in fig.layout.meta
    
    def test_create_filtered_dashboard(self):
        """Test filtered dashboard creation."""
        chart_configs = [
            {
                'type': 'bar',
                'title': 'Filtered Bar Chart',
                'x_col': 'occupation',
                'y_col': 'value'
            }
        ]
        
        fig = self.builder.create_filtered_dashboard(
            self.sample_data, ['category'], chart_configs,
            'Filtered Dashboard'
        )
        
        assert isinstance(fig, go.Figure)
        assert 'filter_options' in fig.layout.meta
        assert len(fig.layout.meta['filter_options']['category']) > 0
    
    def test_export_dashboard_html(self):
        """Test dashboard HTML export."""
        panels = [
            {
                'type': 'bar',
                'title': 'Export Test',
                'data': self.sample_data,
                'config': {'x_col': 'occupation', 'y_col': 'value'}
            }
        ]
        
        fig = self.builder.create_multi_panel_dashboard(
            panels, 'Export Test Dashboard'
        )
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            result = self.builder.export_dashboard_html(fig, temp_path)
            assert result == True
            assert os.path.exists(temp_path)
            
            # Check file content
            with open(temp_path, 'r') as f:
                content = f.read()
                assert 'Export Test Dashboard' in content
                assert 'plotly' in content.lower()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_invalid_layout_handling(self):
        """Test handling of invalid layout types."""
        panels = [{'type': 'bar', 'data': self.sample_data, 'config': {}}]
        
        with pytest.raises(VisualizationError):
            self.builder.create_multi_panel_dashboard(
                panels, 'Test', layout='invalid_layout'
            )
    
    def test_empty_report_data_handling(self):
        """Test handling of empty report data."""
        empty_report = ReportData(
            title="Empty Report",
            description="Empty",
            analysis_results=[],
            visualizations=[],
            metadata={},
            generation_timestamp=datetime.now()
        )
        
        with pytest.raises(VisualizationError):
            self.builder.create_interactive_dashboard(empty_report)


# Integration tests
class TestVisualizationIntegration:
    """Integration tests for visualization components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chart_gen = ChartGenerator()
        self.heatmap_gen = HeatmapGenerator()
        self.ci_plotter = ConfidenceIntervalPlotter()
        self.dashboard_builder = DashboardBuilder()
        
        # Comprehensive test data
        self.test_data = pd.DataFrame({
            'occupation': ['Manager', 'Engineer', 'Nurse', 'Teacher', 'Analyst'] * 3,
            'requirement': ['Physical', 'Physical', 'Physical', 'Physical', 'Physical',
                          'Cognitive', 'Cognitive', 'Cognitive', 'Cognitive', 'Cognitive',
                          'Environmental', 'Environmental', 'Environmental', 'Environmental', 'Environmental'],
            'estimate': [75.2, 68.5, 82.1, 71.3, 79.8, 85.1, 92.3, 78.6, 88.2, 91.5,
                        45.3, 52.1, 38.9, 41.7, 48.2],
            'std_error': [3.2, 2.8, 4.1, 3.5, 2.9, 2.1, 1.8, 3.8, 2.4, 1.9,
                         4.5, 3.9, 5.2, 4.8, 4.1],
            'category': ['Management', 'Technical', 'Healthcare', 'Education', 'Technical'] * 3,
            'reliability': [0.85, 0.92, 0.78, 0.88, 0.91, 0.94, 0.96, 0.82, 0.89, 0.95,
                          0.75, 0.81, 0.68, 0.73, 0.79]
        })
    
    def test_complete_visualization_pipeline(self):
        """Test complete visualization pipeline with multiple chart types."""
        # 1. Create basic charts
        bar_chart = self.chart_gen.create_bar_chart(
            self.test_data[self.test_data['requirement'] == 'Physical'],
            'occupation', 'estimate', 'Physical Requirements by Occupation',
            error_col='std_error'
        )
        assert isinstance(bar_chart, go.Figure)
        
        # 2. Create correlation heatmap
        correlation_data = self.test_data.pivot_table(
            index='occupation', columns='requirement', values='estimate'
        ).corr()
        
        heatmap = self.heatmap_gen.create_correlation_heatmap(
            correlation_data, 'Requirement Correlations'
        )
        assert isinstance(heatmap, go.Figure)
        
        # 3. Create confidence interval plot
        ci_plot = self.ci_plotter.create_error_bar_plot(
            self.test_data[self.test_data['requirement'] == 'Cognitive'],
            'occupation', 'estimate', 'std_error',
            'Cognitive Requirements with Confidence Intervals'
        )
        assert isinstance(ci_plot, go.Figure)
        
        # 4. Create comprehensive dashboard
        panels = [
            {
                'type': 'bar',
                'title': 'Physical Requirements',
                'data': self.test_data[self.test_data['requirement'] == 'Physical'],
                'config': {'x_col': 'occupation', 'y_col': 'estimate'}
            },
            {
                'type': 'scatter',
                'title': 'Reliability vs Estimate',
                'data': self.test_data,
                'config': {'x_col': 'estimate', 'y_col': 'reliability'}
            },
            {
                'type': 'heatmap',
                'title': 'Requirements Heatmap',
                'data': self.test_data,
                'config': {'x_col': 'occupation', 'y_col': 'requirement', 'z_col': 'estimate'}
            }
        ]
        
        dashboard = self.dashboard_builder.create_multi_panel_dashboard(
            panels, 'Comprehensive Occupation Analysis Dashboard',
            layout='grid', cols=2
        )
        assert isinstance(dashboard, go.Figure)
        assert len(dashboard.data) == 3
    
    def test_visualization_with_real_world_data_patterns(self):
        """Test visualizations with realistic data patterns and edge cases."""
        # Create data with missing values and outliers
        realistic_data = self.test_data.copy()
        realistic_data.loc[0, 'estimate'] = np.nan  # Missing value
        realistic_data.loc[1, 'estimate'] = 150.0   # Outlier
        realistic_data.loc[2, 'std_error'] = 0.0    # Zero error
        
        # Test handling of missing data
        bar_chart = self.chart_gen.create_bar_chart(
            realistic_data[realistic_data['requirement'] == 'Physical'].dropna(),
            'occupation', 'estimate', 'Chart with Missing Data Handled'
        )
        assert isinstance(bar_chart, go.Figure)
        
        # Test precision dashboard with edge cases
        precision_dashboard = self.ci_plotter.create_precision_dashboard(
            realistic_data.dropna(), 'estimate', 'std_error', 'category',
            'Precision Dashboard with Edge Cases'
        )
        assert isinstance(precision_dashboard, go.Figure)
    
    def test_dashboard_export_and_styling_consistency(self):
        """Test dashboard export and consistent styling across components."""
        # Create multiple visualizations with consistent styling
        theme = "plotly_white"
        
        chart_gen = ChartGenerator(theme=theme)
        heatmap_gen = HeatmapGenerator()
        dashboard_builder = DashboardBuilder(theme=theme)
        
        # Create charts with consistent styling
        bar_chart = chart_gen.create_bar_chart(
            self.test_data[self.test_data['requirement'] == 'Physical'],
            'occupation', 'estimate', 'Styled Bar Chart'
        )
        
        heatmap = heatmap_gen.create_risk_heatmap(
            self.test_data, 'occupation', 'requirement', 'reliability',
            'Styled Risk Heatmap'
        )
        
        # Combine in dashboard
        panels = [
            {
                'type': 'bar',
                'title': 'Bar Panel',
                'data': self.test_data[self.test_data['requirement'] == 'Physical'],
                'config': {'x_col': 'occupation', 'y_col': 'estimate'}
            }
        ]
        
        dashboard = dashboard_builder.create_multi_panel_dashboard(
            panels, 'Styled Dashboard'
        )
        
        # Test export
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            success = dashboard_builder.export_dashboard_html(dashboard, temp_path)
            assert success == True
            assert os.path.exists(temp_path)
            
            # Verify consistent styling in exported file
            with open(temp_path, 'r') as f:
                content = f.read()
                assert theme in content or 'plotly' in content.lower()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_error_handling_across_components(self):
        """Test error handling consistency across all visualization components."""
        empty_data = pd.DataFrame()
        invalid_data = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
        
        # Test ChartGenerator error handling
        with pytest.raises(VisualizationError):
            self.chart_gen.create_bar_chart(empty_data, 'x', 'y', 'Test')
        
        # Test HeatmapGenerator error handling
        with pytest.raises(VisualizationError):
            self.heatmap_gen.create_risk_heatmap(empty_data, 'x', 'y', 'z', 'Test')
        
        # Test ConfidenceIntervalPlotter error handling
        with pytest.raises(VisualizationError):
            self.ci_plotter.create_error_bar_plot(empty_data, 'x', 'y', 'e', 'Test')
        
        # Test DashboardBuilder error handling
        with pytest.raises(VisualizationError):
            self.dashboard_builder.create_multi_panel_dashboard(
                [], 'Test', layout='invalid'
            )