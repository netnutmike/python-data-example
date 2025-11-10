"""
Confidence interval plotting with uncertainty indicators for statistical analysis.
Provides specialized visualizations for statistical precision and reliability assessment.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from ..interfaces import VisualizationError


class ConfidenceIntervalPlotter:
    """
    Specialized plotter for confidence intervals and statistical precision indicators.
    """
    
    def __init__(self):
        """Initialize the confidence interval plotter."""
        self.font_family = "Arial, sans-serif"
        self.title_font_size = 16
        self.axis_font_size = 12
        self.confidence_colors = {
            'high': 'rgba(0, 128, 0, 0.3)',      # Green for high confidence
            'medium': 'rgba(255, 165, 0, 0.3)',  # Orange for medium confidence
            'low': 'rgba(255, 0, 0, 0.3)'        # Red for low confidence
        }
    
    def _apply_standard_layout(self, fig: go.Figure, title: str,
                             width: int = 800, height: int = 600) -> go.Figure:
        """Apply standardized layout for confidence interval plots."""
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': self.title_font_size, 'family': self.font_family}
            },
            font={'family': self.font_family, 'size': self.axis_font_size},
            width=width,
            height=height,
            margin=dict(l=80, r=80, t=80, b=80),
            showlegend=True
        )
        return fig
    
    def create_error_bar_plot(self, data: pd.DataFrame, x_col: str, y_col: str,
                             error_col: str, title: str,
                             confidence_level: float = 0.95,
                             color_by_precision: bool = True) -> go.Figure:
        """
        Create an error bar plot with confidence intervals.
        
        Args:
            data: DataFrame containing the data
            x_col: Column name for x-axis values
            y_col: Column name for y-axis values (estimates)
            error_col: Column name for error values (standard errors)
            title: Chart title
            confidence_level: Confidence level for intervals (default 0.95)
            color_by_precision: Whether to color bars by precision level
            
        Returns:
            Plotly figure object
        """
        try:
            # Calculate confidence intervals
            from scipy import stats
            t_value = stats.t.ppf((1 + confidence_level) / 2, df=len(data) - 1)
            ci_width = t_value * data[error_col]
            
            # Determine precision categories if color coding is requested
            if color_by_precision:
                # Calculate relative error (CV = standard error / estimate)
                relative_error = data[error_col] / data[y_col].abs()
                precision_categories = pd.cut(
                    relative_error,
                    bins=[0, 0.1, 0.2, float('inf')],
                    labels=['High', 'Medium', 'Low']
                )
                colors = [self.confidence_colors[cat.lower()] for cat in precision_categories]
            else:
                colors = ['rgba(31, 119, 180, 0.7)'] * len(data)
            
            fig = go.Figure()
            
            # Add error bars
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                error_y=dict(
                    type='data',
                    array=ci_width,
                    visible=True,
                    color='rgba(0,0,0,0.5)',
                    thickness=2,
                    width=3
                ),
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors if color_by_precision else 'rgba(31, 119, 180, 0.7)',
                    line=dict(width=1, color='black')
                ),
                name='Estimates with CI',
                hovertemplate='<b>%{x}</b><br>' +
                             f'{y_col}: %{{y:.3f}}<br>' +
                             f'CI Width: ±%{{error_y.array:.3f}}<br>' +
                             f'Confidence: {confidence_level*100:.0f}%<br>' +
                             '<extra></extra>'
            ))
            
            # Add precision legend if color coding is used
            if color_by_precision:
                for precision, color in self.confidence_colors.items():
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(size=10, color=color),
                        name=f'{precision.title()} Precision',
                        showlegend=True
                    ))
            
            fig.update_xaxes(title_text=x_col.replace('_', ' ').title())
            fig.update_yaxes(title_text=y_col.replace('_', ' ').title())
            
            return self._apply_standard_layout(fig, title)
            
        except Exception as e:
            raise VisualizationError(f"Failed to create error bar plot: {str(e)}")
    
    def create_confidence_band_plot(self, data: pd.DataFrame, x_col: str, y_col: str,
                                   lower_ci_col: str, upper_ci_col: str,
                                   title: str, show_points: bool = False) -> go.Figure:
        """
        Create a plot with confidence bands around a trend line.
        
        Args:
            data: DataFrame containing the data
            x_col: Column name for x-axis values
            y_col: Column name for y-axis values (estimates)
            lower_ci_col: Column name for lower confidence interval
            upper_ci_col: Column name for upper confidence interval
            title: Chart title
            show_points: Whether to show individual data points
            
        Returns:
            Plotly figure object
        """
        try:
            fig = go.Figure()
            
            # Sort data by x-axis for proper line plotting
            sorted_data = data.sort_values(x_col)
            
            # Add confidence band
            fig.add_trace(go.Scatter(
                x=sorted_data[x_col],
                y=sorted_data[upper_ci_col],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=sorted_data[x_col],
                y=sorted_data[lower_ci_col],
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.2)',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Confidence Band',
                hovertemplate='<b>%{x}</b><br>' +
                             'Lower CI: %{y:.3f}<br>' +
                             '<extra></extra>'
            ))
            
            # Add main trend line
            fig.add_trace(go.Scatter(
                x=sorted_data[x_col],
                y=sorted_data[y_col],
                mode='lines',
                line=dict(color='blue', width=2),
                name='Estimate',
                hovertemplate='<b>%{x}</b><br>' +
                             f'{y_col}: %{{y:.3f}}<br>' +
                             '<extra></extra>'
            ))
            
            # Add individual points if requested
            if show_points:
                fig.add_trace(go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color='blue',
                        line=dict(width=1, color='white')
                    ),
                    name='Data Points',
                    showlegend=False,
                    hovertemplate='<b>%{x}</b><br>' +
                                 f'{y_col}: %{{y:.3f}}<br>' +
                                 '<extra></extra>'
                ))
            
            fig.update_xaxes(title_text=x_col.replace('_', ' ').title())
            fig.update_yaxes(title_text=y_col.replace('_', ' ').title())
            
            return self._apply_standard_layout(fig, title)
            
        except Exception as e:
            raise VisualizationError(f"Failed to create confidence band plot: {str(e)}")
    
    def create_precision_dashboard(self, data: pd.DataFrame, 
                                 estimate_col: str, error_col: str,
                                 category_col: str, title: str) -> go.Figure:
        """
        Create an interactive dashboard showing statistical precision across categories.
        
        Args:
            data: DataFrame containing the data
            estimate_col: Column name for estimates
            error_col: Column name for standard errors
            category_col: Column name for categories
            title: Dashboard title
            
        Returns:
            Plotly figure object with multiple subplots
        """
        try:
            # Calculate precision metrics
            data = data.copy()
            data['relative_error'] = data[error_col] / data[estimate_col].abs()
            data['precision_score'] = 1 / (1 + data['relative_error'])
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Precision by Category',
                    'Error Distribution',
                    'Precision vs Estimate',
                    'Reliability Summary'
                ],
                specs=[
                    [{"type": "bar"}, {"type": "histogram"}],
                    [{"type": "scatter"}, {"type": "bar"}]
                ]
            )
            
            # 1. Precision by category (bar chart)
            category_precision = data.groupby(category_col)['precision_score'].mean().sort_values(ascending=False)
            fig.add_trace(
                go.Bar(
                    x=category_precision.index,
                    y=category_precision.values,
                    name='Avg Precision',
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
            
            # 2. Error distribution (histogram)
            fig.add_trace(
                go.Histogram(
                    x=data['relative_error'],
                    nbinsx=20,
                    name='Error Distribution',
                    marker_color='lightcoral'
                ),
                row=1, col=2
            )
            
            # 3. Precision vs Estimate (scatter plot)
            fig.add_trace(
                go.Scatter(
                    x=data[estimate_col],
                    y=data['precision_score'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=data['relative_error'],
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title="Relative Error", x=1.1)
                    ),
                    name='Precision vs Estimate',
                    text=data[category_col],
                    hovertemplate='<b>%{text}</b><br>' +
                                 f'{estimate_col}: %{{x:.3f}}<br>' +
                                 'Precision: %{y:.3f}<br>' +
                                 '<extra></extra>'
                ),
                row=2, col=1
            )
            
            # 4. Reliability summary (bar chart)
            reliability_bins = pd.cut(
                data['precision_score'],
                bins=[0, 0.6, 0.8, 1.0],
                labels=['Low', 'Medium', 'High']
            )
            reliability_counts = reliability_bins.value_counts()
            
            fig.add_trace(
                go.Bar(
                    x=reliability_counts.index,
                    y=reliability_counts.values,
                    name='Reliability Levels',
                    marker_color=['red', 'orange', 'green']
                ),
                row=2, col=2
            )
            
            # Update subplot titles and axes
            fig.update_xaxes(title_text="Category", row=1, col=1)
            fig.update_yaxes(title_text="Precision Score", row=1, col=1)
            
            fig.update_xaxes(title_text="Relative Error", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            
            fig.update_xaxes(title_text=estimate_col.replace('_', ' ').title(), row=2, col=1)
            fig.update_yaxes(title_text="Precision Score", row=2, col=1)
            
            fig.update_xaxes(title_text="Reliability Level", row=2, col=2)
            fig.update_yaxes(title_text="Count", row=2, col=2)
            
            return self._apply_standard_layout(fig, title, width=1000, height=800)
            
        except Exception as e:
            raise VisualizationError(f"Failed to create precision dashboard: {str(e)}")
    
    def create_uncertainty_comparison(self, data_list: List[pd.DataFrame],
                                    labels: List[str], estimate_col: str,
                                    error_col: str, title: str) -> go.Figure:
        """
        Create a comparison plot showing uncertainty across different datasets or methods.
        
        Args:
            data_list: List of DataFrames to compare
            labels: Labels for each dataset
            estimate_col: Column name for estimates
            error_col: Column name for errors
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        try:
            fig = go.Figure()
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            for i, (data, label) in enumerate(zip(data_list, labels)):
                color = colors[i % len(colors)]
                
                # Calculate confidence intervals
                ci_width = 1.96 * data[error_col]  # 95% CI
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(data))),
                    y=data[estimate_col],
                    error_y=dict(
                        type='data',
                        array=ci_width,
                        visible=True,
                        color=color,
                        thickness=2,
                        width=3
                    ),
                    mode='markers+lines',
                    marker=dict(size=6, color=color),
                    line=dict(color=color, width=2),
                    name=label,
                    hovertemplate=f'<b>{label}</b><br>' +
                                 'Index: %{x}<br>' +
                                 f'{estimate_col}: %{{y:.3f}}<br>' +
                                 'CI Width: ±%{error_y.array:.3f}<br>' +
                                 '<extra></extra>'
                ))
            
            fig.update_xaxes(title_text="Data Point Index")
            fig.update_yaxes(title_text=estimate_col.replace('_', ' ').title())
            
            return self._apply_standard_layout(fig, title)
            
        except Exception as e:
            raise VisualizationError(f"Failed to create uncertainty comparison: {str(e)}")