"""
Core chart generation system for occupation data reports.
Provides standardized chart creation with interactive capabilities using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from ..interfaces import VisualizationError


class ChartGenerator:
    """
    Core chart generation class supporting bar, pie, scatter, and line charts
    with standardized styling and interactive capabilities.
    """
    
    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize the chart generator with default styling.
        
        Args:
            theme: Plotly theme to use for all charts
        """
        self.theme = theme
        self.default_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        self.font_family = "Arial, sans-serif"
        self.title_font_size = 16
        self.axis_font_size = 12
        self.legend_font_size = 10
    
    def _apply_standard_layout(self, fig: go.Figure, title: str, 
                             width: int = 800, height: int = 600) -> go.Figure:
        """
        Apply standardized layout and styling to a figure.
        
        Args:
            fig: Plotly figure object
            title: Chart title
            width: Chart width in pixels
            height: Chart height in pixels
            
        Returns:
            Styled figure object
        """
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': self.title_font_size, 'family': self.font_family}
            },
            font={'family': self.font_family, 'size': self.axis_font_size},
            template=self.theme,
            width=width,
            height=height,
            margin=dict(l=60, r=60, t=80, b=60),
            legend=dict(
                font={'size': self.legend_font_size},
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        return fig
    
    def create_bar_chart(self, data: pd.DataFrame, x_col: str, y_col: str,
                        title: str, color_col: Optional[str] = None,
                        orientation: str = 'v', show_values: bool = True,
                        error_col: Optional[str] = None) -> go.Figure:
        """
        Create a bar chart with optional error bars and color coding.
        
        Args:
            data: DataFrame containing the data
            x_col: Column name for x-axis values
            y_col: Column name for y-axis values
            title: Chart title
            color_col: Optional column for color coding bars
            orientation: 'v' for vertical, 'h' for horizontal
            show_values: Whether to show values on bars
            error_col: Optional column for error bars (standard errors)
            
        Returns:
            Plotly figure object
        """
        try:
            if color_col and color_col in data.columns:
                fig = px.bar(
                    data, x=x_col, y=y_col, color=color_col,
                    orientation=orientation,
                    color_discrete_sequence=self.default_colors
                )
            else:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=data[x_col] if orientation == 'v' else data[y_col],
                    y=data[y_col] if orientation == 'v' else data[x_col],
                    orientation=orientation,
                    marker_color=self.default_colors[0],
                    error_y=dict(
                        type='data',
                        array=data[error_col] if error_col and error_col in data.columns else None,
                        visible=error_col is not None
                    ) if orientation == 'v' else None,
                    error_x=dict(
                        type='data',
                        array=data[error_col] if error_col and error_col in data.columns else None,
                        visible=error_col is not None
                    ) if orientation == 'h' else None,
                    text=data[y_col] if show_values else None,
                    textposition='auto'
                ))
            
            # Update axis labels
            x_title = x_col.replace('_', ' ').title()
            y_title = y_col.replace('_', ' ').title()
            
            fig.update_xaxes(title_text=x_title)
            fig.update_yaxes(title_text=y_title)
            
            return self._apply_standard_layout(fig, title)
            
        except Exception as e:
            raise VisualizationError(f"Failed to create bar chart: {str(e)}")
    
    def create_pie_chart(self, data: pd.DataFrame, values_col: str, 
                        names_col: str, title: str, 
                        show_percentages: bool = True) -> go.Figure:
        """
        Create a pie chart with customizable labels and styling.
        
        Args:
            data: DataFrame containing the data
            values_col: Column name for pie slice values
            names_col: Column name for pie slice labels
            title: Chart title
            show_percentages: Whether to show percentages on slices
            
        Returns:
            Plotly figure object
        """
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=data[names_col],
                values=data[values_col],
                textinfo='label+percent' if show_percentages else 'label',
                textposition='auto',
                marker=dict(colors=self.default_colors),
                hovertemplate='<b>%{label}</b><br>' +
                             'Value: %{value}<br>' +
                             'Percentage: %{percent}<br>' +
                             '<extra></extra>'
            ))
            
            return self._apply_standard_layout(fig, title)
            
        except Exception as e:
            raise VisualizationError(f"Failed to create pie chart: {str(e)}")
    
    def create_scatter_plot(self, data: pd.DataFrame, x_col: str, y_col: str,
                           title: str, color_col: Optional[str] = None,
                           size_col: Optional[str] = None,
                           trendline: bool = False) -> go.Figure:
        """
        Create a scatter plot with optional color coding, sizing, and trendline.
        
        Args:
            data: DataFrame containing the data
            x_col: Column name for x-axis values
            y_col: Column name for y-axis values
            title: Chart title
            color_col: Optional column for color coding points
            size_col: Optional column for point sizing
            trendline: Whether to add a trendline
            
        Returns:
            Plotly figure object
        """
        try:
            if color_col and color_col in data.columns:
                fig = px.scatter(
                    data, x=x_col, y=y_col, color=color_col,
                    size=size_col if size_col and size_col in data.columns else None,
                    trendline='ols' if trendline else None,
                    color_discrete_sequence=self.default_colors
                )
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='markers',
                    marker=dict(
                        color=self.default_colors[0],
                        size=data[size_col] if size_col and size_col in data.columns else 8,
                        sizemode='diameter',
                        sizeref=2.*max(data[size_col] if size_col and size_col in data.columns else [8])/(40.**2) if size_col else None,
                        sizemin=4
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                 f'{x_col}: %{{x}}<br>' +
                                 f'{y_col}: %{{y}}<br>' +
                                 '<extra></extra>',
                    text=data.index
                ))
                
                # Add trendline if requested
                if trendline:
                    z = np.polyfit(data[x_col], data[y_col], 1)
                    p = np.poly1d(z)
                    fig.add_trace(go.Scatter(
                        x=data[x_col],
                        y=p(data[x_col]),
                        mode='lines',
                        name='Trendline',
                        line=dict(color='red', dash='dash')
                    ))
            
            # Update axis labels
            x_title = x_col.replace('_', ' ').title()
            y_title = y_col.replace('_', ' ').title()
            
            fig.update_xaxes(title_text=x_title)
            fig.update_yaxes(title_text=y_title)
            
            return self._apply_standard_layout(fig, title)
            
        except Exception as e:
            raise VisualizationError(f"Failed to create scatter plot: {str(e)}")
    
    def create_line_chart(self, data: pd.DataFrame, x_col: str, y_col: str,
                         title: str, group_col: Optional[str] = None,
                         show_markers: bool = True) -> go.Figure:
        """
        Create a line chart with optional grouping and markers.
        
        Args:
            data: DataFrame containing the data
            x_col: Column name for x-axis values
            y_col: Column name for y-axis values
            title: Chart title
            group_col: Optional column for grouping lines
            show_markers: Whether to show markers on lines
            
        Returns:
            Plotly figure object
        """
        try:
            if group_col and group_col in data.columns:
                fig = px.line(
                    data, x=x_col, y=y_col, color=group_col,
                    markers=show_markers,
                    color_discrete_sequence=self.default_colors
                )
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='lines+markers' if show_markers else 'lines',
                    line=dict(color=self.default_colors[0]),
                    marker=dict(color=self.default_colors[0]) if show_markers else None
                ))
            
            # Update axis labels
            x_title = x_col.replace('_', ' ').title()
            y_title = y_col.replace('_', ' ').title()
            
            fig.update_xaxes(title_text=x_title)
            fig.update_yaxes(title_text=y_title)
            
            return self._apply_standard_layout(fig, title)
            
        except Exception as e:
            raise VisualizationError(f"Failed to create line chart: {str(e)}")
    
    def create_multi_chart(self, data_list: List[pd.DataFrame], 
                          chart_configs: List[Dict[str, Any]],
                          title: str, subplot_titles: List[str],
                          rows: int = 1, cols: int = 1) -> go.Figure:
        """
        Create multiple charts in subplots.
        
        Args:
            data_list: List of DataFrames for each subplot
            chart_configs: List of configuration dictionaries for each chart
            title: Overall title for the figure
            subplot_titles: Titles for each subplot
            rows: Number of subplot rows
            cols: Number of subplot columns
            
        Returns:
            Plotly figure object with subplots
        """
        try:
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=subplot_titles,
                specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
            )
            
            for i, (data, config) in enumerate(zip(data_list, chart_configs)):
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                chart_type = config.get('type', 'bar')
                
                if chart_type == 'bar':
                    fig.add_trace(
                        go.Bar(
                            x=data[config['x_col']],
                            y=data[config['y_col']],
                            name=config.get('name', f'Chart {i+1}'),
                            marker_color=self.default_colors[i % len(self.default_colors)]
                        ),
                        row=row, col=col
                    )
                elif chart_type == 'scatter':
                    fig.add_trace(
                        go.Scatter(
                            x=data[config['x_col']],
                            y=data[config['y_col']],
                            mode='markers',
                            name=config.get('name', f'Chart {i+1}'),
                            marker=dict(color=self.default_colors[i % len(self.default_colors)])
                        ),
                        row=row, col=col
                    )
                elif chart_type == 'line':
                    fig.add_trace(
                        go.Scatter(
                            x=data[config['x_col']],
                            y=data[config['y_col']],
                            mode='lines+markers',
                            name=config.get('name', f'Chart {i+1}'),
                            line=dict(color=self.default_colors[i % len(self.default_colors)])
                        ),
                        row=row, col=col
                    )
            
            return self._apply_standard_layout(fig, title, width=400*cols, height=400*rows)
            
        except Exception as e:
            raise VisualizationError(f"Failed to create multi-chart: {str(e)}")
    
    def add_confidence_intervals(self, fig: go.Figure, data: pd.DataFrame,
                               x_col: str, y_col: str, error_col: str,
                               fill_color: str = 'rgba(0,100,80,0.2)') -> go.Figure:
        """
        Add confidence interval bands to an existing figure.
        
        Args:
            fig: Existing Plotly figure
            data: DataFrame containing the data
            x_col: Column name for x-axis values
            y_col: Column name for y-axis values (mean)
            error_col: Column name for error values
            fill_color: Color for confidence interval fill
            
        Returns:
            Figure with confidence intervals added
        """
        try:
            upper_bound = data[y_col] + data[error_col]
            lower_bound = data[y_col] - data[error_col]
            
            # Add upper bound
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=upper_bound,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            # Add lower bound with fill
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=lower_bound,
                fill='tonexty',
                fillcolor=fill_color,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Confidence Interval',
                showlegend=True
            ))
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to add confidence intervals: {str(e)}")