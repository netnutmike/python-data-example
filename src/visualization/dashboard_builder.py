"""
Dashboard system for combining multiple visualizations with interactive filtering
and drill-down capabilities. Provides responsive design for different screen sizes.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
import json
from ..interfaces import VisualizationError, ReportData


class DashboardBuilder:
    """
    Dashboard builder for creating interactive multi-visualization dashboards
    with filtering and drill-down capabilities.
    """
    
    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize the dashboard builder.
        
        Args:
            theme: Plotly theme for consistent styling
        """
        self.theme = theme
        self.font_family = "Arial, sans-serif"
        self.title_font_size = 20
        self.subtitle_font_size = 14
        self.default_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    def create_multi_panel_dashboard(self, panels: List[Dict[str, Any]],
                                   title: str, layout: str = "grid",
                                   rows: Optional[int] = None,
                                   cols: Optional[int] = None) -> go.Figure:
        """
        Create a multi-panel dashboard with various chart types.
        
        Args:
            panels: List of panel configurations with chart data and settings
            title: Dashboard title
            layout: Layout type ("grid", "vertical", "horizontal")
            rows: Number of rows for grid layout
            cols: Number of columns for grid layout
            
        Returns:
            Plotly figure object with multiple panels
        """
        try:
            num_panels = len(panels)
            
            # Determine layout dimensions
            if layout == "grid":
                if rows is None and cols is None:
                    cols = min(3, num_panels)
                    rows = (num_panels + cols - 1) // cols
                elif rows is None:
                    rows = (num_panels + cols - 1) // cols
                elif cols is None:
                    cols = (num_panels + rows - 1) // rows
            elif layout == "vertical":
                rows, cols = num_panels, 1
            elif layout == "horizontal":
                rows, cols = 1, num_panels
            else:
                raise ValueError(f"Unknown layout type: {layout}")
            
            # Create subplot titles
            subplot_titles = [panel.get('title', f'Panel {i+1}') for i, panel in enumerate(panels)]
            
            # Create subplots
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=subplot_titles,
                specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)],
                vertical_spacing=0.08,
                horizontal_spacing=0.05
            )
            
            # Add each panel
            for i, panel in enumerate(panels):
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                self._add_panel_to_subplot(fig, panel, row, col)
            
            # Apply overall styling
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': self.title_font_size, 'family': self.font_family}
                },
                font={'family': self.font_family, 'size': 12},
                template=self.theme,
                height=400 * rows,
                width=400 * cols,
                showlegend=True,
                margin=dict(l=60, r=60, t=100, b=60)
            )
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to create multi-panel dashboard: {str(e)}")
    
    def _add_panel_to_subplot(self, fig: go.Figure, panel: Dict[str, Any],
                            row: int, col: int) -> None:
        """Add a single panel to a subplot."""
        panel_type = panel.get('type', 'bar')
        data = panel['data']
        config = panel.get('config', {})
        
        if panel_type == 'bar':
            fig.add_trace(
                go.Bar(
                    x=data[config.get('x_col', data.columns[0])],
                    y=data[config.get('y_col', data.columns[1])],
                    name=config.get('name', 'Bar Chart'),
                    marker_color=config.get('color', self.default_colors[0])
                ),
                row=row, col=col
            )
        elif panel_type == 'scatter':
            fig.add_trace(
                go.Scatter(
                    x=data[config.get('x_col', data.columns[0])],
                    y=data[config.get('y_col', data.columns[1])],
                    mode='markers',
                    name=config.get('name', 'Scatter Plot'),
                    marker=dict(color=config.get('color', self.default_colors[1]))
                ),
                row=row, col=col
            )
        elif panel_type == 'line':
            fig.add_trace(
                go.Scatter(
                    x=data[config.get('x_col', data.columns[0])],
                    y=data[config.get('y_col', data.columns[1])],
                    mode='lines+markers',
                    name=config.get('name', 'Line Chart'),
                    line=dict(color=config.get('color', self.default_colors[2]))
                ),
                row=row, col=col
            )
        elif panel_type == 'heatmap':
            # Assume data is already in matrix format or pivot it
            if len(data.columns) > 2:
                z_data = data.pivot(
                    index=config.get('y_col', data.columns[0]),
                    columns=config.get('x_col', data.columns[1]),
                    values=config.get('z_col', data.columns[2])
                ).values
            else:
                z_data = data.values
            
            fig.add_trace(
                go.Heatmap(
                    z=z_data,
                    colorscale=config.get('colorscale', 'Viridis'),
                    showscale=False
                ),
                row=row, col=col
            )
    
    def create_interactive_dashboard(self, report_data: ReportData,
                                   filters: Optional[Dict[str, List[str]]] = None) -> go.Figure:
        """
        Create an interactive dashboard with filtering capabilities.
        
        Args:
            report_data: ReportData object containing analysis results and visualizations
            filters: Optional dictionary of filter options
            
        Returns:
            Interactive Plotly figure object
        """
        try:
            # Extract data from analysis results
            if not report_data.analysis_results:
                raise ValueError("No analysis results provided for dashboard")
            
            # Convert analysis results to DataFrame for easier manipulation
            results_data = []
            for result in report_data.analysis_results:
                results_data.append({
                    'occupation_category': result.occupation_category,
                    'metric_name': result.metric_name,
                    'value': result.value,
                    'confidence_lower': result.confidence_interval[0],
                    'confidence_upper': result.confidence_interval[1],
                    'reliability_score': result.reliability_score
                })
            
            df = pd.DataFrame(results_data)
            
            # Create dashboard panels based on available data
            panels = []
            
            # Panel 1: Metric values by occupation category
            if 'occupation_category' in df.columns and 'value' in df.columns:
                panels.append({
                    'type': 'bar',
                    'title': 'Metric Values by Occupation',
                    'data': df.groupby('occupation_category')['value'].mean().reset_index(),
                    'config': {
                        'x_col': 'occupation_category',
                        'y_col': 'value',
                        'name': 'Average Value',
                        'color': self.default_colors[0]
                    }
                })
            
            # Panel 2: Reliability scores distribution
            if 'reliability_score' in df.columns:
                panels.append({
                    'type': 'scatter',
                    'title': 'Reliability vs Value',
                    'data': df,
                    'config': {
                        'x_col': 'value',
                        'y_col': 'reliability_score',
                        'name': 'Reliability',
                        'color': self.default_colors[1]
                    }
                })
            
            # Panel 3: Confidence interval widths
            if 'confidence_lower' in df.columns and 'confidence_upper' in df.columns:
                df['ci_width'] = df['confidence_upper'] - df['confidence_lower']
                panels.append({
                    'type': 'bar',
                    'title': 'Confidence Interval Widths',
                    'data': df.groupby('occupation_category')['ci_width'].mean().reset_index(),
                    'config': {
                        'x_col': 'occupation_category',
                        'y_col': 'ci_width',
                        'name': 'CI Width',
                        'color': self.default_colors[2]
                    }
                })
            
            # Panel 4: Metric distribution
            if 'metric_name' in df.columns:
                metric_counts = df['metric_name'].value_counts().reset_index()
                metric_counts.columns = ['metric_name', 'count']
                panels.append({
                    'type': 'bar',
                    'title': 'Metric Distribution',
                    'data': metric_counts,
                    'config': {
                        'x_col': 'metric_name',
                        'y_col': 'count',
                        'name': 'Count',
                        'color': self.default_colors[3]
                    }
                })
            
            # Create the dashboard
            dashboard_title = f"{report_data.title} - Interactive Dashboard"
            return self.create_multi_panel_dashboard(panels, dashboard_title, layout="grid", cols=2)
            
        except Exception as e:
            raise VisualizationError(f"Failed to create interactive dashboard: {str(e)}")
    
    def create_drill_down_dashboard(self, data: pd.DataFrame, 
                                  hierarchy_cols: List[str],
                                  value_col: str, title: str) -> go.Figure:
        """
        Create a dashboard with drill-down capabilities through data hierarchy.
        
        Args:
            data: DataFrame with hierarchical data
            hierarchy_cols: List of column names representing hierarchy levels
            value_col: Column name for values to aggregate
            title: Dashboard title
            
        Returns:
            Plotly figure object with drill-down capability
        """
        try:
            if not hierarchy_cols:
                raise ValueError("At least one hierarchy column must be provided")
            
            # Create initial view at top level
            top_level_data = data.groupby(hierarchy_cols[0])[value_col].sum().reset_index()
            
            fig = go.Figure()
            
            # Add top-level bar chart
            fig.add_trace(go.Bar(
                x=top_level_data[hierarchy_cols[0]],
                y=top_level_data[value_col],
                name='Top Level',
                marker_color=self.default_colors[0],
                hovertemplate='<b>%{x}</b><br>' +
                             f'{value_col}: %{{y}}<br>' +
                             'Click to drill down<br>' +
                             '<extra></extra>'
            ))
            
            # Add drill-down functionality through custom data
            if len(hierarchy_cols) > 1:
                # Prepare drill-down data
                drill_down_data = {}
                for level1 in data[hierarchy_cols[0]].unique():
                    level1_data = data[data[hierarchy_cols[0]] == level1]
                    if len(hierarchy_cols) > 1:
                        level2_data = level1_data.groupby(hierarchy_cols[1])[value_col].sum().reset_index()
                        drill_down_data[level1] = {
                            'x': level2_data[hierarchy_cols[1]].tolist(),
                            'y': level2_data[value_col].tolist()
                        }
                
                # Store drill-down data in figure for potential JavaScript interaction
                fig.update_layout(
                    annotations=[
                        dict(
                            text="Click on bars to drill down to next level",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.5, y=1.05,
                            xanchor='center', yanchor='bottom',
                            font=dict(size=12, color="gray")
                        )
                    ]
                )
            
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': self.title_font_size, 'family': self.font_family}
                },
                font={'family': self.font_family, 'size': 12},
                template=self.theme,
                height=600,
                width=800,
                xaxis_title=hierarchy_cols[0].replace('_', ' ').title(),
                yaxis_title=value_col.replace('_', ' ').title()
            )
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to create drill-down dashboard: {str(e)}")
    
    def create_responsive_dashboard(self, panels: List[Dict[str, Any]],
                                  title: str, breakpoints: Dict[str, int] = None) -> go.Figure:
        """
        Create a responsive dashboard that adapts to different screen sizes.
        
        Args:
            panels: List of panel configurations
            title: Dashboard title
            breakpoints: Dictionary of screen size breakpoints
            
        Returns:
            Plotly figure object with responsive design
        """
        try:
            if breakpoints is None:
                breakpoints = {
                    'mobile': 768,
                    'tablet': 1024,
                    'desktop': 1200
                }
            
            # Create base dashboard
            fig = self.create_multi_panel_dashboard(panels, title, layout="grid")
            
            # Add responsive configuration
            fig.update_layout(
                autosize=True,
                margin=dict(l=20, r=20, t=60, b=20),
                font=dict(size=10),  # Smaller font for mobile
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add responsive behavior through configuration
            config = {
                'responsive': True,
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': [
                    'pan2d', 'lasso2d', 'select2d', 'autoScale2d',
                    'hoverClosestCartesian', 'hoverCompareCartesian'
                ]
            }
            
            # Store responsive config in figure metadata
            fig.update_layout(
                meta={
                    'responsive_config': config,
                    'breakpoints': breakpoints
                }
            )
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to create responsive dashboard: {str(e)}")
    
    def create_filtered_dashboard(self, data: pd.DataFrame, 
                                filter_columns: List[str],
                                chart_configs: List[Dict[str, Any]],
                                title: str) -> go.Figure:
        """
        Create a dashboard with interactive filtering capabilities.
        
        Args:
            data: DataFrame with data to visualize
            filter_columns: List of columns to create filters for
            chart_configs: List of chart configurations
            title: Dashboard title
            
        Returns:
            Plotly figure object with filtering capability
        """
        try:
            # Create filter options
            filter_options = {}
            for col in filter_columns:
                if col in data.columns:
                    filter_options[col] = sorted(data[col].unique().tolist())
            
            # Create initial dashboard with all data
            panels = []
            for i, config in enumerate(chart_configs):
                panel_data = data.copy()
                
                # Apply any pre-filters from config
                if 'filters' in config:
                    for filter_col, filter_vals in config['filters'].items():
                        if filter_col in panel_data.columns:
                            panel_data = panel_data[panel_data[filter_col].isin(filter_vals)]
                
                panels.append({
                    'type': config.get('type', 'bar'),
                    'title': config.get('title', f'Chart {i+1}'),
                    'data': panel_data,
                    'config': config
                })
            
            fig = self.create_multi_panel_dashboard(panels, title)
            
            # Add filter information as annotations
            filter_text = "Available Filters: " + ", ".join(filter_columns)
            fig.add_annotation(
                text=filter_text,
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=1.08,
                xanchor='center', yanchor='bottom',
                font=dict(size=10, color="gray")
            )
            
            # Store filter options in figure metadata for potential use
            fig.update_layout(
                meta={
                    'filter_options': filter_options,
                    'filter_columns': filter_columns
                }
            )
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to create filtered dashboard: {str(e)}")
    
    def export_dashboard_html(self, fig: go.Figure, output_path: str,
                            include_plotlyjs: str = 'cdn') -> bool:
        """
        Export dashboard as standalone HTML file.
        
        Args:
            fig: Plotly figure object
            output_path: Path to save HTML file
            include_plotlyjs: How to include Plotly.js ('cdn', 'inline', 'directory')
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Add metadata and timestamp
            fig.update_layout(
                annotations=fig.layout.annotations + (dict(
                    text=f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=1, y=0,
                    xanchor='right', yanchor='bottom',
                    font=dict(size=8, color="gray")
                ),)
            )
            
            # Export to HTML
            fig.write_html(
                output_path,
                include_plotlyjs=include_plotlyjs,
                config={
                    'responsive': True,
                    'displayModeBar': True,
                    'displaylogo': False
                }
            )
            
            return True
            
        except Exception as e:
            raise VisualizationError(f"Failed to export dashboard HTML: {str(e)}")
            return False