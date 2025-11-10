"""
Specialized heatmap generation for correlation matrices and risk analysis.
Provides advanced heatmap visualizations with statistical precision indicators.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from ..interfaces import VisualizationError


class HeatmapGenerator:
    """
    Specialized heatmap generator for correlation matrices, risk analysis,
    and statistical precision dashboards.
    """
    
    def __init__(self, color_scale: str = "RdBu_r"):
        """
        Initialize the heatmap generator.
        
        Args:
            color_scale: Default color scale for heatmaps
        """
        self.color_scale = color_scale
        self.font_family = "Arial, sans-serif"
        self.title_font_size = 16
        self.axis_font_size = 12
        self.annotation_font_size = 10
    
    def _apply_heatmap_layout(self, fig: go.Figure, title: str,
                             width: int = 800, height: int = 600) -> go.Figure:
        """
        Apply standardized layout for heatmaps.
        
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
            width=width,
            height=height,
            margin=dict(l=100, r=100, t=80, b=100),
            xaxis=dict(side='bottom'),
            yaxis=dict(side='left')
        )
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                                  title: str = "Correlation Matrix",
                                  show_values: bool = True,
                                  mask_diagonal: bool = True,
                                  significance_matrix: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Create a correlation heatmap with optional significance indicators.
        
        Args:
            correlation_matrix: DataFrame with correlation coefficients
            title: Chart title
            show_values: Whether to show correlation values on cells
            mask_diagonal: Whether to mask diagonal values
            significance_matrix: Optional matrix of p-values for significance
            
        Returns:
            Plotly figure object
        """
        try:
            # Mask diagonal if requested
            if mask_diagonal:
                matrix = correlation_matrix.copy()
                np.fill_diagonal(matrix.values, np.nan)
            else:
                matrix = correlation_matrix
            
            # Create annotations for cell values
            annotations = []
            if show_values:
                for i, row in enumerate(matrix.index):
                    for j, col in enumerate(matrix.columns):
                        value = matrix.iloc[i, j]
                        if not pd.isna(value):
                            # Add significance indicator if available
                            text = f"{value:.3f}"
                            if significance_matrix is not None:
                                p_value = significance_matrix.iloc[i, j]
                                if p_value < 0.001:
                                    text += "***"
                                elif p_value < 0.01:
                                    text += "**"
                                elif p_value < 0.05:
                                    text += "*"
                            
                            annotations.append(
                                dict(
                                    x=j, y=i,
                                    text=text,
                                    showarrow=False,
                                    font=dict(
                                        color="white" if abs(value) > 0.5 else "black",
                                        size=self.annotation_font_size
                                    )
                                )
                            )
            
            fig = go.Figure(data=go.Heatmap(
                z=matrix.values,
                x=matrix.columns,
                y=matrix.index,
                colorscale=self.color_scale,
                zmid=0,
                zmin=-1,
                zmax=1,
                colorbar=dict(
                    title="Correlation Coefficient"
                ),
                hoverongaps=False,
                hovertemplate='<b>%{y} vs %{x}</b><br>' +
                             'Correlation: %{z:.3f}<br>' +
                             '<extra></extra>'
            ))
            
            fig.update_layout(annotations=annotations)
            
            return self._apply_heatmap_layout(fig, title)
            
        except Exception as e:
            raise VisualizationError(f"Failed to create correlation heatmap: {str(e)}")
    
    def create_risk_heatmap(self, risk_data: pd.DataFrame,
                           occupations_col: str, risk_factors_col: str,
                           risk_values_col: str, title: str = "Risk Analysis Heatmap",
                           risk_threshold: float = 0.5) -> go.Figure:
        """
        Create a risk analysis heatmap showing risk levels across occupations.
        
        Args:
            risk_data: DataFrame with risk data
            occupations_col: Column name for occupations
            risk_factors_col: Column name for risk factors
            risk_values_col: Column name for risk values
            title: Chart title
            risk_threshold: Threshold for highlighting high-risk cells
            
        Returns:
            Plotly figure object
        """
        try:
            # Pivot data to create matrix format
            risk_matrix = risk_data.pivot(
                index=occupations_col,
                columns=risk_factors_col,
                values=risk_values_col
            )
            
            # Create annotations for risk values
            annotations = []
            for i, occupation in enumerate(risk_matrix.index):
                for j, risk_factor in enumerate(risk_matrix.columns):
                    value = risk_matrix.iloc[i, j]
                    if not pd.isna(value):
                        # Highlight high-risk values
                        text_color = "white" if value > risk_threshold else "black"
                        font_weight = "bold" if value > risk_threshold else "normal"
                        
                        annotations.append(
                            dict(
                                x=j, y=i,
                                text=f"{value:.2f}",
                                showarrow=False,
                                font=dict(
                                    color=text_color,
                                    size=self.annotation_font_size
                                )
                            )
                        )
            
            fig = go.Figure(data=go.Heatmap(
                z=risk_matrix.values,
                x=risk_matrix.columns,
                y=risk_matrix.index,
                colorscale="Reds",
                zmin=0,
                zmax=1,
                colorbar=dict(
                    title="Risk Level"
                ),
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>' +
                             'Risk Factor: %{x}<br>' +
                             'Risk Level: %{z:.3f}<br>' +
                             '<extra></extra>'
            ))
            
            fig.update_layout(annotations=annotations)
            
            return self._apply_heatmap_layout(fig, title, height=max(400, len(risk_matrix.index) * 30))
            
        except Exception as e:
            raise VisualizationError(f"Failed to create risk heatmap: {str(e)}")
    
    def create_precision_heatmap(self, data: pd.DataFrame,
                               occupations_col: str, requirements_col: str,
                               precision_col: str, title: str = "Data Precision Heatmap") -> go.Figure:
        """
        Create a heatmap showing data precision levels across occupations and requirements.
        
        Args:
            data: DataFrame with precision data
            occupations_col: Column name for occupations
            requirements_col: Column name for requirements
            precision_col: Column name for precision scores
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        try:
            # Pivot data to create matrix format
            precision_matrix = data.pivot(
                index=occupations_col,
                columns=requirements_col,
                values=precision_col
            )
            
            # Create custom color scale for precision (green = high precision, red = low precision)
            precision_colorscale = [
                [0.0, "rgb(165,0,38)"],    # Low precision - red
                [0.25, "rgb(215,48,39)"],
                [0.5, "rgb(244,109,67)"],
                [0.75, "rgb(253,174,97)"],
                [1.0, "rgb(26,152,80)"]    # High precision - green
            ]
            
            # Create annotations for precision values
            annotations = []
            for i, occupation in enumerate(precision_matrix.index):
                for j, requirement in enumerate(precision_matrix.columns):
                    value = precision_matrix.iloc[i, j]
                    if not pd.isna(value):
                        # Color text based on precision level
                        text_color = "white" if value < 0.5 else "black"
                        
                        # Add precision category
                        if value >= 0.8:
                            category = "High"
                        elif value >= 0.6:
                            category = "Med"
                        else:
                            category = "Low"
                        
                        annotations.append(
                            dict(
                                x=j, y=i,
                                text=f"{value:.2f}<br>({category})",
                                showarrow=False,
                                font=dict(
                                    color=text_color,
                                    size=self.annotation_font_size - 1
                                )
                            )
                        )
            
            fig = go.Figure(data=go.Heatmap(
                z=precision_matrix.values,
                x=precision_matrix.columns,
                y=precision_matrix.index,
                colorscale=precision_colorscale,
                zmin=0,
                zmax=1,
                colorbar=dict(
                    title="Precision Score",
                    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    ticktext=["0.0 (Low)", "0.2", "0.4", "0.6", "0.8", "1.0 (High)"]
                ),
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>' +
                             'Requirement: %{x}<br>' +
                             'Precision: %{z:.3f}<br>' +
                             '<extra></extra>'
            ))
            
            fig.update_layout(annotations=annotations)
            
            return self._apply_heatmap_layout(fig, title, height=max(400, len(precision_matrix.index) * 25))
            
        except Exception as e:
            raise VisualizationError(f"Failed to create precision heatmap: {str(e)}")
    
    def create_clustered_heatmap(self, data: pd.DataFrame, 
                               cluster_method: str = "ward",
                               title: str = "Clustered Heatmap") -> go.Figure:
        """
        Create a heatmap with hierarchical clustering of rows and columns.
        
        Args:
            data: DataFrame to cluster and visualize
            cluster_method: Clustering method ('ward', 'complete', 'average')
            title: Chart title
            
        Returns:
            Plotly figure object with clustered data
        """
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import pdist
            
            # Perform hierarchical clustering on rows
            row_distances = pdist(data.values, metric='euclidean')
            row_linkage = linkage(row_distances, method=cluster_method)
            row_dendro = dendrogram(row_linkage, no_plot=True)
            row_order = row_dendro['leaves']
            
            # Perform hierarchical clustering on columns
            col_distances = pdist(data.T.values, metric='euclidean')
            col_linkage = linkage(col_distances, method=cluster_method)
            col_dendro = dendrogram(col_linkage, no_plot=True)
            col_order = col_dendro['leaves']
            
            # Reorder data based on clustering
            clustered_data = data.iloc[row_order, col_order]
            
            fig = go.Figure(data=go.Heatmap(
                z=clustered_data.values,
                x=clustered_data.columns,
                y=clustered_data.index,
                colorscale=self.color_scale,
                colorbar=dict(
                    title="Value"
                ),
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>' +
                             'Variable: %{x}<br>' +
                             'Value: %{z:.3f}<br>' +
                             '<extra></extra>'
            ))
            
            return self._apply_heatmap_layout(fig, title)
            
        except ImportError:
            # Fallback to regular heatmap if scipy is not available
            return self.create_correlation_heatmap(data, title, show_values=False)
        except Exception as e:
            raise VisualizationError(f"Failed to create clustered heatmap: {str(e)}")
    
    def create_interactive_heatmap(self, data: pd.DataFrame, title: str,
                                 x_col: str, y_col: str, z_col: str,
                                 additional_info: Optional[List[str]] = None) -> go.Figure:
        """
        Create an interactive heatmap with enhanced hover information.
        
        Args:
            data: DataFrame with heatmap data
            title: Chart title
            x_col: Column for x-axis
            y_col: Column for y-axis
            z_col: Column for z-values (color)
            additional_info: Additional columns to show in hover
            
        Returns:
            Interactive Plotly figure object
        """
        try:
            # Pivot data if needed
            if len(data.columns) > 3:
                heatmap_data = data.pivot(index=y_col, columns=x_col, values=z_col)
            else:
                heatmap_data = data
            
            # Build hover template
            hover_template = '<b>%{y} - %{x}</b><br>' + f'{z_col}: %{{z:.3f}}<br>'
            
            if additional_info:
                for info_col in additional_info:
                    if info_col in data.columns:
                        hover_template += f'{info_col}: %{{customdata[{additional_info.index(info_col)}]}}<br>'
            
            hover_template += '<extra></extra>'
            
            # Prepare custom data for hover
            customdata = None
            if additional_info:
                customdata = []
                for i, row_name in enumerate(heatmap_data.index):
                    row_data = []
                    for j, col_name in enumerate(heatmap_data.columns):
                        cell_info = []
                        for info_col in additional_info:
                            # Find matching row in original data
                            match = data[(data[y_col] == row_name) & (data[x_col] == col_name)]
                            if not match.empty and info_col in match.columns:
                                cell_info.append(match[info_col].iloc[0])
                            else:
                                cell_info.append("N/A")
                        row_data.append(cell_info)
                    customdata.append(row_data)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale=self.color_scale,
                colorbar=dict(
                    title=z_col.replace('_', ' ').title()
                ),
                customdata=customdata,
                hovertemplate=hover_template,
                hoverongaps=False
            ))
            
            return self._apply_heatmap_layout(fig, title)
            
        except Exception as e:
            raise VisualizationError(f"Failed to create interactive heatmap: {str(e)}")