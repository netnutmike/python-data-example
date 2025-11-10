"""
HTML export functionality for occupation data reports.
Creates interactive HTML reports with embedded charts, footnote legends, and metadata display.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, Template
import pandas as pd

from ..interfaces import ExportManagerInterface, ReportData, ExportError


class HTMLExporter:
    """
    HTML exporter class for creating interactive reports with embedded charts,
    responsive design, and comprehensive footnote legends.
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the HTML exporter with template configuration.
        
        Args:
            template_dir: Directory containing HTML templates
        """
        self.template_dir = template_dir or self._get_default_template_dir()
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=True
        )
        
        # Default styling configuration
        self.css_config = {
            'primary_color': '#1f77b4',
            'secondary_color': '#ff7f0e',
            'background_color': '#ffffff',
            'text_color': '#333333',
            'border_color': '#dddddd',
            'font_family': 'Arial, sans-serif'
        }
        
        # Footnote reference mapping
        self.footnote_descriptions = self._load_footnote_descriptions()
    
    def _get_default_template_dir(self) -> str:
        """Get the default template directory path."""
        current_dir = Path(__file__).parent
        template_dir = current_dir / 'templates'
        template_dir.mkdir(exist_ok=True)
        return str(template_dir)
    
    def _load_footnote_descriptions(self) -> Dict[int, str]:
        """Load footnote descriptions for legend display."""
        # This would typically load from a configuration file
        # For now, providing common footnote interpretations
        return {
            1: "Estimate not released because it does not meet BLS or other quality standards",
            2: "Estimate is less than 0.5 percent",
            3: "Estimate is 0.5 percent or greater but less than 1.5 percent",
            4: "Estimate is 1.5 percent or greater but less than 2.5 percent",
            5: "Estimate is 2.5 percent or greater but less than 3.5 percent",
            6: "Estimate is 3.5 percent or greater but less than 4.5 percent",
            7: "Estimate is 4.5 percent or greater but less than 5.5 percent",
            8: "Estimate is 5.5 percent or greater but less than 6.5 percent",
            9: "Estimate is 6.5 percent or greater but less than 7.5 percent",
            10: "Estimate is 7.5 percent or greater but less than 8.5 percent",
            # Add more footnote codes as needed
        }
    
    def export_html_report(self, report_data: ReportData, output_path: str) -> bool:
        """
        Export a complete HTML report with interactive charts and metadata.
        
        Args:
            report_data: ReportData object containing all report information
            output_path: Path where the HTML file should be saved
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert visualizations to HTML
            chart_htmls = self._convert_charts_to_html(report_data.visualizations)
            
            # Prepare template context
            context = {
                'title': report_data.title,
                'description': report_data.description,
                'generation_timestamp': report_data.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_results': report_data.analysis_results,
                'charts': chart_htmls,
                'metadata': report_data.metadata,
                'footnote_legend': self._generate_footnote_legend(report_data),
                'css_config': self.css_config,
                'total_charts': len(chart_htmls),
                'total_results': len(report_data.analysis_results)
            }
            
            # Load and render template
            template = self._get_or_create_template()
            html_content = template.render(**context)
            
            # Write HTML file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return True
            
        except Exception as e:
            raise ExportError(f"Failed to export HTML report: {str(e)}")
    
    def _convert_charts_to_html(self, visualizations: List[go.Figure]) -> List[Dict[str, str]]:
        """
        Convert Plotly figures to HTML format for embedding.
        
        Args:
            visualizations: List of Plotly figure objects
            
        Returns:
            List of dictionaries containing chart HTML and metadata
        """
        chart_htmls = []
        
        for i, fig in enumerate(visualizations):
            try:
                # Configure Plotly for HTML export
                config = {
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                    'responsive': True
                }
                
                # Convert to HTML
                chart_html = pio.to_html(
                    fig,
                    include_plotlyjs='cdn',
                    config=config,
                    div_id=f'chart_{i}',
                    full_html=False
                )
                
                # Extract chart title and type
                chart_title = fig.layout.title.text if fig.layout.title else f'Chart {i+1}'
                chart_type = self._detect_chart_type(fig)
                
                chart_htmls.append({
                    'id': f'chart_{i}',
                    'title': chart_title,
                    'type': chart_type,
                    'html': chart_html,
                    'description': self._generate_chart_description(fig, chart_type)
                })
                
            except Exception as e:
                # Create error placeholder for failed charts
                chart_htmls.append({
                    'id': f'chart_{i}',
                    'title': f'Chart {i+1} (Error)',
                    'type': 'error',
                    'html': f'<div class="chart-error">Error rendering chart: {str(e)}</div>',
                    'description': 'Chart could not be rendered due to an error.'
                })
        
        return chart_htmls
    
    def _detect_chart_type(self, fig: go.Figure) -> str:
        """
        Detect the type of chart from a Plotly figure.
        
        Args:
            fig: Plotly figure object
            
        Returns:
            String describing the chart type
        """
        if not fig.data:
            return 'empty'
        
        trace_types = [trace.type for trace in fig.data]
        
        if 'bar' in trace_types:
            return 'bar'
        elif 'pie' in trace_types:
            return 'pie'
        elif 'scatter' in trace_types:
            # Check if it's a line chart or scatter plot
            modes = [getattr(trace, 'mode', '') for trace in fig.data if hasattr(trace, 'mode')]
            if any('lines' in mode for mode in modes):
                return 'line'
            else:
                return 'scatter'
        elif 'heatmap' in trace_types:
            return 'heatmap'
        else:
            return 'other'
    
    def _generate_chart_description(self, fig: go.Figure, chart_type: str) -> str:
        """
        Generate a description for a chart based on its type and data.
        
        Args:
            fig: Plotly figure object
            chart_type: Type of chart
            
        Returns:
            Description string
        """
        descriptions = {
            'bar': 'Bar chart showing comparative values across categories',
            'pie': 'Pie chart displaying proportional distribution of data',
            'line': 'Line chart showing trends and changes over time or categories',
            'scatter': 'Scatter plot revealing relationships between variables',
            'heatmap': 'Heatmap visualizing correlation or intensity patterns',
            'other': 'Custom visualization displaying data relationships'
        }
        
        base_description = descriptions.get(chart_type, 'Data visualization')
        
        # Add data point count if available
        if fig.data:
            total_points = sum(len(getattr(trace, 'x', [])) for trace in fig.data if hasattr(trace, 'x'))
            if total_points > 0:
                base_description += f' ({total_points} data points)'
        
        return base_description
    
    def _generate_footnote_legend(self, report_data: ReportData) -> List[Dict[str, Any]]:
        """
        Generate footnote legend based on footnotes found in the data.
        
        Args:
            report_data: ReportData object
            
        Returns:
            List of footnote legend entries
        """
        footnote_legend = []
        used_footnotes = set()
        
        # Collect footnotes from analysis results
        for result in report_data.analysis_results:
            for footnote in result.footnote_context:
                if footnote.isdigit():
                    used_footnotes.add(int(footnote))
        
        # Collect footnotes from metadata
        if 'footnotes' in report_data.metadata:
            for footnote in report_data.metadata['footnotes']:
                if isinstance(footnote, int):
                    used_footnotes.add(footnote)
        
        # Build legend entries
        for footnote_code in sorted(used_footnotes):
            if footnote_code in self.footnote_descriptions:
                footnote_legend.append({
                    'code': footnote_code,
                    'description': self.footnote_descriptions[footnote_code],
                    'category': self._categorize_footnote(footnote_code)
                })
        
        return footnote_legend
    
    def _categorize_footnote(self, footnote_code: int) -> str:
        """
        Categorize footnote codes for better organization.
        
        Args:
            footnote_code: Footnote code number
            
        Returns:
            Category string
        """
        if footnote_code == 1:
            return 'quality'
        elif 2 <= footnote_code <= 10:
            return 'precision'
        elif footnote_code > 10:
            return 'interpretation'
        else:
            return 'other'
    
    def _get_or_create_template(self) -> Template:
        """
        Get the HTML template, creating a default one if it doesn't exist.
        
        Returns:
            Jinja2 Template object
        """
        template_path = Path(self.template_dir) / 'report_template.html'
        
        if not template_path.exists():
            self._create_default_template(template_path)
        
        return self.env.get_template('report_template.html')
    
    def _create_default_template(self, template_path: Path) -> None:
        """
        Create a default HTML template for reports.
        
        Args:
            template_path: Path where template should be created
        """
        default_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: {{ css_config.font_family }};
            color: {{ css_config.text_color }};
            background-color: {{ css_config.background_color }};
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            border-bottom: 2px solid {{ css_config.primary_color }};
        }
        
        .header h1 {
            color: {{ css_config.primary_color }};
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }
        
        .header .description {
            font-size: 1.2em;
            color: #666;
            margin: 10px 0;
        }
        
        .header .metadata {
            font-size: 0.9em;
            color: #888;
            margin-top: 15px;
        }
        
        .section {
            margin: 40px 0;
            padding: 20px;
            border: 1px solid {{ css_config.border_color }};
            border-radius: 8px;
            background: #fafafa;
        }
        
        .section h2 {
            color: {{ css_config.primary_color }};
            border-bottom: 1px solid {{ css_config.border_color }};
            padding-bottom: 10px;
            margin-top: 0;
        }
        
        .chart-container {
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .chart-title {
            font-size: 1.3em;
            font-weight: bold;
            color: {{ css_config.primary_color }};
            margin-bottom: 10px;
        }
        
        .chart-description {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 15px;
        }
        
        .analysis-results {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .result-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid {{ css_config.secondary_color }};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .result-card h3 {
            margin: 0 0 10px 0;
            color: {{ css_config.primary_color }};
        }
        
        .result-value {
            font-size: 1.5em;
            font-weight: bold;
            color: {{ css_config.secondary_color }};
            margin: 10px 0;
        }
        
        .confidence-interval {
            font-size: 0.9em;
            color: #666;
            margin: 5px 0;
        }
        
        .footnote-legend {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 30px 0;
        }
        
        .footnote-legend h3 {
            color: {{ css_config.primary_color }};
            margin-top: 0;
        }
        
        .footnote-item {
            margin: 10px 0;
            padding: 8px;
            background: white;
            border-radius: 4px;
            border-left: 3px solid {{ css_config.secondary_color }};
        }
        
        .footnote-code {
            font-weight: bold;
            color: {{ css_config.primary_color }};
        }
        
        .footnote-category {
            display: inline-block;
            background: {{ css_config.secondary_color }};
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-left: 10px;
        }
        
        .summary-stats {
            display: flex;
            justify-content: space-around;
            background: {{ css_config.primary_color }};
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            display: block;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 0 10px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .summary-stats {
                flex-direction: column;
                gap: 15px;
            }
            
            .analysis-results {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <div class="description">{{ description }}</div>
            <div class="metadata">
                Generated on {{ generation_timestamp }}
                {% if metadata.data_source %}
                | Data Source: {{ metadata.data_source }}
                {% endif %}
            </div>
        </div>
        
        <div class="summary-stats">
            <div class="stat-item">
                <span class="stat-value">{{ total_charts }}</span>
                <span class="stat-label">Visualizations</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">{{ total_results }}</span>
                <span class="stat-label">Analysis Results</span>
            </div>
            {% if metadata.total_records %}
            <div class="stat-item">
                <span class="stat-value">{{ "{:,}".format(metadata.total_records) }}</span>
                <span class="stat-label">Data Records</span>
            </div>
            {% endif %}
        </div>
        
        {% if charts %}
        <div class="section">
            <h2>Visualizations</h2>
            {% for chart in charts %}
            <div class="chart-container">
                <div class="chart-title">{{ chart.title }}</div>
                <div class="chart-description">{{ chart.description }}</div>
                {{ chart.html|safe }}
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        {% if analysis_results %}
        <div class="section">
            <h2>Analysis Results</h2>
            <div class="analysis-results">
                {% for result in analysis_results %}
                <div class="result-card">
                    <h3>{{ result.occupation_category }}</h3>
                    <div><strong>{{ result.metric_name }}</strong></div>
                    <div class="result-value">{{ "%.2f"|format(result.value) }}</div>
                    <div class="confidence-interval">
                        95% CI: [{{ "%.2f"|format(result.confidence_interval[0]) }}, 
                                {{ "%.2f"|format(result.confidence_interval[1]) }}]
                    </div>
                    <div class="confidence-interval">
                        Reliability Score: {{ "%.1f"|format(result.reliability_score * 100) }}%
                    </div>
                    {% if result.footnote_context %}
                    <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                        Footnotes: {{ result.footnote_context|join(', ') }}
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        
        {% if footnote_legend %}
        <div class="footnote-legend">
            <h3>Footnote Legend</h3>
            {% for footnote in footnote_legend %}
            <div class="footnote-item">
                <span class="footnote-code">{{ footnote.code }}</span>: {{ footnote.description }}
                <span class="footnote-category">{{ footnote.category }}</span>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        {% if metadata %}
        <div class="section">
            <h2>Report Metadata</h2>
            <div style="font-family: monospace; background: white; padding: 15px; border-radius: 4px; overflow-x: auto;">
                {% for key, value in metadata.items() %}
                <div><strong>{{ key }}:</strong> {{ value }}</div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>'''
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(default_template)
    
    def create_multi_report_index(self, report_files: List[Dict[str, str]], 
                                 output_path: str, title: str = "Occupation Data Reports") -> bool:
        """
        Create an index HTML file linking to multiple reports.
        
        Args:
            report_files: List of dictionaries with 'title', 'description', and 'path' keys
            output_path: Path where the index HTML should be saved
            title: Title for the index page
            
        Returns:
            True if successful, False otherwise
        """
        try:
            index_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 {
            color: #1f77b4;
            text-align: center;
            margin-bottom: 30px;
        }
        .report-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .report-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            background: #fafafa;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .report-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .report-card h3 {
            color: #1f77b4;
            margin-top: 0;
        }
        .report-card p {
            color: #666;
            margin: 10px 0;
        }
        .report-link {
            display: inline-block;
            background: #1f77b4;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            margin-top: 10px;
            transition: background 0.2s;
        }
        .report-link:hover {
            background: #0d5aa7;
        }
        .timestamp {
            text-align: center;
            color: #888;
            margin-top: 30px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <div class="report-grid">
            {% for report in reports %}
            <div class="report-card">
                <h3>{{ report.title }}</h3>
                <p>{{ report.description }}</p>
                <a href="{{ report.path }}" class="report-link">View Report</a>
            </div>
            {% endfor %}
        </div>
        <div class="timestamp">
            Generated on {{ timestamp }}
        </div>
    </div>
</body>
</html>'''
            
            template = Template(index_template)
            html_content = template.render(
                title=title,
                reports=report_files,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return True
            
        except Exception as e:
            raise ExportError(f"Failed to create report index: {str(e)}")