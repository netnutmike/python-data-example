"""
PDF export functionality for occupation data reports.
Creates formatted static PDF reports with proper page layouts, charts, and professional styling.
"""

import os
import io
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, Frame, PageTemplate
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import pandas as pd
from PIL import Image as PILImage

from ..interfaces import ReportData, ExportError


class PDFExporter:
    """
    PDF exporter class for creating formatted static reports with proper page layouts,
    embedded charts, tables, and professional styling.
    """
    
    def __init__(self, page_size: Tuple[float, float] = letter):
        """
        Initialize the PDF exporter with page configuration.
        
        Args:
            page_size: Page size tuple (width, height) in points
        """
        self.page_size = page_size
        self.margin = 0.75 * inch
        self.styles = self._create_custom_styles()
        
        # Color scheme
        self.colors = {
            'primary': colors.Color(31/255, 119/255, 180/255),  # #1f77b4
            'secondary': colors.Color(255/255, 127/255, 14/255),  # #ff7f0e
            'text': colors.Color(51/255, 51/255, 51/255),  # #333333
            'light_gray': colors.Color(245/255, 245/255, 245/255),  # #f5f5f5
            'border': colors.Color(221/255, 221/255, 221/255)  # #dddddd
        }
        
        # Chart configuration
        self.chart_width = 6 * inch
        self.chart_height = 4 * inch
        self.chart_dpi = 150
    
    def _create_custom_styles(self) -> Dict[str, ParagraphStyle]:
        """
        Create custom paragraph styles for the PDF report.
        
        Returns:
            Dictionary of custom styles
        """
        base_styles = getSampleStyleSheet()
        custom_styles = {}
        
        # Title style
        custom_styles['Title'] = ParagraphStyle(
            'CustomTitle',
            parent=base_styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.Color(31/255, 119/255, 180/255),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Heading styles
        custom_styles['Heading1'] = ParagraphStyle(
            'CustomHeading1',
            parent=base_styles['Heading1'],
            fontSize=18,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.Color(31/255, 119/255, 180/255),
            fontName='Helvetica-Bold'
        )
        
        custom_styles['Heading2'] = ParagraphStyle(
            'CustomHeading2',
            parent=base_styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=colors.Color(31/255, 119/255, 180/255),
            fontName='Helvetica-Bold'
        )
        
        # Body text
        custom_styles['Body'] = ParagraphStyle(
            'CustomBody',
            parent=base_styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        )
        
        # Metadata style
        custom_styles['Metadata'] = ParagraphStyle(
            'CustomMetadata',
            parent=base_styles['Normal'],
            fontSize=9,
            textColor=colors.Color(136/255, 136/255, 136/255),
            alignment=TA_CENTER,
            fontName='Helvetica'
        )
        
        # Caption style
        custom_styles['Caption'] = ParagraphStyle(
            'CustomCaption',
            parent=base_styles['Normal'],
            fontSize=10,
            textColor=colors.Color(102/255, 102/255, 102/255),
            alignment=TA_CENTER,
            fontName='Helvetica-Oblique',
            spaceAfter=12
        )
        
        return custom_styles
    
    def export_pdf_report(self, report_data: ReportData, output_path: str) -> bool:
        """
        Export a complete PDF report with formatted layouts and embedded charts.
        
        Args:
            report_data: ReportData object containing all report information
            output_path: Path where the PDF file should be saved
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=self.page_size,
                rightMargin=self.margin,
                leftMargin=self.margin,
                topMargin=self.margin,
                bottomMargin=self.margin
            )
            
            # Build document content
            story = []
            
            # Add title page
            story.extend(self._create_title_page(report_data))
            
            # Add executive summary
            story.extend(self._create_executive_summary(report_data))
            
            # Add analysis results section
            if report_data.analysis_results:
                story.extend(self._create_analysis_section(report_data.analysis_results))
            
            # Add visualizations section
            if report_data.visualizations:
                story.extend(self._create_visualizations_section(report_data.visualizations))
            
            # Add appendices
            story.extend(self._create_appendices(report_data))
            
            # Build PDF
            doc.build(story)
            
            return True
            
        except Exception as e:
            raise ExportError(f"Failed to export PDF report: {str(e)}")
    
    def _create_title_page(self, report_data: ReportData) -> List[Any]:
        """
        Create the title page content.
        
        Args:
            report_data: ReportData object
            
        Returns:
            List of flowable elements for the title page
        """
        elements = []
        
        # Add vertical space
        elements.append(Spacer(1, 2 * inch))
        
        # Title
        elements.append(Paragraph(report_data.title, self.styles['Title']))
        elements.append(Spacer(1, 0.5 * inch))
        
        # Description
        elements.append(Paragraph(report_data.description, self.styles['Body']))
        elements.append(Spacer(1, 1 * inch))
        
        # Metadata table
        metadata_data = [
            ['Generation Date:', report_data.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S')],
            ['Total Analysis Results:', str(len(report_data.analysis_results))],
            ['Total Visualizations:', str(len(report_data.visualizations))]
        ]
        
        if 'data_source' in report_data.metadata:
            metadata_data.append(['Data Source:', str(report_data.metadata['data_source'])])
        
        if 'total_records' in report_data.metadata:
            metadata_data.append(['Total Records:', f"{report_data.metadata['total_records']:,}"])
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
        metadata_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(metadata_table)
        elements.append(PageBreak())
        
        return elements
    
    def _create_executive_summary(self, report_data: ReportData) -> List[Any]:
        """
        Create executive summary section.
        
        Args:
            report_data: ReportData object
            
        Returns:
            List of flowable elements for executive summary
        """
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['Heading1']))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.colors['primary']))
        elements.append(Spacer(1, 12))
        
        # Generate summary statistics
        if report_data.analysis_results:
            total_occupations = len(set(result.occupation_category for result in report_data.analysis_results))
            avg_reliability = sum(result.reliability_score for result in report_data.analysis_results) / len(report_data.analysis_results)
            
            summary_text = f"""
            This report presents a comprehensive analysis of occupational data covering {total_occupations} 
            occupation categories. The analysis includes {len(report_data.analysis_results)} distinct metrics 
            with an average reliability score of {avg_reliability:.1%}.
            
            The report includes {len(report_data.visualizations)} visualizations to support data interpretation 
            and decision-making. All estimates include confidence intervals and reliability assessments to 
            ensure appropriate use of the data.
            """
            
            elements.append(Paragraph(summary_text, self.styles['Body']))
        
        # Key findings
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Key Findings", self.styles['Heading2']))
        
        if report_data.analysis_results:
            # Find top results by value
            top_results = sorted(report_data.analysis_results, key=lambda x: x.value, reverse=True)[:5]
            
            findings_text = "The analysis reveals the following key findings:\n\n"
            for i, result in enumerate(top_results, 1):
                findings_text += f"{i}. {result.occupation_category}: {result.metric_name} = {result.value:.2f}\n"
            
            elements.append(Paragraph(findings_text, self.styles['Body']))
        
        elements.append(PageBreak())
        
        return elements
    
    def _create_analysis_section(self, analysis_results: List[Any]) -> List[Any]:
        """
        Create the analysis results section.
        
        Args:
            analysis_results: List of AnalysisResult objects
            
        Returns:
            List of flowable elements for analysis section
        """
        elements = []
        
        elements.append(Paragraph("Analysis Results", self.styles['Heading1']))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.colors['primary']))
        elements.append(Spacer(1, 12))
        
        # Group results by occupation category
        results_by_category = {}
        for result in analysis_results:
            if result.occupation_category not in results_by_category:
                results_by_category[result.occupation_category] = []
            results_by_category[result.occupation_category].append(result)
        
        # Create tables for each category
        for category, results in results_by_category.items():
            elements.append(Paragraph(category, self.styles['Heading2']))
            
            # Create table data
            table_data = [['Metric', 'Value', 'Confidence Interval', 'Reliability']]
            
            for result in results:
                ci_text = f"[{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]"
                reliability_text = f"{result.reliability_score:.1%}"
                
                table_data.append([
                    result.metric_name,
                    f"{result.value:.2f}",
                    ci_text,
                    reliability_text
                ])
            
            # Create and style table
            table = Table(table_data, colWidths=[2.5*inch, 1*inch, 1.5*inch, 1*inch])
            table.setStyle(TableStyle([
                # Header row
                ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                
                # Data rows
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                ('TOPPADDING', (0, 1), (-1, -1), 8),
                
                # Alternating row colors
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.colors['light_gray']]),
                
                # Grid
                ('GRID', (0, 0), (-1, -1), 0.5, self.colors['border']),
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_visualizations_section(self, visualizations: List[go.Figure]) -> List[Any]:
        """
        Create the visualizations section with embedded charts.
        
        Args:
            visualizations: List of Plotly figure objects
            
        Returns:
            List of flowable elements for visualizations section
        """
        elements = []
        
        elements.append(Paragraph("Visualizations", self.styles['Heading1']))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.colors['primary']))
        elements.append(Spacer(1, 12))
        
        for i, fig in enumerate(visualizations):
            try:
                # Convert Plotly figure to image
                img_bytes = self._convert_figure_to_image(fig)
                
                if img_bytes:
                    # Create image flowable
                    img = Image(io.BytesIO(img_bytes), width=self.chart_width, height=self.chart_height)
                    
                    # Add chart title
                    chart_title = fig.layout.title.text if fig.layout.title else f"Chart {i+1}"
                    elements.append(Paragraph(chart_title, self.styles['Heading2']))
                    
                    # Add chart
                    elements.append(KeepTogether([img]))
                    
                    # Add caption
                    chart_type = self._detect_chart_type(fig)
                    caption_text = f"Figure {i+1}: {chart_type.title()} visualization"
                    elements.append(Paragraph(caption_text, self.styles['Caption']))
                    
                    elements.append(Spacer(1, 20))
                
            except Exception as e:
                # Add error message for failed charts
                error_text = f"Chart {i+1}: Unable to render visualization ({str(e)})"
                elements.append(Paragraph(error_text, self.styles['Body']))
                elements.append(Spacer(1, 12))
        
        return elements
    
    def _convert_figure_to_image(self, fig: go.Figure) -> Optional[bytes]:
        """
        Convert a Plotly figure to PNG image bytes.
        
        Args:
            fig: Plotly figure object
            
        Returns:
            PNG image bytes or None if conversion fails
        """
        try:
            # Configure figure for PDF export
            fig.update_layout(
                width=int(self.chart_width * self.chart_dpi / 72),
                height=int(self.chart_height * self.chart_dpi / 72),
                font=dict(size=10),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            # Convert to PNG bytes
            img_bytes = pio.to_image(fig, format='png', engine='kaleido')
            return img_bytes
            
        except Exception:
            return None
    
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
            return 'bar chart'
        elif 'pie' in trace_types:
            return 'pie chart'
        elif 'scatter' in trace_types:
            modes = [getattr(trace, 'mode', '') for trace in fig.data if hasattr(trace, 'mode')]
            if any('lines' in mode for mode in modes):
                return 'line chart'
            else:
                return 'scatter plot'
        elif 'heatmap' in trace_types:
            return 'heatmap'
        else:
            return 'chart'
    
    def _create_appendices(self, report_data: ReportData) -> List[Any]:
        """
        Create appendices section with metadata and footnotes.
        
        Args:
            report_data: ReportData object
            
        Returns:
            List of flowable elements for appendices
        """
        elements = []
        
        elements.append(PageBreak())
        elements.append(Paragraph("Appendices", self.styles['Heading1']))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.colors['primary']))
        elements.append(Spacer(1, 12))
        
        # Appendix A: Methodology
        elements.append(Paragraph("Appendix A: Methodology", self.styles['Heading2']))
        methodology_text = """
        This report is based on the 2023 Occupational Requirements Survey (ORS) conducted by the 
        Bureau of Labor Statistics. The survey collected data from 56,300 establishments covering 
        145,866,200 civilian workers across various occupational categories.
        
        All estimates include confidence intervals calculated using standard errors provided in the 
        dataset. Reliability scores are computed based on the precision of estimates and the 
        presence of footnote qualifications.
        """
        elements.append(Paragraph(methodology_text, self.styles['Body']))
        elements.append(Spacer(1, 12))
        
        # Appendix B: Data Quality Notes
        elements.append(Paragraph("Appendix B: Data Quality Notes", self.styles['Heading2']))
        
        # Collect footnotes from analysis results
        footnotes_used = set()
        for result in report_data.analysis_results:
            for footnote in result.footnote_context:
                if footnote.isdigit():
                    footnotes_used.add(int(footnote))
        
        if footnotes_used:
            footnote_text = "The following footnote codes appear in this report:\n\n"
            for footnote_code in sorted(footnotes_used):
                footnote_text += f"• Code {footnote_code}: Data quality or interpretation note\n"
            
            elements.append(Paragraph(footnote_text, self.styles['Body']))
        
        # Appendix C: Technical Details
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Appendix C: Technical Details", self.styles['Heading2']))
        
        if report_data.metadata:
            tech_data = []
            for key, value in report_data.metadata.items():
                tech_data.append([key.replace('_', ' ').title(), str(value)])
            
            if tech_data:
                tech_table = Table(tech_data, colWidths=[2*inch, 3*inch])
                tech_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 0.5, self.colors['border']),
                ]))
                
                elements.append(tech_table)
        
        return elements
    
    def create_summary_report(self, multiple_reports: List[ReportData], 
                            output_path: str, title: str = "Occupation Data Summary") -> bool:
        """
        Create a summary PDF combining multiple reports.
        
        Args:
            multiple_reports: List of ReportData objects
            output_path: Path where the summary PDF should be saved
            title: Title for the summary report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create combined report data
            combined_results = []
            combined_visualizations = []
            combined_metadata = {'reports_included': len(multiple_reports)}
            
            for report in multiple_reports:
                combined_results.extend(report.analysis_results)
                combined_visualizations.extend(report.visualizations)
                
            summary_report = ReportData(
                title=title,
                description=f"Summary report combining {len(multiple_reports)} individual reports",
                analysis_results=combined_results,
                visualizations=combined_visualizations,
                metadata=combined_metadata,
                generation_timestamp=datetime.now()
            )
            
            return self.export_pdf_report(summary_report, output_path)
            
        except Exception as e:
            raise ExportError(f"Failed to create summary PDF report: {str(e)}")