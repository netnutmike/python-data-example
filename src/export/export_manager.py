"""
Master export manager that coordinates all export functionality and creates organized output structures.
Combines HTML, PDF, and CSV exporters with master dashboard and file organization capabilities.
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd

from ..interfaces import ExportManagerInterface, ReportData, ExportError
from .html_exporter import HTMLExporter
from .pdf_exporter import PDFExporter
from .csv_exporter import CSVExporter


class ExportManager(ExportManagerInterface):
    """
    Master export manager that coordinates all export functionality,
    creates master dashboards, and organizes output files in structured directories.
    """
    
    def __init__(self, base_output_dir: str = "reports"):
        """
        Initialize the export manager with output configuration.
        
        Args:
            base_output_dir: Base directory for all report outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.html_exporter = HTMLExporter()
        self.pdf_exporter = PDFExporter()
        self.csv_exporter = CSVExporter()
        
        # Directory structure configuration
        self.dir_structure = {
            'html': 'html_reports',
            'pdf': 'pdf_reports', 
            'csv': 'csv_data',
            'assets': 'assets',
            'archive': 'archive'
        }
        
        # File naming conventions
        self.timestamp_format = "%Y%m%d_%H%M%S"
        
    def export_html_report(self, report_data: ReportData, output_path: str) -> bool:
        """
        Export report as interactive HTML using the HTML exporter.
        
        Args:
            report_data: ReportData object containing all report information
            output_path: Path where the HTML file should be saved
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            return self.html_exporter.export_html_report(report_data, output_path)
        except Exception as e:
            raise ExportError(f"HTML export failed: {str(e)}")
    
    def export_pdf_report(self, report_data: ReportData, output_path: str) -> bool:
        """
        Export report as formatted PDF using the PDF exporter.
        
        Args:
            report_data: ReportData object containing all report information
            output_path: Path where the PDF file should be saved
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            return self.pdf_exporter.export_pdf_report(report_data, output_path)
        except Exception as e:
            raise ExportError(f"PDF export failed: {str(e)}")
    
    def export_csv_data(self, data: pd.DataFrame, output_path: str) -> bool:
        """
        Export processed data as CSV using the CSV exporter.
        
        Args:
            data: DataFrame to export
            output_path: Path where the CSV file should be saved
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            return self.csv_exporter.export_csv_data(data, output_path)
        except Exception as e:
            raise ExportError(f"CSV export failed: {str(e)}")
    
    def create_master_dashboard(self, all_reports: List[ReportData]) -> str:
        """
        Create a master dashboard combining all report types with navigation.
        
        Args:
            all_reports: List of ReportData objects to include in dashboard
            
        Returns:
            Path to the created master dashboard HTML file
        """
        try:
            # Create organized directory structure
            output_structure = self.organize_output_files(str(self.base_output_dir))
            
            # Generate timestamp for this dashboard session
            timestamp = datetime.now().strftime(self.timestamp_format)
            dashboard_dir = self.base_output_dir / f"dashboard_{timestamp}"
            dashboard_dir.mkdir(parents=True, exist_ok=True)
            
            # Export all reports in multiple formats
            exported_reports = []
            
            for i, report_data in enumerate(all_reports):
                report_name = self._sanitize_filename(report_data.title)
                
                # Export HTML version
                html_path = dashboard_dir / self.dir_structure['html'] / f"{report_name}.html"
                html_path.parent.mkdir(parents=True, exist_ok=True)
                
                if self.export_html_report(report_data, str(html_path)):
                    # Export PDF version
                    pdf_path = dashboard_dir / self.dir_structure['pdf'] / f"{report_name}.pdf"
                    pdf_path.parent.mkdir(parents=True, exist_ok=True)
                    self.export_pdf_report(report_data, str(pdf_path))
                    
                    # Export analysis results as CSV
                    if report_data.analysis_results:
                        csv_path = dashboard_dir / self.dir_structure['csv'] / f"{report_name}_results.csv"
                        csv_path.parent.mkdir(parents=True, exist_ok=True)
                        self.csv_exporter.export_analysis_results(
                            report_data.analysis_results, str(csv_path)
                        )
                    
                    # Track exported report
                    exported_reports.append({
                        'title': report_data.title,
                        'description': report_data.description,
                        'html_path': str(html_path.relative_to(dashboard_dir)),
                        'pdf_path': str(pdf_path.relative_to(dashboard_dir)),
                        'csv_path': str(csv_path.relative_to(dashboard_dir)) if report_data.analysis_results else None,
                        'generation_time': report_data.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'analysis_count': len(report_data.analysis_results),
                        'visualization_count': len(report_data.visualizations)
                    })
            
            # Create master dashboard HTML
            dashboard_path = dashboard_dir / "index.html"
            self._create_master_dashboard_html(exported_reports, dashboard_path, timestamp)
            
            # Create navigation index
            self._create_navigation_index(exported_reports, dashboard_dir)
            
            # Create report manifest
            self._create_report_manifest(exported_reports, dashboard_dir, timestamp)
            
            # Copy assets if needed
            self._copy_dashboard_assets(dashboard_dir)
            
            return str(dashboard_path)
            
        except Exception as e:
            raise ExportError(f"Failed to create master dashboard: {str(e)}")
    
    def organize_output_files(self, base_path: str) -> Dict[str, str]:
        """
        Organize output files in structured directories with timestamps.
        
        Args:
            base_path: Base path for organizing files
            
        Returns:
            Dictionary mapping directory types to their paths
        """
        try:
            base_dir = Path(base_path)
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Create directory structure
            structure = {}
            for dir_type, dir_name in self.dir_structure.items():
                dir_path = base_dir / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)
                structure[dir_type] = str(dir_path)
            
            return structure
            
        except Exception as e:
            raise ExportError(f"Failed to organize output files: {str(e)}")
    
    def _create_master_dashboard_html(self, reports: List[Dict[str, Any]], 
                                    output_path: Path, timestamp: str) -> None:
        """
        Create the master dashboard HTML file.
        
        Args:
            reports: List of report information dictionaries
            output_path: Path where dashboard HTML should be saved
            timestamp: Timestamp for the dashboard session
        """
        dashboard_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Occupation Data Reports Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1f77b4 0%, #0d5aa7 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
            margin-bottom: 20px;
        }}
        
        .header .stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 30px;
        }}
        
        .stat-item {{
            text-align: center;
        }}
        
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            display: block;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.8;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #1f77b4;
            font-size: 1.8em;
            margin-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
        }}
        
        .reports-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            margin: 30px 0;
        }}
        
        .report-card {{
            background: #fafafa;
            border-radius: 10px;
            padding: 25px;
            border-left: 5px solid #1f77b4;
            transition: all 0.3s ease;
            position: relative;
        }}
        
        .report-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }}
        
        .report-card h3 {{
            color: #1f77b4;
            font-size: 1.3em;
            margin-bottom: 10px;
        }}
        
        .report-card .description {{
            color: #666;
            margin-bottom: 15px;
            line-height: 1.5;
        }}
        
        .report-meta {{
            display: flex;
            justify-content: space-between;
            font-size: 0.9em;
            color: #888;
            margin-bottom: 20px;
        }}
        
        .report-actions {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        
        .btn {{
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            text-decoration: none;
            font-size: 0.9em;
            transition: all 0.2s ease;
            cursor: pointer;
        }}
        
        .btn-primary {{
            background: #1f77b4;
            color: white;
        }}
        
        .btn-primary:hover {{
            background: #0d5aa7;
        }}
        
        .btn-secondary {{
            background: #ff7f0e;
            color: white;
        }}
        
        .btn-secondary:hover {{
            background: #e6720d;
        }}
        
        .btn-tertiary {{
            background: #2ca02c;
            color: white;
        }}
        
        .btn-tertiary:hover {{
            background: #228b22;
        }}
        
        .quick-stats {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 30px;
            border-radius: 10px;
            margin: 30px 0;
        }}
        
        .quick-stats h3 {{
            color: #1f77b4;
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        
        .stats-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        
        .stats-card .number {{
            font-size: 2em;
            font-weight: bold;
            color: #1f77b4;
            display: block;
        }}
        
        .stats-card .label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e0e0e0;
        }}
        
        @media (max-width: 768px) {{
            .header .stats {{
                flex-direction: column;
                gap: 20px;
            }}
            
            .reports-grid {{
                grid-template-columns: 1fr;
            }}
            
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Occupation Data Reports</h1>
            <div class="subtitle">Comprehensive Analysis Dashboard</div>
            <div class="stats">
                <div class="stat-item">
                    <span class="stat-number">{len(reports)}</span>
                    <span class="stat-label">Reports Generated</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{sum(r['analysis_count'] for r in reports)}</span>
                    <span class="stat-label">Analysis Results</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{sum(r['visualization_count'] for r in reports)}</span>
                    <span class="stat-label">Visualizations</span>
                </div>
            </div>
        </div>
        
        <div class="content">
            <div class="quick-stats">
                <h3>Dashboard Overview</h3>
                <div class="stats-grid">
                    <div class="stats-card">
                        <span class="number">{len([r for r in reports if r['html_path']])}</span>
                        <div class="label">HTML Reports</div>
                    </div>
                    <div class="stats-card">
                        <span class="number">{len([r for r in reports if r.get('pdf_path')])}</span>
                        <div class="label">PDF Reports</div>
                    </div>
                    <div class="stats-card">
                        <span class="number">{len([r for r in reports if r.get('csv_path')])}</span>
                        <div class="label">CSV Datasets</div>
                    </div>
                    <div class="stats-card">
                        <span class="number">{timestamp}</span>
                        <div class="label">Session ID</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Available Reports</h2>
                <div class="reports-grid">'''
        
        for report in reports:
            dashboard_html += f'''
                    <div class="report-card">
                        <h3>{report['title']}</h3>
                        <div class="description">{report['description']}</div>
                        <div class="report-meta">
                            <span>Generated: {report['generation_time']}</span>
                            <span>{report['analysis_count']} results, {report['visualization_count']} charts</span>
                        </div>
                        <div class="report-actions">
                            <a href="{report['html_path']}" class="btn btn-primary" target="_blank">View HTML</a>'''
            
            if report.get('pdf_path'):
                dashboard_html += f'''
                            <a href="{report['pdf_path']}" class="btn btn-secondary" target="_blank">Download PDF</a>'''
            
            if report.get('csv_path'):
                dashboard_html += f'''
                            <a href="{report['csv_path']}" class="btn btn-tertiary" download>Download CSV</a>'''
            
            dashboard_html += '''
                        </div>
                    </div>'''
        
        dashboard_html += f'''
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Dashboard generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Occupation Data Reports System | Session: {timestamp}</p>
        </div>
    </div>
</body>
</html>'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
    
    def _create_navigation_index(self, reports: List[Dict[str, Any]], dashboard_dir: Path) -> None:
        """
        Create a simple navigation index for the reports.
        
        Args:
            reports: List of report information dictionaries
            dashboard_dir: Dashboard directory path
        """
        nav_data = []
        for report in reports:
            nav_data.append({
                'title': report['title'],
                'html_file': report['html_path'],
                'pdf_file': report.get('pdf_path', ''),
                'csv_file': report.get('csv_path', ''),
                'generation_time': report['generation_time']
            })
        
        nav_df = pd.DataFrame(nav_data)
        nav_path = dashboard_dir / 'navigation_index.csv'
        
        self.csv_exporter.export_csv_data(nav_df, str(nav_path), {
            'export_type': 'Navigation Index',
            'total_reports': len(reports)
        })
    
    def _create_report_manifest(self, reports: List[Dict[str, Any]], 
                              dashboard_dir: Path, timestamp: str) -> None:
        """
        Create a detailed manifest of all generated reports.
        
        Args:
            reports: List of report information dictionaries
            dashboard_dir: Dashboard directory path
            timestamp: Dashboard session timestamp
        """
        manifest = {
            'dashboard_info': {
                'session_id': timestamp,
                'generation_time': datetime.now().isoformat(),
                'total_reports': len(reports),
                'dashboard_path': str(dashboard_dir)
            },
            'reports': reports,
            'directory_structure': self.dir_structure
        }
        
        manifest_path = dashboard_dir / 'manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    def _copy_dashboard_assets(self, dashboard_dir: Path) -> None:
        """
        Copy any required assets to the dashboard directory.
        
        Args:
            dashboard_dir: Dashboard directory path
        """
        assets_dir = dashboard_dir / self.dir_structure['assets']
        assets_dir.mkdir(exist_ok=True)
        
        # Create a simple CSS file for consistent styling
        css_content = '''
        /* Additional styles for exported reports */
        .report-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .chart-responsive {
            width: 100%;
            height: auto;
        }
        
        @media print {
            .no-print {
                display: none;
            }
        }
        '''
        
        css_path = assets_dir / 'report-styles.css'
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write(css_content)
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe file system usage.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove or replace problematic characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        sanitized = "".join(c if c in safe_chars else "_" for c in filename)
        
        # Remove multiple underscores and trim
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        
        return sanitized.strip("_")[:50]  # Limit length
    
    def archive_old_reports(self, days_old: int = 30) -> int:
        """
        Archive reports older than specified days.
        
        Args:
            days_old: Number of days after which reports should be archived
            
        Returns:
            Number of reports archived
        """
        try:
            archive_count = 0
            current_time = datetime.now()
            archive_dir = self.base_output_dir / self.dir_structure['archive']
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Find old dashboard directories
            for item in self.base_output_dir.iterdir():
                if item.is_dir() and item.name.startswith('dashboard_'):
                    try:
                        # Extract timestamp from directory name
                        timestamp_str = item.name.replace('dashboard_', '')
                        dir_time = datetime.strptime(timestamp_str, self.timestamp_format)
                        
                        # Check if old enough to archive
                        if (current_time - dir_time).days >= days_old:
                            archive_path = archive_dir / item.name
                            shutil.move(str(item), str(archive_path))
                            archive_count += 1
                            
                    except (ValueError, OSError):
                        # Skip directories with invalid timestamps or move errors
                        continue
            
            return archive_count
            
        except Exception as e:
            raise ExportError(f"Failed to archive old reports: {str(e)}")
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """
        Get summary information about all dashboards and reports.
        
        Returns:
            Dictionary containing dashboard summary statistics
        """
        try:
            summary = {
                'total_dashboards': 0,
                'total_reports': 0,
                'latest_dashboard': None,
                'dashboard_list': []
            }
            
            # Scan for dashboard directories
            for item in self.base_output_dir.iterdir():
                if item.is_dir() and item.name.startswith('dashboard_'):
                    summary['total_dashboards'] += 1
                    
                    # Try to read manifest for report count
                    manifest_path = item / 'manifest.json'
                    if manifest_path.exists():
                        try:
                            with open(manifest_path, 'r', encoding='utf-8') as f:
                                manifest = json.load(f)
                                report_count = len(manifest.get('reports', []))
                                summary['total_reports'] += report_count
                                
                                dashboard_info = {
                                    'name': item.name,
                                    'path': str(item),
                                    'report_count': report_count,
                                    'generation_time': manifest['dashboard_info'].get('generation_time')
                                }
                                summary['dashboard_list'].append(dashboard_info)
                                
                        except (json.JSONDecodeError, KeyError):
                            continue
            
            # Find latest dashboard
            if summary['dashboard_list']:
                latest = max(summary['dashboard_list'], 
                           key=lambda x: x.get('generation_time', ''))
                summary['latest_dashboard'] = latest
            
            return summary
            
        except Exception as e:
            raise ExportError(f"Failed to get dashboard summary: {str(e)}")