"""
Command-line interface commands and utilities.
Provides comprehensive CLI functionality with help documentation and usage examples.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..config.settings import ConfigurationManager
from ..integration.pipeline import DataProcessingPipeline
from ..integration.monitoring import get_performance_monitor
from ..reports.report_factory import ReportFactory
from ..interfaces import ConfigurationError


class CLICommands:
    """
    Command-line interface commands for the occupation data reports application.
    Provides structured command handling with comprehensive help and validation.
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize CLI commands.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.report_factory = ReportFactory(self.config_manager)
    
    def list_reports(self, args: argparse.Namespace) -> int:
        """
        List all available report types and their descriptions.
        
        Args:
            args: Command-line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            print("\n" + "="*80)
            print("AVAILABLE REPORT TYPES")
            print("="*80)
            
            # Get report information from factory
            report_info = self.report_factory.get_all_report_info()
            
            if not report_info:
                print("No report types are currently available.")
                return 0
            
            # Group reports by category for better organization
            categories = {
                'distribution': [],
                'analysis': [],
                'assessment': [],
                'insights': []
            }
            
            for report_type, info in report_info.items():
                # Categorize based on report type name
                if 'distribution' in report_type:
                    categories['distribution'].append((report_type, info))
                elif any(word in report_type for word in ['risk', 'demands', 'quality']):
                    categories['assessment'].append((report_type, info))
                elif any(word in report_type for word in ['correlation', 'precision', 'statistical']):
                    categories['analysis'].append((report_type, info))
                else:
                    categories['insights'].append((report_type, info))
            
            # Display reports by category
            for category, reports in categories.items():
                if reports:
                    print(f"\n{category.upper()} REPORTS:")
                    print("-" * 40)
                    
                    for report_type, info in reports:
                        status = "✓ Enabled" if info.get('enabled', True) else "✗ Disabled"
                        print(f"\n  {report_type}")
                        print(f"    Title: {info.get('title', 'N/A')}")
                        print(f"    Description: {info.get('description', 'N/A')}")
                        print(f"    Status: {status}")
                        
                        # Show required columns if available
                        required_cols = info.get('required_columns', [])
                        if required_cols:
                            print(f"    Required Data: {', '.join(required_cols[:3])}{'...' if len(required_cols) > 3 else ''}")
                        
                        # Show supported filters if available
                        filters = info.get('supported_filters', [])
                        if filters:
                            print(f"    Filters: {', '.join(filters[:3])}{'...' if len(filters) > 3 else ''}")
            
            print(f"\nTotal: {len(report_info)} report types available")
            print("\nUsage Examples:")
            print("  python -m src.main --report-type occupation_distribution")
            print("  python -m src.main --generate-all --parallel")
            print("  python -m src.main --report-type environmental_risk --output-format pdf")
            
            return 0
            
        except Exception as e:
            print(f"Error listing reports: {e}", file=sys.stderr)
            return 1
    
    def validate_config(self, args: argparse.Namespace) -> int:
        """
        Validate configuration files and display results.
        
        Args:
            args: Command-line arguments
            
        Returns:
            Exit code (0 for valid, 1 for invalid)
        """
        try:
            print("\n" + "="*60)
            print("CONFIGURATION VALIDATION")
            print("="*60)
            
            config_files = [
                "data_sources.yaml",
                "output.yaml", 
                "visualization.yaml",
                "analysis.yaml",
                "reports.yaml"
            ]
            
            all_valid = True
            validation_results = {}
            
            for config_file in config_files:
                try:
                    print(f"\nValidating {config_file}...")
                    
                    # Load configuration
                    config_data = self.config_manager.load_config(config_file)
                    
                    # Validate configuration
                    validation_result = self.config_manager.validate_config(
                        {config_file.split('.')[0]: config_data}
                    )
                    
                    validation_results[config_file] = validation_result
                    
                    if validation_result.is_valid:
                        print(f"  ✓ {config_file}: Valid")
                    else:
                        print(f"  ✗ {config_file}: Invalid")
                        all_valid = False
                    
                    # Show errors
                    if validation_result.errors:
                        print("    Errors:")
                        for error in validation_result.errors:
                            print(f"      - {error}")
                    
                    # Show warnings
                    if validation_result.warnings:
                        print("    Warnings:")
                        for warning in validation_result.warnings:
                            print(f"      - {warning}")
                            
                except Exception as e:
                    print(f"  ✗ {config_file}: Failed to load - {e}")
                    all_valid = False
            
            # Summary
            print(f"\n{'='*60}")
            if all_valid:
                print("✓ All configurations are valid")
                
                # Show additional information
                print("\nConfiguration Summary:")
                try:
                    data_config = self.config_manager.get_data_source_config()
                    print(f"  Data Source: {data_config.main_dataset_path}")
                    
                    enabled_reports = self.config_manager.get_enabled_reports()
                    print(f"  Enabled Reports: {len(enabled_reports)}")
                    
                    output_config = self.config_manager.get_output_settings()
                    print(f"  Output Directory: {output_config.get('base_output_dir', 'reports')}")
                    
                except Exception as e:
                    print(f"  Warning: Could not load configuration summary: {e}")
                
            else:
                print("✗ Configuration validation failed")
                print("\nPlease fix the errors above before running the application.")
            
            return 0 if all_valid else 1
            
        except Exception as e:
            print(f"Configuration validation failed: {e}", file=sys.stderr)
            return 1
    
    def show_status(self, args: argparse.Namespace) -> int:
        """
        Show application and system status.
        
        Args:
            args: Command-line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            print("\n" + "="*60)
            print("APPLICATION STATUS")
            print("="*60)
            
            # Configuration status
            print("\nConfiguration:")
            try:
                config_validation = self.config_manager.validate_config({
                    'data_sources': self.config_manager.load_config('data_sources.yaml'),
                    'reports': self.config_manager.load_config('reports.yaml')
                })
                
                if config_validation.is_valid:
                    print("  ✓ Configuration: Valid")
                else:
                    print("  ✗ Configuration: Invalid")
                    
            except Exception as e:
                print(f"  ✗ Configuration: Error - {e}")
            
            # Data source status
            print("\nData Sources:")
            try:
                data_config = self.config_manager.get_data_source_config()
                
                main_dataset = Path(data_config.main_dataset_path)
                footnote_dataset = Path(data_config.footnote_dataset_path)
                
                print(f"  Main Dataset: {main_dataset}")
                print(f"    {'✓ Found' if main_dataset.exists() else '✗ Not Found'}")
                if main_dataset.exists():
                    size_mb = main_dataset.stat().st_size / (1024 * 1024)
                    print(f"    Size: {size_mb:.1f} MB")
                
                print(f"  Footnote Dataset: {footnote_dataset}")
                print(f"    {'✓ Found' if footnote_dataset.exists() else '✗ Not Found'}")
                
            except Exception as e:
                print(f"  ✗ Error checking data sources: {e}")
            
            # Report status
            print("\nReports:")
            try:
                enabled_reports = self.config_manager.get_enabled_reports()
                all_reports = self.config_manager.load_config('reports.yaml')
                
                print(f"  Total Available: {len(all_reports)}")
                print(f"  Enabled: {len(enabled_reports)}")
                print(f"  Disabled: {len(all_reports) - len(enabled_reports)}")
                
                if args.verbose:
                    print("  Enabled Reports:")
                    for report_type in enabled_reports:
                        print(f"    - {report_type}")
                
            except Exception as e:
                print(f"  ✗ Error checking reports: {e}")
            
            # System resources
            print("\nSystem Resources:")
            try:
                import psutil
                
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('.')
                
                print(f"  CPU Usage: {cpu_percent:.1f}%")
                print(f"  Memory Usage: {memory.percent:.1f}% ({memory.used / (1024**3):.1f} GB used)")
                print(f"  Disk Usage: {disk.percent:.1f}% ({disk.free / (1024**3):.1f} GB free)")
                
            except ImportError:
                print("  System monitoring not available (psutil not installed)")
            except Exception as e:
                print(f"  Error checking system resources: {e}")
            
            # Performance metrics (if available)
            print("\nPerformance Metrics:")
            try:
                monitor = get_performance_monitor()
                stats = monitor.get_all_operation_stats()
                
                if stats:
                    print(f"  Tracked Operations: {len(stats)}")
                    
                    if args.verbose:
                        for operation, stat in list(stats.items())[:5]:  # Show top 5
                            print(f"    {operation}: {stat['count']} calls, "
                                  f"avg {stat['average_duration']:.3f}s")
                else:
                    print("  No performance data available")
                    
            except Exception as e:
                print(f"  Performance monitoring not available: {e}")
            
            return 0
            
        except Exception as e:
            print(f"Error showing status: {e}", file=sys.stderr)
            return 1
    
    def create_config_template(self, args: argparse.Namespace) -> int:
        """
        Create configuration template files.
        
        Args:
            args: Command-line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            config_dir = Path(args.config_dir or "config")
            config_dir.mkdir(exist_ok=True)
            
            print(f"\nCreating configuration templates in: {config_dir}")
            
            # Create a new configuration manager to generate defaults
            template_manager = ConfigurationManager(str(config_dir))
            
            print("✓ Configuration templates created successfully")
            print("\nGenerated files:")
            
            for config_file in config_dir.glob("*.yaml"):
                print(f"  - {config_file}")
            
            print("\nNext steps:")
            print("1. Review and customize the configuration files")
            print("2. Ensure data source paths are correct")
            print("3. Enable/disable reports as needed")
            print("4. Run validation: python -m src.main --validate-config")
            
            return 0
            
        except Exception as e:
            print(f"Error creating configuration templates: {e}", file=sys.stderr)
            return 1
    
    def run_pipeline_test(self, args: argparse.Namespace) -> int:
        """
        Run a pipeline test with sample data.
        
        Args:
            args: Command-line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            print("\n" + "="*60)
            print("PIPELINE TEST")
            print("="*60)
            
            print("\nInitializing pipeline...")
            pipeline = DataProcessingPipeline(self.config_manager)
            
            # Check if data files exist
            data_config = self.config_manager.get_data_source_config()
            main_dataset = Path(data_config.main_dataset_path)
            
            if not main_dataset.exists():
                print(f"✗ Main dataset not found: {main_dataset}")
                print("Please ensure the data file exists before running the test.")
                return 1
            
            print(f"✓ Found dataset: {main_dataset}")
            
            # Run a minimal pipeline test
            print("\nRunning pipeline test...")
            
            try:
                # Test data loading stage only
                result = pipeline._stage_data_loading()
                
                print("✓ Data loading test passed")
                print(f"  Records loaded: {result.get('records_loaded', 0)}")
                print(f"  Columns found: {result.get('columns_found', 0)}")
                print(f"  Footnote codes: {result.get('footnote_codes_loaded', 0)}")
                
                if result.get('validation_warnings'):
                    print("  Warnings:")
                    for warning in result['validation_warnings']:
                        print(f"    - {warning}")
                
            except Exception as e:
                print(f"✗ Pipeline test failed: {e}")
                return 1
            
            print("\n✓ Pipeline test completed successfully")
            print("The system is ready to generate reports.")
            
            return 0
            
        except Exception as e:
            print(f"Pipeline test failed: {e}", file=sys.stderr)
            return 1
    
    def export_sample_config(self, args: argparse.Namespace) -> int:
        """
        Export a sample configuration for batch processing.
        
        Args:
            args: Command-line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            output_file = args.output or "sample_batch_config.json"
            
            # Create sample batch configuration
            sample_config = {
                "description": "Sample batch configuration for occupation data reports",
                "created": datetime.now().isoformat(),
                "jobs": [
                    {
                        "job_id": "occupation_dist",
                        "report_type": "occupation_distribution",
                        "filters": {
                            "top_n": 20
                        },
                        "config": {
                            "include_charts": True
                        },
                        "output_path": "reports/occupation_distribution",
                        "export_formats": ["html", "pdf", "csv"]
                    },
                    {
                        "job_id": "env_risk",
                        "report_type": "environmental_risk",
                        "filters": {
                            "risk_threshold": 0.7
                        },
                        "config": {
                            "detailed_analysis": True
                        },
                        "output_path": "reports/environmental_risk",
                        "export_formats": ["html", "pdf"]
                    },
                    {
                        "job_id": "data_quality",
                        "report_type": "data_quality",
                        "filters": {},
                        "config": {
                            "confidence_level": 0.95
                        },
                        "output_path": "reports/data_quality",
                        "export_formats": ["html", "csv"]
                    }
                ],
                "global_settings": {
                    "parallel_processing": True,
                    "max_workers": 4,
                    "output_base_dir": "batch_reports",
                    "timestamp_folders": True
                }
            }
            
            # Write to file
            with open(output_file, 'w') as f:
                json.dump(sample_config, f, indent=2)
            
            print(f"✓ Sample batch configuration exported to: {output_file}")
            print("\nUsage:")
            print(f"  python -m src.main --batch-config {output_file}")
            print("\nCustomization:")
            print("  - Modify job_id values for unique identification")
            print("  - Adjust filters and config parameters as needed")
            print("  - Change export_formats to control output types")
            print("  - Set parallel_processing to false for sequential execution")
            
            return 0
            
        except Exception as e:
            print(f"Error exporting sample configuration: {e}", file=sys.stderr)
            return 1


def add_cli_commands(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add CLI command subparsers to the main argument parser.
    
    Args:
        parser: Main argument parser
        
    Returns:
        Updated argument parser with subcommands
    """
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List reports command
    list_parser = subparsers.add_parser(
        'list-reports',
        help='List all available report types'
    )
    list_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information'
    )
    
    # Validate config command
    validate_parser = subparsers.add_parser(
        'validate-config',
        help='Validate configuration files'
    )
    validate_parser.add_argument(
        '--config-dir',
        type=str,
        default='config',
        help='Configuration directory path'
    )
    
    # Status command
    status_parser = subparsers.add_parser(
        'status',
        help='Show application and system status'
    )
    status_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed status information'
    )
    
    # Create config template command
    template_parser = subparsers.add_parser(
        'create-config',
        help='Create configuration template files'
    )
    template_parser.add_argument(
        '--config-dir',
        type=str,
        help='Directory to create configuration files in'
    )
    
    # Pipeline test command
    test_parser = subparsers.add_parser(
        'test-pipeline',
        help='Run a pipeline test with current configuration'
    )
    test_parser.add_argument(
        '--config-dir',
        type=str,
        default='config',
        help='Configuration directory path'
    )
    
    # Export sample config command
    export_parser = subparsers.add_parser(
        'export-sample-config',
        help='Export a sample batch configuration file'
    )
    export_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path for sample configuration'
    )
    
    return parser


def handle_cli_command(args: argparse.Namespace, config_manager: ConfigurationManager) -> int:
    """
    Handle CLI command execution.
    
    Args:
        args: Parsed command-line arguments
        config_manager: Configuration manager instance
        
    Returns:
        Exit code
    """
    cli_commands = CLICommands(config_manager)
    
    command_handlers = {
        'list-reports': cli_commands.list_reports,
        'validate-config': cli_commands.validate_config,
        'status': cli_commands.show_status,
        'create-config': cli_commands.create_config_template,
        'test-pipeline': cli_commands.run_pipeline_test,
        'export-sample-config': cli_commands.export_sample_config
    }
    
    if args.command in command_handlers:
        return command_handlers[args.command](args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1