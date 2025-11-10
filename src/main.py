"""
Main application entry point for the Occupation Data Reports system.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from .config.settings import ConfigurationManager
from .interfaces import ConfigurationError
from .integration.pipeline import DataProcessingPipeline
from .integration.monitoring import initialize_monitoring, get_performance_monitor
from .reports.report_factory import ReportFactory
from .reports.batch_processor import BatchProcessor
from .cli.commands import add_cli_commands, handle_cli_command


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Set up logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Occupation Data Reports - Analyze occupational requirements survey data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --report-type occupation_distribution
  %(prog)s --report-type environmental_risk --output-dir ./custom_reports
  %(prog)s --generate-all --config-dir ./my_config
  %(prog)s --list-reports
  %(prog)s --validate-config
        """
    )
    
    # Main operation arguments
    parser.add_argument(
        "--report-type",
        type=str,
        help="Type of report to generate (use --list-reports to see available types)"
    )
    
    parser.add_argument(
        "--generate-all",
        action="store_true",
        help="Generate all enabled report types"
    )
    
    parser.add_argument(
        "--list-reports",
        action="store_true",
        help="List all available report types and their descriptions"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Directory containing configuration files (default: config)"
    )
    
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration files and exit"
    )
    
    # Data source arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing input data files (default: current directory)"
    )
    
    parser.add_argument(
        "--dataset-file",
        type=str,
        help="Path to main dataset CSV file (overrides config)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for generated reports (overrides config)"
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["html", "pdf", "csv", "all"],
        default="all",
        help="Output format for reports (default: all)"
    )
    
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Don't create timestamped subdirectories for output"
    )
    
    # Processing arguments
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing for multiple reports"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of worker processes for parallel execution (default: 4)"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: log to console only)"
    )
    
    # Development and debugging arguments
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually generating reports"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (equivalent to --log-level DEBUG)"
    )
    
    # Batch processing arguments
    parser.add_argument(
        "--batch-config",
        type=str,
        help="Path to batch configuration JSON file"
    )
    
    parser.add_argument(
        "--template",
        type=str,
        choices=["comprehensive", "safety_focused", "workforce_analysis", "statistical_analysis", "basic"],
        help="Use a predefined batch template"
    )
    
    # Add CLI subcommands
    parser = add_cli_commands(parser)
    
    return parser


class OccupationDataReportsApp:
    """Main application class for the Occupation Data Reports system."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_manager = ConfigurationManager(config_dir)
        self.logger = logging.getLogger(__name__)
    
    def list_available_reports(self) -> None:
        """List all available report types and their descriptions."""
        try:
            reports_config = self.config_manager.load_config("reports.yaml")
            
            print("\nAvailable Report Types:")
            print("=" * 50)
            
            for report_type, config in reports_config.items():
                status = "✓ Enabled" if config.get("enabled", True) else "✗ Disabled"
                print(f"\n{report_type}")
                print(f"  Title: {config.get('title', 'N/A')}")
                print(f"  Description: {config.get('description', 'N/A')}")
                print(f"  Status: {status}")
                
                if config.get("custom_parameters"):
                    print(f"  Parameters: {config['custom_parameters']}")
            
            print(f"\nTotal: {len(reports_config)} report types available")
            
        except Exception as e:
            self.logger.error(f"Failed to list reports: {e}")
            sys.exit(1)
    
    def validate_configuration(self) -> bool:
        """Validate all configuration files."""
        try:
            config_files = ["data_sources.yaml", "output.yaml", "visualization.yaml", 
                          "analysis.yaml", "reports.yaml"]
            
            all_valid = True
            
            for config_file in config_files:
                try:
                    config_data = self.config_manager.load_config(config_file)
                    validation_result = self.config_manager.validate_config({config_file.split('.')[0]: config_data})
                    
                    print(f"\n{config_file}:")
                    if validation_result.is_valid:
                        print("  ✓ Valid")
                    else:
                        print("  ✗ Invalid")
                        all_valid = False
                    
                    if validation_result.errors:
                        print("  Errors:")
                        for error in validation_result.errors:
                            print(f"    - {error}")
                    
                    if validation_result.warnings:
                        print("  Warnings:")
                        for warning in validation_result.warnings:
                            print(f"    - {warning}")
                            
                except Exception as e:
                    print(f"  ✗ Failed to load: {e}")
                    all_valid = False
            
            print(f"\nOverall configuration status: {'✓ Valid' if all_valid else '✗ Invalid'}")
            return all_valid
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def generate_report(self, report_type: str, **kwargs) -> bool:
        """Generate a specific report type."""
        try:
            self.logger.info(f"Generating report: {report_type}")
            
            # Validate report type exists
            report_config = self.config_manager.get_report_config(report_type)
            
            if not report_config.get("enabled", True):
                self.logger.warning(f"Report type '{report_type}' is disabled in configuration")
                return False
            
            self.logger.info(f"Report configuration loaded: {report_config['title']}")
            
            # Initialize pipeline
            pipeline = DataProcessingPipeline(self.config_manager)
            
            # Get configuration parameters
            dataset_path = kwargs.get('dataset_file')
            output_dir = kwargs.get('output_dir')
            export_formats = kwargs.get('output_format', 'all')
            
            if export_formats == 'all':
                export_formats = ['html', 'pdf', 'csv']
            else:
                export_formats = [export_formats]
            
            # Run pipeline for single report
            result = pipeline.run_pipeline(
                dataset_path=dataset_path,
                output_dir=output_dir,
                report_types=[report_type],
                export_formats=export_formats
            )
            
            if result['success']:
                self.logger.info(f"Successfully generated {report_type} report")
                return True
            else:
                self.logger.error(f"Failed to generate {report_type} report: {result.get('error', 'Unknown error')}")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to generate report '{report_type}': {e}")
            return False
    
    def generate_all_reports(self, **kwargs) -> bool:
        """Generate all enabled reports."""
        try:
            enabled_reports = self.config_manager.get_enabled_reports()
            
            if not enabled_reports:
                self.logger.warning("No reports are enabled in configuration")
                return False
            
            self.logger.info(f"Generating {len(enabled_reports)} enabled reports")
            
            # Initialize pipeline
            pipeline = DataProcessingPipeline(self.config_manager)
            
            # Get configuration parameters
            dataset_path = kwargs.get('dataset_file')
            output_dir = kwargs.get('output_dir')
            export_formats = kwargs.get('output_format', 'all')
            parallel = kwargs.get('parallel', False)
            
            if export_formats == 'all':
                export_formats = ['html', 'pdf', 'csv']
            else:
                export_formats = [export_formats]
            
            # Use batch processing if parallel is enabled and multiple reports
            if parallel and len(enabled_reports) > 1:
                self.logger.info("Using parallel batch processing for multiple reports")
                
                # Create batch processor
                batch_processor = BatchProcessor(
                    max_workers=kwargs.get('max_workers', 4)
                )
                
                # Add all reports to batch
                for report_type in enabled_reports:
                    batch_processor.add_job(
                        job_id=report_type,
                        report_type=report_type,
                        data=None,  # Will be loaded by pipeline
                        output_path=output_dir,
                        export_formats=export_formats
                    )
                
                # Process batch
                batch_results = batch_processor.process_batch(parallel=True)
                success_count = batch_results.get('completed', 0)
                
            else:
                # Run pipeline for all reports
                result = pipeline.run_pipeline(
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    report_types=enabled_reports,
                    export_formats=export_formats
                )
                
                if result['success']:
                    success_count = result['reports_generated']
                else:
                    self.logger.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")
                    return False
            
            self.logger.info(f"Successfully generated {success_count}/{len(enabled_reports)} reports")
            return success_count == len(enabled_reports)
            
        except Exception as e:
            self.logger.error(f"Failed to generate all reports: {e}")
            return False
    
    def run(self, args: argparse.Namespace) -> int:
        """Run the application with parsed command-line arguments."""
        try:
            # Handle CLI subcommands first
            if hasattr(args, 'command') and args.command:
                return handle_cli_command(args, self.config_manager)
            
            # Handle legacy special operations
            if args.list_reports:
                self.list_available_reports()
                return 0
            
            if args.validate_config:
                is_valid = self.validate_configuration()
                return 0 if is_valid else 1
            
            # Handle batch processing
            if args.batch_config:
                return self.run_batch_from_config(args.batch_config, args)
            
            if args.template:
                return self.run_batch_from_template(args.template, args)
            
            # Validate that we have something to do
            if not args.report_type and not args.generate_all:
                self.logger.error("Must specify either --report-type, --generate-all, --batch-config, or --template")
                return 1
            
            # Handle dry run
            if args.dry_run:
                self.logger.info("DRY RUN MODE - No reports will be generated")
                if args.report_type:
                    self.logger.info(f"Would generate report: {args.report_type}")
                elif args.generate_all:
                    enabled_reports = self.config_manager.get_enabled_reports()
                    self.logger.info(f"Would generate {len(enabled_reports)} reports: {', '.join(enabled_reports)}")
                return 0
            
            # Generate reports
            success = False
            kwargs = {
                'dataset_file': args.dataset_file,
                'output_dir': args.output_dir,
                'output_format': args.output_format,
                'parallel': args.parallel,
                'max_workers': args.max_workers
            }
            
            if args.report_type:
                success = self.generate_report(args.report_type, **kwargs)
            elif args.generate_all:
                success = self.generate_all_reports(**kwargs)
            
            return 0 if success else 1
            
        except KeyboardInterrupt:
            self.logger.info("Operation cancelled by user")
            return 130
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            return 1
    
    def run_batch_from_config(self, config_path: str, args: argparse.Namespace) -> int:
        """
        Run batch processing from a configuration file.
        
        Args:
            config_path: Path to batch configuration JSON file
            args: Command-line arguments
            
        Returns:
            Exit code
        """
        try:
            import json
            
            self.logger.info(f"Loading batch configuration from: {config_path}")
            
            with open(config_path, 'r') as f:
                batch_config = json.load(f)
            
            # Initialize pipeline
            pipeline = DataProcessingPipeline(self.config_manager)
            
            # Create batch processor
            global_settings = batch_config.get('global_settings', {})
            max_workers = global_settings.get('max_workers', args.max_workers)
            
            batch_processor = BatchProcessor(max_workers=max_workers)
            
            # Create batch from configuration
            success = batch_processor.create_batch_from_config(batch_config, None)
            
            if not success:
                self.logger.error("Failed to create batch from configuration")
                return 1
            
            # Process batch
            parallel = global_settings.get('parallel_processing', args.parallel)
            results = batch_processor.process_batch(parallel=parallel)
            
            # Report results
            self.logger.info(f"Batch processing completed: {results['completed']} successful, "
                           f"{results['failed']} failed, {results['cancelled']} cancelled")
            
            return 0 if results['failed'] == 0 else 1
            
        except Exception as e:
            self.logger.error(f"Batch processing from config failed: {e}")
            return 1
    
    def run_batch_from_template(self, template_name: str, args: argparse.Namespace) -> int:
        """
        Run batch processing from a predefined template.
        
        Args:
            template_name: Name of the batch template
            args: Command-line arguments
            
        Returns:
            Exit code
        """
        try:
            self.logger.info(f"Running batch template: {template_name}")
            
            # Initialize pipeline to load data
            pipeline = DataProcessingPipeline(self.config_manager)
            
            # Load data first
            data_config = self.config_manager.get_data_source_config()
            dataset_path = args.dataset_file or data_config.main_dataset_path
            
            # Create report factory and batch processor from template
            batch_processor = self.report_factory.create_batch_from_template(template_name, None)
            
            if not batch_processor:
                self.logger.error(f"Failed to create batch from template: {template_name}")
                return 1
            
            # Process batch
            results = batch_processor.process_batch(parallel=args.parallel)
            
            # Report results
            self.logger.info(f"Template batch processing completed: {results['completed']} successful, "
                           f"{results['failed']} failed, {results['cancelled']} cancelled")
            
            return 0 if results['failed'] == 0 else 1
            
        except Exception as e:
            self.logger.error(f"Batch processing from template failed: {e}")
            return 1


def main() -> int:
    """Main entry point for the application."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else args.log_level
    setup_logging(log_level, args.log_file)
    
    # Initialize monitoring system
    try:
        performance_monitor, logging_manager = initialize_monitoring(
            log_dir="logs",
            app_name="occupation_reports",
            start_monitoring=True
        )
        
        # Record application start
        performance_monitor.record_metric(
            name="application_start",
            value=1,
            unit="event",
            category="lifecycle"
        )
        
    except Exception as e:
        logging.warning(f"Failed to initialize monitoring: {e}")
    
    # Create and run the application
    try:
        app = OccupationDataReportsApp(args.config_dir)
        
        # Start application timer
        app_timer = get_performance_monitor().start_timer("application_runtime")
        
        result = app.run(args)
        
        # Stop application timer
        runtime = get_performance_monitor().stop_timer(app_timer)
        
        # Record application completion
        get_performance_monitor().record_metric(
            name="application_completion",
            value=runtime,
            unit="seconds",
            category="lifecycle",
            metadata={"exit_code": result}
        )
        
        return result
        
    except ConfigurationError as e:
        logging.error(f"Configuration error: {e}")
        get_performance_monitor().record_error(e, {"error_type": "configuration"})
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        get_performance_monitor().record_error(e, {"error_type": "unexpected"})
        return 1
    finally:
        # Stop monitoring and export metrics
        try:
            get_performance_monitor().stop_monitoring()
            
            # Export performance metrics
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_file = f"logs/performance_metrics_{timestamp}.json"
            get_performance_monitor().export_metrics(metrics_file)
            
        except Exception as e:
            logging.warning(f"Failed to export performance metrics: {e}")


if __name__ == "__main__":
    sys.exit(main())