"""
End-to-end data processing pipeline for occupation data reports.
Integrates all components with comprehensive error handling and monitoring.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from ..data_processing.csv_loader import CSVLoader
from ..data_processing.footnote_processor import FootnoteProcessor
from ..data_processing.data_cleaner import DataCleaner
from ..data_processing.data_processor import DataProcessor
from ..analysis.statistical_analyzer import StatisticalAnalyzer
from ..analysis.occupation_analyzer import OccupationAnalyzer
from ..analysis.correlation_analyzer import CorrelationAnalyzer
from ..analysis.environmental_risk_analyzer import EnvironmentalRiskAnalyzer
from ..analysis.physical_demands_analyzer import PhysicalDemandsAnalyzer
from ..analysis.cognitive_requirements_analyzer import CognitiveRequirementsAnalyzer
from ..analysis.additive_category_analyzer import AdditiveCategoryAnalyzer
from ..analysis.quality_assessor import QualityAssessor
from ..analysis.establishment_analyzer import EstablishmentAnalyzer
from ..visualization.chart_generator import ChartGenerator
from ..visualization.heatmap_generator import HeatmapGenerator
from ..visualization.confidence_interval_plotter import ConfidenceIntervalPlotter
from ..visualization.dashboard_builder import DashboardBuilder
from ..export.export_manager import ExportManager
from ..export.html_exporter import HTMLExporter
from ..export.pdf_exporter import PDFExporter
from ..export.csv_exporter import CSVExporter
from ..reports.report_factory import ReportFactory
from ..reports.batch_processor import BatchProcessor
from ..config.settings import ConfigurationManager
from ..interfaces import (
    ReportData, ValidationResult, DataProcessingError, 
    VisualizationError, ExportError, ConfigurationError
)


class PipelineStage:
    """Represents a stage in the data processing pipeline."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status: str = "pending"
        self.error_message: Optional[str] = None
        self.metrics: Dict[str, Any] = {}
    
    def start(self):
        """Mark stage as started."""
        self.start_time = datetime.now()
        self.status = "running"
    
    def complete(self, metrics: Optional[Dict[str, Any]] = None):
        """Mark stage as completed."""
        self.end_time = datetime.now()
        self.status = "completed"
        if metrics:
            self.metrics.update(metrics)
    
    def fail(self, error_message: str):
        """Mark stage as failed."""
        self.end_time = datetime.now()
        self.status = "failed"
        self.error_message = error_message
    
    @property
    def duration(self) -> Optional[float]:
        """Get stage duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class DataProcessingPipeline:
    """
    End-to-end data processing pipeline that integrates all components
    with comprehensive error handling, recovery mechanisms, and monitoring.
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the data processing pipeline.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.logger = logging.getLogger(__name__)
        
        # Pipeline stages
        self.stages: List[PipelineStage] = []
        self.current_stage: Optional[PipelineStage] = None
        
        # Component instances
        self.csv_loader: Optional[CSVLoader] = None
        self.footnote_processor: Optional[FootnoteProcessor] = None
        self.data_cleaner: Optional[DataCleaner] = None
        self.data_processor: Optional[DataProcessor] = None
        self.analyzers: Dict[str, Any] = {}
        self.visualizers: Dict[str, Any] = {}
        self.exporters: Dict[str, Any] = {}
        self.report_factory: Optional[ReportFactory] = None
        
        # Pipeline data
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.footnote_data: Optional[Dict[int, Any]] = None
        self.analysis_results: Dict[str, Any] = {}
        self.reports: Dict[str, ReportData] = {}
        
        # Error handling and recovery
        self.error_recovery_enabled = True
        self.max_retry_attempts = 3
        self.retry_delay = 1.0  # seconds
        
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize pipeline stages and components."""
        try:
            # Define pipeline stages
            self.stages = [
                PipelineStage("initialization", "Initialize components and validate configuration"),
                PipelineStage("data_loading", "Load CSV data files and validate structure"),
                PipelineStage("footnote_processing", "Process footnote codes and interpretations"),
                PipelineStage("data_cleaning", "Clean and standardize data"),
                PipelineStage("data_validation", "Validate processed data quality"),
                PipelineStage("statistical_analysis", "Perform statistical calculations"),
                PipelineStage("specialized_analysis", "Run specialized analysis modules"),
                PipelineStage("visualization", "Generate charts and visualizations"),
                PipelineStage("report_generation", "Generate structured reports"),
                PipelineStage("export", "Export reports in multiple formats"),
                PipelineStage("finalization", "Finalize pipeline and cleanup")
            ]
            
            # Initialize components
            self._initialize_components()
            
            self.logger.info("Pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Data processing components
            self.csv_loader = CSVLoader()
            self.footnote_processor = FootnoteProcessor()
            self.data_cleaner = DataCleaner()
            self.data_processor = DataProcessor()
            
            # Analysis components
            self.analyzers = {
                'statistical': StatisticalAnalyzer(),
                'occupation': OccupationAnalyzer(),
                'correlation': CorrelationAnalyzer(),
                'environmental_risk': EnvironmentalRiskAnalyzer(),
                'physical_demands': PhysicalDemandsAnalyzer(),
                'cognitive_requirements': CognitiveRequirementsAnalyzer(),
                'additive_category': AdditiveCategoryAnalyzer(),
                'quality': QualityAssessor(),
                'establishment': EstablishmentAnalyzer()
            }
            
            # Visualization components
            self.visualizers = {
                'chart': ChartGenerator(),
                'heatmap': HeatmapGenerator(),
                'confidence_interval': ConfidenceIntervalPlotter(),
                'dashboard': DashboardBuilder()
            }
            
            # Export components
            self.exporters = {
                'html': HTMLExporter(),
                'pdf': PDFExporter(),
                'csv': CSVExporter(),
                'manager': ExportManager()
            }
            
            # Report factory
            self.report_factory = ReportFactory(self.config_manager)
            
            self.logger.debug("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    def run_pipeline(self, 
                    dataset_path: Optional[str] = None,
                    footnote_path: Optional[str] = None,
                    output_dir: Optional[str] = None,
                    report_types: Optional[List[str]] = None,
                    export_formats: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the complete data processing pipeline.
        
        Args:
            dataset_path: Path to main dataset CSV file
            footnote_path: Path to footnote codes CSV file
            output_dir: Output directory for reports
            report_types: List of report types to generate
            export_formats: List of export formats
            
        Returns:
            Dictionary containing pipeline results and metrics
        """
        pipeline_start_time = datetime.now()
        
        try:
            self.logger.info("Starting end-to-end data processing pipeline")
            
            # Stage 1: Initialization
            self._run_stage("initialization", self._stage_initialization)
            
            # Stage 2: Data Loading
            self._run_stage("data_loading", self._stage_data_loading, 
                          dataset_path, footnote_path)
            
            # Stage 3: Footnote Processing
            self._run_stage("footnote_processing", self._stage_footnote_processing)
            
            # Stage 4: Data Cleaning
            self._run_stage("data_cleaning", self._stage_data_cleaning)
            
            # Stage 5: Data Validation
            self._run_stage("data_validation", self._stage_data_validation)
            
            # Stage 6: Statistical Analysis
            self._run_stage("statistical_analysis", self._stage_statistical_analysis)
            
            # Stage 7: Specialized Analysis
            self._run_stage("specialized_analysis", self._stage_specialized_analysis)
            
            # Stage 8: Visualization
            self._run_stage("visualization", self._stage_visualization)
            
            # Stage 9: Report Generation
            self._run_stage("report_generation", self._stage_report_generation,
                          report_types)
            
            # Stage 10: Export
            self._run_stage("export", self._stage_export, output_dir, export_formats)
            
            # Stage 11: Finalization
            self._run_stage("finalization", self._stage_finalization)
            
            # Calculate final metrics
            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            
            results = {
                'success': True,
                'pipeline_duration': pipeline_duration,
                'stages_completed': len([s for s in self.stages if s.status == "completed"]),
                'total_stages': len(self.stages),
                'data_records_processed': len(self.processed_data) if self.processed_data is not None else 0,
                'reports_generated': len(self.reports),
                'analysis_results': len(self.analysis_results),
                'stage_details': [
                    {
                        'name': stage.name,
                        'status': stage.status,
                        'duration': stage.duration,
                        'metrics': stage.metrics,
                        'error': stage.error_message
                    }
                    for stage in self.stages
                ]
            }
            
            self.logger.info(f"Pipeline completed successfully in {pipeline_duration:.2f} seconds")
            return results
            
        except Exception as e:
            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            
            self.logger.error(f"Pipeline failed after {pipeline_duration:.2f} seconds: {str(e)}")
            
            return {
                'success': False,
                'error': str(e),
                'pipeline_duration': pipeline_duration,
                'stages_completed': len([s for s in self.stages if s.status == "completed"]),
                'total_stages': len(self.stages),
                'stage_details': [
                    {
                        'name': stage.name,
                        'status': stage.status,
                        'duration': stage.duration,
                        'metrics': stage.metrics,
                        'error': stage.error_message
                    }
                    for stage in self.stages
                ]
            }
    
    def _run_stage(self, stage_name: str, stage_function, *args, **kwargs):
        """
        Run a pipeline stage with error handling and recovery.
        
        Args:
            stage_name: Name of the stage to run
            stage_function: Function to execute for the stage
            *args: Arguments to pass to the stage function
            **kwargs: Keyword arguments to pass to the stage function
        """
        stage = next((s for s in self.stages if s.name == stage_name), None)
        if not stage:
            raise ValueError(f"Unknown pipeline stage: {stage_name}")
        
        self.current_stage = stage
        stage.start()
        
        self.logger.info(f"Starting stage: {stage.name} - {stage.description}")
        
        attempt = 1
        while attempt <= self.max_retry_attempts:
            try:
                result = stage_function(*args, **kwargs)
                stage.complete(result if isinstance(result, dict) else {})
                self.logger.info(f"Completed stage: {stage.name}")
                return result
                
            except Exception as e:
                self.logger.error(f"Stage {stage.name} failed (attempt {attempt}): {str(e)}")
                
                if attempt < self.max_retry_attempts and self.error_recovery_enabled:
                    self.logger.info(f"Retrying stage {stage.name} in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    attempt += 1
                else:
                    stage.fail(str(e))
                    raise
    
    def _stage_initialization(self) -> Dict[str, Any]:
        """Initialize pipeline components and validate configuration."""
        try:
            # Validate configuration
            config_validation = self.config_manager.validate_config({
                'data_sources': self.config_manager.load_config('data_sources.yaml'),
                'analysis': self.config_manager.load_config('analysis.yaml'),
                'visualization': self.config_manager.load_config('visualization.yaml'),
                'output': self.config_manager.load_config('output.yaml')
            })
            
            if not config_validation.is_valid:
                raise ConfigurationError(f"Configuration validation failed: {config_validation.errors}")
            
            # Test component initialization
            component_count = (
                len(self.analyzers) + 
                len(self.visualizers) + 
                len(self.exporters) + 
                4  # data processing components
            )
            
            return {
                'config_valid': True,
                'components_initialized': component_count,
                'config_warnings': config_validation.warnings
            }
            
        except Exception as e:
            raise ConfigurationError(f"Initialization failed: {str(e)}")
    
    def _stage_data_loading(self, dataset_path: Optional[str] = None, 
                           footnote_path: Optional[str] = None) -> Dict[str, Any]:
        """Load CSV data files and validate structure."""
        try:
            # Get data source configuration
            data_config = self.config_manager.get_data_source_config()
            
            # Use provided paths or fall back to configuration
            main_path = dataset_path or data_config.main_dataset_path
            footnote_path = footnote_path or data_config.footnote_dataset_path
            
            # Load main dataset
            self.logger.info(f"Loading main dataset from: {main_path}")
            self.raw_data = self.csv_loader.load_dataset(main_path)
            
            # Validate dataset structure
            validation_result = self.csv_loader.validate_columns(self.raw_data)
            if not validation_result.is_valid:
                raise DataProcessingError(f"Dataset validation failed: {validation_result.errors}")
            
            # Load footnote data
            self.logger.info(f"Loading footnote data from: {footnote_path}")
            self.footnote_data = self.csv_loader.load_footnotes(footnote_path)
            
            return {
                'records_loaded': len(self.raw_data),
                'columns_found': len(self.raw_data.columns),
                'footnote_codes_loaded': len(self.footnote_data),
                'validation_warnings': validation_result.warnings
            }
            
        except Exception as e:
            raise DataProcessingError(f"Data loading failed: {str(e)}")
    
    def _stage_footnote_processing(self) -> Dict[str, Any]:
        """Process footnote codes and interpretations."""
        try:
            if self.raw_data is None or self.footnote_data is None:
                raise DataProcessingError("Raw data or footnote data not loaded")
            
            # Process footnotes in the dataset
            processed_count = self.footnote_processor.process_footnotes(
                self.raw_data, self.footnote_data
            )
            
            # Interpret range estimates
            range_interpretations = self.footnote_processor.interpret_range_estimates(self.raw_data)
            
            return {
                'footnotes_processed': processed_count,
                'range_estimates_interpreted': len(range_interpretations),
                'footnote_coverage': processed_count / len(self.raw_data) * 100
            }
            
        except Exception as e:
            raise DataProcessingError(f"Footnote processing failed: {str(e)}")
    
    def _stage_data_cleaning(self) -> Dict[str, Any]:
        """Clean and standardize data."""
        try:
            if self.raw_data is None:
                raise DataProcessingError("Raw data not available for cleaning")
            
            # Clean occupation names
            occupation_cleaning_stats = self.data_cleaner.clean_occupation_names(self.raw_data)
            
            # Standardize estimates
            estimate_cleaning_stats = self.data_cleaner.standardize_estimates(self.raw_data)
            
            # Handle missing data
            missing_data_stats = self.data_cleaner.handle_missing_data(self.raw_data)
            
            # Create processed data copy
            self.processed_data = self.raw_data.copy()
            
            return {
                'occupation_names_cleaned': occupation_cleaning_stats.get('cleaned_count', 0),
                'estimates_standardized': estimate_cleaning_stats.get('standardized_count', 0),
                'missing_data_handled': missing_data_stats.get('handled_count', 0),
                'data_quality_score': self._calculate_data_quality_score()
            }
            
        except Exception as e:
            raise DataProcessingError(f"Data cleaning failed: {str(e)}")
    
    def _stage_data_validation(self) -> Dict[str, Any]:
        """Validate processed data quality."""
        try:
            if self.processed_data is None:
                raise DataProcessingError("Processed data not available for validation")
            
            # Comprehensive data validation
            validation_result = self.data_processor.validate_processed_data(self.processed_data)
            
            if not validation_result.is_valid:
                if self.error_recovery_enabled:
                    self.logger.warning("Data validation issues found, attempting recovery...")
                    # Attempt data recovery/correction
                    self.processed_data = self.data_processor.attempt_data_recovery(self.processed_data)
                else:
                    raise DataProcessingError(f"Data validation failed: {validation_result.errors}")
            
            return {
                'validation_passed': validation_result.is_valid,
                'records_validated': validation_result.record_count,
                'validation_errors': len(validation_result.errors),
                'validation_warnings': len(validation_result.warnings)
            }
            
        except Exception as e:
            raise DataProcessingError(f"Data validation failed: {str(e)}")
    
    def _stage_statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical calculations."""
        try:
            if self.processed_data is None:
                raise DataProcessingError("Processed data not available for analysis")
            
            # Run statistical analysis
            statistical_results = self.analyzers['statistical'].analyze_dataset(self.processed_data)
            self.analysis_results['statistical'] = statistical_results
            
            # Calculate confidence intervals
            confidence_intervals = self.analyzers['statistical'].calculate_confidence_intervals(
                self.processed_data
            )
            self.analysis_results['confidence_intervals'] = confidence_intervals
            
            return {
                'statistical_metrics_calculated': len(statistical_results),
                'confidence_intervals_calculated': len(confidence_intervals),
                'analysis_coverage': self._calculate_analysis_coverage()
            }
            
        except Exception as e:
            raise DataProcessingError(f"Statistical analysis failed: {str(e)}")
    
    def _stage_specialized_analysis(self) -> Dict[str, Any]:
        """Run specialized analysis modules."""
        try:
            if self.processed_data is None:
                raise DataProcessingError("Processed data not available for specialized analysis")
            
            analysis_count = 0
            
            # Run each specialized analyzer
            for analyzer_name, analyzer in self.analyzers.items():
                if analyzer_name == 'statistical':  # Skip, already done
                    continue
                
                try:
                    self.logger.debug(f"Running {analyzer_name} analysis")
                    results = analyzer.analyze(self.processed_data)
                    self.analysis_results[analyzer_name] = results
                    analysis_count += 1
                except Exception as e:
                    self.logger.warning(f"Specialized analysis {analyzer_name} failed: {str(e)}")
                    if not self.error_recovery_enabled:
                        raise
            
            return {
                'specialized_analyses_completed': analysis_count,
                'total_analysis_results': len(self.analysis_results)
            }
            
        except Exception as e:
            raise DataProcessingError(f"Specialized analysis failed: {str(e)}")
    
    def _stage_visualization(self) -> Dict[str, Any]:
        """Generate charts and visualizations."""
        try:
            if not self.analysis_results:
                raise VisualizationError("No analysis results available for visualization")
            
            visualization_count = 0
            
            # Generate visualizations for each analysis result
            for analysis_type, results in self.analysis_results.items():
                try:
                    # Choose appropriate visualizer based on analysis type
                    if 'correlation' in analysis_type:
                        visualizer = self.visualizers['heatmap']
                    elif 'confidence' in analysis_type:
                        visualizer = self.visualizers['confidence_interval']
                    else:
                        visualizer = self.visualizers['chart']
                    
                    # Generate visualization
                    viz_result = visualizer.create_visualization(results)
                    if viz_result:
                        visualization_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Visualization for {analysis_type} failed: {str(e)}")
                    if not self.error_recovery_enabled:
                        raise
            
            return {
                'visualizations_created': visualization_count,
                'visualization_types': list(self.visualizers.keys())
            }
            
        except Exception as e:
            raise VisualizationError(f"Visualization generation failed: {str(e)}")
    
    def _stage_report_generation(self, report_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate structured reports."""
        try:
            if self.processed_data is None:
                raise DataProcessingError("Processed data not available for report generation")
            
            # Determine which reports to generate
            if report_types is None:
                report_types = self.config_manager.get_enabled_reports()
            
            reports_generated = 0
            
            # Generate each requested report
            for report_type in report_types:
                try:
                    self.logger.debug(f"Generating {report_type} report")
                    report_data = self.report_factory.generate_report(
                        report_type, self.processed_data
                    )
                    
                    if report_data:
                        self.reports[report_type] = report_data
                        reports_generated += 1
                        
                except Exception as e:
                    self.logger.warning(f"Report generation for {report_type} failed: {str(e)}")
                    if not self.error_recovery_enabled:
                        raise
            
            return {
                'reports_generated': reports_generated,
                'report_types': list(self.reports.keys()),
                'total_analysis_results': sum(len(r.analysis_results) for r in self.reports.values())
            }
            
        except Exception as e:
            raise DataProcessingError(f"Report generation failed: {str(e)}")
    
    def _stage_export(self, output_dir: Optional[str] = None, 
                     export_formats: Optional[List[str]] = None) -> Dict[str, Any]:
        """Export reports in multiple formats."""
        try:
            if not self.reports:
                raise ExportError("No reports available for export")
            
            # Get output configuration
            output_config = self.config_manager.get_output_settings()
            output_dir = output_dir or output_config.get('base_output_dir', 'reports')
            export_formats = export_formats or ['html', 'pdf', 'csv']
            
            # Create output directory
            output_path = Path(output_dir)
            if output_config.get('timestamp_folders', True):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = output_path / timestamp
            
            output_path.mkdir(parents=True, exist_ok=True)
            
            exports_completed = 0
            export_results = {}
            
            # Export each report in each format
            for report_type, report_data in self.reports.items():
                export_results[report_type] = {}
                
                for format_type in export_formats:
                    try:
                        exporter = self.exporters.get(format_type)
                        if not exporter:
                            self.logger.warning(f"No exporter available for format: {format_type}")
                            continue
                        
                        output_file = output_path / f"{report_type}.{format_type}"
                        success = exporter.export_report(report_data, str(output_file))
                        
                        export_results[report_type][format_type] = {
                            'success': success,
                            'file_path': str(output_file) if success else None
                        }
                        
                        if success:
                            exports_completed += 1
                            
                    except Exception as e:
                        self.logger.warning(f"Export of {report_type} to {format_type} failed: {str(e)}")
                        export_results[report_type][format_type] = {
                            'success': False,
                            'error': str(e)
                        }
                        
                        if not self.error_recovery_enabled:
                            raise
            
            # Create master dashboard
            try:
                dashboard_path = output_path / "index.html"
                self.visualizers['dashboard'].create_master_dashboard(
                    list(self.reports.values()), str(dashboard_path)
                )
            except Exception as e:
                self.logger.warning(f"Master dashboard creation failed: {str(e)}")
            
            return {
                'exports_completed': exports_completed,
                'output_directory': str(output_path),
                'export_results': export_results,
                'formats_used': export_formats
            }
            
        except Exception as e:
            raise ExportError(f"Export failed: {str(e)}")
    
    def _stage_finalization(self) -> Dict[str, Any]:
        """Finalize pipeline and cleanup."""
        try:
            # Calculate final metrics
            total_processing_time = sum(
                stage.duration for stage in self.stages 
                if stage.duration is not None
            )
            
            # Cleanup temporary data if needed
            cleanup_count = 0
            
            # Log final summary
            self.logger.info("Pipeline finalization completed")
            
            return {
                'total_processing_time': total_processing_time,
                'cleanup_operations': cleanup_count,
                'pipeline_success': True
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline finalization failed: {str(e)}")
            return {
                'pipeline_success': False,
                'finalization_error': str(e)
            }
    
    def _calculate_data_quality_score(self) -> float:
        """Calculate overall data quality score."""
        if self.processed_data is None:
            return 0.0
        
        # Simple quality score based on completeness and validity
        total_cells = self.processed_data.size
        missing_cells = self.processed_data.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells
        
        return completeness * 100
    
    def _calculate_analysis_coverage(self) -> float:
        """Calculate analysis coverage percentage."""
        if not self.analysis_results or self.processed_data is None:
            return 0.0
        
        # Calculate based on number of analysis types completed
        total_analyzers = len(self.analyzers)
        completed_analyses = len(self.analysis_results)
        
        return (completed_analyses / total_analyzers) * 100
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics."""
        return {
            'current_stage': self.current_stage.name if self.current_stage else None,
            'stages': [
                {
                    'name': stage.name,
                    'status': stage.status,
                    'duration': stage.duration,
                    'error': stage.error_message
                }
                for stage in self.stages
            ],
            'data_loaded': self.raw_data is not None,
            'data_processed': self.processed_data is not None,
            'analysis_completed': len(self.analysis_results),
            'reports_generated': len(self.reports),
            'error_recovery_enabled': self.error_recovery_enabled
        }
    
    def enable_error_recovery(self, enabled: bool = True):
        """Enable or disable error recovery mechanisms."""
        self.error_recovery_enabled = enabled
        self.logger.info(f"Error recovery {'enabled' if enabled else 'disabled'}")
    
    def set_retry_parameters(self, max_attempts: int = 3, delay: float = 1.0):
        """Set retry parameters for error recovery."""
        self.max_retry_attempts = max_attempts
        self.retry_delay = delay
        self.logger.info(f"Retry parameters set: {max_attempts} attempts, {delay}s delay")