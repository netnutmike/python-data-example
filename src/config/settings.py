"""
Configuration management for the occupation data reports application.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from ..interfaces import ConfigurationManagerInterface, ValidationResult, ConfigurationError


@dataclass
class DataSourceConfig:
    """Configuration for data source files."""
    main_dataset_path: str = "2023-complete-dataset.csv"
    footnote_dataset_path: str = "2023-complete-dataset-footnote-codes.csv"
    field_descriptions_path: str = "2023-complete-dataset-field names and description.csv"
    encoding: str = "utf-8"


@dataclass
class OutputConfig:
    """Configuration for output settings."""
    base_output_dir: str = "reports"
    html_enabled: bool = True
    pdf_enabled: bool = True
    csv_enabled: bool = True
    timestamp_folders: bool = True
    include_metadata: bool = True


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    chart_theme: str = "plotly_white"
    color_palette: str = "viridis"
    figure_width: int = 800
    figure_height: int = 600
    interactive_charts: bool = True
    confidence_interval_alpha: float = 0.05


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    confidence_level: float = 0.95
    min_sample_size: int = 30
    correlation_threshold: float = 0.3
    risk_score_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.risk_score_weights is None:
            self.risk_score_weights = {
                "extreme_cold": 0.2,
                "extreme_heat": 0.2,
                "hazardous_contaminants": 0.3,
                "heavy_vibrations": 0.15,
                "heights": 0.15
            }


@dataclass
class ReportConfig:
    """Configuration for specific report types."""
    report_type: str
    title: str
    description: str
    enabled: bool = True
    custom_parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_parameters is None:
            self.custom_parameters = {}


class ConfigurationManager(ConfigurationManagerInterface):
    """Implementation of configuration management."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._config_cache: Dict[str, Any] = {}
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize default configuration files if they don't exist."""
        default_configs = {
            "data_sources.yaml": asdict(DataSourceConfig()),
            "output.yaml": asdict(OutputConfig()),
            "visualization.yaml": asdict(VisualizationConfig()),
            "analysis.yaml": asdict(AnalysisConfig()),
            "reports.yaml": self._get_default_report_configs()
        }
        
        for filename, config_data in default_configs.items():
            config_path = self.config_dir / filename
            if not config_path.exists():
                with open(config_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def _get_default_report_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default configurations for all report types."""
        return {
            "occupation_distribution": {
                "report_type": "occupation_distribution",
                "title": "Occupation Distribution Analysis",
                "description": "Analysis of occupation frequency and characteristics",
                "enabled": True,
                "custom_parameters": {"top_n": 20}
            },
            "environmental_risk": {
                "report_type": "environmental_risk",
                "title": "Environmental Risk Assessment",
                "description": "Analysis of workplace environmental conditions and risks",
                "enabled": True,
                "custom_parameters": {"risk_threshold": 0.7}
            },
            "skills_training": {
                "report_type": "skills_training",
                "title": "Skills and Training Requirements",
                "description": "Analysis of education, training, and skill requirements",
                "enabled": True,
                "custom_parameters": {}
            },
            "work_autonomy": {
                "report_type": "work_autonomy",
                "title": "Work Pace and Autonomy Analysis",
                "description": "Analysis of job flexibility and worker control patterns",
                "enabled": True,
                "custom_parameters": {}
            },
            "public_interaction": {
                "report_type": "public_interaction",
                "title": "Public Interaction Requirements",
                "description": "Analysis of customer service and communication requirements",
                "enabled": True,
                "custom_parameters": {}
            },
            "physical_demands": {
                "report_type": "physical_demands",
                "title": "Physical Demands Assessment",
                "description": "Analysis of physical requirements and ergonomic considerations",
                "enabled": True,
                "custom_parameters": {}
            },
            "data_quality": {
                "report_type": "data_quality",
                "title": "Data Quality and Reliability Assessment",
                "description": "Analysis of data confidence levels and reliability",
                "enabled": True,
                "custom_parameters": {}
            },
            "cognitive_requirements": {
                "report_type": "cognitive_requirements",
                "title": "Cognitive and Mental Requirements",
                "description": "Analysis of cognitive demands and mental requirements",
                "enabled": True,
                "custom_parameters": {}
            },
            "additive_analysis": {
                "report_type": "additive_analysis",
                "title": "Additive Category Analysis",
                "description": "Analysis of combined occupational requirements",
                "enabled": True,
                "custom_parameters": {}
            },
            "statistical_precision": {
                "report_type": "statistical_precision",
                "title": "Statistical Precision Dashboard",
                "description": "Visualization of confidence intervals and data precision",
                "enabled": True,
                "custom_parameters": {}
            },
            "correlation_analysis": {
                "report_type": "correlation_analysis",
                "title": "Cross-Requirement Correlation Analysis",
                "description": "Analysis of relationships between requirement types",
                "enabled": True,
                "custom_parameters": {}
            },
            "workforce_insights": {
                "report_type": "workforce_insights",
                "title": "Establishment-Level Workforce Insights",
                "description": "Policy-focused analysis of workforce representation",
                "enabled": True,
                "custom_parameters": {}
            }
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            full_path = self.config_dir / config_path
            if not full_path.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")
            
            # Check cache first
            cache_key = str(full_path)
            if cache_key in self._config_cache:
                return self._config_cache[cache_key]
            
            with open(full_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    config_data = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported configuration file format: {config_path}")
            
            # Cache the loaded configuration
            self._config_cache[cache_key] = config_data
            return config_data
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {str(e)}")
    
    def get_report_config(self, report_type: str) -> Dict[str, Any]:
        """Get configuration for specific report type."""
        reports_config = self.load_config("reports.yaml")
        
        if report_type not in reports_config:
            raise ConfigurationError(f"Report type '{report_type}' not found in configuration")
        
        return reports_config[report_type]
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration parameters."""
        errors = []
        warnings = []
        
        # Validate data source paths
        if "data_sources" in config:
            data_config = config["data_sources"]
            required_paths = ["main_dataset_path", "footnote_dataset_path", "field_descriptions_path"]
            
            for path_key in required_paths:
                if path_key not in data_config:
                    errors.append(f"Missing required data source path: {path_key}")
                elif not Path(data_config[path_key]).exists():
                    warnings.append(f"Data source file not found: {data_config[path_key]}")
        
        # Validate analysis parameters
        if "analysis" in config:
            analysis_config = config["analysis"]
            
            if "confidence_level" in analysis_config:
                confidence_level = analysis_config["confidence_level"]
                if not (0 < confidence_level < 1):
                    errors.append("Confidence level must be between 0 and 1")
            
            if "min_sample_size" in analysis_config:
                min_sample_size = analysis_config["min_sample_size"]
                if min_sample_size < 1:
                    errors.append("Minimum sample size must be positive")
        
        # Validate visualization settings
        if "visualization" in config:
            viz_config = config["visualization"]
            
            if "figure_width" in viz_config and viz_config["figure_width"] < 100:
                warnings.append("Figure width is very small, may affect readability")
            
            if "figure_height" in viz_config and viz_config["figure_height"] < 100:
                warnings.append("Figure height is very small, may affect readability")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            record_count=len(config)
        )
    
    def get_output_settings(self) -> Dict[str, Any]:
        """Get output format and directory settings."""
        return self.load_config("output.yaml")
    
    def get_data_source_config(self) -> DataSourceConfig:
        """Get data source configuration as typed object."""
        config_data = self.load_config("data_sources.yaml")
        return DataSourceConfig(**config_data)
    
    def get_analysis_config(self) -> AnalysisConfig:
        """Get analysis configuration as typed object."""
        config_data = self.load_config("analysis.yaml")
        return AnalysisConfig(**config_data)
    
    def get_visualization_config(self) -> VisualizationConfig:
        """Get visualization configuration as typed object."""
        config_data = self.load_config("visualization.yaml")
        return VisualizationConfig(**config_data)
    
    def get_enabled_reports(self) -> List[str]:
        """Get list of enabled report types."""
        reports_config = self.load_config("reports.yaml")
        return [report_type for report_type, config in reports_config.items() 
                if config.get("enabled", True)]
    
    def update_config(self, config_path: str, updates: Dict[str, Any]) -> None:
        """Update configuration file with new values."""
        try:
            current_config = self.load_config(config_path)
            current_config.update(updates)
            
            full_path = self.config_dir / config_path
            with open(full_path, 'w') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    yaml.dump(current_config, f, default_flow_style=False, indent=2)
                elif config_path.endswith('.json'):
                    json.dump(current_config, f, indent=2)
            
            # Clear cache for updated file
            cache_key = str(full_path)
            if cache_key in self._config_cache:
                del self._config_cache[cache_key]
                
        except Exception as e:
            raise ConfigurationError(f"Failed to update configuration {config_path}: {str(e)}")