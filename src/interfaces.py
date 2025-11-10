"""
Base interfaces and abstract classes for the occupation data reports application.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Type alias for DataFrame to avoid pandas dependency in interfaces
DataFrame = Any  # Will be pandas.DataFrame when pandas is available


@dataclass
class OccupationRecord:
    """Core data structure for occupation survey records."""
    series_id: str
    series_title: str
    soc_code: str
    occupation: str
    requirement_type: str
    estimate_code: str
    estimate_text: str
    category_code: str
    category: str
    additive_code: str
    additive: str
    datatype_code: str
    datatype: str
    estimate: float
    standard_error: Optional[float]
    data_footnote: Optional[int]
    standard_error_footnote: Optional[int]
    series_footnote: Optional[int]


@dataclass
class FootnoteReference:
    """Reference structure for footnote codes and interpretations."""
    code: int
    description: str
    precision_level: str
    interpretation_guidance: str


@dataclass
class AnalysisResult:
    """Result structure for analysis operations."""
    occupation_category: str
    metric_name: str
    value: float
    confidence_interval: Tuple[float, float]
    reliability_score: float
    footnote_context: List[str]


@dataclass
class ValidationResult:
    """Result structure for data validation operations."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    record_count: int


@dataclass
class ReportData:
    """Container for report generation data."""
    title: str
    description: str
    analysis_results: List[AnalysisResult]
    visualizations: List[Any]  # Will be Figure objects from plotting libraries
    metadata: Dict[str, Any]
    generation_timestamp: datetime


class DataProcessorInterface(ABC):
    """Abstract interface for data processing operations."""
    
    @abstractmethod
    def load_dataset(self, file_path: str) -> DataFrame:
        """Load the main occupational dataset from CSV file."""
        pass
    
    @abstractmethod
    def load_footnotes(self, footnote_path: str) -> Dict[int, FootnoteReference]:
        """Load and parse footnote reference data."""
        pass
    
    @abstractmethod
    def validate_columns(self, df: DataFrame) -> ValidationResult:
        """Validate that all required columns are present and properly formatted."""
        pass
    
    @abstractmethod
    def process_footnotes(self, df: DataFrame) -> DataFrame:
        """Process and interpret footnote codes in the dataset."""
        pass
    
    @abstractmethod
    def clean_occupation_names(self, df: DataFrame) -> DataFrame:
        """Standardize occupation names and categories."""
        pass
    
    @abstractmethod
    def handle_estimate_ranges(self, df: DataFrame) -> DataFrame:
        """Convert range estimates to numeric values."""
        pass


class AnalysisEngineInterface(ABC):
    """Abstract interface for statistical analysis operations."""
    
    @abstractmethod
    def calculate_confidence_intervals(self, estimates: Any, std_errors: Any) -> DataFrame:
        """Calculate confidence intervals for estimates."""
        pass
    
    @abstractmethod
    def analyze_occupation_distribution(self, df: DataFrame) -> List[AnalysisResult]:
        """Analyze the distribution of occupations in the dataset."""
        pass
    
    @abstractmethod
    def calculate_risk_scores(self, environmental_data: DataFrame) -> DataFrame:
        """Calculate risk scores for environmental conditions."""
        pass
    
    @abstractmethod
    def perform_correlation_analysis(self, df: DataFrame) -> DataFrame:
        """Perform correlation analysis between requirement types."""
        pass
    
    @abstractmethod
    def analyze_physical_demands(self, df: DataFrame) -> List[AnalysisResult]:
        """Analyze physical demand requirements across occupations."""
        pass
    
    @abstractmethod
    def assess_data_quality(self, df: DataFrame) -> Dict[str, Any]:
        """Assess the quality and reliability of the dataset."""
        pass


class VisualizationEngineInterface(ABC):
    """Abstract interface for visualization operations."""
    
    @abstractmethod
    def create_distribution_chart(self, data: DataFrame, chart_type: str) -> Any:
        """Create distribution charts (bar, pie, etc.)."""
        pass
    
    @abstractmethod
    def generate_heatmap(self, correlation_matrix: DataFrame) -> Any:
        """Generate correlation or risk heatmaps."""
        pass
    
    @abstractmethod
    def create_confidence_interval_plot(self, data: DataFrame) -> Any:
        """Create plots showing confidence intervals."""
        pass
    
    @abstractmethod
    def build_interactive_dashboard(self, report_data: ReportData) -> Any:
        """Build interactive dashboards with multiple visualizations."""
        pass
    
    @abstractmethod
    def create_risk_matrix(self, risk_data: DataFrame) -> Any:
        """Create risk assessment matrices."""
        pass


class ExportManagerInterface(ABC):
    """Abstract interface for report export operations."""
    
    @abstractmethod
    def export_html_report(self, report_data: ReportData, output_path: str) -> bool:
        """Export report as interactive HTML."""
        pass
    
    @abstractmethod
    def export_pdf_report(self, report_data: ReportData, output_path: str) -> bool:
        """Export report as formatted PDF."""
        pass
    
    @abstractmethod
    def export_csv_data(self, data: DataFrame, output_path: str) -> bool:
        """Export processed data as CSV."""
        pass
    
    @abstractmethod
    def create_master_dashboard(self, all_reports: List[ReportData]) -> str:
        """Create a master dashboard combining all report types."""
        pass
    
    @abstractmethod
    def organize_output_files(self, base_path: str) -> Dict[str, str]:
        """Organize output files in structured directories."""
        pass


class ConfigurationManagerInterface(ABC):
    """Abstract interface for configuration management."""
    
    @abstractmethod
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        pass
    
    @abstractmethod
    def get_report_config(self, report_type: str) -> Dict[str, Any]:
        """Get configuration for specific report type."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration parameters."""
        pass
    
    @abstractmethod
    def get_output_settings(self) -> Dict[str, Any]:
        """Get output format and directory settings."""
        pass


# Custom exceptions for the application
class DataProcessingError(Exception):
    """Raised when data processing encounters unrecoverable errors."""
    pass


class FootnoteInterpretationError(Exception):
    """Raised when footnote codes cannot be properly interpreted."""
    pass


class VisualizationError(Exception):
    """Raised when chart generation fails."""
    pass


class ExportError(Exception):
    """Raised when report export operations fail."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass