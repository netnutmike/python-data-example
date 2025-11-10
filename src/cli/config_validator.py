"""
Configuration validation utilities for the CLI.
Provides comprehensive validation and helpful error messages.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from ..interfaces import ValidationResult


@dataclass
class ConfigValidationRule:
    """Configuration validation rule."""
    field_path: str
    rule_type: str  # 'required', 'type', 'range', 'choices', 'file_exists'
    expected_value: Any = None
    error_message: str = ""
    warning_message: str = ""


class ConfigValidator:
    """
    Comprehensive configuration validator with detailed error reporting
    and suggestions for fixing configuration issues.
    """
    
    def __init__(self):
        """Initialize the configuration validator."""
        self.validation_rules = self._define_validation_rules()
    
    def _define_validation_rules(self) -> Dict[str, List[ConfigValidationRule]]:
        """Define validation rules for each configuration file."""
        return {
            'data_sources.yaml': [
                ConfigValidationRule(
                    'main_dataset_path',
                    'required',
                    error_message="Main dataset path is required"
                ),
                ConfigValidationRule(
                    'main_dataset_path',
                    'file_exists',
                    error_message="Main dataset file does not exist",
                    warning_message="Main dataset file not found - ensure path is correct"
                ),
                ConfigValidationRule(
                    'footnote_dataset_path',
                    'required',
                    error_message="Footnote dataset path is required"
                ),
                ConfigValidationRule(
                    'footnote_dataset_path',
                    'file_exists',
                    error_message="Footnote dataset file does not exist",
                    warning_message="Footnote dataset file not found - ensure path is correct"
                ),
                ConfigValidationRule(
                    'encoding',
                    'type',
                    expected_value=str,
                    error_message="Encoding must be a string"
                )
            ],
            
            'output.yaml': [
                ConfigValidationRule(
                    'base_output_dir',
                    'required',
                    error_message="Base output directory is required"
                ),
                ConfigValidationRule(
                    'base_output_dir',
                    'type',
                    expected_value=str,
                    error_message="Base output directory must be a string"
                ),
                ConfigValidationRule(
                    'html_enabled',
                    'type',
                    expected_value=bool,
                    error_message="html_enabled must be a boolean"
                ),
                ConfigValidationRule(
                    'pdf_enabled',
                    'type',
                    expected_value=bool,
                    error_message="pdf_enabled must be a boolean"
                ),
                ConfigValidationRule(
                    'csv_enabled',
                    'type',
                    expected_value=bool,
                    error_message="csv_enabled must be a boolean"
                )
            ],
            
            'visualization.yaml': [
                ConfigValidationRule(
                    'chart_theme',
                    'choices',
                    expected_value=['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn'],
                    error_message="Invalid chart theme"
                ),
                ConfigValidationRule(
                    'figure_width',
                    'type',
                    expected_value=int,
                    error_message="Figure width must be an integer"
                ),
                ConfigValidationRule(
                    'figure_width',
                    'range',
                    expected_value=(100, 2000),
                    warning_message="Figure width should be between 100 and 2000 pixels"
                ),
                ConfigValidationRule(
                    'figure_height',
                    'type',
                    expected_value=int,
                    error_message="Figure height must be an integer"
                ),
                ConfigValidationRule(
                    'figure_height',
                    'range',
                    expected_value=(100, 2000),
                    warning_message="Figure height should be between 100 and 2000 pixels"
                ),
                ConfigValidationRule(
                    'confidence_interval_alpha',
                    'type',
                    expected_value=float,
                    error_message="Confidence interval alpha must be a float"
                ),
                ConfigValidationRule(
                    'confidence_interval_alpha',
                    'range',
                    expected_value=(0.01, 0.1),
                    error_message="Confidence interval alpha must be between 0.01 and 0.1"
                )
            ],
            
            'analysis.yaml': [
                ConfigValidationRule(
                    'confidence_level',
                    'type',
                    expected_value=float,
                    error_message="Confidence level must be a float"
                ),
                ConfigValidationRule(
                    'confidence_level',
                    'range',
                    expected_value=(0.8, 0.99),
                    error_message="Confidence level must be between 0.8 and 0.99"
                ),
                ConfigValidationRule(
                    'min_sample_size',
                    'type',
                    expected_value=int,
                    error_message="Minimum sample size must be an integer"
                ),
                ConfigValidationRule(
                    'min_sample_size',
                    'range',
                    expected_value=(1, 1000),
                    warning_message="Minimum sample size should be between 1 and 1000"
                ),
                ConfigValidationRule(
                    'correlation_threshold',
                    'type',
                    expected_value=float,
                    error_message="Correlation threshold must be a float"
                ),
                ConfigValidationRule(
                    'correlation_threshold',
                    'range',
                    expected_value=(0.1, 0.9),
                    warning_message="Correlation threshold should be between 0.1 and 0.9"
                )
            ],
            
            'reports.yaml': [
                # Dynamic validation based on report types
            ]
        }
    
    def validate_config_file(self, file_path: str, config_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate a single configuration file.
        
        Args:
            file_path: Path to the configuration file
            config_data: Loaded configuration data
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        
        file_name = Path(file_path).name
        rules = self.validation_rules.get(file_name, [])
        
        for rule in rules:
            try:
                field_value = self._get_nested_value(config_data, rule.field_path)
                validation_result = self._apply_rule(rule, field_value, config_data)
                
                if validation_result['error']:
                    errors.append(f"{rule.field_path}: {validation_result['error']}")
                
                if validation_result['warning']:
                    warnings.append(f"{rule.field_path}: {validation_result['warning']}")
                    
            except KeyError:
                if rule.rule_type == 'required':
                    errors.append(f"Missing required field: {rule.field_path}")
            except Exception as e:
                errors.append(f"Validation error for {rule.field_path}: {str(e)}")
        
        # Special validation for reports.yaml
        if file_name == 'reports.yaml':
            report_errors, report_warnings = self._validate_reports_config(config_data)
            errors.extend(report_errors)
            warnings.extend(report_warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            record_count=len(config_data)
        )
    
    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = field_path.split('.')
        value = data
        
        for key in keys:
            value = value[key]
        
        return value
    
    def _apply_rule(self, rule: ConfigValidationRule, field_value: Any, 
                   config_data: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """
        Apply a validation rule to a field value.
        
        Args:
            rule: Validation rule to apply
            field_value: Value to validate
            config_data: Full configuration data for context
            
        Returns:
            Dictionary with 'error' and 'warning' keys
        """
        result = {'error': None, 'warning': None}
        
        if rule.rule_type == 'required':
            if field_value is None:
                result['error'] = rule.error_message
        
        elif rule.rule_type == 'type':
            if field_value is not None and not isinstance(field_value, rule.expected_value):
                result['error'] = rule.error_message
        
        elif rule.rule_type == 'range':
            if field_value is not None:
                min_val, max_val = rule.expected_value
                if not (min_val <= field_value <= max_val):
                    message = rule.error_message or rule.warning_message
                    if rule.error_message:
                        result['error'] = message
                    else:
                        result['warning'] = message
        
        elif rule.rule_type == 'choices':
            if field_value is not None and field_value not in rule.expected_value:
                result['error'] = f"{rule.error_message}. Valid choices: {', '.join(map(str, rule.expected_value))}"
        
        elif rule.rule_type == 'file_exists':
            if field_value is not None:
                file_path = Path(field_value)
                if not file_path.exists():
                    if rule.error_message:
                        result['error'] = rule.error_message
                    elif rule.warning_message:
                        result['warning'] = rule.warning_message
        
        return result
    
    def _validate_reports_config(self, reports_config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Validate reports configuration with dynamic rules.
        
        Args:
            reports_config: Reports configuration data
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Known report types
        known_report_types = {
            'occupation_distribution', 'environmental_risk', 'physical_demands',
            'skills_training', 'work_pace_autonomy', 'public_interaction',
            'data_quality', 'cognitive_requirements', 'additive_category',
            'statistical_precision', 'correlation_analysis', 'workforce_insights',
            'comprehensive_summary', 'establishment_coverage'
        }
        
        enabled_count = 0
        
        for report_type, config in reports_config.items():
            # Check if report type is known
            if report_type not in known_report_types:
                warnings.append(f"Unknown report type: {report_type}")
            
            # Validate report configuration structure
            if not isinstance(config, dict):
                errors.append(f"Report {report_type} configuration must be a dictionary")
                continue
            
            # Check required fields
            required_fields = ['report_type', 'title', 'description']
            for field in required_fields:
                if field not in config:
                    errors.append(f"Report {report_type} missing required field: {field}")
            
            # Check field types
            if 'enabled' in config and not isinstance(config['enabled'], bool):
                errors.append(f"Report {report_type} 'enabled' field must be boolean")
            
            if 'custom_parameters' in config and not isinstance(config['custom_parameters'], dict):
                errors.append(f"Report {report_type} 'custom_parameters' field must be a dictionary")
            
            # Count enabled reports
            if config.get('enabled', True):
                enabled_count += 1
        
        # Check if at least one report is enabled
        if enabled_count == 0:
            warnings.append("No reports are enabled - at least one report should be enabled")
        
        return errors, warnings
    
    def generate_fix_suggestions(self, file_path: str, validation_result: ValidationResult) -> List[str]:
        """
        Generate suggestions for fixing configuration errors.
        
        Args:
            file_path: Path to the configuration file
            validation_result: Validation result with errors
            
        Returns:
            List of fix suggestions
        """
        suggestions = []
        file_name = Path(file_path).name
        
        for error in validation_result.errors:
            if "Missing required field" in error:
                field_name = error.split(": ")[-1]
                suggestions.append(f"Add the required field '{field_name}' to {file_name}")
            
            elif "file does not exist" in error:
                suggestions.append(f"Ensure the data file path is correct and the file exists")
                suggestions.append(f"Check file permissions and accessibility")
            
            elif "must be" in error:
                suggestions.append(f"Check the data type for the field mentioned in the error")
                suggestions.append(f"Refer to the configuration documentation for correct types")
            
            elif "Invalid chart theme" in error:
                suggestions.append("Use one of the supported chart themes: plotly, plotly_white, plotly_dark, ggplot2, seaborn")
            
            elif "between" in error:
                suggestions.append("Adjust the value to be within the specified range")
        
        # General suggestions
        if validation_result.errors:
            suggestions.extend([
                f"Review the {file_name} file for syntax errors (YAML formatting)",
                "Use a YAML validator to check file structure",
                "Compare with the sample configuration files"
            ])
        
        return suggestions
    
    def validate_all_configs(self, config_dir: str) -> Dict[str, ValidationResult]:
        """
        Validate all configuration files in a directory.
        
        Args:
            config_dir: Directory containing configuration files
            
        Returns:
            Dictionary mapping file names to validation results
        """
        config_path = Path(config_dir)
        results = {}
        
        config_files = ['data_sources.yaml', 'output.yaml', 'visualization.yaml', 
                       'analysis.yaml', 'reports.yaml']
        
        for config_file in config_files:
            file_path = config_path / config_file
            
            if not file_path.exists():
                results[config_file] = ValidationResult(
                    is_valid=False,
                    errors=[f"Configuration file not found: {config_file}"],
                    warnings=[],
                    record_count=0
                )
                continue
            
            try:
                with open(file_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                if config_data is None:
                    config_data = {}
                
                results[config_file] = self.validate_config_file(str(file_path), config_data)
                
            except yaml.YAMLError as e:
                results[config_file] = ValidationResult(
                    is_valid=False,
                    errors=[f"YAML parsing error: {str(e)}"],
                    warnings=[],
                    record_count=0
                )
            except Exception as e:
                results[config_file] = ValidationResult(
                    is_valid=False,
                    errors=[f"Failed to load configuration: {str(e)}"],
                    warnings=[],
                    record_count=0
                )
        
        return results
    
    def create_validation_report(self, config_dir: str) -> str:
        """
        Create a comprehensive validation report.
        
        Args:
            config_dir: Directory containing configuration files
            
        Returns:
            Formatted validation report as string
        """
        results = self.validate_all_configs(config_dir)
        
        report_lines = []
        report_lines.append("CONFIGURATION VALIDATION REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Configuration Directory: {config_dir}")
        report_lines.append(f"Validation Date: {Path().cwd()}")
        report_lines.append("")
        
        overall_valid = True
        
        for config_file, result in results.items():
            report_lines.append(f"File: {config_file}")
            report_lines.append("-" * 30)
            
            if result.is_valid:
                report_lines.append("✓ VALID")
            else:
                report_lines.append("✗ INVALID")
                overall_valid = False
            
            if result.errors:
                report_lines.append("Errors:")
                for error in result.errors:
                    report_lines.append(f"  - {error}")
            
            if result.warnings:
                report_lines.append("Warnings:")
                for warning in result.warnings:
                    report_lines.append(f"  - {warning}")
            
            # Add fix suggestions for invalid configurations
            if not result.is_valid:
                suggestions = self.generate_fix_suggestions(config_file, result)
                if suggestions:
                    report_lines.append("Suggestions:")
                    for suggestion in suggestions:
                        report_lines.append(f"  • {suggestion}")
            
            report_lines.append("")
        
        # Overall summary
        report_lines.append("SUMMARY")
        report_lines.append("-" * 20)
        if overall_valid:
            report_lines.append("✓ All configurations are valid")
        else:
            invalid_count = sum(1 for r in results.values() if not r.is_valid)
            report_lines.append(f"✗ {invalid_count} configuration(s) have errors")
        
        return "\n".join(report_lines)