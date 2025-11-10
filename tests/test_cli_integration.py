"""
Integration tests for command-line interface functionality.
Tests CLI commands, argument parsing, and end-to-end command execution.
"""

import pytest
import tempfile
import shutil
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import argparse

from src.main import create_argument_parser, main, OccupationDataReportsApp
from src.cli.commands import CLICommands, handle_cli_command
from src.cli.config_validator import ConfigValidator
from src.config.settings import ConfigurationManager


class TestCLIArgumentParsing:
    """Test command-line argument parsing and validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = create_argument_parser()
    
    def test_basic_argument_parsing(self):
        """Test basic argument parsing functionality."""
        # Test single report generation
        args = self.parser.parse_args(['--report-type', 'occupation_distribution'])
        assert args.report_type == 'occupation_distribution'
        assert args.generate_all is False
        
        # Test generate all reports
        args = self.parser.parse_args(['--generate-all'])
        assert args.generate_all is True
        assert args.report_type is None
        
        # Test with output options
        args = self.parser.parse_args([
            '--report-type', 'environmental_risk',
            '--output-dir', '/tmp/reports',
            '--output-format', 'pdf'
        ])
        assert args.report_type == 'environmental_risk'
        assert args.output_dir == '/tmp/reports'
        assert args.output_format == 'pdf'
    
    def test_parallel_processing_arguments(self):
        """Test parallel processing argument parsing."""
        args = self.parser.parse_args([
            '--generate-all',
            '--parallel',
            '--max-workers', '8'
        ])
        assert args.parallel is True
        assert args.max_workers == 8
    
    def test_logging_arguments(self):
        """Test logging-related argument parsing."""
        args = self.parser.parse_args([
            '--report-type', 'data_quality',
            '--log-level', 'DEBUG',
            '--log-file', '/tmp/app.log',
            '--verbose'
        ])
        assert args.log_level == 'DEBUG'
        assert args.log_file == '/tmp/app.log'
        assert args.verbose is True
    
    def test_batch_processing_arguments(self):
        """Test batch processing argument parsing."""
        args = self.parser.parse_args([
            '--batch-config', 'batch.json'
        ])
        assert args.batch_config == 'batch.json'
        
        args = self.parser.parse_args([
            '--template', 'comprehensive'
        ])
        assert args.template == 'comprehensive'
    
    def test_cli_subcommands(self):
        """Test CLI subcommand parsing."""
        # Test list-reports command
        args = self.parser.parse_args(['list-reports', '--verbose'])
        assert args.command == 'list-reports'
        assert args.verbose is True
        
        # Test validate-config command
        args = self.parser.parse_args(['validate-config', '--config-dir', '/custom/config'])
        assert args.command == 'validate-config'
        assert args.config_dir == '/custom/config'
        
        # Test status command
        args = self.parser.parse_args(['status'])
        assert args.command == 'status'
    
    def test_invalid_arguments(self):
        """Test handling of invalid arguments."""
        # Test invalid output format
        with pytest.raises(SystemExit):
            self.parser.parse_args(['--report-type', 'test', '--output-format', 'invalid'])
        
        # Test invalid log level
        with pytest.raises(SystemExit):
            self.parser.parse_args(['--log-level', 'INVALID'])
        
        # Test invalid template
        with pytest.raises(SystemExit):
            self.parser.parse_args(['--template', 'nonexistent'])


class TestCLICommands:
    """Test CLI command implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test configuration
        self.config_dir = self.temp_path / "config"
        self.config_dir.mkdir()
        self.create_test_config()
        
        self.config_manager = ConfigurationManager(str(self.config_dir))
        self.cli_commands = CLICommands(self.config_manager)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_config(self):
        """Create test configuration files."""
        import yaml
        
        configs = {
            'data_sources.yaml': {
                'main_dataset_path': 'test_data.csv',
                'footnote_dataset_path': 'test_footnotes.csv'
            },
            'output.yaml': {
                'base_output_dir': 'reports',
                'html_enabled': True
            },
            'visualization.yaml': {
                'chart_theme': 'plotly_white'
            },
            'analysis.yaml': {
                'confidence_level': 0.95
            },
            'reports.yaml': {
                'test_report': {
                    'report_type': 'occupation_distribution',
                    'title': 'Test Report',
                    'description': 'Test report description',
                    'enabled': True,
                    'custom_parameters': {}
                }
            }
        }
        
        for filename, config_data in configs.items():
            with open(self.config_dir / filename, 'w') as f:
                yaml.dump(config_data, f)
    
    def test_list_reports_command(self):
        """Test list-reports command functionality."""
        args = argparse.Namespace(verbose=False)
        
        # Capture stdout
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = self.cli_commands.list_reports(args)
        
        assert result == 0  # Success exit code
        output = captured_output.getvalue()
        assert 'AVAILABLE REPORT TYPES' in output
        assert 'test_report' in output
    
    def test_validate_config_command(self):
        """Test validate-config command functionality."""
        args = argparse.Namespace()
        
        # Capture stdout
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = self.cli_commands.validate_config(args)
        
        assert result == 0  # Should succeed with valid config
        output = captured_output.getvalue()
        assert 'CONFIGURATION VALIDATION' in output
        assert 'Valid' in output
    
    def test_status_command(self):
        """Test status command functionality."""
        args = argparse.Namespace(verbose=False)
        
        # Capture stdout
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = self.cli_commands.show_status(args)
        
        assert result == 0
        output = captured_output.getvalue()
        assert 'APPLICATION STATUS' in output
        assert 'Configuration:' in output
        assert 'Data Sources:' in output
    
    def test_create_config_template_command(self):
        """Test create-config command functionality."""
        template_dir = self.temp_path / "new_config"
        args = argparse.Namespace(config_dir=str(template_dir))
        
        # Capture stdout
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = self.cli_commands.create_config_template(args)
        
        assert result == 0
        assert template_dir.exists()
        
        # Check that config files were created
        config_files = list(template_dir.glob("*.yaml"))
        assert len(config_files) > 0
        
        output = captured_output.getvalue()
        assert 'Configuration templates created successfully' in output
    
    def test_export_sample_config_command(self):
        """Test export-sample-config command functionality."""
        output_file = self.temp_path / "sample_config.json"
        args = argparse.Namespace(output=str(output_file))
        
        # Capture stdout
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = self.cli_commands.export_sample_config(args)
        
        assert result == 0
        assert output_file.exists()
        
        # Verify the exported configuration
        with open(output_file, 'r') as f:
            config_data = json.load(f)
        
        assert 'jobs' in config_data
        assert 'global_settings' in config_data
        assert len(config_data['jobs']) > 0
        
        output = captured_output.getvalue()
        assert 'Sample batch configuration exported' in output
    
    @patch('src.cli.commands.DataProcessingPipeline')
    def test_pipeline_test_command(self, mock_pipeline_class):
        """Test test-pipeline command functionality."""
        # Create a sample data file for testing
        import pandas as pd
        
        sample_data = pd.DataFrame({
            'SERIES ID': ['TEST001'],
            'SOC 2018 CODE': ['111011'],
            'OCCUPATION': ['Test Occupation'],
            'ESTIMATE': [50.0]
        })
        
        data_file = self.temp_path / "test_data.csv"
        sample_data.to_csv(data_file, index=False)
        
        # Update config to point to the test file
        import yaml
        with open(self.config_dir / 'data_sources.yaml', 'w') as f:
            yaml.dump({
                'main_dataset_path': str(data_file),
                'footnote_dataset_path': 'test_footnotes.csv'
            }, f)
        
        # Mock the pipeline stage
        mock_pipeline = Mock()
        mock_pipeline._stage_data_loading.return_value = {
            'records_loaded': 1,
            'columns_found': 4,
            'footnote_codes_loaded': 0,
            'validation_warnings': []
        }
        mock_pipeline_class.return_value = mock_pipeline
        
        args = argparse.Namespace(config_dir=str(self.config_dir))
        
        # Capture stdout
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = self.cli_commands.run_pipeline_test(args)
        
        assert result == 0
        output = captured_output.getvalue()
        assert 'PIPELINE TEST' in output
        assert 'Pipeline test completed successfully' in output


class TestConfigValidator:
    """Test configuration validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.validator = ConfigValidator()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_valid_configuration_validation(self):
        """Test validation of valid configuration."""
        valid_config = {
            'main_dataset_path': 'test.csv',
            'footnote_dataset_path': 'footnotes.csv',
            'encoding': 'utf-8'
        }
        
        result = self.validator.validate_config_file('data_sources.yaml', valid_config)
        
        # Should have warnings about missing files but no errors in structure
        assert len(result.errors) == 0  # No structural errors
        assert result.is_valid is True
    
    def test_invalid_configuration_validation(self):
        """Test validation of invalid configuration."""
        invalid_config = {
            'main_dataset_path': 123,  # Should be string
            'encoding': ['utf-8']  # Should be string, not list
            # Missing required footnote_dataset_path
        }
        
        result = self.validator.validate_config_file('data_sources.yaml', invalid_config)
        
        assert len(result.errors) > 0
        assert result.is_valid is False
    
    def test_reports_configuration_validation(self):
        """Test validation of reports configuration."""
        reports_config = {
            'valid_report': {
                'report_type': 'occupation_distribution',
                'title': 'Valid Report',
                'description': 'A valid report configuration',
                'enabled': True,
                'custom_parameters': {}
            },
            'invalid_report': {
                'title': 'Missing required fields',
                'enabled': 'not_boolean'  # Should be boolean
            }
        }
        
        result = self.validator.validate_config_file('reports.yaml', reports_config)
        
        assert len(result.errors) > 0  # Should have errors for invalid_report
        assert result.is_valid is False
    
    def test_validation_report_generation(self):
        """Test comprehensive validation report generation."""
        # Create test config directory with mixed valid/invalid configs
        config_dir = self.temp_path / "test_config"
        config_dir.mkdir()
        
        import yaml
        
        # Valid config
        with open(config_dir / 'data_sources.yaml', 'w') as f:
            yaml.dump({
                'main_dataset_path': 'test.csv',
                'footnote_dataset_path': 'footnotes.csv',
                'encoding': 'utf-8'
            }, f)
        
        # Invalid config
        with open(config_dir / 'analysis.yaml', 'w') as f:
            yaml.dump({
                'confidence_level': 1.5,  # Invalid: should be < 1
                'min_sample_size': -10  # Invalid: should be positive
            }, f)
        
        # Generate validation report
        report = self.validator.create_validation_report(str(config_dir))
        
        assert 'CONFIGURATION VALIDATION REPORT' in report
        assert 'data_sources.yaml' in report
        assert 'analysis.yaml' in report
        assert 'INVALID' in report  # Should show invalid configs
    
    def test_fix_suggestions_generation(self):
        """Test generation of fix suggestions for configuration errors."""
        validation_result = Mock()
        validation_result.errors = [
            "Missing required field: main_dataset_path",
            "confidence_level: must be between 0.8 and 0.99",
            "main_dataset_path: file does not exist"
        ]
        
        suggestions = self.validator.generate_fix_suggestions(
            'data_sources.yaml', validation_result
        )
        
        assert len(suggestions) > 0
        
        # Check that suggestions are relevant to the errors
        suggestion_text = ' '.join(suggestions)
        assert 'required field' in suggestion_text or 'main_dataset_path' in suggestion_text
        assert 'file' in suggestion_text or 'path' in suggestion_text


class TestEndToEndCLIWorkflows:
    """Test complete end-to-end CLI workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create complete test environment
        self.setup_test_environment()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def setup_test_environment(self):
        """Set up a complete test environment with data and config."""
        # Create config directory
        self.config_dir = self.temp_path / "config"
        self.config_dir.mkdir()
        
        # Create sample data files
        import pandas as pd
        
        sample_data = pd.DataFrame({
            'SERIES ID': ['TEST001', 'TEST002'],
            'SERIES TITLE': ['Test Series 1', 'Test Series 2'],
            'SOC 2018 CODE': ['111011', '291141'],
            'OCCUPATION': ['Test Occupation 1', 'Test Occupation 2'],
            'REQUIREMENT': ['Test Requirement', 'Test Requirement'],
            'ESTIMATE CODE': ['001', '002'],
            'ESTIMATE TEXT': ['Test Estimate 1', 'Test Estimate 2'],
            'CATEGORY CODE': ['01', '02'],
            'CATEGORY': ['Test Category 1', 'Test Category 2'],
            'ADDITIVE CODE': ['01', '02'],
            'ADDITIVE': ['Test Additive 1', 'Test Additive 2'],
            'DATATYPE CODE': ['01', '01'],
            'DATATYPE': ['Percentage', 'Percentage'],
            'ESTIMATE': [50.0, 75.0],
            'STANDARD ERROR': [2.5, 3.0],
            'DATA FOOTNOTE': [None, None],
            'STANDARD ERROR FOOTNOTE': [None, None],
            'SERIES FOOTNOTE': [None, None]
        })
        
        self.data_file = self.temp_path / "test_data.csv"
        sample_data.to_csv(self.data_file, index=False)
        
        footnotes = pd.DataFrame({
            'Footnote code': [1, 2],
            'Footnote text': ['Test footnote 1', 'Test footnote 2']
        })
        
        self.footnote_file = self.temp_path / "test_footnotes.csv"
        footnotes.to_csv(self.footnote_file, index=False)
        
        # Create configuration files
        self.create_complete_config()
    
    def create_complete_config(self):
        """Create complete configuration for testing."""
        import yaml
        
        configs = {
            'data_sources.yaml': {
                'main_dataset_path': str(self.data_file),
                'footnote_dataset_path': str(self.footnote_file),
                'encoding': 'utf-8'
            },
            'output.yaml': {
                'base_output_dir': str(self.temp_path / 'reports'),
                'html_enabled': True,
                'pdf_enabled': False,  # Disable to avoid dependencies
                'csv_enabled': True,
                'timestamp_folders': False
            },
            'visualization.yaml': {
                'chart_theme': 'plotly_white',
                'figure_width': 800,
                'figure_height': 600,
                'interactive_charts': True
            },
            'analysis.yaml': {
                'confidence_level': 0.95,
                'min_sample_size': 10,
                'correlation_threshold': 0.3
            },
            'reports.yaml': {
                'test_report': {
                    'report_type': 'occupation_distribution',
                    'title': 'Test Occupation Distribution',
                    'description': 'Test report for CLI integration',
                    'enabled': True,
                    'custom_parameters': {'top_n': 5}
                }
            }
        }
        
        for filename, config_data in configs.items():
            with open(self.config_dir / filename, 'w') as f:
                yaml.dump(config_data, f)
    
    @patch('src.main.initialize_monitoring')
    @patch('src.main.DataProcessingPipeline')
    def test_complete_report_generation_workflow(self, mock_pipeline_class, mock_monitoring):
        """Test complete report generation workflow through CLI."""
        # Mock monitoring initialization
        mock_monitor = Mock()
        mock_logging_manager = Mock()
        mock_monitoring.return_value = (mock_monitor, mock_logging_manager)
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.run_pipeline.return_value = {
            'success': True,
            'reports_generated': 1,
            'pipeline_duration': 5.0,
            'data_records_processed': 2
        }
        mock_pipeline_class.return_value = mock_pipeline
        
        # Test command line arguments
        test_args = [
            '--report-type', 'test_report',
            '--config-dir', str(self.config_dir),
            '--output-dir', str(self.temp_path / 'output'),
            '--output-format', 'html'
        ]
        
        # Mock sys.argv
        with patch.object(sys, 'argv', ['main.py'] + test_args):
            result = main()
        
        assert result == 0  # Success exit code
        
        # Verify pipeline was called with correct parameters
        mock_pipeline.run_pipeline.assert_called_once()
        call_args = mock_pipeline.run_pipeline.call_args
        assert call_args[1]['report_types'] == ['test_report']
        assert call_args[1]['export_formats'] == ['html']
    
    @patch('src.main.initialize_monitoring')
    def test_cli_subcommand_workflow(self, mock_monitoring):
        """Test CLI subcommand execution workflow."""
        # Mock monitoring
        mock_monitor = Mock()
        mock_logging_manager = Mock()
        mock_monitoring.return_value = (mock_monitor, mock_logging_manager)
        
        # Test list-reports subcommand
        test_args = ['list-reports', '--config-dir', str(self.config_dir)]
        
        with patch.object(sys, 'argv', ['main.py'] + test_args):
            result = main()
        
        assert result == 0  # Success exit code
    
    @patch('src.main.initialize_monitoring')
    def test_configuration_validation_workflow(self, mock_monitoring):
        """Test configuration validation workflow through CLI."""
        # Mock monitoring
        mock_monitor = Mock()
        mock_logging_manager = Mock()
        mock_monitoring.return_value = (mock_monitor, mock_logging_manager)
        
        # Test validate-config subcommand
        test_args = ['validate-config', '--config-dir', str(self.config_dir)]
        
        with patch.object(sys, 'argv', ['main.py'] + test_args):
            result = main()
        
        assert result == 0  # Should succeed with valid config
    
    def test_batch_configuration_workflow(self):
        """Test batch configuration creation and usage workflow."""
        # Create batch configuration file
        batch_config = {
            "description": "Test batch configuration",
            "jobs": [
                {
                    "job_id": "test_job",
                    "report_type": "occupation_distribution",
                    "filters": {"top_n": 3},
                    "config": {},
                    "output_path": str(self.temp_path / "batch_output"),
                    "export_formats": ["csv"]
                }
            ],
            "global_settings": {
                "parallel_processing": False,
                "max_workers": 1
            }
        }
        
        batch_file = self.temp_path / "test_batch.json"
        with open(batch_file, 'w') as f:
            json.dump(batch_config, f, indent=2)
        
        # Test that the batch configuration is valid JSON
        assert batch_file.exists()
        
        # Verify the configuration can be loaded
        with open(batch_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config['description'] == "Test batch configuration"
        assert len(loaded_config['jobs']) == 1
        assert loaded_config['jobs'][0]['job_id'] == "test_job"


if __name__ == '__main__':
    # Run CLI integration tests
    pytest.main([__file__, '-v', '--tb=short'])