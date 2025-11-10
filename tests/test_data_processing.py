"""
Unit tests for data processing components.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.data_processing.csv_loader import CSVLoader
from src.data_processing.footnote_processor import FootnoteProcessor
from src.data_processing.data_cleaner import DataCleaner
from src.interfaces import DataProcessingError, FootnoteInterpretationError


class TestCSVLoader:
    """Test cases for CSVLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = CSVLoader()
        
        # Create sample data for testing
        self.sample_data = {
            'SERIES ID': ['ORUC1000000000001197', 'ORUC1000000000001198'],
            'SERIES TITLE': ['Test series 1', 'Test series 2'],
            'SOC 2018 CODE': ['000000', '111011'],
            'OCCUPATION': ['All workers', 'Chief executives'],
            'REQUIREMENT': ['Cognitive and mental requirements', 'Physical demands'],
            'ESTIMATE CODE': ['01197', '01198'],
            'ESTIMATE TEXT': ['Test estimate 1', 'Test estimate 2'],
            'CATEGORY CODE': ['090', '091'],
            'CATEGORY': ['Test category 1', 'Test category 2'],
            'ADDITIVE CODE': ['090', '091'],
            'ADDITIVE': ['Test additive 1', 'Test additive 2'],
            'DATATYPE CODE': ['01', '02'],
            'DATATYPE': ['Percentage', 'Hours'],
            'ESTIMATE': [79.2, 45.5],
            'STANDARD ERROR': [2.1, 3.2],
            'DATA FOOTNOTE': [7, None],
            'STANDARD ERROR FOOTNOTE': [6, None],
            'SERIES FOOTNOTE': [None, None]
        }
        
        self.sample_footnotes = {
            'Footnote code': [1, 7, 16, 26],
            'Footnote text': [
                'Estimate is less than 0.5 percent.',
                'This estimate is the mode for the category group.',
                'Estimate is less than 5 percent.',
                'Estimate is greater than 50 percent.'
            ]
        }
    
    def test_validate_columns_valid_data(self):
        """Test column validation with valid data."""
        df = pd.DataFrame(self.sample_data)
        result = self.loader.validate_columns(df)
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.record_count == 2
    
    def test_validate_columns_missing_columns(self):
        """Test column validation with missing required columns."""
        incomplete_data = {col: self.sample_data[col] for col in list(self.sample_data.keys())[:10]}
        df = pd.DataFrame(incomplete_data)
        result = self.loader.validate_columns(df)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert 'Missing required columns' in result.errors[0]
    
    def test_validate_columns_empty_dataset(self):
        """Test column validation with empty dataset."""
        df = pd.DataFrame(columns=self.loader.REQUIRED_COLUMNS)
        result = self.loader.validate_columns(df)
        
        assert not result.is_valid
        assert 'Dataset is empty' in result.errors
    
    def test_load_dataset_file_not_found(self):
        """Test loading non-existent dataset file."""
        with pytest.raises(DataProcessingError, match="Dataset file not found"):
            self.loader.load_dataset("nonexistent_file.csv")
    
    def test_load_footnotes_valid_data(self):
        """Test loading valid footnote data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            footnote_df = pd.DataFrame(self.sample_footnotes)
            footnote_df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            result_df = self.loader.load_footnotes(temp_path)
            assert len(result_df) == 4
            assert 'Footnote code' in result_df.columns
            assert 'Footnote text' in result_df.columns
            assert result_df['Footnote code'].dtype == int
        finally:
            os.unlink(temp_path)
    
    def test_load_footnotes_missing_columns(self):
        """Test loading footnote data with missing columns."""
        invalid_footnotes = {'Code': [1, 2], 'Description': ['Test 1', 'Test 2']}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame(invalid_footnotes).to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            with pytest.raises(DataProcessingError, match="Missing required footnote columns"):
                self.loader.load_footnotes(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_get_dataset_summary(self):
        """Test dataset summary generation."""
        df = pd.DataFrame(self.sample_data)
        summary = self.loader.get_dataset_summary(df)
        
        assert summary['total_records'] == 2
        assert summary['total_columns'] == 18
        assert summary['unique_soc_codes'] == 2
        assert summary['unique_occupations'] == 2
        assert len(summary['requirement_types']) == 2


class TestFootnoteProcessor:
    """Test cases for FootnoteProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = FootnoteProcessor()
        
        self.sample_footnotes_df = pd.DataFrame({
            'Footnote code': [1, 7, 16, 26, 36],
            'Footnote text': [
                'Estimate is less than 0.5 percent.',
                'This estimate is the mode for the category group.',
                'Estimate is less than 5 percent.',
                'Estimate is greater than 50 percent.',
                'Estimate is greater than 99.5 percent.'
            ]
        })
    
    def test_load_footnote_mapping(self):
        """Test loading footnote mapping from DataFrame."""
        mapping = self.processor.load_footnote_mapping(self.sample_footnotes_df)
        
        assert len(mapping) == 5
        assert 1 in mapping
        assert 7 in mapping
        assert mapping[1].code == 1
        assert 'less than 0.5 percent' in mapping[1].description
    
    def test_convert_range_estimate_midpoint(self):
        """Test range estimate conversion using midpoint method."""
        # Load footnotes first
        self.processor.load_footnote_mapping(self.sample_footnotes_df)
        
        # Test range footnotes
        result = self.processor.convert_range_estimate(1, 'midpoint')  # less than 0.5%
        assert result == 0.25
        
        result = self.processor.convert_range_estimate(26, 'midpoint')  # greater than 50%
        assert result == 75.0
        
        # Test non-range footnote
        result = self.processor.convert_range_estimate(7, 'midpoint')  # mode footnote
        assert result is None
    
    def test_convert_range_estimate_conservative(self):
        """Test range estimate conversion using conservative method."""
        result = self.processor.convert_range_estimate(1, 'conservative')  # less than 0.5%
        assert result == 0.0
        
        result = self.processor.convert_range_estimate(26, 'conservative')  # greater than 50%
        assert result == 50.0
    
    def test_convert_range_estimate_optimistic(self):
        """Test range estimate conversion using optimistic method."""
        result = self.processor.convert_range_estimate(1, 'optimistic')  # less than 0.5%
        assert result == 0.5
        
        result = self.processor.convert_range_estimate(26, 'optimistic')  # greater than 50%
        assert result == 100.0
    
    def test_convert_range_estimate_invalid_method(self):
        """Test range estimate conversion with invalid method."""
        with pytest.raises(ValueError, match="Unknown conversion method"):
            self.processor.convert_range_estimate(1, 'invalid_method')
    
    def test_process_footnotes_in_dataframe(self):
        """Test processing footnotes in a DataFrame."""
        # Load footnotes first
        self.processor.load_footnote_mapping(self.sample_footnotes_df)
        
        test_data = pd.DataFrame({
            'DATA FOOTNOTE': [1, 7, None],
            'ESTIMATE': [0.3, 45.2, 78.9]
        })
        
        result_df = self.processor.process_footnotes_in_dataframe(test_data)
        
        assert 'DATA FOOTNOTE_INTERPRETATION' in result_df.columns
        assert 'DATA FOOTNOTE_PRECISION' in result_df.columns
        assert 'DATA FOOTNOTE_RANGE_ESTIMATE' in result_df.columns
        
        # Check that range estimate was calculated for footnote 1
        assert result_df.loc[0, 'DATA FOOTNOTE_RANGE_ESTIMATE'] == 0.25
        
        # Check that non-range footnote has no range estimate
        assert pd.isna(result_df.loc[1, 'DATA FOOTNOTE_RANGE_ESTIMATE'])
    
    def test_get_footnote_summary(self):
        """Test footnote summary generation."""
        self.processor.load_footnote_mapping(self.sample_footnotes_df)
        summary = self.processor.get_footnote_summary()
        
        assert summary['total_footnotes'] == 5
        assert summary['range_footnotes'] > 0
        assert 'precision_levels' in summary
        assert 'available_codes' in summary
    
    def test_validate_footnote_codes(self):
        """Test footnote code validation."""
        self.processor.load_footnote_mapping(self.sample_footnotes_df)
        
        test_data = pd.DataFrame({
            'DATA FOOTNOTE': [1, 7, 99, 'invalid'],  # Mix of valid, missing, and invalid
            'ESTIMATE': [0.3, 45.2, 78.9, 12.5]
        })
        
        results = self.processor.validate_footnote_codes(test_data)
        
        assert 1 in results['valid_codes']
        assert 7 in results['valid_codes']
        assert 99 in results['missing_codes']
        assert 'invalid' in results['invalid_codes']


class TestDataCleaner:
    """Test cases for DataCleaner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = DataCleaner()
        
        self.sample_data = pd.DataFrame({
            'SOC 2018 CODE': ['000000', '111011', '291141', 'invalid'],
            'OCCUPATION': ['All workers', 'Chief Executives', 'Registered Nurses', 'Test Mgrs'],
            'ESTIMATE': [79.2, '-', 45.5, 'invalid'],
            'STANDARD ERROR': [2.1, '-', None, 3.2]
        })
    
    def test_clean_occupation_names(self):
        """Test occupation name standardization."""
        result_df = self.cleaner.clean_occupation_names(self.sample_data)
        
        assert 'OCCUPATION_CLEANED' in result_df.columns
        assert 'Managers' in result_df.loc[3, 'OCCUPATION_CLEANED']  # mgrs -> managers
    
    def test_normalize_estimate_values(self):
        """Test estimate value normalization."""
        result_df = self.cleaner.normalize_estimate_values(self.sample_data)
        
        assert 'ESTIMATE_NUMERIC' in result_df.columns
        assert 'ESTIMATE_MISSING' in result_df.columns
        
        # Check numeric conversion
        assert result_df.loc[0, 'ESTIMATE_NUMERIC'] == 79.2
        
        # Check missing data handling
        assert pd.isna(result_df.loc[1, 'ESTIMATE_NUMERIC'])  # '-' should become None
        assert result_df.loc[1, 'ESTIMATE_MISSING'] == True
    
    def test_validate_soc_codes(self):
        """Test SOC code validation."""
        result = self.cleaner.validate_soc_codes(self.sample_data)
        
        assert result.record_count == 4
        assert len(result.warnings) > 0  # Should warn about invalid format
    
    def test_add_soc_code_metadata(self):
        """Test SOC code metadata addition."""
        result_df = self.cleaner.add_soc_code_metadata(self.sample_data)
        
        assert 'SOC_MAJOR_GROUP' in result_df.columns
        assert 'SOC_MAJOR_GROUP_TITLE' in result_df.columns
        assert 'SOC_DETAIL_LEVEL' in result_df.columns
        
        # Check specific values
        assert result_df.loc[0, 'SOC_MAJOR_GROUP'] == '00'  # All workers
        assert result_df.loc[1, 'SOC_MAJOR_GROUP'] == '11'  # Management
        assert result_df.loc[0, 'SOC_DETAIL_LEVEL'] == 'all_workers'
    
    def test_clean_text_fields(self):
        """Test text field cleaning."""
        test_data = pd.DataFrame({
            'OCCUPATION': ['  Test Occupation  ', 'Another   Job', None],
            'REQUIREMENT': ['Physical demands...', 'Cognitive requirements;', 'Normal text']
        })
        
        result_df = self.cleaner.clean_text_fields(test_data, ['OCCUPATION', 'REQUIREMENT'])
        
        assert 'OCCUPATION_CLEANED' in result_df.columns
        assert 'REQUIREMENT_CLEANED' in result_df.columns
        
        # Check whitespace normalization
        assert result_df.loc[0, 'OCCUPATION_CLEANED'] == 'Test Occupation'
        assert result_df.loc[1, 'OCCUPATION_CLEANED'] == 'Another Job'
        
        # Check punctuation removal
        assert not result_df.loc[0, 'REQUIREMENT_CLEANED'].endswith('...')
        assert not result_df.loc[1, 'REQUIREMENT_CLEANED'].endswith(';')
    
    def test_get_cleaning_summary(self):
        """Test cleaning summary generation."""
        original_df = self.sample_data.copy()
        cleaned_df = self.cleaner.clean_occupation_names(self.sample_data)
        cleaned_df = self.cleaner.normalize_estimate_values(cleaned_df)
        cleaned_df = self.cleaner.add_soc_code_metadata(cleaned_df)
        
        summary = self.cleaner.get_cleaning_summary(original_df, cleaned_df)
        
        assert summary['original_records'] == len(original_df)
        assert summary['cleaned_records'] == len(cleaned_df)
        assert summary['columns_added'] > 0
        assert summary['occupation_names_standardized'] == True
        assert summary['estimates_normalized'] == True
        assert summary['soc_codes_validated'] == True


# Integration tests
class TestDataProcessingIntegration:
    """Integration tests for data processing components."""
    
    def test_full_processing_pipeline(self):
        """Test the complete data processing pipeline."""
        # Create sample data files
        sample_data = {
            'SERIES ID': ['ORUC1000000000001197', 'ORUC1000000000001198'],
            'SERIES TITLE': ['Test series 1', 'Test series 2'],
            'SOC 2018 CODE': ['000000', '111011'],
            'OCCUPATION': ['All workers', 'Chief executives'],
            'REQUIREMENT': ['Cognitive and mental requirements', 'Physical demands'],
            'ESTIMATE CODE': ['01197', '01198'],
            'ESTIMATE TEXT': ['Test estimate 1', 'Test estimate 2'],
            'CATEGORY CODE': ['090', '091'],
            'CATEGORY': ['Test category 1', 'Test category 2'],
            'ADDITIVE CODE': ['090', '091'],
            'ADDITIVE': ['Test additive 1', 'Test additive 2'],
            'DATATYPE CODE': ['01', '02'],
            'DATATYPE': ['Percentage', 'Hours'],
            'ESTIMATE': [79.2, '-'],
            'STANDARD ERROR': [2.1, '-'],
            'DATA FOOTNOTE': [7, 1],
            'STANDARD ERROR FOOTNOTE': [6, None],
            'SERIES FOOTNOTE': [None, None]
        }
        
        sample_footnotes = {
            'Footnote code': [1, 6, 7],
            'Footnote text': [
                'Estimate is less than 0.5 percent.',
                'Standard error is less than 0.5.',
                'This estimate is the mode for the category group.'
            ]
        }
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as data_file:
            pd.DataFrame(sample_data).to_csv(data_file.name, index=False)
            data_path = data_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as footnote_file:
            pd.DataFrame(sample_footnotes).to_csv(footnote_file.name, index=False)
            footnote_path = footnote_file.name
        
        try:
            # Initialize components
            loader = CSVLoader()
            processor = FootnoteProcessor()
            cleaner = DataCleaner()
            
            # Load data
            df = loader.load_dataset(data_path)
            footnote_df = loader.load_footnotes(footnote_path)
            
            # Process footnotes
            footnote_mapping = processor.load_footnote_mapping(footnote_df)
            df_with_footnotes = processor.process_footnotes_in_dataframe(df)
            
            # Clean data
            df_cleaned = cleaner.clean_occupation_names(df_with_footnotes)
            df_cleaned = cleaner.normalize_estimate_values(df_cleaned)
            df_cleaned = cleaner.add_soc_code_metadata(df_cleaned)
            
            # Verify the pipeline worked
            assert len(df_cleaned) == 2
            assert 'OCCUPATION_CLEANED' in df_cleaned.columns
            assert 'ESTIMATE_NUMERIC' in df_cleaned.columns
            assert 'SOC_MAJOR_GROUP' in df_cleaned.columns
            assert 'DATA FOOTNOTE_INTERPRETATION' in df_cleaned.columns
            
            # Verify specific processing
            assert df_cleaned.loc[0, 'ESTIMATE_NUMERIC'] == 79.2
            assert pd.isna(df_cleaned.loc[1, 'ESTIMATE_NUMERIC'])  # '-' should be None
            assert df_cleaned.loc[1, 'DATA FOOTNOTE_RANGE_ESTIMATE'] == 0.25  # footnote 1 range
            
        finally:
            # Clean up temporary files
            os.unlink(data_path)
            os.unlink(footnote_path)

class TestDataProcessor:
    """Test cases for the complete DataProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.data_processing.data_processor import DataProcessor
        self.processor = DataProcessor()
        
        # Sample data for testing
        self.sample_data = {
            'SERIES ID': ['ORUC1000000000001197', 'ORUC1000000000001198'],
            'SERIES TITLE': ['Test series 1', 'Test series 2'],
            'SOC 2018 CODE': ['000000', '111011'],
            'OCCUPATION': ['All workers', 'Chief executives'],
            'REQUIREMENT': ['Cognitive and mental requirements', 'Physical demands'],
            'ESTIMATE CODE': ['01197', '01198'],
            'ESTIMATE TEXT': ['Test estimate 1', 'Test estimate 2'],
            'CATEGORY CODE': ['090', '091'],
            'CATEGORY': ['Test category 1', 'Test category 2'],
            'ADDITIVE CODE': ['090', '091'],
            'ADDITIVE': ['Test additive 1', 'Test additive 2'],
            'DATATYPE CODE': ['01', '02'],
            'DATATYPE': ['Percentage', 'Hours'],
            'ESTIMATE': [79.2, '-'],
            'STANDARD ERROR': [2.1, '-'],
            'DATA FOOTNOTE': [7, 1],
            'STANDARD ERROR FOOTNOTE': [6, None],
            'SERIES FOOTNOTE': [None, None]
        }
        
        self.sample_footnotes = {
            'Footnote code': [1, 6, 7],
            'Footnote text': [
                'Estimate is less than 0.5 percent.',
                'Standard error is less than 0.5.',
                'This estimate is the mode for the category group.'
            ]
        }
    
    def test_complete_processing_pipeline(self):
        """Test the complete data processing pipeline through DataProcessor."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as data_file:
            pd.DataFrame(self.sample_data).to_csv(data_file.name, index=False)
            data_path = data_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as footnote_file:
            pd.DataFrame(self.sample_footnotes).to_csv(footnote_file.name, index=False)
            footnote_path = footnote_file.name
        
        try:
            # Process complete dataset
            result_df = self.processor.process_complete_dataset(data_path, footnote_path)
            
            # Verify all processing steps were applied
            assert len(result_df) == 2
            
            # Check footnote processing
            assert 'DATA FOOTNOTE_INTERPRETATION' in result_df.columns
            assert 'DATA FOOTNOTE_RANGE_ESTIMATE' in result_df.columns
            
            # Check occupation cleaning
            assert 'OCCUPATION_CLEANED' in result_df.columns
            
            # Check estimate normalization
            assert 'ESTIMATE_NUMERIC' in result_df.columns
            assert 'ESTIMATE_MISSING' in result_df.columns
            
            # Check SOC metadata
            assert 'SOC_MAJOR_GROUP' in result_df.columns
            assert 'SOC_DETAIL_LEVEL' in result_df.columns
            
            # Check text cleaning
            assert 'SERIES TITLE_CLEANED' in result_df.columns
            
            # Verify specific values
            assert result_df.loc[0, 'ESTIMATE_NUMERIC'] == 79.2
            assert pd.isna(result_df.loc[1, 'ESTIMATE_NUMERIC'])  # '-' should be None
            assert result_df.loc[1, 'DATA FOOTNOTE_RANGE_ESTIMATE'] == 0.25  # footnote 1 range
            
        finally:
            # Clean up temporary files
            os.unlink(data_path)
            os.unlink(footnote_path)
    
    def test_get_processing_summary(self):
        """Test processing summary generation."""
        # Create a simple processed DataFrame
        df = pd.DataFrame(self.sample_data)
        df['DATA FOOTNOTE_INTERPRETATION'] = ['test', 'test']
        df['OCCUPATION_CLEANED'] = ['All Workers', 'Chief Executives']
        df['ESTIMATE_NUMERIC'] = [79.2, None]
        df['SOC_MAJOR_GROUP'] = ['00', '11']
        
        summary = self.processor.get_processing_summary(df)
        
        assert 'dataset_summary' in summary
        assert 'footnote_summary' in summary
        assert 'soc_validation' in summary
        assert 'processing_columns' in summary
        
        # Check processing columns detection
        processing_cols = summary['processing_columns']
        assert len(processing_cols['footnote_columns']) > 0
        assert len(processing_cols['cleaned_columns']) > 0
        assert len(processing_cols['numeric_columns']) > 0
        assert len(processing_cols['metadata_columns']) > 0