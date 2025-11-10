"""
Concrete implementation of the DataProcessor interface.
"""

import pandas as pd
import logging
from typing import Dict, Optional
from pathlib import Path

from ..interfaces import DataProcessorInterface, ValidationResult, FootnoteReference
from .csv_loader import CSVLoader
from .footnote_processor import FootnoteProcessor
from .data_cleaner import DataCleaner


class DataProcessor(DataProcessorInterface):
    """
    Concrete implementation of the data processing interface.
    
    This class combines all data processing components into a unified interface
    that implements the DataProcessorInterface from the interfaces module.
    """
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize the data processor with all component classes.
        
        Args:
            encoding: Default file encoding to use
        """
        self.logger = logging.getLogger(__name__)
        self.csv_loader = CSVLoader(encoding)
        self.footnote_processor = FootnoteProcessor()
        self.data_cleaner = DataCleaner()
        
        # Cache for loaded footnote mapping
        self._footnote_mapping: Optional[Dict[int, FootnoteReference]] = None
    
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load the main occupational dataset from CSV file.
        
        Args:
            file_path: Path to the main dataset CSV file
            
        Returns:
            pandas.DataFrame: Loaded and validated dataset
        """
        self.logger.info(f"Loading dataset from: {file_path}")
        return self.csv_loader.load_dataset(file_path)
    
    def load_footnotes(self, footnote_path: str) -> Dict[int, FootnoteReference]:
        """
        Load and parse footnote reference data.
        
        Args:
            footnote_path: Path to the footnote codes CSV file
            
        Returns:
            Dict mapping footnote codes to FootnoteReference objects
        """
        self.logger.info(f"Loading footnotes from: {footnote_path}")
        
        # Load the footnote DataFrame
        footnote_df = self.csv_loader.load_footnotes(footnote_path)
        
        # Process into footnote mapping
        self._footnote_mapping = self.footnote_processor.load_footnote_mapping(footnote_df)
        
        return self._footnote_mapping
    
    def validate_columns(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate that all required columns are present and properly formatted.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult: Validation results with errors and warnings
        """
        return self.csv_loader.validate_columns(df)
    
    def process_footnotes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and interpret footnote codes in the dataset.
        
        Args:
            df: DataFrame containing footnote columns
            
        Returns:
            DataFrame with processed footnote interpretations
        """
        self.logger.info("Processing footnotes in dataset")
        
        if self._footnote_mapping is None:
            self.logger.warning("No footnote mapping loaded. Footnote processing may be incomplete.")
        
        return self.footnote_processor.process_footnotes_in_dataframe(df)
    
    def clean_occupation_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize occupation names and categories.
        
        Args:
            df: DataFrame containing occupation data
            
        Returns:
            DataFrame with standardized occupation names
        """
        self.logger.info("Cleaning and standardizing occupation names")
        return self.data_cleaner.clean_occupation_names(df)
    
    def handle_estimate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert range estimates to numeric values.
        
        Args:
            df: DataFrame containing estimate data
            
        Returns:
            DataFrame with normalized estimate values
        """
        self.logger.info("Handling estimate ranges and normalizing values")
        return self.data_cleaner.normalize_estimate_values(df)
    
    def process_complete_dataset(self, dataset_path: str, footnote_path: str, 
                               field_desc_path: Optional[str] = None) -> pd.DataFrame:
        """
        Process the complete dataset through all cleaning and standardization steps.
        
        Args:
            dataset_path: Path to the main dataset CSV file
            footnote_path: Path to the footnote codes CSV file
            field_desc_path: Optional path to field descriptions CSV file
            
        Returns:
            Fully processed DataFrame
        """
        self.logger.info("Starting complete dataset processing pipeline")
        
        # Load footnotes first
        self.load_footnotes(footnote_path)
        
        # Load main dataset
        df = self.load_dataset(dataset_path)
        
        # Load field descriptions if provided
        if field_desc_path:
            field_descriptions = self.csv_loader.load_field_descriptions(field_desc_path)
            self.logger.info(f"Loaded {len(field_descriptions)} field descriptions")
        
        # Process footnotes
        df = self.process_footnotes(df)
        
        # Clean occupation names
        df = self.clean_occupation_names(df)
        
        # Handle estimate ranges and normalize values
        df = self.handle_estimate_ranges(df)
        
        # Add SOC code metadata
        df = self.data_cleaner.add_soc_code_metadata(df)
        
        # Clean text fields
        df = self.data_cleaner.clean_text_fields(df)
        
        # Validate SOC codes
        soc_validation = self.data_cleaner.validate_soc_codes(df)
        if not soc_validation.is_valid:
            self.logger.warning(f"SOC code validation issues: {soc_validation.errors}")
        
        for warning in soc_validation.warnings:
            self.logger.warning(f"SOC validation warning: {warning}")
        
        self.logger.info(f"Dataset processing complete: {len(df)} records processed")
        
        return df
    
    def get_processing_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Get a comprehensive summary of the processed dataset.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary containing processing summary statistics
        """
        summary = {
            'dataset_summary': self.csv_loader.get_dataset_summary(df),
            'footnote_summary': self.footnote_processor.get_footnote_summary(),
            'soc_validation': self.data_cleaner.validate_soc_codes(df),
            'processing_columns': {
                'footnote_columns': [col for col in df.columns if 'FOOTNOTE' in col and 'INTERPRETATION' in col],
                'cleaned_columns': [col for col in df.columns if 'CLEANED' in col],
                'numeric_columns': [col for col in df.columns if 'NUMERIC' in col],
                'metadata_columns': [col for col in df.columns if col.startswith('SOC_')]
            }
        }
        
        return summary