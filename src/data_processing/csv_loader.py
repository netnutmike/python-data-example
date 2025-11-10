"""
CSV data loader with validation for the occupation data reports application.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from ..interfaces import ValidationResult, DataProcessingError


class CSVLoader:
    """
    Handles loading and validation of CSV data files for the ORS dataset.
    
    This class implements the CSVLoader functionality as specified in task 2.1:
    - Load the main dataset and metadata files
    - Validate all 18 required fields are present
    - Handle file encoding and parsing errors gracefully
    """
    
    # Expected columns for the main dataset (18 required fields)
    REQUIRED_COLUMNS = [
        'SERIES ID',
        'SERIES TITLE', 
        'SOC 2018 CODE',
        'OCCUPATION',
        'REQUIREMENT',
        'ESTIMATE CODE',
        'ESTIMATE TEXT',
        'CATEGORY CODE',
        'CATEGORY',
        'ADDITIVE CODE',
        'ADDITIVE',
        'DATATYPE CODE',
        'DATATYPE',
        'ESTIMATE',
        'STANDARD ERROR',
        'DATA FOOTNOTE',
        'STANDARD ERROR FOOTNOTE',
        'SERIES FOOTNOTE'
    ]
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize the CSV loader.
        
        Args:
            encoding: Default file encoding to use
        """
        self.encoding = encoding
        self.logger = logging.getLogger(__name__)
    
    def load_dataset(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load the main occupational dataset from CSV file.
        
        Args:
            file_path: Path to the main dataset CSV file
            
        Returns:
            pandas.DataFrame: Loaded dataset
            
        Raises:
            DataProcessingError: If file cannot be loaded or validated
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DataProcessingError(f"Dataset file not found: {file_path}")
        
        try:
            self.logger.info(f"Loading dataset from: {file_path}")
            
            # Try different encodings if the default fails
            encodings_to_try = [self.encoding, 'utf-8-sig', 'latin-1', 'cp1252']
            
            df = None
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    self.logger.info(f"Successfully loaded dataset with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    self.logger.warning(f"Failed to load with encoding {encoding}, trying next...")
                    continue
            
            if df is None:
                raise DataProcessingError(f"Could not load file with any supported encoding: {encodings_to_try}")
            
            # Validate the loaded dataset
            validation_result = self.validate_columns(df)
            if not validation_result.is_valid:
                error_msg = f"Dataset validation failed: {'; '.join(validation_result.errors)}"
                raise DataProcessingError(error_msg)
            
            # Log warnings if any
            for warning in validation_result.warnings:
                self.logger.warning(warning)
            
            self.logger.info(f"Dataset loaded successfully: {len(df)} records, {len(df.columns)} columns")
            return df
            
        except pd.errors.EmptyDataError:
            raise DataProcessingError(f"Dataset file is empty: {file_path}")
        except pd.errors.ParserError as e:
            raise DataProcessingError(f"Failed to parse CSV file: {e}")
        except Exception as e:
            raise DataProcessingError(f"Unexpected error loading dataset: {e}")
    
    def load_footnotes(self, footnote_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load footnote reference data from CSV file.
        
        Args:
            footnote_path: Path to the footnote codes CSV file
            
        Returns:
            pandas.DataFrame: Footnote reference data
            
        Raises:
            DataProcessingError: If footnote file cannot be loaded
        """
        footnote_path = Path(footnote_path)
        
        if not footnote_path.exists():
            raise DataProcessingError(f"Footnote file not found: {footnote_path}")
        
        try:
            self.logger.info(f"Loading footnotes from: {footnote_path}")
            
            # Try different encodings
            encodings_to_try = [self.encoding, 'utf-8-sig', 'latin-1', 'cp1252']
            
            df = None
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(footnote_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise DataProcessingError(f"Could not load footnote file with any supported encoding")
            
            # Validate footnote structure
            expected_footnote_columns = ['Footnote code', 'Footnote text']
            missing_cols = [col for col in expected_footnote_columns if col not in df.columns]
            
            if missing_cols:
                raise DataProcessingError(f"Missing required footnote columns: {missing_cols}")
            
            # Ensure footnote codes are integers
            try:
                df['Footnote code'] = pd.to_numeric(df['Footnote code'], errors='coerce')
                invalid_codes = df[df['Footnote code'].isna()]
                if not invalid_codes.empty:
                    self.logger.warning(f"Found {len(invalid_codes)} invalid footnote codes")
                    df = df.dropna(subset=['Footnote code'])
                
                df['Footnote code'] = df['Footnote code'].astype(int)
            except Exception as e:
                raise DataProcessingError(f"Failed to process footnote codes: {e}")
            
            self.logger.info(f"Footnotes loaded successfully: {len(df)} footnote codes")
            return df
            
        except Exception as e:
            if isinstance(e, DataProcessingError):
                raise
            raise DataProcessingError(f"Unexpected error loading footnotes: {e}")
    
    def load_field_descriptions(self, field_desc_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load field descriptions from CSV file.
        
        Args:
            field_desc_path: Path to the field descriptions CSV file
            
        Returns:
            pandas.DataFrame: Field descriptions data
            
        Raises:
            DataProcessingError: If field descriptions file cannot be loaded
        """
        field_desc_path = Path(field_desc_path)
        
        if not field_desc_path.exists():
            raise DataProcessingError(f"Field descriptions file not found: {field_desc_path}")
        
        try:
            self.logger.info(f"Loading field descriptions from: {field_desc_path}")
            
            # Try different encodings
            encodings_to_try = [self.encoding, 'utf-8-sig', 'latin-1', 'cp1252']
            
            df = None
            for encoding in encodings_to_try:
                try:
                    # Skip the header information and read from where the actual field data starts
                    df = pd.read_csv(field_desc_path, encoding=encoding, skiprows=12)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise DataProcessingError(f"Could not load field descriptions with any supported encoding")
            
            # Clean up the dataframe - remove empty rows
            df = df.dropna(how='all')
            
            self.logger.info(f"Field descriptions loaded successfully: {len(df)} field descriptions")
            return df
            
        except Exception as e:
            if isinstance(e, DataProcessingError):
                raise
            raise DataProcessingError(f"Unexpected error loading field descriptions: {e}")
    
    def validate_columns(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate that all required columns are present and properly formatted.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult: Validation results with errors and warnings
        """
        errors = []
        warnings = []
        
        # Check for required columns
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for extra columns (not necessarily an error, but worth noting)
        extra_columns = [col for col in df.columns if col not in self.REQUIRED_COLUMNS]
        if extra_columns:
            warnings.append(f"Found unexpected columns: {extra_columns}")
        
        # Check for empty dataset
        if len(df) == 0:
            errors.append("Dataset is empty")
        
        # Validate data types for key columns
        if 'ESTIMATE' in df.columns:
            # Check if ESTIMATE column has numeric values (allowing for '-' as missing)
            non_numeric_estimates = df[
                (df['ESTIMATE'] != '-') & 
                (~pd.to_numeric(df['ESTIMATE'], errors='coerce').notna())
            ]
            if not non_numeric_estimates.empty:
                warnings.append(f"Found {len(non_numeric_estimates)} non-numeric estimate values")
        
        if 'SOC 2018 CODE' in df.columns:
            # Check SOC code format (should be 6 digits, but allow for '000000' for aggregates)
            invalid_soc_codes = df[
                ~df['SOC 2018 CODE'].astype(str).str.match(r'^\d{6}$')
            ]
            if not invalid_soc_codes.empty:
                warnings.append(f"Found {len(invalid_soc_codes)} invalid SOC code formats")
        
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            warnings.append(f"Found {empty_rows} completely empty rows")
        
        # Check data footnote values
        if 'DATA FOOTNOTE' in df.columns:
            footnote_values = df['DATA FOOTNOTE'].dropna()
            if not footnote_values.empty:
                try:
                    numeric_footnotes = pd.to_numeric(footnote_values, errors='coerce')
                    invalid_footnotes = footnote_values[numeric_footnotes.isna()]
                    if not invalid_footnotes.empty:
                        warnings.append(f"Found {len(invalid_footnotes)} invalid footnote codes")
                except Exception:
                    warnings.append("Could not validate footnote codes")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            record_count=len(df)
        )
    
    def get_dataset_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate a summary of the loaded dataset.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dict containing dataset summary statistics
        """
        summary = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'unique_soc_codes': df['SOC 2018 CODE'].nunique() if 'SOC 2018 CODE' in df.columns else 0,
            'unique_occupations': df['OCCUPATION'].nunique() if 'OCCUPATION' in df.columns else 0,
            'requirement_types': df['REQUIREMENT'].unique().tolist() if 'REQUIREMENT' in df.columns else [],
            'estimate_range': self._get_estimate_range(df) if 'ESTIMATE' in df.columns else None,
            'missing_estimates': (df['ESTIMATE'] == '-').sum() if 'ESTIMATE' in df.columns else 0,
            'records_with_footnotes': df['DATA FOOTNOTE'].notna().sum() if 'DATA FOOTNOTE' in df.columns else 0
        }
        
        return summary
    
    def _get_estimate_range(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Get estimate range, handling non-numeric values safely.
        
        Args:
            df: DataFrame containing estimate data
            
        Returns:
            Dictionary with min/max estimates or None if no numeric values
        """
        try:
            # Filter out non-numeric values and missing data
            numeric_estimates = pd.to_numeric(df['ESTIMATE'], errors='coerce')
            numeric_estimates = numeric_estimates.dropna()
            
            if len(numeric_estimates) > 0:
                return {
                    'min': numeric_estimates.min(),
                    'max': numeric_estimates.max(),
                    'numeric_count': len(numeric_estimates),
                    'non_numeric_count': len(df) - len(numeric_estimates)
                }
            else:
                return {
                    'min': None,
                    'max': None,
                    'numeric_count': 0,
                    'non_numeric_count': len(df)
                }
        except Exception:
            return {
                'min': None,
                'max': None,
                'numeric_count': 0,
                'non_numeric_count': len(df)
            }