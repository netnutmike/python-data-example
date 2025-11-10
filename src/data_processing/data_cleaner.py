"""
Data cleaning and standardization for the occupation data reports application.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from ..interfaces import ValidationResult, DataProcessingError


class DataCleaner:
    """
    Handles data cleaning and standardization operations.
    
    This class implements the DataCleaner functionality as specified in task 2.3:
    - Occupation name standardization
    - Estimate value normalization and missing data handling
    - SOC code validation against 2018 classification system
    """
    
    # Standard SOC 2018 major groups (first 2 digits)
    VALID_SOC_MAJOR_GROUPS = {
        '11': 'Management Occupations',
        '13': 'Business and Financial Operations Occupations',
        '15': 'Computer and Mathematical Occupations',
        '17': 'Architecture and Engineering Occupations',
        '19': 'Life, Physical, and Social Science Occupations',
        '21': 'Community and Social Service Occupations',
        '23': 'Legal Occupations',
        '25': 'Educational Instruction and Library Occupations',
        '27': 'Arts, Design, Entertainment, Sports, and Media Occupations',
        '29': 'Healthcare Practitioners and Technical Occupations',
        '31': 'Healthcare Support Occupations',
        '33': 'Protective Service Occupations',
        '35': 'Food Preparation and Serving Related Occupations',
        '37': 'Building and Grounds Cleaning and Maintenance Occupations',
        '39': 'Personal Care and Service Occupations',
        '41': 'Sales and Related Occupations',
        '43': 'Office and Administrative Support Occupations',
        '45': 'Farming, Fishing, and Forestry Occupations',
        '47': 'Construction and Extraction Occupations',
        '49': 'Installation, Maintenance, and Repair Occupations',
        '51': 'Production Occupations',
        '53': 'Transportation and Material Moving Occupations',
        '55': 'Military Specific Occupations'
    }
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.logger = logging.getLogger(__name__)
        self._occupation_name_mappings: Dict[str, str] = {}
        self._initialize_occupation_mappings()
    
    def _initialize_occupation_mappings(self):
        """Initialize common occupation name standardization mappings."""
        # Common abbreviations and variations
        self._occupation_name_mappings = {
            # Common abbreviations
            'mgrs': 'managers',
            'mgr': 'manager',
            'admin': 'administrative',
            'asst': 'assistant',
            'assoc': 'associate',
            'tech': 'technician',
            'techs': 'technicians',
            'rep': 'representative',
            'reps': 'representatives',
            'spec': 'specialist',
            'specs': 'specialists',
            'coord': 'coordinator',
            'coords': 'coordinators',
            'supv': 'supervisor',
            'supvs': 'supervisors',
            
            # Common variations
            'healthcare': 'health care',
            'childcare': 'child care',
            'homecare': 'home care',
            'daycare': 'day care',
            
            # Standardize punctuation
            'n.e.c.': 'nec',
            'n.o.s.': 'nos',
        }
    
    def clean_occupation_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize occupation names and categories.
        
        Args:
            df: DataFrame containing occupation data
            
        Returns:
            DataFrame with standardized occupation names
        """
        df_cleaned = df.copy()
        
        if 'OCCUPATION' not in df_cleaned.columns:
            self.logger.warning("OCCUPATION column not found, skipping occupation name cleaning")
            return df_cleaned
        
        self.logger.info("Cleaning and standardizing occupation names")
        
        # Create a cleaned occupation column
        df_cleaned['OCCUPATION_CLEANED'] = df_cleaned['OCCUPATION'].apply(self._standardize_occupation_name)
        
        # Log cleaning statistics
        original_unique = df_cleaned['OCCUPATION'].nunique()
        cleaned_unique = df_cleaned['OCCUPATION_CLEANED'].nunique()
        
        self.logger.info(f"Occupation name cleaning: {original_unique} -> {cleaned_unique} unique names")
        
        return df_cleaned
    
    def _standardize_occupation_name(self, occupation_name: str) -> str:
        """
        Standardize a single occupation name.
        
        Args:
            occupation_name: Original occupation name
            
        Returns:
            Standardized occupation name
        """
        if pd.isna(occupation_name):
            return occupation_name
        
        # Convert to string and strip whitespace
        name = str(occupation_name).strip()
        
        # Convert to lowercase for processing
        name_lower = name.lower()
        
        # Apply common mappings
        for abbrev, full_form in self._occupation_name_mappings.items():
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            name_lower = re.sub(pattern, full_form, name_lower)
        
        # Standardize whitespace
        name_lower = re.sub(r'\s+', ' ', name_lower)
        
        # Remove extra punctuation but keep essential ones
        name_lower = re.sub(r'[^\w\s\-,./()]', '', name_lower)
        
        # Convert back to title case, preserving some common acronyms
        name_standardized = self._apply_title_case(name_lower)
        
        return name_standardized
    
    def _apply_title_case(self, text: str) -> str:
        """
        Apply title case while preserving common acronyms and special cases.
        
        Args:
            text: Text to convert
            
        Returns:
            Text in proper title case
        """
        # Common acronyms that should stay uppercase
        acronyms = {'CEO', 'CFO', 'CTO', 'IT', 'HR', 'PR', 'QA', 'RN', 'LPN', 'EMT', 'CPA', 'MD', 'PhD'}
        
        # Words that should stay lowercase (articles, prepositions, etc.)
        lowercase_words = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'if', 'in', 'of', 'on', 'or', 'the', 'to', 'up'}
        
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            word_upper = word.upper()
            
            # Check if it's a known acronym
            if word_upper in acronyms:
                result.append(word_upper)
            # First word or not a lowercase word
            elif i == 0 or word.lower() not in lowercase_words:
                result.append(word.capitalize())
            else:
                result.append(word.lower())
        
        return ' '.join(result)
    
    def normalize_estimate_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize estimate values and handle missing data.
        
        Args:
            df: DataFrame containing estimate data
            
        Returns:
            DataFrame with normalized estimate values
        """
        df_normalized = df.copy()
        
        if 'ESTIMATE' not in df_normalized.columns:
            self.logger.warning("ESTIMATE column not found, skipping estimate normalization")
            return df_normalized
        
        self.logger.info("Normalizing estimate values and handling missing data")
        
        # Create normalized estimate column
        df_normalized['ESTIMATE_NUMERIC'] = df_normalized['ESTIMATE'].apply(self._convert_estimate_to_numeric)
        
        # Handle standard errors similarly
        if 'STANDARD ERROR' in df_normalized.columns:
            df_normalized['STANDARD_ERROR_NUMERIC'] = df_normalized['STANDARD ERROR'].apply(self._convert_estimate_to_numeric)
        
        # Add flags for missing data
        df_normalized['ESTIMATE_MISSING'] = df_normalized['ESTIMATE'].apply(lambda x: x == '-' or pd.isna(x))
        
        if 'STANDARD ERROR' in df_normalized.columns:
            df_normalized['STANDARD_ERROR_MISSING'] = df_normalized['STANDARD ERROR'].apply(lambda x: x == '-' or pd.isna(x))
        
        # Log normalization statistics
        total_records = len(df_normalized)
        missing_estimates = df_normalized['ESTIMATE_MISSING'].sum()
        
        self.logger.info(f"Estimate normalization: {missing_estimates}/{total_records} missing estimates")
        
        return df_normalized
    
    def _convert_estimate_to_numeric(self, value) -> Optional[float]:
        """
        Convert estimate value to numeric, handling missing data markers.
        
        Args:
            value: Estimate value to convert
            
        Returns:
            Numeric value or None for missing data
        """
        if pd.isna(value) or value == '-' or value == '':
            return None
        
        try:
            # Handle string values
            if isinstance(value, str):
                # Remove any whitespace
                value = value.strip()
                
                # Handle common non-numeric markers
                if value in ['-', 'N/A', 'n/a', 'NA', 'null', 'NULL']:
                    return None
                
                # Try to convert to float
                return float(value)
            
            # Handle numeric values
            return float(value)
            
        except (ValueError, TypeError):
            self.logger.warning(f"Could not convert estimate value to numeric: {value}")
            return None
    
    def validate_soc_codes(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate SOC codes against 2018 classification system.
        
        Args:
            df: DataFrame containing SOC codes
            
        Returns:
            ValidationResult with SOC code validation details
        """
        errors = []
        warnings = []
        
        if 'SOC 2018 CODE' not in df.columns:
            errors.append("SOC 2018 CODE column not found")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings, record_count=0)
        
        self.logger.info("Validating SOC codes against 2018 classification system")
        
        soc_codes = df['SOC 2018 CODE'].dropna()
        total_codes = len(soc_codes)
        
        # Check format (6 digits)
        invalid_format = soc_codes[~soc_codes.astype(str).str.match(r'^\d{6}$')]
        if not invalid_format.empty:
            warnings.append(f"Found {len(invalid_format)} SOC codes with invalid format")
        
        # Check major groups
        valid_format_codes = soc_codes[soc_codes.astype(str).str.match(r'^\d{6}$')]
        invalid_major_groups = []
        
        for code in valid_format_codes.unique():
            code_str = str(code)
            major_group = code_str[:2]
            
            # Allow '00' for aggregate categories
            if major_group != '00' and major_group not in self.VALID_SOC_MAJOR_GROUPS:
                invalid_major_groups.append(code)
        
        if invalid_major_groups:
            warnings.append(f"Found {len(invalid_major_groups)} SOC codes with unrecognized major groups")
        
        # Check for common issues
        duplicate_codes = soc_codes.duplicated().sum()
        if duplicate_codes > 0:
            warnings.append(f"Found {duplicate_codes} duplicate SOC code entries")
        
        # Summary statistics
        unique_major_groups = set()
        for code in valid_format_codes.unique():
            code_str = str(code)
            if len(code_str) >= 2:
                unique_major_groups.add(code_str[:2])
        
        self.logger.info(f"SOC validation: {total_codes} codes, {len(unique_major_groups)} major groups")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            record_count=total_codes
        )
    
    def add_soc_code_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add SOC code metadata columns.
        
        Args:
            df: DataFrame containing SOC codes
            
        Returns:
            DataFrame with additional SOC metadata columns
        """
        df_with_metadata = df.copy()
        
        if 'SOC 2018 CODE' not in df_with_metadata.columns:
            self.logger.warning("SOC 2018 CODE column not found, skipping metadata addition")
            return df_with_metadata
        
        self.logger.info("Adding SOC code metadata")
        
        # Add major group information
        df_with_metadata['SOC_MAJOR_GROUP'] = df_with_metadata['SOC 2018 CODE'].apply(self._get_soc_major_group)
        df_with_metadata['SOC_MAJOR_GROUP_TITLE'] = df_with_metadata['SOC_MAJOR_GROUP'].apply(
            lambda x: self.VALID_SOC_MAJOR_GROUPS.get(x, 'Unknown') if x != '00' else 'All Workers'
        )
        
        # Add detail level information
        df_with_metadata['SOC_DETAIL_LEVEL'] = df_with_metadata['SOC 2018 CODE'].apply(self._get_soc_detail_level)
        
        return df_with_metadata
    
    def _get_soc_major_group(self, soc_code) -> Optional[str]:
        """Extract major group from SOC code."""
        if pd.isna(soc_code):
            return None
        
        code_str = str(soc_code)
        if len(code_str) >= 2:
            return code_str[:2]
        return None
    
    def _get_soc_detail_level(self, soc_code) -> Optional[str]:
        """Determine the detail level of a SOC code."""
        if pd.isna(soc_code):
            return None
        
        code_str = str(soc_code)
        
        if code_str == '000000':
            return 'all_workers'
        elif code_str.endswith('0000'):
            return 'major_group'
        elif code_str.endswith('000'):
            return 'minor_group'
        elif code_str.endswith('00'):
            return 'broad_occupation'
        else:
            return 'detailed_occupation'
    
    def clean_text_fields(self, df: pd.DataFrame, text_columns: List[str] = None) -> pd.DataFrame:
        """
        Clean and standardize text fields.
        
        Args:
            df: DataFrame to clean
            text_columns: List of text columns to clean
            
        Returns:
            DataFrame with cleaned text fields
        """
        if text_columns is None:
            text_columns = ['SERIES TITLE', 'OCCUPATION', 'REQUIREMENT', 'ESTIMATE TEXT', 'CATEGORY', 'ADDITIVE', 'DATATYPE']
        
        df_cleaned = df.copy()
        
        for col in text_columns:
            if col in df_cleaned.columns:
                df_cleaned[f"{col}_CLEANED"] = df_cleaned[col].apply(self._clean_text_field)
        
        return df_cleaned
    
    def _clean_text_field(self, text) -> str:
        """
        Clean a single text field.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return text
        
        text_str = str(text).strip()
        
        # Standardize whitespace
        text_str = re.sub(r'\s+', ' ', text_str)
        
        # Remove leading/trailing punctuation
        text_str = text_str.strip('.,;:')
        
        return text_str
    
    def get_cleaning_summary(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate a summary of cleaning operations performed.
        
        Args:
            original_df: Original DataFrame before cleaning
            cleaned_df: DataFrame after cleaning
            
        Returns:
            Dictionary containing cleaning summary statistics
        """
        summary = {
            'original_records': len(original_df),
            'cleaned_records': len(cleaned_df),
            'columns_added': len(cleaned_df.columns) - len(original_df.columns),
            'occupation_names_standardized': False,
            'estimates_normalized': False,
            'soc_codes_validated': False,
            'missing_estimates': 0,
            'invalid_soc_codes': 0
        }
        
        # Check what cleaning operations were performed
        if 'OCCUPATION_CLEANED' in cleaned_df.columns:
            summary['occupation_names_standardized'] = True
            original_unique = original_df['OCCUPATION'].nunique() if 'OCCUPATION' in original_df.columns else 0
            cleaned_unique = cleaned_df['OCCUPATION_CLEANED'].nunique()
            summary['occupation_standardization'] = {
                'original_unique': original_unique,
                'cleaned_unique': cleaned_unique,
                'reduction': original_unique - cleaned_unique
            }
        
        if 'ESTIMATE_NUMERIC' in cleaned_df.columns:
            summary['estimates_normalized'] = True
            summary['missing_estimates'] = cleaned_df['ESTIMATE_MISSING'].sum() if 'ESTIMATE_MISSING' in cleaned_df.columns else 0
        
        if 'SOC_MAJOR_GROUP' in cleaned_df.columns:
            summary['soc_codes_validated'] = True
            summary['unique_major_groups'] = cleaned_df['SOC_MAJOR_GROUP'].nunique()
        
        return summary