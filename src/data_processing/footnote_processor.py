"""
Footnote processing system for the occupation data reports application.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from ..interfaces import FootnoteReference, DataProcessingError, FootnoteInterpretationError


class FootnoteProcessor:
    """
    Handles footnote code interpretation and range estimate conversion.
    
    This class implements the FootnoteProcessor functionality as specified in task 2.2:
    - Load and interpret footnote codes (codes 1-36) with precision indicators
    - Convert range estimates to numeric values
    - Map footnote codes with precision indicators
    """
    
    def __init__(self):
        """Initialize the footnote processor."""
        self.logger = logging.getLogger(__name__)
        self._footnote_mapping: Dict[int, FootnoteReference] = {}
        self._range_conversions: Dict[int, Tuple[float, float]] = {}
        self._initialize_range_conversions()
    
    def _initialize_range_conversions(self):
        """Initialize the range conversion mappings for footnote codes."""
        # Range conversions based on footnote codes
        # For "less than" footnotes, we use (0, threshold)
        # For "greater than" footnotes, we use (threshold, 100)
        self._range_conversions = {
            1: (0.0, 0.5),      # less than 0.5 percent
            16: (0.0, 5.0),     # less than 5 percent
            17: (0.0, 10.0),    # less than 10 percent
            18: (0.0, 15.0),    # less than 15 percent
            19: (0.0, 20.0),    # less than 20 percent
            20: (0.0, 25.0),    # less than 25 percent
            21: (0.0, 30.0),    # less than 30 percent
            22: (0.0, 35.0),    # less than 35 percent
            23: (0.0, 40.0),    # less than 40 percent
            24: (0.0, 45.0),    # less than 45 percent
            25: (0.0, 50.0),    # less than 50 percent
            26: (50.0, 100.0),  # greater than 50 percent
            27: (55.0, 100.0),  # greater than 55 percent
            28: (60.0, 100.0),  # greater than 60 percent
            29: (65.0, 100.0),  # greater than 65 percent
            30: (70.0, 100.0),  # greater than 70 percent
            31: (75.0, 100.0),  # greater than 75 percent
            32: (80.0, 100.0),  # greater than 80 percent
            33: (85.0, 100.0),  # greater than 85 percent
            34: (90.0, 100.0),  # greater than 90 percent
            35: (95.0, 100.0),  # greater than 95 percent
            36: (99.5, 100.0),  # greater than 99.5 percent
        }
    
    def load_footnote_mapping(self, footnote_df: pd.DataFrame) -> Dict[int, FootnoteReference]:
        """
        Load and create footnote reference mapping from DataFrame.
        
        Args:
            footnote_df: DataFrame containing footnote codes and descriptions
            
        Returns:
            Dict mapping footnote codes to FootnoteReference objects
            
        Raises:
            FootnoteInterpretationError: If footnote data cannot be processed
        """
        try:
            self.logger.info("Loading footnote mapping from DataFrame")
            
            if 'Footnote code' not in footnote_df.columns or 'Footnote text' not in footnote_df.columns:
                raise FootnoteInterpretationError("Required footnote columns not found")
            
            footnote_mapping = {}
            
            for _, row in footnote_df.iterrows():
                code = int(row['Footnote code'])
                text = str(row['Footnote text'])
                
                # Determine precision level based on footnote content
                precision_level = self._determine_precision_level(code, text)
                
                # Generate interpretation guidance
                interpretation_guidance = self._generate_interpretation_guidance(code, text)
                
                footnote_ref = FootnoteReference(
                    code=code,
                    description=text,
                    precision_level=precision_level,
                    interpretation_guidance=interpretation_guidance
                )
                
                footnote_mapping[code] = footnote_ref
            
            self._footnote_mapping = footnote_mapping
            self.logger.info(f"Loaded {len(footnote_mapping)} footnote codes")
            
            return footnote_mapping
            
        except Exception as e:
            raise FootnoteInterpretationError(f"Failed to load footnote mapping: {e}")
    
    def _determine_precision_level(self, code: int, text: str) -> str:
        """
        Determine the precision level for a footnote code.
        
        Args:
            code: Footnote code
            text: Footnote description text
            
        Returns:
            String indicating precision level
        """
        text_lower = text.lower()
        
        # High precision issues
        if any(phrase in text_lower for phrase in ['less than 0.05', 'less than 0.5']):
            return 'very_low_precision'
        
        # Range estimates (low precision)
        if code in self._range_conversions:
            return 'low_precision'
        
        # Standard error related
        if 'standard error' in text_lower:
            if 'not available' in text_lower:
                return 'no_precision_data'
            else:
                return 'precision_indicator'
        
        # Special cases and methodology notes
        if any(phrase in text_lower for phrase in ['mode', 'demonstration', 'prerequisite']):
            return 'methodological_note'
        
        # Default for other footnotes
        return 'standard_precision'
    
    def _generate_interpretation_guidance(self, code: int, text: str) -> str:
        """
        Generate interpretation guidance for a footnote code.
        
        Args:
            code: Footnote code
            text: Footnote description text
            
        Returns:
            String with interpretation guidance
        """
        text_lower = text.lower()
        
        # Range estimates
        if code in self._range_conversions:
            min_val, max_val = self._range_conversions[code]
            return f"Use range estimate: {min_val}% to {max_val}%. Consider midpoint ({(min_val + max_val) / 2}%) for calculations."
        
        # Very small values
        if 'less than 0.5 percent' in text_lower:
            return "Very small percentage, treat as approximately 0.25% for calculations."
        elif 'less than 0.05 hours' in text_lower:
            return "Very small time value, treat as approximately 0.025 hours for calculations."
        elif 'less than 0.05 pounds' in text_lower:
            return "Very small weight value, treat as approximately 0.025 pounds for calculations."
        
        # Standard error guidance
        if 'standard error' in text_lower:
            if 'not available' in text_lower:
                return "No precision data available. Use estimate with caution in statistical analyses."
            else:
                return "Standard error is very small, indicating high precision estimate."
        
        # Methodological notes
        if 'mode' in text_lower:
            return "This is the most common value in the category group."
        elif 'demonstration' in text_lower:
            return "Short preparation time (4 hours or less) required."
        elif 'prerequisite' in text_lower:
            if 'no time' in text_lower or 'combined' in text_lower:
                return "Time requirement is either not applicable or included in other requirements."
            else:
                return "Time requirement is separate from other education/training requirements."
        
        # Standing/sitting clarifications
        if 'standing' in text_lower and 'walking' in text_lower:
            return "Standing time includes both standing and walking activities."
        elif 'sitting' in text_lower and 'lying' in text_lower:
            return "Sitting time includes both sitting and lying down activities."
        
        # Flexibility indicators
        if 'flexibility' in text_lower or 'choose' in text_lower:
            return "Workers have flexibility in this requirement - no fixed assignment."
        
        # Literacy requirements
        if 'literacy' in text_lower and 'minimum education' in text_lower:
            return "Workers with minimum education requirements are excluded from literacy percentages."
        
        # Default guidance
        return "Apply footnote context when interpreting this estimate."
    
    def convert_range_estimate(self, footnote_code: int, method: str = 'midpoint') -> Optional[float]:
        """
        Convert a range estimate footnote to a numeric value.
        
        Args:
            footnote_code: The footnote code indicating a range
            method: Conversion method ('midpoint', 'conservative', 'optimistic')
            
        Returns:
            Numeric value representing the range, or None if not a range footnote
        """
        if footnote_code not in self._range_conversions:
            return None
        
        min_val, max_val = self._range_conversions[footnote_code]
        
        if method == 'midpoint':
            return (min_val + max_val) / 2
        elif method == 'conservative':
            # For "less than" use the minimum, for "greater than" use the threshold
            return min_val if footnote_code <= 25 else min_val
        elif method == 'optimistic':
            # For "less than" use the maximum, for "greater than" use the maximum
            return max_val if footnote_code <= 25 else max_val
        else:
            raise ValueError(f"Unknown conversion method: {method}")
    
    def process_footnotes_in_dataframe(self, df: pd.DataFrame, 
                                     footnote_columns: List[str] = None,
                                     add_interpretation_columns: bool = True) -> pd.DataFrame:
        """
        Process footnotes in a DataFrame and add interpretation columns.
        
        Args:
            df: DataFrame containing footnote columns
            footnote_columns: List of footnote column names to process
            add_interpretation_columns: Whether to add interpretation columns
            
        Returns:
            DataFrame with processed footnotes
        """
        if footnote_columns is None:
            footnote_columns = ['DATA FOOTNOTE', 'STANDARD ERROR FOOTNOTE', 'SERIES FOOTNOTE']
        
        df_processed = df.copy()
        
        for col in footnote_columns:
            if col not in df_processed.columns:
                continue
            
            if add_interpretation_columns:
                # Add interpretation columns
                interp_col = f"{col}_INTERPRETATION"
                precision_col = f"{col}_PRECISION"
                range_estimate_col = f"{col}_RANGE_ESTIMATE"
                
                df_processed[interp_col] = df_processed[col].apply(
                    lambda x: self._get_footnote_interpretation(x) if pd.notna(x) else None
                )
                
                df_processed[precision_col] = df_processed[col].apply(
                    lambda x: self._get_footnote_precision(x) if pd.notna(x) else None
                )
                
                df_processed[range_estimate_col] = df_processed[col].apply(
                    lambda x: self.convert_range_estimate(int(float(x))) if pd.notna(x) and str(x).replace('.', '').isdigit() else None
                )
        
        return df_processed
    
    def _get_footnote_interpretation(self, footnote_code: Union[int, str]) -> Optional[str]:
        """Get interpretation text for a footnote code."""
        try:
            code = int(footnote_code)
            if code in self._footnote_mapping:
                return self._footnote_mapping[code].interpretation_guidance
        except (ValueError, TypeError):
            pass
        return None
    
    def _get_footnote_precision(self, footnote_code: Union[int, str]) -> Optional[str]:
        """Get precision level for a footnote code."""
        try:
            code = int(footnote_code)
            if code in self._footnote_mapping:
                return self._footnote_mapping[code].precision_level
        except (ValueError, TypeError):
            pass
        return None
    
    def get_footnote_summary(self) -> Dict[str, any]:
        """
        Get a summary of loaded footnotes.
        
        Returns:
            Dictionary containing footnote summary statistics
        """
        if not self._footnote_mapping:
            return {'total_footnotes': 0, 'range_footnotes': 0, 'precision_levels': {}}
        
        precision_counts = {}
        for footnote in self._footnote_mapping.values():
            precision_counts[footnote.precision_level] = precision_counts.get(footnote.precision_level, 0) + 1
        
        return {
            'total_footnotes': len(self._footnote_mapping),
            'range_footnotes': len(self._range_conversions),
            'precision_levels': precision_counts,
            'available_codes': sorted(self._footnote_mapping.keys())
        }
    
    def validate_footnote_codes(self, df: pd.DataFrame, 
                              footnote_columns: List[str] = None) -> Dict[str, List[int]]:
        """
        Validate footnote codes in a DataFrame against loaded footnote mapping.
        
        Args:
            df: DataFrame to validate
            footnote_columns: List of footnote columns to check
            
        Returns:
            Dictionary with validation results
        """
        if footnote_columns is None:
            footnote_columns = ['DATA FOOTNOTE', 'STANDARD ERROR FOOTNOTE', 'SERIES FOOTNOTE']
        
        results = {
            'valid_codes': [],
            'invalid_codes': [],
            'missing_codes': []
        }
        
        for col in footnote_columns:
            if col not in df.columns:
                continue
            
            # Get all non-null footnote codes
            footnote_values = df[col].dropna()
            
            for value in footnote_values.unique():
                try:
                    code = int(value)
                    if code in self._footnote_mapping:
                        if code not in results['valid_codes']:
                            results['valid_codes'].append(code)
                    else:
                        if code not in results['missing_codes']:
                            results['missing_codes'].append(code)
                except (ValueError, TypeError):
                    if value not in results['invalid_codes']:
                        results['invalid_codes'].append(value)
        
        # Sort the results
        results['valid_codes'].sort()
        results['missing_codes'].sort()
        
        return results