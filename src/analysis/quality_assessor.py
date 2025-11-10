"""
Data quality assessment system for occupation data reports.
Implements reliability scoring, precision categorization, and data completeness analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

from ..interfaces import AnalysisResult, ValidationResult
from .statistical_analyzer import StatisticalAnalyzer


@dataclass
class QualityMetrics:
    """Container for data quality metrics."""
    reliability_score: float
    precision_level: str
    completeness_score: float
    coverage_score: float
    footnote_quality_score: float
    overall_quality_score: float
    quality_category: str
    recommendations: List[str]


@dataclass
class DataCompletenessResult:
    """Result structure for data completeness analysis."""
    total_records: int
    complete_records: int
    completeness_percentage: float
    missing_by_column: Dict[str, int]
    missing_patterns: Dict[str, int]
    critical_missing: List[str]


class QualityAssessor:
    """
    Data quality assessment system for occupation data.
    
    Implements reliability scoring, precision level categorization based on footnotes
    and standard errors, and data completeness and coverage analysis.
    """
    
    def __init__(self, statistical_analyzer: Optional[StatisticalAnalyzer] = None):
        """
        Initialize the quality assessor.
        
        Args:
            statistical_analyzer: Optional StatisticalAnalyzer instance for calculations
        """
        self.logger = logging.getLogger(__name__)
        self.statistical_analyzer = statistical_analyzer or StatisticalAnalyzer()
        
        # Quality thresholds
        self.reliability_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.3
        }
        
        self.completeness_thresholds = {
            'excellent': 0.95,
            'good': 0.85,
            'fair': 0.70,
            'poor': 0.50
        }
        
        # Critical columns that must be present for quality analysis
        self.critical_columns = [
            'SOC 2018 CODE',
            'OCCUPATION',
            'REQUIREMENT',
            'ESTIMATE',
            'STANDARD ERROR'
        ]
        
        # Footnote precision mapping
        self.footnote_precision_mapping = {
            # Range estimates (low precision)
            1: 'very_low', 16: 'low', 17: 'low', 18: 'low', 19: 'low', 20: 'low',
            21: 'low', 22: 'low', 23: 'low', 24: 'low', 25: 'low',
            26: 'low', 27: 'low', 28: 'low', 29: 'low', 30: 'low',
            31: 'low', 32: 'low', 33: 'low', 34: 'low', 35: 'low', 36: 'low',
            
            # Standard error related (medium to high precision)
            2: 'high', 3: 'high', 4: 'medium', 5: 'medium',
            
            # Methodological notes (medium precision)
            6: 'medium', 7: 'medium', 8: 'medium', 9: 'medium', 10: 'medium',
            11: 'medium', 12: 'medium', 13: 'medium', 14: 'medium', 15: 'medium'
        }
    
    def assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment.
        
        Args:
            df: DataFrame containing occupation data
            
        Returns:
            Dictionary containing comprehensive quality assessment results
        """
        self.logger.info("Starting comprehensive data quality assessment")
        
        # Basic data validation
        validation_result = self._validate_data_structure(df)
        
        # Completeness analysis
        completeness_result = self.analyze_data_completeness(df)
        
        # Reliability scoring
        reliability_scores = self.calculate_reliability_scores(df)
        
        # Precision categorization
        precision_analysis = self.categorize_precision_levels(df)
        
        # Coverage analysis
        coverage_analysis = self.analyze_data_coverage(df)
        
        # Overall quality metrics
        overall_metrics = self._calculate_overall_quality_metrics(
            validation_result, completeness_result, reliability_scores, 
            precision_analysis, coverage_analysis
        )
        
        quality_assessment = {
            'validation_result': validation_result,
            'completeness_analysis': completeness_result,
            'reliability_scores': reliability_scores,
            'precision_analysis': precision_analysis,
            'coverage_analysis': coverage_analysis,
            'overall_metrics': overall_metrics,
            'assessment_timestamp': pd.Timestamp.now(),
            'total_records_assessed': len(df)
        }
        
        self.logger.info(f"Quality assessment completed for {len(df)} records")
        return quality_assessment
    
    def calculate_reliability_scores(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate reliability scores for the dataset.
        
        Args:
            df: DataFrame containing occupation data
            
        Returns:
            Dictionary containing reliability analysis results
        """
        if 'ESTIMATE' not in df.columns or 'STANDARD ERROR' not in df.columns:
            return {'error': 'Required columns (ESTIMATE, STANDARD ERROR) not found'}
        
        estimates = df['ESTIMATE']
        std_errors = df['STANDARD ERROR']
        footnote_codes = df.get('DATA FOOTNOTE', pd.Series(dtype=float))
        
        # Only calculate reliability for rows with valid estimates and standard errors
        valid_mask = estimates.notna() & std_errors.notna()
        valid_estimates = estimates[valid_mask]
        valid_std_errors = std_errors[valid_mask]
        valid_footnotes = footnote_codes[valid_mask] if footnote_codes is not None else None
        
        if len(valid_estimates) == 0:
            # No valid data to calculate reliability scores
            reliability_scores = pd.Series(dtype=float)
            valid_scores = pd.Series(dtype=float)
        else:
            # Calculate individual reliability scores
            reliability_scores_valid = self.statistical_analyzer.calculate_reliability_scores(
                valid_estimates, valid_std_errors, valid_footnotes
            )
            
            # Create full series with NaN for invalid entries
            reliability_scores = pd.Series(index=estimates.index, dtype=float)
            reliability_scores[valid_mask] = reliability_scores_valid
            
            # Filter out NaN scores for analysis
            valid_scores = reliability_scores.dropna()
        
        if len(valid_scores) == 0:
            # No valid scores to analyze
            reliability_summary = {
                'mean_reliability': np.nan,
                'median_reliability': np.nan,
                'std_reliability': np.nan,
                'min_reliability': np.nan,
                'max_reliability': np.nan,
                'reliability_distribution': {},
                'high_reliability_count': 0,
                'low_reliability_count': 0,
                'total_assessed': 0
            }
        else:
            # Categorize reliability levels
            reliability_categories = pd.cut(
                valid_scores,
                bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
                labels=['poor', 'fair', 'good', 'very_good', 'excellent'],
                include_lowest=True
            )
            
            # Calculate summary statistics
            reliability_summary = {
                'mean_reliability': valid_scores.mean(),
                'median_reliability': valid_scores.median(),
                'std_reliability': valid_scores.std(),
                'min_reliability': valid_scores.min(),
                'max_reliability': valid_scores.max(),
                'reliability_distribution': reliability_categories.value_counts().to_dict(),
                'high_reliability_count': (valid_scores >= 0.7).sum(),
                'low_reliability_count': (valid_scores < 0.5).sum(),
                'total_assessed': len(valid_scores)
            }
        
        # Identify records with reliability issues
        if len(valid_scores) > 0:
            low_reliability_mask = reliability_scores < 0.5
            low_reliability_records = df[low_reliability_mask].copy()
            low_reliability_records['reliability_score'] = reliability_scores[low_reliability_mask]
        else:
            low_reliability_records = pd.DataFrame()
        
        return {
            'individual_scores': reliability_scores,
            'summary_statistics': reliability_summary,
            'low_reliability_records': low_reliability_records,
            'reliability_by_occupation': self._analyze_reliability_by_group(
                df, reliability_scores, 'OCCUPATION'
            ),
            'reliability_by_requirement': self._analyze_reliability_by_group(
                df, reliability_scores, 'REQUIREMENT'
            )
        }
    
    def categorize_precision_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Categorize precision levels based on footnotes and standard errors.
        
        Args:
            df: DataFrame containing occupation data
            
        Returns:
            Dictionary containing precision level analysis
        """
        precision_analysis = {
            'footnote_precision': {},
            'statistical_precision': {},
            'combined_precision': {},
            'precision_summary': {}
        }
        
        # Analyze footnote-based precision
        if 'DATA FOOTNOTE' in df.columns:
            footnote_precision = df['DATA FOOTNOTE'].apply(
                lambda x: self.footnote_precision_mapping.get(x, 'unknown') 
                if pd.notna(x) else 'no_footnote'
            )
            
            precision_analysis['footnote_precision'] = {
                'distribution': footnote_precision.value_counts().to_dict(),
                'high_precision_count': (footnote_precision == 'high').sum(),
                'low_precision_count': (footnote_precision.isin(['low', 'very_low'])).sum(),
                'unknown_precision_count': (footnote_precision == 'unknown').sum()
            }
        
        # Analyze statistical precision (coefficient of variation)
        if 'ESTIMATE' in df.columns and 'STANDARD ERROR' in df.columns:
            estimates = df['ESTIMATE']
            std_errors = df['STANDARD ERROR']
            
            # Calculate coefficient of variation
            cv = np.abs(std_errors / estimates)
            cv = cv.replace([np.inf, -np.inf], np.nan)
            
            # Categorize statistical precision
            statistical_precision = pd.cut(
                cv,
                bins=[0, 0.1, 0.3, 0.5, np.inf],
                labels=['high', 'medium', 'low', 'very_low'],
                include_lowest=True
            )
            
            precision_analysis['statistical_precision'] = {
                'cv_mean': cv.mean(),
                'cv_median': cv.median(),
                'cv_std': cv.std(),
                'distribution': statistical_precision.value_counts().to_dict(),
                'high_precision_count': (statistical_precision == 'high').sum(),
                'low_precision_count': (statistical_precision.isin(['low', 'very_low'])).sum()
            }
            
            # Combined precision assessment
            if 'DATA FOOTNOTE' in df.columns:
                combined_precision = self._combine_precision_assessments(
                    footnote_precision, statistical_precision
                )
                precision_analysis['combined_precision'] = {
                    'distribution': combined_precision.value_counts().to_dict(),
                    'overall_high_precision': (combined_precision == 'high').sum(),
                    'overall_low_precision': (combined_precision.isin(['low', 'very_low'])).sum()
                }
        
        # Generate precision summary
        precision_analysis['precision_summary'] = self._generate_precision_summary(precision_analysis)
        
        return precision_analysis
    
    def analyze_data_completeness(self, df: pd.DataFrame) -> DataCompletenessResult:
        """
        Analyze data completeness and coverage.
        
        Args:
            df: DataFrame containing occupation data
            
        Returns:
            DataCompletenessResult with completeness analysis
        """
        total_records = len(df)
        
        # Calculate missing values by column
        missing_by_column = df.isnull().sum().to_dict()
        
        # Calculate completeness for each column
        completeness_by_column = {
            col: (total_records - missing_count) / total_records * 100 if total_records > 0 else 0
            for col, missing_count in missing_by_column.items()
        }
        
        # Identify complete records (no missing values in critical columns)
        critical_columns_present = [col for col in self.critical_columns if col in df.columns]
        if critical_columns_present:
            complete_records = df[critical_columns_present].dropna()
            complete_count = len(complete_records)
        else:
            complete_count = 0
        
        completeness_percentage = (complete_count / total_records * 100) if total_records > 0 else 0
        
        # Analyze missing data patterns
        missing_patterns = self._analyze_missing_patterns(df)
        
        # Identify critical missing data
        critical_missing = [
            col for col in critical_columns_present 
            if missing_by_column.get(col, 0) > total_records * 0.1  # More than 10% missing
        ]
        
        return DataCompletenessResult(
            total_records=total_records,
            complete_records=complete_count,
            completeness_percentage=completeness_percentage,
            missing_by_column=missing_by_column,
            missing_patterns=missing_patterns,
            critical_missing=critical_missing
        )
    
    def analyze_data_coverage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data coverage across different dimensions.
        
        Args:
            df: DataFrame containing occupation data
            
        Returns:
            Dictionary containing coverage analysis results
        """
        coverage_analysis = {}
        
        # SOC code coverage
        if 'SOC 2018 CODE' in df.columns:
            soc_codes = df['SOC 2018 CODE'].dropna()
            coverage_analysis['soc_coverage'] = {
                'unique_soc_codes': soc_codes.nunique(),
                'total_soc_observations': len(soc_codes),
                'soc_distribution': soc_codes.value_counts().head(20).to_dict(),
                'coverage_breadth': soc_codes.nunique() / len(soc_codes) * 100 if len(soc_codes) > 0 else 0
            }
        
        # Requirement type coverage
        if 'REQUIREMENT' in df.columns:
            requirements = df['REQUIREMENT'].dropna()
            coverage_analysis['requirement_coverage'] = {
                'unique_requirements': requirements.nunique(),
                'requirement_distribution': requirements.value_counts().to_dict(),
                'balanced_coverage': self._assess_requirement_balance(requirements)
            }
        
        # Occupation coverage
        if 'OCCUPATION' in df.columns:
            occupations = df['OCCUPATION'].dropna()
            coverage_analysis['occupation_coverage'] = {
                'unique_occupations': occupations.nunique(),
                'occupation_distribution': occupations.value_counts().head(20).to_dict(),
                'coverage_diversity': occupations.nunique() / len(occupations) * 100 if len(occupations) > 0 else 0
            }
        
        # Estimate range coverage
        if 'ESTIMATE' in df.columns:
            estimates = df['ESTIMATE'].dropna()
            coverage_analysis['estimate_coverage'] = {
                'estimate_range': {
                    'min': estimates.min(),
                    'max': estimates.max(),
                    'mean': estimates.mean(),
                    'median': estimates.median()
                },
                'estimate_distribution': self._analyze_estimate_distribution(estimates),
                'outlier_count': len(self.statistical_analyzer.identify_outliers(estimates))
            }
        
        return coverage_analysis
    
    def generate_quality_recommendations(self, quality_assessment: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on quality assessment results.
        
        Args:
            quality_assessment: Results from assess_data_quality()
            
        Returns:
            List of quality improvement recommendations
        """
        recommendations = []
        
        # Completeness recommendations
        completeness = quality_assessment.get('completeness_analysis')
        if completeness and completeness.completeness_percentage < 90:
            recommendations.append(
                f"Data completeness is {completeness.completeness_percentage:.1f}%. "
                "Consider data collection improvements for critical missing fields."
            )
        
        if completeness and completeness.critical_missing:
            recommendations.append(
                f"Critical columns with significant missing data: {', '.join(completeness.critical_missing)}. "
                "Prioritize data collection for these fields."
            )
        
        # Reliability recommendations
        reliability = quality_assessment.get('reliability_scores', {}).get('summary_statistics', {})
        if reliability.get('mean_reliability', 0) < 0.7:
            recommendations.append(
                f"Average reliability score is {reliability.get('mean_reliability', 0):.2f}. "
                "Consider increasing sample sizes or improving measurement precision."
            )
        
        low_reliability_count = reliability.get('low_reliability_count', 0)
        total_assessed = reliability.get('total_assessed', 1)
        if low_reliability_count / total_assessed > 0.2:
            recommendations.append(
                f"{low_reliability_count} records ({low_reliability_count/total_assessed*100:.1f}%) "
                "have low reliability. Review data collection methods for these cases."
            )
        
        # Precision recommendations
        precision = quality_assessment.get('precision_analysis', {})
        footnote_precision = precision.get('footnote_precision', {})
        if footnote_precision.get('low_precision_count', 0) > footnote_precision.get('high_precision_count', 0):
            recommendations.append(
                "More records have low precision footnotes than high precision. "
                "Consider targeted data collection for low-precision estimates."
            )
        
        # Coverage recommendations
        coverage = quality_assessment.get('coverage_analysis', {})
        soc_coverage = coverage.get('soc_coverage', {})
        if soc_coverage.get('coverage_breadth', 100) < 50:
            recommendations.append(
                f"SOC code coverage breadth is {soc_coverage.get('coverage_breadth', 0):.1f}%. "
                "Consider expanding occupational representation in the sample."
            )
        
        return recommendations
    
    def _validate_data_structure(self, df: pd.DataFrame) -> ValidationResult:
        """Validate the basic structure of the dataset."""
        errors = []
        warnings = []
        
        # Check for required columns
        missing_critical = [col for col in self.critical_columns if col not in df.columns]
        if missing_critical:
            errors.extend([f"Missing critical column: {col}" for col in missing_critical])
        
        # Check data types and quality issues
        if 'ESTIMATE' in df.columns:
            estimates = df['ESTIMATE']
            
            # Check for non-numeric values
            non_numeric_estimates = estimates.apply(
                lambda x: not isinstance(x, (int, float)) and pd.notna(x)
            ).sum()
            if non_numeric_estimates > 0:
                warnings.append(f"{non_numeric_estimates} non-numeric values in ESTIMATE column")
            
            # Check for negative or extreme values
            numeric_estimates = estimates[pd.to_numeric(estimates, errors='coerce').notna()]
            if len(numeric_estimates) > 0:
                negative_count = (numeric_estimates < 0).sum()
                extreme_count = (numeric_estimates > 1000).sum()
                
                if negative_count > 0:
                    warnings.append(f"{negative_count} negative values in ESTIMATE column")
                if extreme_count > 0:
                    warnings.append(f"{extreme_count} extreme values (>1000) in ESTIMATE column")
        
        # Check for duplicate records
        if len(df) > 0 and len(df) != len(df.drop_duplicates()):
            warnings.append(f"{len(df) - len(df.drop_duplicates())} duplicate records found")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            record_count=len(df)
        )
    
    def _analyze_reliability_by_group(self, df: pd.DataFrame, reliability_scores: pd.Series, 
                                    group_column: str) -> Dict[str, Any]:
        """Analyze reliability scores by group."""
        if group_column not in df.columns:
            return {}
        
        df_with_reliability = df.copy()
        df_with_reliability['reliability_score'] = reliability_scores
        
        # Filter out rows with NaN reliability scores
        valid_reliability = df_with_reliability.dropna(subset=['reliability_score'])
        
        if len(valid_reliability) == 0:
            return {
                'group_statistics': {},
                'lowest_reliability_groups': [],
                'highest_reliability_groups': []
            }
        
        group_stats = valid_reliability.groupby(group_column)['reliability_score'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(3)
        
        if len(group_stats) == 0:
            return {
                'group_statistics': {},
                'lowest_reliability_groups': [],
                'highest_reliability_groups': []
            }
        
        return {
            'group_statistics': group_stats.to_dict('index'),
            'lowest_reliability_groups': group_stats.nsmallest(min(5, len(group_stats)), 'mean').index.tolist(),
            'highest_reliability_groups': group_stats.nlargest(min(5, len(group_stats)), 'mean').index.tolist()
        }
    
    def _combine_precision_assessments(self, footnote_precision: pd.Series, 
                                     statistical_precision: pd.Series) -> pd.Series:
        """Combine footnote and statistical precision assessments."""
        # Create a combined precision score
        precision_scores = {
            'high': 4, 'medium': 3, 'low': 2, 'very_low': 1, 'unknown': 2, 'no_footnote': 3
        }
        
        footnote_scores = footnote_precision.map(precision_scores).fillna(2)
        statistical_scores = statistical_precision.map(precision_scores).fillna(2)
        
        # Convert to numpy arrays to avoid categorical issues
        footnote_array = np.array(footnote_scores.values, dtype=float)
        statistical_array = np.array(statistical_scores.values, dtype=float)
        
        # Take the minimum (most conservative) assessment
        combined_scores = np.minimum(footnote_array, statistical_array)
        
        # Convert back to categories
        score_to_category = {4: 'high', 3: 'medium', 2: 'low', 1: 'very_low'}
        return pd.Series(combined_scores).map(score_to_category)
    
    def _generate_precision_summary(self, precision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of precision analysis."""
        summary = {}
        
        # Footnote precision summary
        footnote_precision = precision_analysis.get('footnote_precision', {})
        if footnote_precision:
            total_footnote = sum(footnote_precision.get('distribution', {}).values())
            high_pct = footnote_precision.get('high_precision_count', 0) / total_footnote * 100 if total_footnote > 0 else 0
            summary['footnote_high_precision_percentage'] = high_pct
        
        # Statistical precision summary
        statistical_precision = precision_analysis.get('statistical_precision', {})
        if statistical_precision:
            total_statistical = sum(statistical_precision.get('distribution', {}).values())
            high_pct = statistical_precision.get('high_precision_count', 0) / total_statistical * 100 if total_statistical > 0 else 0
            summary['statistical_high_precision_percentage'] = high_pct
        
        # Combined precision summary
        combined_precision = precision_analysis.get('combined_precision', {})
        if combined_precision:
            total_combined = sum(combined_precision.get('distribution', {}).values())
            high_pct = combined_precision.get('overall_high_precision', 0) / total_combined * 100 if total_combined > 0 else 0
            summary['overall_high_precision_percentage'] = high_pct
        
        return summary
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze patterns in missing data."""
        # Create a pattern string for each row
        missing_patterns = df.isnull().apply(
            lambda row: ''.join(['1' if missing else '0' for missing in row]), axis=1
        )
        
        # Count occurrences of each pattern
        pattern_counts = missing_patterns.value_counts().head(10).to_dict()
        
        return pattern_counts
    
    def _assess_requirement_balance(self, requirements: pd.Series) -> Dict[str, Any]:
        """Assess the balance of requirement type coverage."""
        req_counts = requirements.value_counts()
        
        # Calculate balance metrics
        total_reqs = len(req_counts)
        expected_per_req = len(requirements) / total_reqs if total_reqs > 0 else 0
        
        # Calculate coefficient of variation for balance
        cv = req_counts.std() / req_counts.mean() if req_counts.mean() > 0 else np.inf
        
        return {
            'total_requirement_types': total_reqs,
            'expected_observations_per_type': expected_per_req,
            'actual_range': {'min': req_counts.min(), 'max': req_counts.max()},
            'balance_coefficient_variation': cv,
            'is_balanced': cv < 0.5  # Arbitrary threshold for "balanced"
        }
    
    def _analyze_estimate_distribution(self, estimates: pd.Series) -> Dict[str, Any]:
        """Analyze the distribution of estimate values."""
        return {
            'percentiles': {
                '5th': estimates.quantile(0.05),
                '25th': estimates.quantile(0.25),
                '50th': estimates.quantile(0.50),
                '75th': estimates.quantile(0.75),
                '95th': estimates.quantile(0.95)
            },
            'zero_values': (estimates == 0).sum(),
            'negative_values': (estimates < 0).sum(),
            'extreme_values': (estimates > 100).sum(),  # Assuming percentages
            'distribution_skewness': estimates.skew(),
            'distribution_kurtosis': estimates.kurtosis()
        }
    
    def _calculate_overall_quality_metrics(self, validation_result: ValidationResult,
                                         completeness_result: DataCompletenessResult,
                                         reliability_scores: Dict[str, Any],
                                         precision_analysis: Dict[str, Any],
                                         coverage_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality metrics and scores."""
        # Validation score (0-1)
        validation_score = 1.0 if validation_result.is_valid else 0.5
        
        # Completeness score (0-1)
        completeness_score = completeness_result.completeness_percentage / 100
        
        # Reliability score (0-1)
        reliability_summary = reliability_scores.get('summary_statistics', {})
        reliability_score = reliability_summary.get('mean_reliability', 0)
        
        # Precision score (0-1)
        precision_summary = precision_analysis.get('precision_summary', {})
        precision_score = precision_summary.get('overall_high_precision_percentage', 0) / 100
        
        # Coverage score (0-1) - simplified metric
        soc_coverage = coverage_analysis.get('soc_coverage', {})
        coverage_breadth = soc_coverage.get('coverage_breadth', 0)
        coverage_score = min(coverage_breadth / 50, 1.0)  # Normalize to 50% as good coverage
        
        # Calculate weighted overall score
        weights = {
            'validation': 0.2,
            'completeness': 0.25,
            'reliability': 0.25,
            'precision': 0.15,
            'coverage': 0.15
        }
        
        overall_score = (
            validation_score * weights['validation'] +
            completeness_score * weights['completeness'] +
            reliability_score * weights['reliability'] +
            precision_score * weights['precision'] +
            coverage_score * weights['coverage']
        )
        
        # Determine overall quality category
        if overall_score >= 0.9:
            quality_category = 'excellent'
        elif overall_score >= 0.7:
            quality_category = 'good'
        elif overall_score >= 0.5:
            quality_category = 'fair'
        else:
            quality_category = 'poor'
        
        return {
            'validation_score': validation_score,
            'completeness_score': completeness_score,
            'reliability_score': reliability_score,
            'precision_score': precision_score,
            'coverage_score': coverage_score,
            'overall_score': overall_score,
            'quality_category': quality_category,
            'component_weights': weights
        }