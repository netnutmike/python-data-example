"""
Core statistical analysis functionality for occupation data reports.
Implements confidence intervals, precision metrics, and population weighting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import warnings

from ..interfaces import AnalysisResult


class StatisticalAnalyzer:
    """
    Core statistical analyzer for occupation data.
    Handles confidence intervals, precision metrics, and population weighting.
    """
    
    # Total civilian workforce from BLS data
    TOTAL_CIVILIAN_WORKERS = 145_866_200
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the statistical analyzer.
        
        Args:
            confidence_level: Confidence level for interval calculations (default 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.z_score = stats.norm.ppf(1 - self.alpha / 2)
    
    def calculate_confidence_intervals(
        self, 
        estimates: pd.Series, 
        standard_errors: pd.Series,
        confidence_level: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Calculate confidence intervals for estimates using standard errors.
        
        Args:
            estimates: Series of estimate values
            standard_errors: Series of standard error values
            confidence_level: Override default confidence level
            
        Returns:
            DataFrame with columns: estimate, std_error, lower_ci, upper_ci, margin_error
        """
        if confidence_level is not None:
            z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        else:
            z_score = self.z_score
        
        # Handle missing standard errors
        std_errors_clean = standard_errors.fillna(0)
        
        # Calculate margin of error
        margin_error = z_score * std_errors_clean
        
        # Calculate confidence intervals
        lower_ci = estimates - margin_error
        upper_ci = estimates + margin_error
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'estimate': estimates,
            'std_error': standard_errors,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'margin_error': margin_error,
            'confidence_level': confidence_level or self.confidence_level
        })
        
        return result_df
    
    def calculate_reliability_scores(
        self, 
        estimates: pd.Series, 
        standard_errors: pd.Series,
        footnote_codes: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate reliability scores based on standard errors and footnote codes.
        
        Args:
            estimates: Series of estimate values
            standard_errors: Series of standard error values
            footnote_codes: Optional series of footnote codes for additional context
            
        Returns:
            Series of reliability scores (0-1, higher is more reliable)
        """
        # Handle missing values
        std_errors_clean = standard_errors.fillna(np.inf)
        estimates_clean = estimates.fillna(0)
        
        # Calculate coefficient of variation (CV)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cv = np.abs(std_errors_clean / estimates_clean)
        
        # Handle division by zero and infinite values
        cv = cv.replace([np.inf, -np.inf], 1.0)
        cv = cv.fillna(1.0)
        
        # Convert CV to reliability score (inverse relationship)
        # CV < 0.1 = high reliability (0.9-1.0)
        # CV 0.1-0.3 = medium reliability (0.5-0.9)
        # CV > 0.3 = low reliability (0.0-0.5)
        reliability_scores = np.where(
            cv <= 0.1, 
            0.9 + 0.1 * (0.1 - cv) / 0.1,  # High reliability
            np.where(
                cv <= 0.3,
                0.5 + 0.4 * (0.3 - cv) / 0.2,  # Medium reliability
                0.5 * np.maximum(0, (1.0 - cv) / 0.7)  # Low reliability
            )
        )
        
        # Adjust for footnote codes if provided
        if footnote_codes is not None and len(footnote_codes) > 0:
            # Footnote codes indicating lower precision get reduced reliability
            low_precision_codes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Range estimates
            medium_precision_codes = [11, 12, 13, 14, 15]  # Suppressed data
            
            footnote_adjustment = footnote_codes.apply(
                lambda x: 0.7 if x in low_precision_codes 
                         else 0.8 if x in medium_precision_codes 
                         else 1.0
            )
            
            # Only apply adjustment if we have matching indices
            if len(footnote_adjustment) == len(reliability_scores):
                reliability_scores = reliability_scores * footnote_adjustment
        
        return pd.Series(reliability_scores, index=estimates.index)
    
    def apply_population_weighting(
        self, 
        data: pd.DataFrame,
        estimate_column: str = 'estimate',
        soc_code_column: str = 'soc_code'
    ) -> pd.DataFrame:
        """
        Apply population weighting based on civilian workforce representation.
        
        Args:
            data: DataFrame containing occupation data
            estimate_column: Name of the estimate column
            soc_code_column: Name of the SOC code column
            
        Returns:
            DataFrame with additional population-weighted columns
        """
        result_df = data.copy()
        
        # Calculate occupation frequencies
        occupation_counts = data[soc_code_column].value_counts()
        total_observations = len(data)
        
        # Calculate representation weights
        # Assumes equal representation across occupations in the absence of specific weights
        occupation_weights = occupation_counts / total_observations
        
        # Map weights back to original data
        result_df['occupation_weight'] = result_df[soc_code_column].map(occupation_weights)
        
        # Calculate population-weighted estimates
        result_df['weighted_estimate'] = (
            result_df[estimate_column] * result_df['occupation_weight']
        )
        
        # Calculate estimated worker counts
        result_df['estimated_workers'] = (
            result_df['weighted_estimate'] * self.TOTAL_CIVILIAN_WORKERS / 100
        ).round().astype(int)
        
        return result_df
    
    def calculate_precision_metrics(
        self, 
        estimates: pd.Series, 
        standard_errors: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate overall precision metrics for a set of estimates.
        
        Args:
            estimates: Series of estimate values
            standard_errors: Series of standard error values
            
        Returns:
            Dictionary containing precision metrics
        """
        # Clean data
        valid_mask = (~estimates.isna()) & (~standard_errors.isna()) & (standard_errors > 0)
        clean_estimates = estimates[valid_mask]
        clean_std_errors = standard_errors[valid_mask]
        
        if len(clean_estimates) == 0:
            return {
                'mean_cv': np.nan,
                'median_cv': np.nan,
                'high_precision_pct': 0.0,
                'medium_precision_pct': 0.0,
                'low_precision_pct': 0.0,
                'valid_estimates': 0
            }
        
        # Calculate coefficient of variation
        cv = np.abs(clean_std_errors / clean_estimates)
        cv = cv.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Calculate precision categories
        high_precision = (cv <= 0.1).sum()
        medium_precision = ((cv > 0.1) & (cv <= 0.3)).sum()
        low_precision = (cv > 0.3).sum()
        total_valid = len(cv)
        
        return {
            'mean_cv': cv.mean(),
            'median_cv': cv.median(),
            'high_precision_pct': (high_precision / total_valid * 100) if total_valid > 0 else 0.0,
            'medium_precision_pct': (medium_precision / total_valid * 100) if total_valid > 0 else 0.0,
            'low_precision_pct': (low_precision / total_valid * 100) if total_valid > 0 else 0.0,
            'valid_estimates': total_valid
        }
    
    def calculate_statistical_summary(
        self, 
        data: pd.DataFrame,
        group_by_column: str,
        estimate_column: str = 'estimate',
        std_error_column: str = 'standard_error'
    ) -> pd.DataFrame:
        """
        Calculate comprehensive statistical summary by group.
        
        Args:
            data: DataFrame containing the data
            group_by_column: Column to group by
            estimate_column: Name of the estimate column
            std_error_column: Name of the standard error column
            
        Returns:
            DataFrame with statistical summaries by group
        """
        def group_stats(group):
            estimates = group[estimate_column]
            std_errors = group[std_error_column]
            
            # Basic statistics
            stats_dict = {
                'count': len(group),
                'mean_estimate': estimates.mean(),
                'median_estimate': estimates.median(),
                'std_estimate': estimates.std(),
                'min_estimate': estimates.min(),
                'max_estimate': estimates.max(),
                'mean_std_error': std_errors.mean(),
                'median_std_error': std_errors.median()
            }
            
            # Precision metrics
            precision_metrics = self.calculate_precision_metrics(estimates, std_errors)
            stats_dict.update(precision_metrics)
            
            # Reliability scores
            reliability_scores = self.calculate_reliability_scores(estimates, std_errors)
            stats_dict['mean_reliability'] = reliability_scores.mean()
            stats_dict['median_reliability'] = reliability_scores.median()
            
            return pd.Series(stats_dict)
        
        summary_df = data.groupby(group_by_column).apply(group_stats).reset_index()
        
        return summary_df
    
    def identify_outliers(
        self, 
        estimates: pd.Series, 
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.Series:
        """
        Identify statistical outliers in estimates.
        
        Args:
            estimates: Series of estimate values
            method: Method to use ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean series indicating outliers
        """
        clean_estimates = estimates.dropna()
        
        if method == 'iqr':
            Q1 = clean_estimates.quantile(0.25)
            Q3 = clean_estimates.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (estimates < lower_bound) | (estimates > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(clean_estimates, nan_policy='omit'))
            outliers = pd.Series(False, index=estimates.index)
            outliers.loc[clean_estimates.index] = z_scores > threshold
            
        elif method == 'modified_zscore':
            median = clean_estimates.median()
            mad = np.median(np.abs(clean_estimates - median))
            modified_z_scores = 0.6745 * (clean_estimates - median) / mad
            outliers = pd.Series(False, index=estimates.index)
            outliers.loc[clean_estimates.index] = np.abs(modified_z_scores) > threshold
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outliers.fillna(False)
    
    def bootstrap_confidence_interval(
        self, 
        data: pd.Series, 
        statistic_func: callable = np.mean,
        n_bootstrap: int = 1000,
        confidence_level: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence intervals for a statistic.
        
        Args:
            data: Series of data values
            statistic_func: Function to calculate statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Override default confidence level
            
        Returns:
            Tuple of (statistic_value, lower_ci, upper_ci)
        """
        clean_data = data.dropna()
        if len(clean_data) == 0:
            return np.nan, np.nan, np.nan
        
        # Calculate original statistic
        original_stat = statistic_func(clean_data)
        
        # Bootstrap sampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(clean_data, size=len(clean_data), replace=True)
            bootstrap_stats.append(statistic_func(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        conf_level = confidence_level or self.confidence_level
        alpha = 1 - conf_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_ci = np.percentile(bootstrap_stats, lower_percentile)
        upper_ci = np.percentile(bootstrap_stats, upper_percentile)
        
        return original_stat, lower_ci, upper_ci