"""
Correlation analysis functionality for cross-requirement analysis.
Implements correlation matrices, significance testing, and strength categorization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import warnings

from ..interfaces import AnalysisResult


class CorrelationAnalyzer:
    """
    Analyzer for cross-requirement correlation analysis.
    Handles correlation matrices, significance testing, and relationship categorization.
    """
    
    def __init__(self, significance_level: float = 0.05, min_sample_size: int = 30):
        """
        Initialize the correlation analyzer.
        
        Args:
            significance_level: Alpha level for significance testing (default 0.05)
            min_sample_size: Minimum sample size for reliable correlations (default 30)
        """
        self.significance_level = significance_level
        self.min_sample_size = min_sample_size
    
    def calculate_correlation_matrix(
        self, 
        data: pd.DataFrame,
        requirement_column: str = 'requirement_type',
        soc_code_column: str = 'soc_code',
        estimate_column: str = 'estimate',
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between requirement types.
        
        Args:
            data: DataFrame containing occupation data
            requirement_column: Name of the requirement type column
            soc_code_column: Name of the SOC code column
            estimate_column: Name of the estimate column
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            DataFrame containing correlation matrix
        """
        # Create pivot table with requirements as columns and occupations as rows
        pivot_data = data.pivot_table(
            index=soc_code_column,
            columns=requirement_column,
            values=estimate_column,
            aggfunc='mean',
            fill_value=np.nan
        )
        
        # Calculate correlation matrix based on method
        if method == 'pearson':
            correlation_matrix = pivot_data.corr(method='pearson')
        elif method == 'spearman':
            correlation_matrix = pivot_data.corr(method='spearman')
        elif method == 'kendall':
            correlation_matrix = pivot_data.corr(method='kendall')
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        return correlation_matrix
    
    def calculate_correlation_with_significance(
        self, 
        data: pd.DataFrame,
        requirement_column: str = 'requirement_type',
        soc_code_column: str = 'soc_code',
        estimate_column: str = 'estimate',
        method: str = 'pearson'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Calculate correlation matrix with significance testing and sample sizes.
        
        Args:
            data: DataFrame containing occupation data
            requirement_column: Name of the requirement type column
            soc_code_column: Name of the SOC code column
            estimate_column: Name of the estimate column
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Tuple of (correlation_matrix, p_value_matrix, sample_size_matrix)
        """
        # Create pivot table
        pivot_data = data.pivot_table(
            index=soc_code_column,
            columns=requirement_column,
            values=estimate_column,
            aggfunc='mean',
            fill_value=np.nan
        )
        
        requirements = pivot_data.columns.tolist()
        n_requirements = len(requirements)
        
        # Initialize matrices
        correlation_matrix = pd.DataFrame(
            np.nan, index=requirements, columns=requirements
        )
        p_value_matrix = pd.DataFrame(
            np.nan, index=requirements, columns=requirements
        )
        sample_size_matrix = pd.DataFrame(
            np.nan, index=requirements, columns=requirements
        )
        
        # Calculate pairwise correlations
        for i, req1 in enumerate(requirements):
            for j, req2 in enumerate(requirements):
                if i == j:
                    # Diagonal elements
                    correlation_matrix.loc[req1, req2] = 1.0
                    p_value_matrix.loc[req1, req2] = 0.0
                    sample_size_matrix.loc[req1, req2] = (~pivot_data[req1].isna()).sum()
                else:
                    # Off-diagonal elements
                    x = pivot_data[req1].dropna()
                    y = pivot_data[req2].dropna()
                    
                    # Find common indices (occupations with both requirements)
                    common_idx = x.index.intersection(y.index)
                    if len(common_idx) >= self.min_sample_size:
                        x_common = x.loc[common_idx]
                        y_common = y.loc[common_idx]
                        
                        # Remove any remaining NaN values
                        valid_mask = (~x_common.isna()) & (~y_common.isna())
                        x_valid = x_common[valid_mask]
                        y_valid = y_common[valid_mask]
                        
                        if len(x_valid) >= self.min_sample_size:
                            try:
                                if method == 'pearson':
                                    corr, p_val = pearsonr(x_valid, y_valid)
                                elif method == 'spearman':
                                    corr, p_val = spearmanr(x_valid, y_valid)
                                elif method == 'kendall':
                                    corr, p_val = kendalltau(x_valid, y_valid)
                                else:
                                    raise ValueError(f"Unknown correlation method: {method}")
                                
                                correlation_matrix.loc[req1, req2] = corr
                                p_value_matrix.loc[req1, req2] = p_val
                                sample_size_matrix.loc[req1, req2] = len(x_valid)
                                
                            except Exception as e:
                                warnings.warn(f"Could not calculate correlation between {req1} and {req2}: {e}")
        
        return correlation_matrix, p_value_matrix, sample_size_matrix
    
    def categorize_correlation_strength(self, correlation_value: float) -> str:
        """
        Categorize correlation strength based on absolute value.
        
        Args:
            correlation_value: Correlation coefficient
            
        Returns:
            String describing correlation strength
        """
        abs_corr = abs(correlation_value)
        
        if abs_corr >= 0.9:
            return "Very Strong"
        elif abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.5:
            return "Moderate"
        elif abs_corr >= 0.3:
            return "Weak"
        elif abs_corr >= 0.1:
            return "Very Weak"
        else:
            return "Negligible"
    
    def identify_significant_correlations(
        self, 
        correlation_matrix: pd.DataFrame,
        p_value_matrix: pd.DataFrame,
        sample_size_matrix: pd.DataFrame,
        min_correlation: float = 0.3,
        max_p_value: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify statistically significant correlations above threshold.
        
        Args:
            correlation_matrix: Matrix of correlation coefficients
            p_value_matrix: Matrix of p-values
            sample_size_matrix: Matrix of sample sizes
            min_correlation: Minimum absolute correlation to include
            max_p_value: Maximum p-value for significance (default: self.significance_level)
            
        Returns:
            List of significant correlation dictionaries
        """
        if max_p_value is None:
            max_p_value = self.significance_level
        
        significant_correlations = []
        requirements = correlation_matrix.index.tolist()
        
        for i, req1 in enumerate(requirements):
            for j, req2 in enumerate(requirements[i+1:], i+1):  # Avoid duplicates
                corr = correlation_matrix.loc[req1, req2]
                p_val = p_value_matrix.loc[req1, req2]
                sample_size = sample_size_matrix.loc[req1, req2]
                
                if (not pd.isna(corr) and not pd.isna(p_val) and 
                    abs(corr) >= min_correlation and p_val <= max_p_value):
                    
                    significant_correlations.append({
                        'requirement_1': req1,
                        'requirement_2': req2,
                        'correlation': corr,
                        'p_value': p_val,
                        'sample_size': int(sample_size) if not pd.isna(sample_size) else 0,
                        'strength_category': self.categorize_correlation_strength(corr),
                        'direction': 'Positive' if corr > 0 else 'Negative',
                        'is_significant': True
                    })
        
        # Sort by absolute correlation value (descending)
        significant_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return significant_correlations
    
    def analyze_requirement_clusters(
        self, 
        correlation_matrix: pd.DataFrame,
        clustering_threshold: float = 0.6
    ) -> Dict[str, List[str]]:
        """
        Identify clusters of highly correlated requirements.
        
        Args:
            correlation_matrix: Matrix of correlation coefficients
            clustering_threshold: Minimum correlation for clustering
            
        Returns:
            Dictionary mapping cluster names to requirement lists
        """
        # Use absolute correlations for clustering
        abs_corr_matrix = correlation_matrix.abs()
        
        # Simple clustering based on correlation threshold
        requirements = correlation_matrix.index.tolist()
        clusters = {}
        assigned_requirements = set()
        cluster_id = 1
        
        for req in requirements:
            if req not in assigned_requirements:
                # Find all requirements highly correlated with this one
                highly_correlated = []
                for other_req in requirements:
                    if (other_req != req and 
                        not pd.isna(abs_corr_matrix.loc[req, other_req]) and
                        abs_corr_matrix.loc[req, other_req] >= clustering_threshold):
                        highly_correlated.append(other_req)
                
                if highly_correlated:
                    # Create cluster
                    cluster_name = f"Cluster_{cluster_id}"
                    cluster_requirements = [req] + highly_correlated
                    clusters[cluster_name] = cluster_requirements
                    
                    # Mark as assigned
                    assigned_requirements.update(cluster_requirements)
                    cluster_id += 1
                else:
                    # Singleton cluster
                    clusters[f"Singleton_{req}"] = [req]
                    assigned_requirements.add(req)
        
        return clusters
    
    def calculate_partial_correlations(
        self, 
        data: pd.DataFrame,
        target_requirement: str,
        control_requirements: List[str],
        requirement_column: str = 'requirement_type',
        soc_code_column: str = 'soc_code',
        estimate_column: str = 'estimate'
    ) -> pd.DataFrame:
        """
        Calculate partial correlations controlling for specified requirements.
        
        Args:
            data: DataFrame containing occupation data
            target_requirement: Target requirement to analyze
            control_requirements: Requirements to control for
            requirement_column: Name of the requirement type column
            soc_code_column: Name of the SOC code column
            estimate_column: Name of the estimate column
            
        Returns:
            DataFrame with partial correlation results
        """
        # Create pivot table
        pivot_data = data.pivot_table(
            index=soc_code_column,
            columns=requirement_column,
            values=estimate_column,
            aggfunc='mean',
            fill_value=np.nan
        )
        
        # Check if target and control requirements exist
        available_requirements = set(pivot_data.columns)
        if target_requirement not in available_requirements:
            raise ValueError(f"Target requirement '{target_requirement}' not found in data")
        
        missing_controls = set(control_requirements) - available_requirements
        if missing_controls:
            raise ValueError(f"Control requirements not found: {missing_controls}")
        
        # Calculate partial correlations
        partial_results = []
        other_requirements = [req for req in available_requirements 
                            if req != target_requirement and req not in control_requirements]
        
        for other_req in other_requirements:
            # Get data for analysis
            analysis_data = pivot_data[[target_requirement, other_req] + control_requirements].dropna()
            
            if len(analysis_data) >= self.min_sample_size:
                try:
                    # Simple partial correlation using linear regression residuals
                    # Note: sklearn is optional, fallback to basic correlation if not available
                    try:
                        from sklearn.linear_model import LinearRegression
                    except ImportError:
                        # Fallback to simple correlation if sklearn not available
                        partial_corr, partial_p = pearsonr(
                            analysis_data[target_requirement], 
                            analysis_data[other_req]
                        )
                        partial_results.append({
                            'target_requirement': target_requirement,
                            'other_requirement': other_req,
                            'partial_correlation': partial_corr,
                            'partial_p_value': partial_p,
                            'sample_size': len(analysis_data),
                            'control_requirements': ', '.join(control_requirements),
                            'strength_category': self.categorize_correlation_strength(partial_corr),
                            'note': 'Fallback to simple correlation (sklearn not available)'
                        })
                        continue
                    
                    # Control variables
                    X_control = analysis_data[control_requirements].values
                    
                    # Target variable residuals
                    reg_target = LinearRegression().fit(X_control, analysis_data[target_requirement])
                    target_residuals = analysis_data[target_requirement] - reg_target.predict(X_control)
                    
                    # Other variable residuals
                    reg_other = LinearRegression().fit(X_control, analysis_data[other_req])
                    other_residuals = analysis_data[other_req] - reg_other.predict(X_control)
                    
                    # Correlation of residuals
                    partial_corr, partial_p = pearsonr(target_residuals, other_residuals)
                    
                    partial_results.append({
                        'target_requirement': target_requirement,
                        'other_requirement': other_req,
                        'partial_correlation': partial_corr,
                        'partial_p_value': partial_p,
                        'sample_size': len(analysis_data),
                        'control_requirements': ', '.join(control_requirements),
                        'strength_category': self.categorize_correlation_strength(partial_corr)
                    })
                    
                except Exception as e:
                    warnings.warn(f"Could not calculate partial correlation for {other_req}: {e}")
        
        return pd.DataFrame(partial_results)
    
    def generate_correlation_summary(
        self, 
        data: pd.DataFrame,
        requirement_column: str = 'requirement_type',
        soc_code_column: str = 'soc_code',
        estimate_column: str = 'estimate'
    ) -> Dict[str, Any]:
        """
        Generate comprehensive correlation analysis summary.
        
        Args:
            data: DataFrame containing occupation data
            requirement_column: Name of the requirement type column
            soc_code_column: Name of the SOC code column
            estimate_column: Name of the estimate column
            
        Returns:
            Dictionary containing comprehensive correlation summary
        """
        # Calculate correlation matrices
        corr_matrix, p_val_matrix, sample_matrix = self.calculate_correlation_with_significance(
            data, requirement_column, soc_code_column, estimate_column
        )
        
        # Identify significant correlations
        significant_corrs = self.identify_significant_correlations(
            corr_matrix, p_val_matrix, sample_matrix
        )
        
        # Analyze requirement clusters
        clusters = self.analyze_requirement_clusters(corr_matrix)
        
        # Calculate summary statistics
        corr_values = corr_matrix.values
        corr_values_clean = corr_values[~np.isnan(corr_values)]
        # Remove diagonal (perfect correlations)
        corr_values_clean = corr_values_clean[corr_values_clean != 1.0]
        
        # Strength distribution
        strength_counts = {}
        for corr_val in corr_values_clean:
            strength = self.categorize_correlation_strength(corr_val)
            strength_counts[strength] = strength_counts.get(strength, 0) + 1
        
        return {
            'overview': {
                'total_requirements': len(corr_matrix),
                'total_correlations': len(corr_values_clean),
                'significant_correlations': len(significant_corrs),
                'mean_correlation': np.mean(np.abs(corr_values_clean)),
                'median_correlation': np.median(np.abs(corr_values_clean)),
                'max_correlation': np.max(np.abs(corr_values_clean)),
                'min_correlation': np.min(np.abs(corr_values_clean))
            },
            'strength_distribution': strength_counts,
            'significant_correlations': significant_corrs[:10],  # Top 10
            'requirement_clusters': clusters,
            'correlation_matrix': corr_matrix.round(3),
            'p_value_matrix': p_val_matrix.round(4),
            'sample_size_matrix': sample_matrix.astype(int),
            'strongest_positive_correlations': [
                corr for corr in significant_corrs 
                if corr['direction'] == 'Positive'
            ][:5],
            'strongest_negative_correlations': [
                corr for corr in significant_corrs 
                if corr['direction'] == 'Negative'
            ][:5]
        }
    
    def identify_unusual_patterns(
        self, 
        correlation_matrix: pd.DataFrame,
        p_value_matrix: pd.DataFrame,
        expected_correlations: Optional[Dict[Tuple[str, str], str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify unusual correlation patterns that break typical expectations.
        
        Args:
            correlation_matrix: Matrix of correlation coefficients
            p_value_matrix: Matrix of p-values
            expected_correlations: Optional dictionary of expected correlation directions
            
        Returns:
            List of unusual pattern dictionaries
        """
        unusual_patterns = []
        requirements = correlation_matrix.index.tolist()
        
        # Look for unexpected patterns
        for i, req1 in enumerate(requirements):
            for j, req2 in enumerate(requirements[i+1:], i+1):
                corr = correlation_matrix.loc[req1, req2]
                p_val = p_value_matrix.loc[req1, req2]
                
                if not pd.isna(corr) and not pd.isna(p_val):
                    # Check for unexpected strong correlations
                    if abs(corr) > 0.7 and p_val <= self.significance_level:
                        # Check if this was expected
                        expected_direction = None
                        if expected_correlations:
                            expected_direction = expected_correlations.get((req1, req2)) or \
                                               expected_correlations.get((req2, req1))
                        
                        pattern_type = "unexpected_strong"
                        if expected_direction:
                            actual_direction = "positive" if corr > 0 else "negative"
                            if actual_direction != expected_direction.lower():
                                pattern_type = "direction_mismatch"
                        
                        unusual_patterns.append({
                            'requirement_1': req1,
                            'requirement_2': req2,
                            'correlation': corr,
                            'p_value': p_val,
                            'pattern_type': pattern_type,
                            'strength_category': self.categorize_correlation_strength(corr),
                            'expected_direction': expected_direction,
                            'actual_direction': 'Positive' if corr > 0 else 'Negative'
                        })
        
        return unusual_patterns