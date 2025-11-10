"""
Unit tests for statistical analysis components.
Tests confidence intervals, correlation analysis, and population weighting.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import warnings

from src.analysis.statistical_analyzer import StatisticalAnalyzer
from src.analysis.occupation_analyzer import OccupationAnalyzer
from src.analysis.correlation_analyzer import CorrelationAnalyzer
from src.interfaces import AnalysisResult


class TestStatisticalAnalyzer:
    """Test cases for StatisticalAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        # Sample data for testing
        self.sample_estimates = pd.Series([10.5, 25.3, 45.7, 78.2, 12.1])
        self.sample_std_errors = pd.Series([1.2, 2.5, 3.1, 4.2, 1.8])
        self.sample_footnotes = pd.Series([1, 7, None, 26, 16])
        
        # Sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'soc_code': ['111011', '111011', '291141', '291141', '431011'],
            'occupation': ['Chief Executives', 'Chief Executives', 'Registered Nurses', 
                          'Registered Nurses', 'First-Line Supervisors'],
            'estimate': [79.2, 45.5, 67.8, 23.4, 56.1],
            'standard_error': [2.1, 3.2, 2.8, 4.1, 2.9],
            'requirement_type': ['Physical demands', 'Cognitive requirements', 
                                'Physical demands', 'Environmental conditions', 'Physical demands']
        })
    
    def test_initialization(self):
        """Test StatisticalAnalyzer initialization."""
        analyzer = StatisticalAnalyzer(confidence_level=0.99)
        assert analyzer.confidence_level == 0.99
        assert abs(analyzer.alpha - 0.01) < 1e-10
        assert analyzer.TOTAL_CIVILIAN_WORKERS == 145_866_200
    
    def test_calculate_confidence_intervals_basic(self):
        """Test basic confidence interval calculations."""
        result_df = self.analyzer.calculate_confidence_intervals(
            self.sample_estimates, self.sample_std_errors
        )
        
        # Check structure
        assert len(result_df) == 5
        expected_columns = ['estimate', 'std_error', 'lower_ci', 'upper_ci', 'margin_error', 'confidence_level']
        assert all(col in result_df.columns for col in expected_columns)
        
        # Check calculations for first row
        z_score = 1.96  # Approximately for 95% confidence
        expected_margin = z_score * 1.2
        assert abs(result_df.loc[0, 'margin_error'] - expected_margin) < 0.01
        assert abs(result_df.loc[0, 'lower_ci'] - (10.5 - expected_margin)) < 0.01
        assert abs(result_df.loc[0, 'upper_ci'] - (10.5 + expected_margin)) < 0.01
    
    def test_calculate_confidence_intervals_custom_level(self):
        """Test confidence intervals with custom confidence level."""
        result_df = self.analyzer.calculate_confidence_intervals(
            self.sample_estimates, self.sample_std_errors, confidence_level=0.90
        )
        
        # Check that confidence level is updated
        assert all(result_df['confidence_level'] == 0.90)
        
        # Margin should be smaller for 90% vs 95%
        margin_90 = result_df.loc[0, 'margin_error']
        
        result_95 = self.analyzer.calculate_confidence_intervals(
            self.sample_estimates, self.sample_std_errors, confidence_level=0.95
        )
        margin_95 = result_95.loc[0, 'margin_error']
        
        assert margin_90 < margin_95
    
    def test_calculate_confidence_intervals_missing_std_errors(self):
        """Test confidence intervals with missing standard errors."""
        estimates_with_na = pd.Series([10.5, 25.3, 45.7])
        std_errors_with_na = pd.Series([1.2, np.nan, 3.1])
        
        result_df = self.analyzer.calculate_confidence_intervals(
            estimates_with_na, std_errors_with_na
        )
        
        # Missing std error should result in zero margin
        assert result_df.loc[1, 'margin_error'] == 0.0
        assert result_df.loc[1, 'lower_ci'] == 25.3
        assert result_df.loc[1, 'upper_ci'] == 25.3
    
    def test_calculate_reliability_scores_basic(self):
        """Test basic reliability score calculations."""
        reliability_scores = self.analyzer.calculate_reliability_scores(
            self.sample_estimates, self.sample_std_errors
        )
        
        assert len(reliability_scores) == 5
        assert all(0 <= score <= 1 for score in reliability_scores)
        
        # Lower CV should have higher reliability
        cv_0 = abs(self.sample_std_errors.iloc[0] / self.sample_estimates.iloc[0])
        cv_3 = abs(self.sample_std_errors.iloc[3] / self.sample_estimates.iloc[3])
        
        if cv_0 < cv_3:
            assert reliability_scores.iloc[0] > reliability_scores.iloc[3]
    
    def test_calculate_reliability_scores_with_footnotes(self):
        """Test reliability scores with footnote adjustments."""
        reliability_with_footnotes = self.analyzer.calculate_reliability_scores(
            self.sample_estimates, self.sample_std_errors, self.sample_footnotes
        )
        
        reliability_without_footnotes = self.analyzer.calculate_reliability_scores(
            self.sample_estimates, self.sample_std_errors
        )
        
        # Footnote codes should generally reduce reliability
        # (except for non-precision affecting footnotes)
        assert len(reliability_with_footnotes) == len(reliability_without_footnotes)
    
    def test_apply_population_weighting(self):
        """Test population weighting calculations."""
        weighted_df = self.analyzer.apply_population_weighting(
            self.sample_data, 'estimate', 'soc_code'
        )
        
        # Check new columns
        expected_columns = ['occupation_weight', 'weighted_estimate', 'estimated_workers']
        assert all(col in weighted_df.columns for col in expected_columns)
        
        # Check that weights sum to 1
        unique_weights = weighted_df.drop_duplicates('soc_code')['occupation_weight']
        assert abs(unique_weights.sum() - 1.0) < 0.01
        
        # Check estimated workers calculation
        total_estimated = weighted_df['estimated_workers'].sum()
        assert total_estimated > 0
        assert total_estimated <= self.analyzer.TOTAL_CIVILIAN_WORKERS
    
    def test_calculate_precision_metrics(self):
        """Test precision metrics calculation."""
        metrics = self.analyzer.calculate_precision_metrics(
            self.sample_estimates, self.sample_std_errors
        )
        
        expected_keys = ['mean_cv', 'median_cv', 'high_precision_pct', 
                        'medium_precision_pct', 'low_precision_pct', 'valid_estimates']
        assert all(key in metrics for key in expected_keys)
        
        # Check that percentages sum to 100
        total_pct = (metrics['high_precision_pct'] + 
                    metrics['medium_precision_pct'] + 
                    metrics['low_precision_pct'])
        assert abs(total_pct - 100.0) < 0.01
        
        assert metrics['valid_estimates'] == 5
    
    def test_calculate_statistical_summary(self):
        """Test statistical summary by group."""
        summary_df = self.analyzer.calculate_statistical_summary(
            self.sample_data, 'soc_code', 'estimate', 'standard_error'
        )
        
        # Check structure
        assert len(summary_df) == 3  # Three unique SOC codes
        assert 'count' in summary_df.columns
        assert 'mean_estimate' in summary_df.columns
        assert 'mean_reliability' in summary_df.columns
        
        # Check calculations
        chief_exec_data = self.sample_data[self.sample_data['soc_code'] == '111011']
        chief_exec_summary = summary_df[summary_df['soc_code'] == '111011'].iloc[0]
        
        assert chief_exec_summary['count'] == 2
        assert abs(chief_exec_summary['mean_estimate'] - chief_exec_data['estimate'].mean()) < 0.01
    
    def test_identify_outliers_iqr(self):
        """Test outlier identification using IQR method."""
        # Create data with clear outliers
        data_with_outliers = pd.Series([1, 2, 3, 4, 5, 100, 2, 3, 4, 1])
        
        outliers = self.analyzer.identify_outliers(data_with_outliers, method='iqr')
        
        assert len(outliers) == len(data_with_outliers)
        assert outliers.iloc[5] == True  # 100 should be an outlier
        assert outliers.iloc[0] == False  # 1 should not be an outlier
    
    def test_identify_outliers_zscore(self):
        """Test outlier identification using z-score method."""
        data_with_outliers = pd.Series([1, 2, 3, 4, 5, 100, 2, 3, 4, 1])
        
        outliers = self.analyzer.identify_outliers(data_with_outliers, method='zscore', threshold=2.0)
        
        assert len(outliers) == len(data_with_outliers)
        assert outliers.iloc[5] == True  # 100 should be an outlier
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval calculation."""
        np.random.seed(42)  # For reproducible results
        
        data = pd.Series([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
        
        stat, lower_ci, upper_ci = self.analyzer.bootstrap_confidence_interval(
            data, statistic_func=np.mean, n_bootstrap=100
        )
        
        # Check that statistic is reasonable
        assert abs(stat - data.mean()) < 0.01
        
        # Check that confidence interval makes sense
        assert lower_ci < stat < upper_ci
        assert lower_ci > data.min()
        assert upper_ci < data.max()
    
    def test_bootstrap_confidence_interval_empty_data(self):
        """Test bootstrap with empty data."""
        empty_data = pd.Series([])
        
        stat, lower_ci, upper_ci = self.analyzer.bootstrap_confidence_interval(empty_data)
        
        assert pd.isna(stat)
        assert pd.isna(lower_ci)
        assert pd.isna(upper_ci)


class TestOccupationAnalyzer:
    """Test cases for OccupationAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stats_analyzer = StatisticalAnalyzer()
        self.analyzer = OccupationAnalyzer(self.stats_analyzer)
        
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'soc_code': ['111011', '111011', '291141', '291141', '431011', '431011'],
            'occupation': ['Chief Executives', 'Chief Executives', 'Registered Nurses', 
                          'Registered Nurses', 'First-Line Supervisors', 'First-Line Supervisors'],
            'estimate': [79.2, 45.5, 67.8, 23.4, 56.1, 34.7],
            'standard_error': [2.1, 3.2, 2.8, 4.1, 2.9, 3.5],
            'requirement_type': ['Physical demands', 'Cognitive requirements', 
                                'Physical demands', 'Environmental conditions', 
                                'Physical demands', 'Cognitive requirements']
        })
    
    def test_calculate_frequency_distribution(self):
        """Test frequency distribution calculation."""
        freq_dist = self.analyzer.calculate_frequency_distribution(self.sample_data)
        
        # Check structure
        assert len(freq_dist) == 3  # Three unique occupations
        expected_columns = ['soc_code', 'occupation', 'frequency', 'relative_frequency',
                           'mean_estimate', 'reliability_score']
        assert all(col in freq_dist.columns for col in expected_columns)
        
        # Check calculations
        assert freq_dist['frequency'].sum() == 6  # Total observations
        assert abs(freq_dist['relative_frequency'].sum() - 1.0) < 0.01
        
        # Check sorting (should be by frequency, descending)
        assert freq_dist['frequency'].iloc[0] >= freq_dist['frequency'].iloc[1]
    
    def test_calculate_diversity_metrics(self):
        """Test diversity metrics calculation."""
        diversity_metrics = self.analyzer.calculate_diversity_metrics(self.sample_data)
        
        expected_keys = ['total_occupations', 'total_observations', 'shannon_diversity',
                        'simpson_diversity', 'effective_occupations', 'evenness',
                        'gini_coefficient', 'concentration_ratio_top10pct']
        assert all(key in diversity_metrics for key in expected_keys)
        
        # Check basic values
        assert diversity_metrics['total_occupations'] == 3
        assert diversity_metrics['total_observations'] == 6
        assert 0 <= diversity_metrics['evenness'] <= 1
        assert 0 <= diversity_metrics['gini_coefficient'] <= 1
    
    def test_identify_top_n_occupations(self):
        """Test top N occupation identification."""
        top_occupations = self.analyzer.identify_top_n_occupations(
            self.sample_data, n=2, sort_by='frequency'
        )
        
        # Check structure
        assert len(top_occupations) == 2
        assert all(isinstance(result, AnalysisResult) for result in top_occupations)
        
        # Check that results are sorted by frequency
        first_result = top_occupations[0]
        second_result = top_occupations[1]
        assert first_result.value >= second_result.value
        
        # Check AnalysisResult structure
        assert first_result.occupation_category is not None
        assert first_result.confidence_interval is not None
        assert 0 <= first_result.reliability_score <= 1
        assert len(first_result.footnote_context) > 0
    
    def test_identify_top_n_occupations_by_estimate(self):
        """Test top N occupation identification by mean estimate."""
        top_occupations = self.analyzer.identify_top_n_occupations(
            self.sample_data, n=2, sort_by='mean_estimate'
        )
        
        assert len(top_occupations) == 2
        
        # Should be sorted by mean estimate (descending)
        first_mean = top_occupations[0].value
        second_mean = top_occupations[1].value
        assert first_mean >= second_mean
    
    def test_analyze_occupation_categories(self):
        """Test occupation analysis by categories."""
        category_analysis = self.analyzer.analyze_occupation_categories(
            self.sample_data, 'requirement_type'
        )
        
        # Check structure
        expected_columns = ['category', 'unique_occupations', 'total_observations',
                           'mean_estimate', 'mean_reliability']
        assert all(col in category_analysis.columns for col in expected_columns)
        
        # Check that all requirement types are included
        requirement_types = set(self.sample_data['requirement_type'].unique())
        category_types = set(category_analysis['category'].unique())
        assert requirement_types == category_types
    
    def test_calculate_occupation_similarity(self):
        """Test occupation similarity calculation."""
        similarity_df = self.analyzer.calculate_occupation_similarity(
            self.sample_data, similarity_threshold=0.5
        )
        
        # Check structure
        if not similarity_df.empty:
            expected_columns = ['occupation_1', 'occupation_2', 'similarity_score', 'similarity_category']
            assert all(col in similarity_df.columns for col in expected_columns)
            
            # Check similarity scores are within valid range
            assert all(0 <= score <= 1 for score in similarity_df['similarity_score'])
            
            # Check that similarity scores meet threshold
            assert all(score >= 0.5 for score in similarity_df['similarity_score'])
    
    def test_generate_occupation_summary(self):
        """Test comprehensive occupation summary generation."""
        summary = self.analyzer.generate_occupation_summary(self.sample_data)
        
        # Check main sections
        expected_sections = ['overview', 'diversity_metrics', 'estimate_statistics',
                           'standard_error_statistics', 'top_occupations_by_frequency',
                           'top_occupations_by_estimate', 'population_weighting']
        assert all(section in summary for section in expected_sections)
        
        # Check overview
        overview = summary['overview']
        assert overview['total_observations'] == 6
        assert overview['unique_occupations'] == 3
        assert overview['civilian_workers_represented'] == StatisticalAnalyzer.TOTAL_CIVILIAN_WORKERS
        
        # Check that top occupations lists are not empty
        assert len(summary['top_occupations_by_frequency']) > 0
        assert len(summary['top_occupations_by_estimate']) > 0


class TestCorrelationAnalyzer:
    """Test cases for CorrelationAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = CorrelationAnalyzer(significance_level=0.05, min_sample_size=3)
        
        # Sample data for testing correlations
        self.sample_data = pd.DataFrame({
            'soc_code': ['111011', '111011', '111011', '291141', '291141', '291141',
                        '431011', '431011', '431011'],
            'requirement_type': ['Physical demands', 'Cognitive requirements', 'Environmental conditions',
                               'Physical demands', 'Cognitive requirements', 'Environmental conditions',
                               'Physical demands', 'Cognitive requirements', 'Environmental conditions'],
            'estimate': [79.2, 45.5, 30.1, 67.8, 23.4, 15.2, 56.1, 34.7, 25.8]
        })
    
    def test_calculate_correlation_matrix_pearson(self):
        """Test Pearson correlation matrix calculation."""
        corr_matrix = self.analyzer.calculate_correlation_matrix(
            self.sample_data, method='pearson'
        )
        
        # Check structure
        requirements = self.sample_data['requirement_type'].unique()
        assert corr_matrix.shape == (len(requirements), len(requirements))
        assert list(corr_matrix.index) == list(corr_matrix.columns)
        
        # Check diagonal elements are 1
        for req in requirements:
            assert abs(corr_matrix.loc[req, req] - 1.0) < 0.01
        
        # Check symmetry
        for i, req1 in enumerate(requirements):
            for j, req2 in enumerate(requirements):
                if i != j:
                    assert abs(corr_matrix.loc[req1, req2] - corr_matrix.loc[req2, req1]) < 0.01
    
    def test_calculate_correlation_matrix_spearman(self):
        """Test Spearman correlation matrix calculation."""
        corr_matrix = self.analyzer.calculate_correlation_matrix(
            self.sample_data, method='spearman'
        )
        
        requirements = self.sample_data['requirement_type'].unique()
        assert corr_matrix.shape == (len(requirements), len(requirements))
        
        # Diagonal should still be 1
        for req in requirements:
            assert abs(corr_matrix.loc[req, req] - 1.0) < 0.01
    
    def test_calculate_correlation_with_significance(self):
        """Test correlation calculation with significance testing."""
        corr_matrix, p_val_matrix, sample_matrix = self.analyzer.calculate_correlation_with_significance(
            self.sample_data
        )
        
        requirements = self.sample_data['requirement_type'].unique()
        
        # Check all matrices have same shape
        assert corr_matrix.shape == p_val_matrix.shape == sample_matrix.shape
        
        # Check diagonal elements
        for req in requirements:
            assert abs(corr_matrix.loc[req, req] - 1.0) < 0.01
            assert p_val_matrix.loc[req, req] == 0.0
            assert sample_matrix.loc[req, req] > 0
    
    def test_categorize_correlation_strength(self):
        """Test correlation strength categorization."""
        assert self.analyzer.categorize_correlation_strength(0.95) == "Very Strong"
        assert self.analyzer.categorize_correlation_strength(0.75) == "Strong"
        assert self.analyzer.categorize_correlation_strength(0.55) == "Moderate"
        assert self.analyzer.categorize_correlation_strength(0.35) == "Weak"
        assert self.analyzer.categorize_correlation_strength(0.15) == "Very Weak"
        assert self.analyzer.categorize_correlation_strength(0.05) == "Negligible"
        
        # Test negative correlations
        assert self.analyzer.categorize_correlation_strength(-0.85) == "Strong"
    
    def test_identify_significant_correlations(self):
        """Test identification of significant correlations."""
        # Create correlation matrices with known values
        requirements = ['Physical demands', 'Cognitive requirements', 'Environmental conditions']
        
        corr_matrix = pd.DataFrame({
            'Physical demands': [1.0, 0.8, 0.2],
            'Cognitive requirements': [0.8, 1.0, -0.6],
            'Environmental conditions': [0.2, -0.6, 1.0]
        }, index=requirements)
        
        p_val_matrix = pd.DataFrame({
            'Physical demands': [0.0, 0.01, 0.3],
            'Cognitive requirements': [0.01, 0.0, 0.02],
            'Environmental conditions': [0.3, 0.02, 0.0]
        }, index=requirements)
        
        sample_matrix = pd.DataFrame({
            'Physical demands': [50, 50, 50],
            'Cognitive requirements': [50, 50, 50],
            'Environmental conditions': [50, 50, 50]
        }, index=requirements)
        
        significant_corrs = self.analyzer.identify_significant_correlations(
            corr_matrix, p_val_matrix, sample_matrix, min_correlation=0.3
        )
        
        # Should find 2 significant correlations (0.8 and -0.6)
        assert len(significant_corrs) == 2
        
        # Check structure
        for corr in significant_corrs:
            assert 'correlation' in corr
            assert 'p_value' in corr
            assert 'strength_category' in corr
            assert 'direction' in corr
            assert corr['is_significant'] == True
        
        # Check sorting (by absolute correlation, descending)
        assert abs(significant_corrs[0]['correlation']) >= abs(significant_corrs[1]['correlation'])
    
    def test_analyze_requirement_clusters(self):
        """Test requirement clustering analysis."""
        # Create correlation matrix with clear clusters
        requirements = ['Phys_A', 'Phys_B', 'Cog_A', 'Cog_B']
        
        corr_matrix = pd.DataFrame({
            'Phys_A': [1.0, 0.8, 0.2, 0.1],
            'Phys_B': [0.8, 1.0, 0.1, 0.2],
            'Cog_A': [0.2, 0.1, 1.0, 0.7],
            'Cog_B': [0.1, 0.2, 0.7, 1.0]
        }, index=requirements)
        
        clusters = self.analyzer.analyze_requirement_clusters(
            corr_matrix, clustering_threshold=0.6
        )
        
        # Should identify clusters
        assert len(clusters) >= 1
        
        # Check that highly correlated requirements are clustered together
        cluster_found = False
        for cluster_name, cluster_reqs in clusters.items():
            if 'Phys_A' in cluster_reqs and 'Phys_B' in cluster_reqs:
                cluster_found = True
                break
        assert cluster_found
    
    def test_generate_correlation_summary(self):
        """Test comprehensive correlation summary generation."""
        summary = self.analyzer.generate_correlation_summary(self.sample_data)
        
        # Check main sections
        expected_sections = ['overview', 'strength_distribution', 'significant_correlations',
                           'requirement_clusters', 'correlation_matrix', 'p_value_matrix',
                           'sample_size_matrix']
        assert all(section in summary for section in expected_sections)
        
        # Check overview
        overview = summary['overview']
        assert overview['total_requirements'] > 0
        assert overview['total_correlations'] >= 0
        assert 'mean_correlation' in overview
        assert 'max_correlation' in overview
        
        # Check matrices
        assert isinstance(summary['correlation_matrix'], pd.DataFrame)
        assert isinstance(summary['p_value_matrix'], pd.DataFrame)
        assert isinstance(summary['sample_size_matrix'], pd.DataFrame)
    
    def test_identify_unusual_patterns(self):
        """Test identification of unusual correlation patterns."""
        # Create correlation matrix with some unusual patterns
        requirements = ['Physical demands', 'Cognitive requirements']
        
        corr_matrix = pd.DataFrame({
            'Physical demands': [1.0, 0.9],  # Unexpectedly high correlation
            'Cognitive requirements': [0.9, 1.0]
        }, index=requirements)
        
        p_val_matrix = pd.DataFrame({
            'Physical demands': [0.0, 0.001],
            'Cognitive requirements': [0.001, 0.0]
        }, index=requirements)
        
        unusual_patterns = self.analyzer.identify_unusual_patterns(
            corr_matrix, p_val_matrix
        )
        
        # Should identify the high correlation as unusual
        assert len(unusual_patterns) >= 1
        
        if unusual_patterns:
            pattern = unusual_patterns[0]
            assert 'pattern_type' in pattern
            assert 'correlation' in pattern
            assert 'strength_category' in pattern


class TestStatisticalAnalysisIntegration:
    """Integration tests for statistical analysis components."""
    
    def setup_method(self):
        """Set up test fixtures for integration testing."""
        self.stats_analyzer = StatisticalAnalyzer()
        self.occupation_analyzer = OccupationAnalyzer(self.stats_analyzer)
        self.correlation_analyzer = CorrelationAnalyzer()
        
        # Larger sample dataset for integration testing
        np.random.seed(42)
        n_occupations = 10
        n_requirements = 5
        n_observations = 100
        
        occupations = [f"Occupation_{i:02d}" for i in range(n_occupations)]
        requirements = ['Physical demands', 'Cognitive requirements', 'Environmental conditions',
                       'Education requirements', 'Social skills']
        
        # Generate realistic data
        data_rows = []
        for _ in range(n_observations):
            occ = np.random.choice(occupations)
            req = np.random.choice(requirements)
            
            # Create some correlation structure
            base_estimate = np.random.normal(50, 20)
            if req == 'Physical demands':
                estimate = max(0, min(100, base_estimate + np.random.normal(0, 10)))
            elif req == 'Cognitive requirements':
                estimate = max(0, min(100, base_estimate + np.random.normal(5, 15)))
            else:
                estimate = max(0, min(100, base_estimate + np.random.normal(0, 12)))
            
            std_error = max(0.1, estimate * 0.05 + np.random.normal(0, 1))
            
            data_rows.append({
                'soc_code': occ,
                'occupation': occ.replace('_', ' '),
                'requirement_type': req,
                'estimate': round(estimate, 1),
                'standard_error': round(std_error, 2)
            })
        
        self.integration_data = pd.DataFrame(data_rows)
    
    def test_complete_statistical_pipeline(self):
        """Test complete statistical analysis pipeline."""
        # Step 1: Calculate confidence intervals
        ci_data = self.stats_analyzer.calculate_confidence_intervals(
            self.integration_data['estimate'],
            self.integration_data['standard_error']
        )
        
        assert len(ci_data) == len(self.integration_data)
        assert all(ci_data['lower_ci'] <= ci_data['estimate'])
        assert all(ci_data['estimate'] <= ci_data['upper_ci'])
        
        # Step 2: Calculate reliability scores
        reliability_scores = self.stats_analyzer.calculate_reliability_scores(
            self.integration_data['estimate'],
            self.integration_data['standard_error']
        )
        
        assert len(reliability_scores) == len(self.integration_data)
        assert all(0 <= score <= 1 for score in reliability_scores)
        
        # Step 3: Apply population weighting
        weighted_data = self.stats_analyzer.apply_population_weighting(
            self.integration_data
        )
        
        assert 'occupation_weight' in weighted_data.columns
        assert 'estimated_workers' in weighted_data.columns
        
        # Step 4: Occupation distribution analysis
        freq_dist = self.occupation_analyzer.calculate_frequency_distribution(
            self.integration_data
        )
        
        assert len(freq_dist) <= self.integration_data['soc_code'].nunique()
        
        # Step 5: Correlation analysis
        corr_summary = self.correlation_analyzer.generate_correlation_summary(
            self.integration_data
        )
        
        assert 'correlation_matrix' in corr_summary
        assert 'significant_correlations' in corr_summary
    
    def test_precision_and_reliability_consistency(self):
        """Test consistency between precision metrics and reliability scores."""
        # Calculate precision metrics
        precision_metrics = self.stats_analyzer.calculate_precision_metrics(
            self.integration_data['estimate'],
            self.integration_data['standard_error']
        )
        
        # Calculate reliability scores
        reliability_scores = self.stats_analyzer.calculate_reliability_scores(
            self.integration_data['estimate'],
            self.integration_data['standard_error']
        )
        
        # High precision should correlate with high reliability
        mean_reliability = reliability_scores.mean()
        
        # If most estimates are high precision, mean reliability should be high
        if precision_metrics['high_precision_pct'] > 50:
            assert mean_reliability > 0.5
    
    def test_population_weighting_consistency(self):
        """Test population weighting calculations for consistency."""
        weighted_data = self.stats_analyzer.apply_population_weighting(
            self.integration_data
        )
        
        # Check that weights are properly normalized
        unique_weights = weighted_data.drop_duplicates('soc_code')['occupation_weight']
        assert abs(unique_weights.sum() - 1.0) < 0.01
        
        # Check that estimated workers sum is reasonable
        total_estimated = weighted_data['estimated_workers'].sum()
        assert 0 < total_estimated <= StatisticalAnalyzer.TOTAL_CIVILIAN_WORKERS * 2  # Allow some margin
    
    def test_correlation_and_occupation_analysis_consistency(self):
        """Test consistency between correlation and occupation analyses."""
        # Get occupation similarity from occupation analyzer
        similarity_df = self.occupation_analyzer.calculate_occupation_similarity(
            self.integration_data, similarity_threshold=0.3
        )
        
        # Get correlation analysis
        corr_summary = self.correlation_analyzer.generate_correlation_summary(
            self.integration_data
        )
        
        # Both should identify relationships, though from different perspectives
        has_similarities = not similarity_df.empty
        has_correlations = len(corr_summary['significant_correlations']) > 0
        
        # At least one should find relationships in this dataset
        assert has_similarities or has_correlations
    
    def test_statistical_summary_completeness(self):
        """Test that statistical summaries include all expected components."""
        # Occupation summary
        occ_summary = self.occupation_analyzer.generate_occupation_summary(
            self.integration_data
        )
        
        # Should have comprehensive information
        assert occ_summary['overview']['total_observations'] == len(self.integration_data)
        assert occ_summary['overview']['unique_occupations'] > 0
        assert len(occ_summary['top_occupations_by_frequency']) > 0
        
        # Correlation summary
        corr_summary = self.correlation_analyzer.generate_correlation_summary(
            self.integration_data
        )
        
        # Should have correlation matrix and statistics
        assert corr_summary['overview']['total_requirements'] > 0
        assert isinstance(corr_summary['correlation_matrix'], pd.DataFrame)
        assert 'strength_distribution' in corr_summary


# Performance and edge case tests
class TestStatisticalAnalysisEdgeCases:
    """Test edge cases and performance scenarios."""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        empty_df = pd.DataFrame(columns=['soc_code', 'occupation', 'estimate', 'standard_error'])
        
        analyzer = StatisticalAnalyzer()
        
        # Should handle empty data gracefully
        precision_metrics = analyzer.calculate_precision_metrics(
            pd.Series([], dtype=float), pd.Series([], dtype=float)
        )
        
        assert precision_metrics['valid_estimates'] == 0
        assert precision_metrics['high_precision_pct'] == 0.0
    
    def test_single_observation_handling(self):
        """Test handling of single observation datasets."""
        single_obs_data = pd.DataFrame({
            'soc_code': ['111011'],
            'occupation': ['Chief Executive'],
            'estimate': [75.5],
            'standard_error': [2.1]
        })
        
        analyzer = OccupationAnalyzer()
        
        # Should handle single observation
        freq_dist = analyzer.calculate_frequency_distribution(single_obs_data)
        assert len(freq_dist) == 1
        assert freq_dist.iloc[0]['frequency'] == 1
    
    def test_missing_data_handling(self):
        """Test handling of datasets with missing values."""
        data_with_missing = pd.DataFrame({
            'soc_code': ['111011', '291141', '431011'],
            'occupation': ['Chief Executive', None, 'Supervisor'],
            'estimate': [75.5, np.nan, 45.2],
            'standard_error': [2.1, 3.2, np.nan]
        })
        
        analyzer = StatisticalAnalyzer()
        
        # Should handle missing values
        ci_data = analyzer.calculate_confidence_intervals(
            data_with_missing['estimate'],
            data_with_missing['standard_error']
        )
        
        assert len(ci_data) == 3
        # Missing standard error should result in zero margin
        assert ci_data.loc[2, 'margin_error'] == 0.0
    
    def test_extreme_values_handling(self):
        """Test handling of extreme values."""
        extreme_data = pd.DataFrame({
            'soc_code': ['111011', '291141', '431011'],
            'estimate': [0.001, 99.999, 50.0],  # Very small, very large, normal
            'standard_error': [0.0001, 10.0, 2.5]  # Very small, very large, normal
        })
        
        analyzer = StatisticalAnalyzer()
        
        # Should handle extreme values
        reliability_scores = analyzer.calculate_reliability_scores(
            extreme_data['estimate'],
            extreme_data['standard_error']
        )
        
        assert len(reliability_scores) == 3
        assert all(0 <= score <= 1 for score in reliability_scores)
        
        # Very large standard error should result in low reliability
        assert reliability_scores.iloc[1] < reliability_scores.iloc[2]