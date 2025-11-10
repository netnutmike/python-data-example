"""
Unit tests for data quality assessment and establishment analysis components.
Tests data quality scoring accuracy, establishment coverage calculations,
and reliability assessment against known quality indicators.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import warnings
from datetime import datetime

from src.analysis.quality_assessor import QualityAssessor, QualityMetrics, DataCompletenessResult
from src.analysis.establishment_analyzer import EstablishmentAnalyzer, EstablishmentCoverage, WorkforceCoverage, PolicyImplication
from src.analysis.statistical_analyzer import StatisticalAnalyzer
from src.interfaces import ValidationResult, AnalysisResult


class TestQualityAssessor:
    """Test cases for QualityAssessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stats_analyzer = StatisticalAnalyzer()
        self.assessor = QualityAssessor(self.stats_analyzer)
        
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'SOC 2018 CODE': ['111011', '111011', '291141', '291141', '431011', '431011'],
            'OCCUPATION': ['Chief Executives', 'Chief Executives', 'Registered Nurses', 
                          'Registered Nurses', 'First-Line Supervisors', 'First-Line Supervisors'],
            'REQUIREMENT': ['Physical demands', 'Cognitive requirements', 
                           'Physical demands', 'Environmental conditions', 
                           'Physical demands', 'Cognitive requirements'],
            'ESTIMATE': [79.2, 45.5, 67.8, 23.4, 56.1, 34.7],
            'STANDARD ERROR': [2.1, 3.2, 2.8, 4.1, 2.9, 3.5],
            'DATA FOOTNOTE': [1, 7, None, 26, 16, None]
        })
        
        # Data with quality issues for testing
        self.problematic_data = pd.DataFrame({
            'SOC 2018 CODE': ['111011', '111011', None, '291141', '431011'],
            'OCCUPATION': ['Chief Executives', None, 'Registered Nurses', 
                          'Registered Nurses', 'First-Line Supervisors'],
            'REQUIREMENT': ['Physical demands', 'Cognitive requirements', 
                           None, 'Environmental conditions', 'Physical demands'],
            'ESTIMATE': [79.2, np.nan, 67.8, -5.0, 150.0],  # Missing, negative, extreme values
            'STANDARD ERROR': [2.1, 3.2, np.nan, 4.1, np.nan],
            'DATA FOOTNOTE': [1, 7, None, 26, 16]
        })
    
    def test_initialization(self):
        """Test QualityAssessor initialization."""
        assessor = QualityAssessor()
        assert assessor.statistical_analyzer is not None
        assert len(assessor.reliability_thresholds) == 4
        assert len(assessor.completeness_thresholds) == 4
        assert len(assessor.critical_columns) == 5
        assert len(assessor.footnote_precision_mapping) > 0
    
    def test_assess_data_quality_basic(self):
        """Test basic data quality assessment."""
        quality_assessment = self.assessor.assess_data_quality(self.sample_data)
        
        # Check structure
        expected_keys = ['validation_result', 'completeness_analysis', 'reliability_scores',
                        'precision_analysis', 'coverage_analysis', 'overall_metrics',
                        'assessment_timestamp', 'total_records_assessed']
        assert all(key in quality_assessment for key in expected_keys)
        
        # Check basic values
        assert quality_assessment['total_records_assessed'] == 6
        assert isinstance(quality_assessment['assessment_timestamp'], pd.Timestamp)
        
        # Validation should pass for good data
        validation = quality_assessment['validation_result']
        assert validation.is_valid == True
        assert len(validation.errors) == 0
    
    def test_assess_data_quality_problematic_data(self):
        """Test data quality assessment with problematic data."""
        quality_assessment = self.assessor.assess_data_quality(self.problematic_data)
        
        # Validation should identify issues
        validation = quality_assessment['validation_result']
        assert len(validation.warnings) > 0  # Should have warnings about data issues
        
        # Completeness should be lower
        completeness = quality_assessment['completeness_analysis']
        assert completeness.completeness_percentage < 100
        assert len(completeness.missing_by_column) > 0
    
    def test_calculate_reliability_scores(self):
        """Test reliability score calculations."""
        reliability_results = self.assessor.calculate_reliability_scores(self.sample_data)
        
        # Check structure
        expected_keys = ['individual_scores', 'summary_statistics', 'low_reliability_records',
                        'reliability_by_occupation', 'reliability_by_requirement']
        assert all(key in reliability_results for key in expected_keys)
        
        # Check individual scores
        individual_scores = reliability_results['individual_scores']
        assert len(individual_scores) == 6
        assert all(0 <= score <= 1 for score in individual_scores)
        
        # Check summary statistics
        summary = reliability_results['summary_statistics']
        assert 'mean_reliability' in summary
        assert 'reliability_distribution' in summary
        assert summary['total_assessed'] == 6
    
    def test_calculate_reliability_scores_missing_columns(self):
        """Test reliability scores with missing required columns."""
        incomplete_data = self.sample_data.drop(columns=['STANDARD ERROR'])
        
        reliability_results = self.assessor.calculate_reliability_scores(incomplete_data)
        
        # Should return error message
        assert 'error' in reliability_results
        assert 'Required columns' in reliability_results['error']
    
    def test_categorize_precision_levels(self):
        """Test precision level categorization."""
        precision_analysis = self.assessor.categorize_precision_levels(self.sample_data)
        
        # Check structure
        expected_keys = ['footnote_precision', 'statistical_precision', 
                        'combined_precision', 'precision_summary']
        assert all(key in precision_analysis for key in expected_keys)
        
        # Check footnote precision
        footnote_precision = precision_analysis['footnote_precision']
        assert 'distribution' in footnote_precision
        assert 'high_precision_count' in footnote_precision
        
        # Check statistical precision
        statistical_precision = precision_analysis['statistical_precision']
        assert 'cv_mean' in statistical_precision
        assert 'distribution' in statistical_precision
    
    def test_analyze_data_completeness(self):
        """Test data completeness analysis."""
        completeness_result = self.assessor.analyze_data_completeness(self.sample_data)
        
        # Check structure
        assert isinstance(completeness_result, DataCompletenessResult)
        assert completeness_result.total_records == 6
        assert completeness_result.completeness_percentage > 0
        assert isinstance(completeness_result.missing_by_column, dict)
        assert isinstance(completeness_result.missing_patterns, dict)
        assert isinstance(completeness_result.critical_missing, list)
        
        # For good data, completeness should be high
        assert completeness_result.completeness_percentage > 80
    
    def test_analyze_data_completeness_problematic_data(self):
        """Test data completeness analysis with problematic data."""
        completeness_result = self.assessor.analyze_data_completeness(self.problematic_data)
        
        # Should identify missing data
        assert completeness_result.completeness_percentage < 100
        assert len(completeness_result.critical_missing) > 0
        
        # Should identify missing patterns
        assert len(completeness_result.missing_patterns) > 0
    
    def test_analyze_data_coverage(self):
        """Test data coverage analysis."""
        coverage_analysis = self.assessor.analyze_data_coverage(self.sample_data)
        
        # Check structure
        expected_keys = ['soc_coverage', 'requirement_coverage', 
                        'occupation_coverage', 'estimate_coverage']
        assert all(key in coverage_analysis for key in expected_keys)
        
        # Check SOC coverage
        soc_coverage = coverage_analysis['soc_coverage']
        assert soc_coverage['unique_soc_codes'] == 3
        assert soc_coverage['total_soc_observations'] == 6
        
        # Check requirement coverage
        req_coverage = coverage_analysis['requirement_coverage']
        assert req_coverage['unique_requirements'] == 3
        assert 'balanced_coverage' in req_coverage
    
    def test_generate_quality_recommendations(self):
        """Test quality recommendation generation."""
        quality_assessment = self.assessor.assess_data_quality(self.problematic_data)
        recommendations = self.assessor.generate_quality_recommendations(quality_assessment)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should have recommendations for problematic data
        recommendation_text = ' '.join(recommendations)
        assert any(keyword in recommendation_text.lower() 
                  for keyword in ['completeness', 'reliability', 'precision', 'missing'])
    
    def test_validate_data_structure(self):
        """Test data structure validation."""
        # Test with good data
        validation_result = self.assessor._validate_data_structure(self.sample_data)
        assert validation_result.is_valid == True
        assert len(validation_result.errors) == 0
        
        # Test with missing critical columns
        incomplete_data = self.sample_data.drop(columns=['SOC 2018 CODE'])
        validation_result = self.assessor._validate_data_structure(incomplete_data)
        assert validation_result.is_valid == False
        assert len(validation_result.errors) > 0
    
    def test_combine_precision_assessments(self):
        """Test combination of footnote and statistical precision assessments."""
        footnote_precision = pd.Series(['high', 'medium', 'low', 'very_low', 'unknown'])
        statistical_precision = pd.Series(['high', 'low', 'medium', 'high', 'medium'])
        
        combined = self.assessor._combine_precision_assessments(
            footnote_precision, statistical_precision
        )
        
        assert len(combined) == 5
        # Should take the more conservative (lower) assessment
        assert combined.iloc[0] == 'high'  # high + high = high
        assert combined.iloc[1] == 'low'   # medium + low = low
    
    def test_calculate_overall_quality_metrics(self):
        """Test overall quality metrics calculation."""
        # Create mock inputs
        validation_result = ValidationResult(True, [], [], 100)
        completeness_result = DataCompletenessResult(100, 95, 95.0, {}, {}, [])
        reliability_scores = {'summary_statistics': {'mean_reliability': 0.8}}
        precision_analysis = {'precision_summary': {'overall_high_precision_percentage': 70}}
        coverage_analysis = {'soc_coverage': {'coverage_breadth': 60}}
        
        overall_metrics = self.assessor._calculate_overall_quality_metrics(
            validation_result, completeness_result, reliability_scores,
            precision_analysis, coverage_analysis
        )
        
        # Check structure
        expected_keys = ['validation_score', 'completeness_score', 'reliability_score',
                        'precision_score', 'coverage_score', 'overall_score',
                        'quality_category', 'component_weights']
        assert all(key in overall_metrics for key in expected_keys)
        
        # Check score ranges
        assert 0 <= overall_metrics['overall_score'] <= 1
        assert overall_metrics['quality_category'] in ['excellent', 'good', 'fair', 'poor']


class TestEstablishmentAnalyzer:
    """Test cases for EstablishmentAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = EstablishmentAnalyzer()
        
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'SOC 2018 CODE': ['111011', '111011', '291141', '291141', '431011', '431011'],
            'OCCUPATION': ['Chief Executives', 'Chief Executives', 'Registered Nurses', 
                          'Registered Nurses', 'First-Line Supervisors', 'First-Line Supervisors'],
            'REQUIREMENT': ['Physical demands', 'Cognitive requirements', 
                           'Physical demands', 'Environmental conditions', 
                           'Physical demands', 'Cognitive requirements'],
            'ESTIMATE': [79.2, 45.5, 67.8, 23.4, 56.1, 34.7],
            'STANDARD ERROR': [2.1, 3.2, 2.8, 4.1, 2.9, 3.5]
        })
    
    def test_initialization(self):
        """Test EstablishmentAnalyzer initialization."""
        assert self.analyzer.TOTAL_ESTABLISHMENTS_SURVEYED == 56_300
        assert self.analyzer.TOTAL_CIVILIAN_WORKERS == 145_866_200
        assert self.analyzer.SURVEY_YEAR == 2023
        assert len(self.analyzer.industry_categories) > 0
        assert len(self.analyzer.size_categories) == 4
    
    def test_analyze_sample_representativeness(self):
        """Test sample representativeness analysis."""
        representativeness = self.analyzer.analyze_sample_representativeness(self.sample_data)
        
        # Check structure
        expected_keys = ['sample_statistics', 'industry_representation', 
                        'occupation_representation', 'geographic_representation',
                        'sample_quality', 'overall_representativeness_score',
                        'analysis_timestamp']
        assert all(key in representativeness for key in expected_keys)
        
        # Check sample statistics
        sample_stats = representativeness['sample_statistics']
        assert sample_stats['total_observations'] == 6
        assert sample_stats['unique_establishments_estimated'] == 56_300
        
        # Check representativeness score
        score = representativeness['overall_representativeness_score']
        assert 0 <= score <= 1
    
    def test_calculate_workforce_coverage_statistics(self):
        """Test workforce coverage statistics calculation."""
        coverage_stats = self.analyzer.calculate_workforce_coverage_statistics(self.sample_data)
        
        # Check structure
        assert isinstance(coverage_stats, WorkforceCoverage)
        assert coverage_stats.total_civilian_workers == 145_866_200
        assert coverage_stats.represented_workers > 0
        assert 0 <= coverage_stats.coverage_percentage <= 100
        assert isinstance(coverage_stats.occupation_coverage, dict)
        assert isinstance(coverage_stats.requirement_coverage, dict)
        assert isinstance(coverage_stats.weighted_representation, dict)
    
    def test_assess_policy_implications(self):
        """Test policy implications assessment."""
        coverage_stats = self.analyzer.calculate_workforce_coverage_statistics(self.sample_data)
        policy_implications = self.analyzer.assess_policy_implications(self.sample_data, coverage_stats)
        
        # Check structure
        assert isinstance(policy_implications, list)
        assert len(policy_implications) > 0
        
        # Check PolicyImplication structure
        for implication in policy_implications:
            assert isinstance(implication, PolicyImplication)
            assert implication.policy_area is not None
            assert implication.affected_workers > 0
            assert implication.confidence_level in ['high', 'medium', 'low']
            assert isinstance(implication.key_findings, list)
            assert isinstance(implication.recommendations, list)
            assert isinstance(implication.data_limitations, list)
    
    def test_generate_coverage_report(self):
        """Test comprehensive coverage report generation."""
        coverage_report = self.analyzer.generate_coverage_report(self.sample_data)
        
        # Check structure
        expected_keys = ['executive_summary', 'sample_representativeness', 
                        'workforce_coverage', 'policy_implications',
                        'policy_data_quality', 'coverage_gaps',
                        'policy_recommendations', 'report_metadata']
        assert all(key in coverage_report for key in expected_keys)
        
        # Check executive summary
        exec_summary = coverage_report['executive_summary']
        assert 'key_findings' in exec_summary
        assert 'data_strengths' in exec_summary
        assert 'policy_applications' in exec_summary
        assert 'recommended_uses' in exec_summary
        assert 'cautions' in exec_summary
        
        # Check metadata
        metadata = coverage_report['report_metadata']
        assert metadata['total_establishments_surveyed'] == 56_300
        assert metadata['total_civilian_workers'] == 145_866_200
        assert metadata['survey_year'] == 2023
        assert metadata['data_records_analyzed'] == 6
    
    def test_calculate_sample_statistics(self):
        """Test sample statistics calculation."""
        sample_stats = self.analyzer._calculate_sample_statistics(self.sample_data)
        
        # Check basic statistics
        assert sample_stats['total_observations'] == 6
        assert sample_stats['unique_establishments_estimated'] == 56_300
        assert sample_stats['observations_per_establishment'] > 0
        
        # Check occupation and requirement statistics
        assert sample_stats['unique_occupations'] == 3
        assert sample_stats['unique_requirements'] == 3
        assert isinstance(sample_stats['occupation_observations'], dict)
        assert isinstance(sample_stats['requirement_observations'], dict)
    
    def test_analyze_industry_representation(self):
        """Test industry representation analysis."""
        industry_rep = self.analyzer._analyze_industry_representation(self.sample_data)
        
        # Check structure
        expected_keys = ['methodology_note', 'estimated_industry_coverage',
                        'industry_balance_score', 'underrepresented_industries',
                        'overrepresented_industries']
        assert all(key in industry_rep for key in expected_keys)
        
        # Check that methodology note explains limitations
        assert 'inferred' in industry_rep['methodology_note'].lower()
        
        # Check balance score
        assert 0 <= industry_rep['industry_balance_score'] <= 1
    
    def test_analyze_occupation_representation(self):
        """Test occupation representation analysis."""
        occupation_rep = self.analyzer._analyze_occupation_representation(self.sample_data)
        
        # Check structure
        expected_keys = ['total_unique_occupations', 'occupation_distribution',
                        'occupation_balance_score', 'underrepresented_occupations',
                        'overrepresented_occupations', 'coverage_breadth']
        assert all(key in occupation_rep for key in expected_keys)
        
        # Check values
        assert occupation_rep['total_unique_occupations'] == 3
        assert 0 <= occupation_rep['occupation_balance_score'] <= 1
        assert 0 <= occupation_rep['coverage_breadth'] <= 100
    
    def test_assess_safety_policy_implications(self):
        """Test safety policy implications assessment."""
        coverage_stats = WorkforceCoverage(
            total_civilian_workers=145_866_200,
            represented_workers=100_000_000,
            coverage_percentage=68.5,
            occupation_coverage={},
            requirement_coverage={},
            weighted_representation={}
        )
        
        safety_implications = self.analyzer._assess_safety_policy_implications(
            self.sample_data, coverage_stats
        )
        
        # Should return list of PolicyImplication objects
        assert isinstance(safety_implications, list)
        
        # If environmental/safety requirements found, should have implications
        if any('environmental' in req.lower() or 'physical' in req.lower() 
               for req in self.sample_data['REQUIREMENT']):
            assert len(safety_implications) > 0
            
            for implication in safety_implications:
                assert isinstance(implication, PolicyImplication)
                assert 'safety' in implication.policy_area.lower()
    
    def test_identify_coverage_gaps(self):
        """Test coverage gaps identification."""
        coverage_stats = WorkforceCoverage(
            total_civilian_workers=145_866_200,
            represented_workers=100_000_000,
            coverage_percentage=68.5,
            occupation_coverage={'Occupation A': 10, 'Occupation B': 5, 'Occupation C': 1},
            requirement_coverage={'Requirement A': 50, 'Requirement B': 30, 'Requirement C': 5},
            weighted_representation={}
        )
        
        coverage_gaps = self.analyzer._identify_coverage_gaps(self.sample_data, coverage_stats)
        
        # Check structure
        expected_keys = ['occupational_gaps', 'requirement_gaps', 'demographic_gaps',
                        'geographic_gaps', 'temporal_gaps']
        assert all(key in coverage_gaps for key in expected_keys)
        
        # Should identify underrepresented items
        assert isinstance(coverage_gaps['occupational_gaps'], list)
        assert isinstance(coverage_gaps['requirement_gaps'], list)
        
        # Should have standard gaps for this survey type
        assert len(coverage_gaps['demographic_gaps']) > 0
        assert len(coverage_gaps['geographic_gaps']) > 0
        assert len(coverage_gaps['temporal_gaps']) > 0
    
    def test_calculate_balance_score(self):
        """Test balance score calculation."""
        # Balanced distribution
        balanced_series = pd.Series([10, 10, 10, 10])
        balance_score = self.analyzer._calculate_balance_score(balanced_series)
        assert balance_score > 0.8  # Should be high for balanced data
        
        # Unbalanced distribution
        unbalanced_series = pd.Series([100, 1, 1, 1])
        balance_score = self.analyzer._calculate_balance_score(unbalanced_series)
        assert balance_score < 0.5  # Should be low for unbalanced data
        
        # Empty series
        empty_series = pd.Series([])
        balance_score = self.analyzer._calculate_balance_score(empty_series)
        assert balance_score == 0.0
    
    def test_calculate_response_completeness(self):
        """Test response completeness calculation."""
        # Complete data
        completeness = self.analyzer._calculate_response_completeness(self.sample_data)
        assert completeness == 100.0
        
        # Data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'ESTIMATE'] = np.nan
        data_with_missing.loc[1, 'STANDARD ERROR'] = np.nan
        
        completeness = self.analyzer._calculate_response_completeness(data_with_missing)
        assert completeness < 100.0
        
        # Empty data
        empty_data = pd.DataFrame()
        completeness = self.analyzer._calculate_response_completeness(empty_data)
        assert completeness == 0.0
    
    def test_calculate_data_consistency_score(self):
        """Test data consistency score calculation."""
        # Good data should have high consistency
        consistency_score = self.analyzer._calculate_data_consistency_score(self.sample_data)
        assert 0.5 <= consistency_score <= 1.0
        
        # Data with inconsistencies
        inconsistent_data = self.sample_data.copy()
        inconsistent_data.loc[0, 'ESTIMATE'] = -50.0  # Negative estimate
        inconsistent_data.loc[1, 'ESTIMATE'] = 2000.0  # Extreme estimate
        
        consistency_score = self.analyzer._calculate_data_consistency_score(inconsistent_data)
        assert consistency_score < 0.9  # Should be lower due to inconsistencies


class TestQualityAssessmentIntegration:
    """Integration tests for quality assessment components."""
    
    def setup_method(self):
        """Set up test fixtures for integration testing."""
        self.stats_analyzer = StatisticalAnalyzer()
        self.quality_assessor = QualityAssessor(self.stats_analyzer)
        self.establishment_analyzer = EstablishmentAnalyzer()
        
        # Larger sample dataset for integration testing
        np.random.seed(42)
        n_observations = 50
        
        occupations = ['Chief Executives', 'Registered Nurses', 'Software Developers', 
                      'Teachers', 'Sales Representatives']
        requirements = ['Physical demands', 'Cognitive requirements', 'Environmental conditions',
                       'Education requirements', 'Social skills']
        soc_codes = ['111011', '291141', '151252', '252031', '419012']
        
        # Generate realistic data with some quality issues
        data_rows = []
        for i in range(n_observations):
            occ_idx = i % len(occupations)
            req_idx = i % len(requirements)
            
            # Introduce some missing data (10% chance)
            estimate = np.random.normal(50, 20) if np.random.random() > 0.1 else np.nan
            std_error = max(0.1, abs(np.random.normal(3, 1))) if not pd.isna(estimate) else np.nan
            
            # Introduce some footnotes
            footnote = np.random.choice([1, 7, 16, 26, None], p=[0.1, 0.1, 0.1, 0.1, 0.6])
            
            data_rows.append({
                'SOC 2018 CODE': soc_codes[occ_idx],
                'OCCUPATION': occupations[occ_idx],
                'REQUIREMENT': requirements[req_idx],
                'ESTIMATE': round(estimate, 1) if not pd.isna(estimate) else np.nan,
                'STANDARD ERROR': round(std_error, 2) if not pd.isna(std_error) else np.nan,
                'DATA FOOTNOTE': footnote
            })
        
        self.integration_data = pd.DataFrame(data_rows)
    
    def test_complete_quality_assessment_pipeline(self):
        """Test complete quality assessment pipeline."""
        # Step 1: Comprehensive quality assessment
        quality_assessment = self.quality_assessor.assess_data_quality(self.integration_data)
        
        # Should complete without errors
        assert 'overall_metrics' in quality_assessment
        assert quality_assessment['total_records_assessed'] == 50
        
        # Step 2: Generate recommendations
        recommendations = self.quality_assessor.generate_quality_recommendations(quality_assessment)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Step 3: Establishment analysis
        coverage_report = self.establishment_analyzer.generate_coverage_report(self.integration_data)
        assert 'executive_summary' in coverage_report
        assert 'policy_implications' in coverage_report
    
    def test_quality_metrics_consistency(self):
        """Test consistency between different quality metrics."""
        quality_assessment = self.quality_assessor.assess_data_quality(self.integration_data)
        
        # Reliability and precision should be related
        reliability_summary = quality_assessment['reliability_scores']['summary_statistics']
        precision_summary = quality_assessment['precision_analysis']['precision_summary']
        
        mean_reliability = reliability_summary.get('mean_reliability', 0)
        high_precision_pct = precision_summary.get('overall_high_precision_percentage', 0)
        
        # If precision is high, reliability should generally be high too
        if high_precision_pct > 70:
            assert mean_reliability > 0.5
    
    def test_establishment_coverage_consistency(self):
        """Test consistency in establishment coverage calculations."""
        coverage_stats = self.establishment_analyzer.calculate_workforce_coverage_statistics(
            self.integration_data
        )
        
        # Coverage percentage should be reasonable
        assert 0 <= coverage_stats.coverage_percentage <= 100
        
        # Represented workers should not exceed total civilian workers
        assert coverage_stats.represented_workers <= coverage_stats.total_civilian_workers
        
        # Occupation and requirement coverage should have entries
        assert len(coverage_stats.occupation_coverage) > 0
        assert len(coverage_stats.requirement_coverage) > 0
    
    def test_policy_implications_completeness(self):
        """Test completeness of policy implications assessment."""
        coverage_stats = self.establishment_analyzer.calculate_workforce_coverage_statistics(
            self.integration_data
        )
        
        policy_implications = self.establishment_analyzer.assess_policy_implications(
            self.integration_data, coverage_stats
        )
        
        # Should have multiple policy areas covered
        policy_areas = [impl.policy_area for impl in policy_implications]
        assert len(set(policy_areas)) >= 3  # At least 3 different policy areas
        
        # Each implication should have substantive content
        for implication in policy_implications:
            assert len(implication.key_findings) > 0
            assert len(implication.recommendations) > 0
            assert implication.affected_workers > 0


class TestQualityAssessmentEdgeCases:
    """Test edge cases and error handling for quality assessment."""
    
    def test_empty_data_quality_assessment(self):
        """Test quality assessment with empty data."""
        empty_df = pd.DataFrame(columns=['SOC 2018 CODE', 'OCCUPATION', 'REQUIREMENT', 
                                        'ESTIMATE', 'STANDARD ERROR'])
        
        assessor = QualityAssessor()
        quality_assessment = assessor.assess_data_quality(empty_df)
        
        # Should handle empty data gracefully
        assert quality_assessment['total_records_assessed'] == 0
        assert quality_assessment['validation_result'].is_valid == True  # No errors, just empty
    
    def test_missing_critical_columns(self):
        """Test quality assessment with missing critical columns."""
        incomplete_data = pd.DataFrame({
            'OCCUPATION': ['Chief Executive'],
            'ESTIMATE': [75.5]
            # Missing SOC 2018 CODE, REQUIREMENT, STANDARD ERROR
        })
        
        assessor = QualityAssessor()
        quality_assessment = assessor.assess_data_quality(incomplete_data)
        
        # Should identify missing critical columns
        validation = quality_assessment['validation_result']
        assert validation.is_valid == False
        assert len(validation.errors) > 0
    
    def test_all_missing_estimates(self):
        """Test quality assessment when all estimates are missing."""
        data_no_estimates = pd.DataFrame({
            'SOC 2018 CODE': ['111011', '291141'],
            'OCCUPATION': ['Chief Executive', 'Nurse'],
            'REQUIREMENT': ['Physical demands', 'Cognitive requirements'],
            'ESTIMATE': [np.nan, np.nan],
            'STANDARD ERROR': [np.nan, np.nan]
        })
        
        assessor = QualityAssessor()
        reliability_results = assessor.calculate_reliability_scores(data_no_estimates)
        
        # Should handle all missing estimates
        summary = reliability_results['summary_statistics']
        assert summary['total_assessed'] == 0  # No valid estimates to assess
    
    def test_extreme_footnote_codes(self):
        """Test handling of unknown or extreme footnote codes."""
        data_extreme_footnotes = pd.DataFrame({
            'SOC 2018 CODE': ['111011', '291141'],
            'OCCUPATION': ['Chief Executive', 'Nurse'],
            'REQUIREMENT': ['Physical demands', 'Cognitive requirements'],
            'ESTIMATE': [75.5, 45.2],
            'STANDARD ERROR': [2.1, 3.2],
            'DATA FOOTNOTE': [999, -1]  # Unknown footnote codes
        })
        
        assessor = QualityAssessor()
        precision_analysis = assessor.categorize_precision_levels(data_extreme_footnotes)
        
        # Should handle unknown footnote codes gracefully
        footnote_precision = precision_analysis['footnote_precision']
        assert 'unknown_precision_count' in footnote_precision
        assert footnote_precision['unknown_precision_count'] == 2
    
    def test_establishment_analysis_minimal_data(self):
        """Test establishment analysis with minimal data."""
        minimal_data = pd.DataFrame({
            'SOC 2018 CODE': ['111011'],
            'OCCUPATION': ['Chief Executive'],
            'REQUIREMENT': ['Physical demands'],
            'ESTIMATE': [75.5],
            'STANDARD ERROR': [2.1]
        })
        
        analyzer = EstablishmentAnalyzer()
        coverage_report = analyzer.generate_coverage_report(minimal_data)
        
        # Should handle minimal data
        assert coverage_report['report_metadata']['data_records_analyzed'] == 1
        assert len(coverage_report['policy_implications']) > 0  # Should still generate implications
    
    def test_zero_standard_errors(self):
        """Test handling of zero standard errors."""
        data_zero_se = pd.DataFrame({
            'SOC 2018 CODE': ['111011', '291141'],
            'OCCUPATION': ['Chief Executive', 'Nurse'],
            'REQUIREMENT': ['Physical demands', 'Cognitive requirements'],
            'ESTIMATE': [75.5, 45.2],
            'STANDARD ERROR': [0.0, 0.0]  # Zero standard errors
        })
        
        assessor = QualityAssessor()
        reliability_results = assessor.calculate_reliability_scores(data_zero_se)
        
        # Should handle zero standard errors
        individual_scores = reliability_results['individual_scores']
        assert len(individual_scores) == 2
        # Zero standard error should result in high reliability (perfect precision)
        assert all(score > 0.9 for score in individual_scores)