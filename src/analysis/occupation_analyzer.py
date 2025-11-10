"""
Occupation distribution analysis functionality.
Implements frequency distributions, diversity metrics, and top-N identification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import math

from ..interfaces import AnalysisResult
from .statistical_analyzer import StatisticalAnalyzer


class OccupationAnalyzer:
    """
    Analyzer for occupation distribution patterns and characteristics.
    Handles frequency distributions, diversity metrics, and top-N analysis.
    """
    
    def __init__(self, statistical_analyzer: Optional[StatisticalAnalyzer] = None):
        """
        Initialize the occupation analyzer.
        
        Args:
            statistical_analyzer: Optional StatisticalAnalyzer instance
        """
        self.stats_analyzer = statistical_analyzer or StatisticalAnalyzer()
    
    def calculate_frequency_distribution(
        self, 
        data: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation',
        estimate_column: str = 'estimate',
        std_error_column: str = 'standard_error'
    ) -> pd.DataFrame:
        """
        Calculate frequency distribution of SOC codes with statistical measures.
        
        Args:
            data: DataFrame containing occupation data
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            estimate_column: Name of the estimate column
            std_error_column: Name of the standard error column
            
        Returns:
            DataFrame with frequency distribution and statistics
        """
        # Group by SOC code and occupation
        grouped = data.groupby([soc_code_column, occupation_column]).agg({
            estimate_column: ['count', 'mean', 'std', 'min', 'max'],
            std_error_column: ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        grouped.columns = [f"{col[1]}_{col[0]}" if col[1] != '' else col[0] 
                          for col in grouped.columns]
        
        # Reset index to get SOC code and occupation as columns
        freq_dist = grouped.reset_index()
        
        # Calculate additional metrics
        freq_dist['frequency'] = freq_dist['count_ESTIMATE']
        freq_dist['relative_frequency'] = (
            freq_dist['frequency'] / freq_dist['frequency'].sum()
        )
        freq_dist['cumulative_frequency'] = freq_dist['frequency'].cumsum()
        freq_dist['cumulative_relative_frequency'] = (
            freq_dist['cumulative_frequency'] / freq_dist['frequency'].sum()
        )
        
        # Calculate confidence intervals for mean estimates
        ci_data = self.stats_analyzer.calculate_confidence_intervals(
            freq_dist['mean_ESTIMATE'],
            freq_dist['mean_STANDARD ERROR']
        )
        
        freq_dist['estimate_lower_ci'] = ci_data['lower_ci']
        freq_dist['estimate_upper_ci'] = ci_data['upper_ci']
        freq_dist['estimate_margin_error'] = ci_data['margin_error']
        
        # Calculate reliability scores
        freq_dist['reliability_score'] = self.stats_analyzer.calculate_reliability_scores(
            freq_dist['mean_ESTIMATE'],
            freq_dist['mean_STANDARD ERROR']
        )
        
        # Sort by frequency (descending)
        freq_dist = freq_dist.sort_values('frequency', ascending=False).reset_index(drop=True)
        
        return freq_dist
    
    def calculate_diversity_metrics(
        self, 
        data: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        estimate_column: str = 'estimate'
    ) -> Dict[str, float]:
        """
        Calculate diversity metrics for occupation distribution.
        
        Args:
            data: DataFrame containing occupation data
            soc_code_column: Name of the SOC code column
            estimate_column: Name of the estimate column
            
        Returns:
            Dictionary containing diversity metrics
        """
        # Get occupation frequencies
        occupation_counts = data[soc_code_column].value_counts()
        total_observations = len(data)
        
        # Calculate proportions
        proportions = occupation_counts / total_observations
        
        # Shannon Diversity Index (entropy)
        shannon_diversity = -sum(p * math.log(p) for p in proportions if p > 0)
        
        # Simpson Diversity Index
        simpson_diversity = 1 - sum(p**2 for p in proportions)
        
        # Effective number of species (exponential of Shannon)
        effective_occupations = math.exp(shannon_diversity)
        
        # Evenness (Shannon diversity / log of number of occupations)
        max_diversity = math.log(len(occupation_counts))
        evenness = shannon_diversity / max_diversity if max_diversity > 0 else 0
        
        # Gini coefficient for inequality
        sorted_counts = sorted(occupation_counts.values)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (2 * sum((i + 1) * count for i, count in enumerate(sorted_counts))) / (n * sum(sorted_counts)) - (n + 1) / n
        
        # Concentration ratio (top 10% of occupations)
        top_10_pct_count = max(1, int(0.1 * len(occupation_counts)))
        concentration_ratio = occupation_counts.head(top_10_pct_count).sum() / total_observations
        
        return {
            'total_occupations': len(occupation_counts),
            'total_observations': total_observations,
            'shannon_diversity': shannon_diversity,
            'simpson_diversity': simpson_diversity,
            'effective_occupations': effective_occupations,
            'evenness': evenness,
            'gini_coefficient': gini,
            'concentration_ratio_top10pct': concentration_ratio,
            'most_common_occupation_pct': proportions.iloc[0] * 100,
            'least_common_occupation_pct': proportions.iloc[-1] * 100
        }
    
    def identify_top_n_occupations(
        self, 
        data: pd.DataFrame,
        n: int = 20,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation',
        estimate_column: str = 'estimate',
        std_error_column: str = 'standard_error',
        sort_by: str = 'frequency'
    ) -> List[AnalysisResult]:
        """
        Identify top N occupations with confidence intervals.
        
        Args:
            data: DataFrame containing occupation data
            n: Number of top occupations to return
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            estimate_column: Name of the estimate column
            std_error_column: Name of the standard error column
            sort_by: Criteria to sort by ('frequency', 'mean_estimate', 'reliability')
            
        Returns:
            List of AnalysisResult objects for top N occupations
        """
        # Calculate frequency distribution
        freq_dist = self.calculate_frequency_distribution(
            data, soc_code_column, occupation_column, estimate_column, std_error_column
        )
        
        # Sort by specified criteria
        if sort_by == 'frequency':
            sorted_dist = freq_dist.sort_values('frequency', ascending=False)
        elif sort_by == 'mean_estimate':
            sorted_dist = freq_dist.sort_values('mean_ESTIMATE', ascending=False)
        elif sort_by == 'reliability':
            sorted_dist = freq_dist.sort_values('reliability_score', ascending=False)
        else:
            raise ValueError(f"Unknown sort criteria: {sort_by}")
        
        # Get top N
        top_n = sorted_dist.head(n)
        
        # Convert to AnalysisResult objects
        results = []
        for _, row in top_n.iterrows():
            result = AnalysisResult(
                occupation_category=f"{row[soc_code_column]} - {row[occupation_column]}",
                metric_name=f"top_{sort_by}",
                value=row['mean_ESTIMATE'] if sort_by != 'frequency' else row['frequency'],
                confidence_interval=(row['estimate_lower_ci'], row['estimate_upper_ci']),
                reliability_score=row['reliability_score'],
                footnote_context=[
                    f"Frequency: {row['frequency']} observations",
                    f"Relative frequency: {row['relative_frequency']:.1%}",
                    f"Mean estimate: {row['mean_ESTIMATE']:.2f}",
                    f"Standard error: {row['mean_STANDARD ERROR']:.4f}"
                ]
            )
            results.append(result)
        
        return results
    
    def analyze_occupation_categories(
        self, 
        data: pd.DataFrame,
        category_column: str = 'requirement_type',
        soc_code_column: str = 'soc_code',
        estimate_column: str = 'estimate',
        std_error_column: str = 'standard_error'
    ) -> pd.DataFrame:
        """
        Analyze occupation distribution by requirement categories.
        
        Args:
            data: DataFrame containing occupation data
            category_column: Name of the category column
            soc_code_column: Name of the SOC code column
            estimate_column: Name of the estimate column
            std_error_column: Name of the standard error column
            
        Returns:
            DataFrame with category-wise analysis
        """
        # Group by category and SOC code
        category_analysis = data.groupby([category_column, soc_code_column]).agg({
            estimate_column: ['count', 'mean', 'std'],
            std_error_column: ['mean']
        }).round(4)
        
        # Flatten column names
        category_analysis.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] 
                                   for col in category_analysis.columns]
        
        # Reset index
        category_analysis = category_analysis.reset_index()
        
        # Calculate statistics by category
        category_stats = []
        for category in data[category_column].unique():
            category_data = data[data[category_column] == category]
            
            # Basic statistics
            stats = {
                'category': category,
                'unique_occupations': category_data[soc_code_column].nunique(),
                'total_observations': len(category_data),
                'mean_estimate': category_data[estimate_column].mean(),
                'median_estimate': category_data[estimate_column].median(),
                'std_estimate': category_data[estimate_column].std(),
                'mean_std_error': category_data[std_error_column].mean()
            }
            
            # Diversity metrics for this category
            diversity_metrics = self.calculate_diversity_metrics(
                category_data, soc_code_column, estimate_column
            )
            stats.update({f"category_{k}": v for k, v in diversity_metrics.items()})
            
            # Confidence intervals
            ci_data = self.stats_analyzer.calculate_confidence_intervals(
                category_data[estimate_column],
                category_data[std_error_column]
            )
            stats['mean_lower_ci'] = ci_data['lower_ci'].mean()
            stats['mean_upper_ci'] = ci_data['upper_ci'].mean()
            
            # Reliability scores
            reliability_scores = self.stats_analyzer.calculate_reliability_scores(
                category_data[estimate_column],
                category_data[std_error_column]
            )
            stats['mean_reliability'] = reliability_scores.mean()
            
            category_stats.append(stats)
        
        return pd.DataFrame(category_stats)
    
    def calculate_occupation_similarity(
        self, 
        data: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        requirement_column: str = 'requirement_type',
        estimate_column: str = 'estimate',
        similarity_threshold: float = 0.8
    ) -> pd.DataFrame:
        """
        Calculate similarity between occupations based on requirement profiles.
        
        Args:
            data: DataFrame containing occupation data
            soc_code_column: Name of the SOC code column
            requirement_column: Name of the requirement column
            estimate_column: Name of the estimate column
            similarity_threshold: Minimum similarity score to include
            
        Returns:
            DataFrame with occupation similarity pairs
        """
        # Create occupation-requirement matrix
        pivot_data = data.pivot_table(
            index=soc_code_column,
            columns=requirement_column,
            values=estimate_column,
            aggfunc='mean',
            fill_value=0
        )
        
        # Calculate correlation matrix (similarity)
        correlation_matrix = pivot_data.T.corr()
        
        # Extract similarity pairs above threshold
        similarity_pairs = []
        occupations = correlation_matrix.index.tolist()
        
        for i, occ1 in enumerate(occupations):
            for j, occ2 in enumerate(occupations[i+1:], i+1):
                similarity = correlation_matrix.loc[occ1, occ2]
                if similarity >= similarity_threshold:
                    similarity_pairs.append({
                        'occupation_1': occ1,
                        'occupation_2': occ2,
                        'similarity_score': similarity,
                        'similarity_category': self._categorize_similarity(similarity)
                    })
        
        # Sort by similarity score
        similarity_df = pd.DataFrame(similarity_pairs)
        if not similarity_df.empty:
            similarity_df = similarity_df.sort_values('similarity_score', ascending=False)
        
        return similarity_df
    
    def _categorize_similarity(self, similarity_score: float) -> str:
        """Categorize similarity scores into descriptive labels."""
        if similarity_score >= 0.9:
            return "Very High"
        elif similarity_score >= 0.8:
            return "High"
        elif similarity_score >= 0.6:
            return "Moderate"
        elif similarity_score >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def generate_occupation_summary(
        self, 
        data: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation',
        estimate_column: str = 'estimate',
        std_error_column: str = 'standard_error'
    ) -> Dict[str, Any]:
        """
        Generate comprehensive summary of occupation distribution.
        
        Args:
            data: DataFrame containing occupation data
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            estimate_column: Name of the estimate column
            std_error_column: Name of the standard error column
            
        Returns:
            Dictionary containing comprehensive occupation summary
        """
        # Basic statistics
        total_observations = len(data)
        unique_occupations = data[soc_code_column].nunique()
        
        # Frequency distribution
        freq_dist = self.calculate_frequency_distribution(
            data, soc_code_column, occupation_column, estimate_column, std_error_column
        )
        
        # Diversity metrics
        diversity_metrics = self.calculate_diversity_metrics(
            data, soc_code_column, estimate_column
        )
        
        # Top occupations
        top_10_by_frequency = self.identify_top_n_occupations(
            data, n=10, sort_by='frequency'
        )
        
        top_10_by_estimate = self.identify_top_n_occupations(
            data, n=10, sort_by='mean_estimate'
        )
        
        # Statistical summary
        estimate_stats = data[estimate_column].describe()
        std_error_stats = data[std_error_column].describe()
        
        # Population weighting
        weighted_data = self.stats_analyzer.apply_population_weighting(
            data, estimate_column, soc_code_column
        )
        
        return {
            'overview': {
                'total_observations': total_observations,
                'unique_occupations': unique_occupations,
                'observations_per_occupation': total_observations / unique_occupations,
                'civilian_workers_represented': StatisticalAnalyzer.TOTAL_CIVILIAN_WORKERS
            },
            'diversity_metrics': diversity_metrics,
            'estimate_statistics': {
                'mean': estimate_stats['mean'],
                'median': estimate_stats['50%'],
                'std': estimate_stats['std'],
                'min': estimate_stats['min'],
                'max': estimate_stats['max'],
                'q25': estimate_stats['25%'],
                'q75': estimate_stats['75%']
            },
            'standard_error_statistics': {
                'mean': std_error_stats['mean'],
                'median': std_error_stats['50%'],
                'std': std_error_stats['std'],
                'min': std_error_stats['min'],
                'max': std_error_stats['max']
            },
            'top_occupations_by_frequency': [
                {
                    'occupation': result.occupation_category,
                    'value': result.value,
                    'reliability': result.reliability_score
                } for result in top_10_by_frequency
            ],
            'top_occupations_by_estimate': [
                {
                    'occupation': result.occupation_category,
                    'value': result.value,
                    'confidence_interval': result.confidence_interval,
                    'reliability': result.reliability_score
                } for result in top_10_by_estimate
            ],
            'population_weighting': {
                'total_weighted_estimate': weighted_data['weighted_estimate'].sum(),
                'estimated_total_workers': weighted_data['estimated_workers'].sum(),
                'mean_occupation_weight': weighted_data['occupation_weight'].mean()
            }
        }