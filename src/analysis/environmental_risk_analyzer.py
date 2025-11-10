"""
Environmental risk analysis functionality for occupation data reports.
Implements risk scoring system for environmental conditions and occupation ranking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings

from ..interfaces import AnalysisResult
from .statistical_analyzer import StatisticalAnalyzer


class EnvironmentalRiskAnalyzer:
    """
    Analyzer for environmental risk conditions across occupations.
    Handles risk scoring, weighted calculations, and occupation ranking.
    """
    
    # Environmental condition categories and their risk weights
    ENVIRONMENTAL_CONDITIONS = {
        'extreme_cold': {
            'keywords': ['extreme cold', 'cold temperature', 'freezing'],
            'base_weight': 0.8,
            'description': 'Exposure to extreme cold temperatures'
        },
        'extreme_heat': {
            'keywords': ['extreme heat', 'hot temperature', 'high temperature'],
            'base_weight': 0.8,
            'description': 'Exposure to extreme heat temperatures'
        },
        'hazardous_contaminants': {
            'keywords': ['hazardous contaminants', 'toxic substances', 'chemical exposure'],
            'base_weight': 1.0,
            'description': 'Exposure to hazardous chemical contaminants'
        },
        'heavy_vibrations': {
            'keywords': ['heavy vibrations', 'vibration exposure', 'mechanical vibration'],
            'base_weight': 0.6,
            'description': 'Exposure to heavy mechanical vibrations'
        },
        'heights': {
            'keywords': ['heights', 'elevated work', 'high elevation'],
            'base_weight': 0.9,
            'description': 'Working at dangerous heights'
        },
        'loud_noise': {
            'keywords': ['loud noise', 'noise exposure', 'high decibel'],
            'base_weight': 0.7,
            'description': 'Exposure to loud noise levels'
        },
        'radiation': {
            'keywords': ['radiation', 'radioactive', 'ionizing radiation'],
            'base_weight': 1.0,
            'description': 'Exposure to radiation sources'
        },
        'confined_spaces': {
            'keywords': ['confined spaces', 'enclosed areas', 'cramped conditions'],
            'base_weight': 0.7,
            'description': 'Working in confined or enclosed spaces'
        }
    }
    
    def __init__(self, statistical_analyzer: Optional[StatisticalAnalyzer] = None):
        """
        Initialize the environmental risk analyzer.
        
        Args:
            statistical_analyzer: Optional StatisticalAnalyzer instance
        """
        self.stats_analyzer = statistical_analyzer or StatisticalAnalyzer()
    
    def identify_environmental_conditions(
        self, 
        data: pd.DataFrame,
        requirement_column: str = 'requirement_type',
        category_column: str = 'category'
    ) -> pd.DataFrame:
        """
        Identify environmental condition records in the dataset.
        
        Args:
            data: DataFrame containing occupation data
            requirement_column: Name of the requirement type column
            category_column: Name of the category column
            
        Returns:
            DataFrame filtered to environmental condition records
        """
        # Filter for environmental conditions
        environmental_mask = data[requirement_column].str.contains(
            'environmental|temperature|hazardous|vibration|height|noise|radiation|confined',
            case=False, na=False
        )
        
        environmental_data = data[environmental_mask].copy()
        
        # Categorize environmental conditions
        environmental_data['risk_category'] = environmental_data.apply(
            lambda row: self._categorize_environmental_condition(
                row[requirement_column], row[category_column]
            ), axis=1
        )
        
        # Filter out uncategorized conditions
        environmental_data = environmental_data[
            environmental_data['risk_category'] != 'other'
        ].copy()
        
        return environmental_data
    
    def _categorize_environmental_condition(
        self, 
        requirement_type: str, 
        category: str
    ) -> str:
        """
        Categorize environmental condition based on requirement type and category.
        
        Args:
            requirement_type: The requirement type text
            category: The category text
            
        Returns:
            Environmental risk category name
        """
        combined_text = f"{requirement_type} {category}".lower()
        
        for condition_name, condition_info in self.ENVIRONMENTAL_CONDITIONS.items():
            for keyword in condition_info['keywords']:
                if keyword.lower() in combined_text:
                    return condition_name
        
        return 'other'
    
    def calculate_risk_scores(
        self, 
        environmental_data: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation',
        estimate_column: str = 'estimate',
        std_error_column: str = 'standard_error'
    ) -> pd.DataFrame:
        """
        Calculate weighted risk scores for environmental conditions.
        
        Args:
            environmental_data: DataFrame with environmental condition data
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            estimate_column: Name of the estimate column
            std_error_column: Name of the standard error column
            
        Returns:
            DataFrame with risk scores by occupation
        """
        if environmental_data.empty:
            return pd.DataFrame()
        
        # Calculate confidence intervals for estimates
        ci_data = self.stats_analyzer.calculate_confidence_intervals(
            environmental_data[estimate_column],
            environmental_data[std_error_column]
        )
        
        environmental_data = environmental_data.copy()
        environmental_data['lower_ci'] = ci_data['lower_ci']
        environmental_data['upper_ci'] = ci_data['upper_ci']
        environmental_data['margin_error'] = ci_data['margin_error']
        
        # Calculate reliability scores
        environmental_data['reliability_score'] = self.stats_analyzer.calculate_reliability_scores(
            environmental_data[estimate_column],
            environmental_data[std_error_column]
        )
        
        # Calculate weighted risk scores
        environmental_data['base_weight'] = environmental_data['risk_category'].map(
            lambda x: self.ENVIRONMENTAL_CONDITIONS.get(x, {}).get('base_weight', 0.5)
        )
        
        # Risk score = estimate * base_weight * reliability_score
        environmental_data['risk_score'] = (
            environmental_data[estimate_column] * 
            environmental_data['base_weight'] * 
            environmental_data['reliability_score']
        )
        
        # Adjust risk score based on confidence interval width
        # Wider intervals get lower risk scores due to uncertainty
        ci_width = environmental_data['upper_ci'] - environmental_data['lower_ci']
        ci_adjustment = 1 / (1 + ci_width / environmental_data[estimate_column].abs())
        environmental_data['adjusted_risk_score'] = (
            environmental_data['risk_score'] * ci_adjustment
        )
        
        return environmental_data
    
    def rank_occupations_by_risk(
        self, 
        risk_data: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation',
        top_n: Optional[int] = None
    ) -> List[AnalysisResult]:
        """
        Rank occupations by overall environmental risk exposure.
        
        Args:
            risk_data: DataFrame with calculated risk scores
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            top_n: Number of top risky occupations to return (None for all)
            
        Returns:
            List of AnalysisResult objects ranked by risk
        """
        if risk_data.empty:
            return []
        
        # Aggregate risk scores by occupation
        occupation_risk = risk_data.groupby([soc_code_column, occupation_column]).agg({
            'adjusted_risk_score': ['sum', 'mean', 'count'],
            'reliability_score': 'mean',
            'risk_category': lambda x: list(x.unique()),
            'estimate': 'mean',
            'lower_ci': 'mean',
            'upper_ci': 'mean'
        }).round(4)
        
        # Flatten column names
        occupation_risk.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] 
                                 for col in occupation_risk.columns]
        
        # Reset index
        occupation_risk = occupation_risk.reset_index()
        
        # Calculate overall risk score (sum of individual risk scores)
        occupation_risk['overall_risk_score'] = occupation_risk['sum_adjusted_risk_score']
        
        # Sort by overall risk score
        occupation_risk = occupation_risk.sort_values(
            'overall_risk_score', ascending=False
        ).reset_index(drop=True)
        
        # Limit to top N if specified
        if top_n is not None:
            occupation_risk = occupation_risk.head(top_n)
        
        # Convert to AnalysisResult objects
        results = []
        for _, row in occupation_risk.iterrows():
            risk_categories = row['<lambda>_risk_category']
            
            result = AnalysisResult(
                occupation_category=f"{row[soc_code_column]} - {row[occupation_column]}",
                metric_name="environmental_risk_score",
                value=row['overall_risk_score'],
                confidence_interval=(row['mean_lower_ci'], row['mean_upper_ci']),
                reliability_score=row['mean_reliability_score'],
                footnote_context=[
                    f"Risk conditions: {', '.join(risk_categories)}",
                    f"Number of risk factors: {row['count_adjusted_risk_score']}",
                    f"Average risk per factor: {row['mean_adjusted_risk_score']:.2f}",
                    f"Mean estimate: {row['mean_estimate']:.2f}%",
                    f"Overall reliability: {row['mean_reliability_score']:.3f}"
                ]
            )
            results.append(result)
        
        return results
    
    def analyze_risk_by_category(
        self, 
        risk_data: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation'
    ) -> Dict[str, Any]:
        """
        Analyze environmental risk patterns by risk category.
        
        Args:
            risk_data: DataFrame with calculated risk scores
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            
        Returns:
            Dictionary with risk analysis by category
        """
        if risk_data.empty:
            return {}
        
        category_analysis = {}
        
        for category in risk_data['risk_category'].unique():
            category_data = risk_data[risk_data['risk_category'] == category]
            
            # Basic statistics
            stats = {
                'total_occupations': category_data[soc_code_column].nunique(),
                'total_observations': len(category_data),
                'mean_estimate': category_data['estimate'].mean(),
                'median_estimate': category_data['estimate'].median(),
                'std_estimate': category_data['estimate'].std(),
                'mean_risk_score': category_data['adjusted_risk_score'].mean(),
                'max_risk_score': category_data['adjusted_risk_score'].max(),
                'mean_reliability': category_data['reliability_score'].mean()
            }
            
            # Top occupations for this category
            top_occupations = category_data.nlargest(5, 'adjusted_risk_score')
            stats['top_occupations'] = [
                {
                    'soc_code': row[soc_code_column],
                    'occupation': row[occupation_column],
                    'risk_score': row['adjusted_risk_score'],
                    'estimate': row['estimate'],
                    'reliability': row['reliability_score']
                }
                for _, row in top_occupations.iterrows()
            ]
            
            # Risk distribution
            risk_scores = category_data['adjusted_risk_score']
            stats['risk_distribution'] = {
                'q25': risk_scores.quantile(0.25),
                'q50': risk_scores.quantile(0.50),
                'q75': risk_scores.quantile(0.75),
                'q90': risk_scores.quantile(0.90),
                'q95': risk_scores.quantile(0.95)
            }
            
            category_analysis[category] = stats
        
        return category_analysis
    
    def create_risk_matrix(
        self, 
        risk_data: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation'
    ) -> pd.DataFrame:
        """
        Create a risk matrix showing occupations vs environmental conditions.
        
        Args:
            risk_data: DataFrame with calculated risk scores
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            
        Returns:
            DataFrame with occupations as rows and risk categories as columns
        """
        if risk_data.empty:
            return pd.DataFrame()
        
        # Create occupation identifier
        risk_data = risk_data.copy()
        risk_data['occupation_id'] = (
            risk_data[soc_code_column] + " - " + risk_data[occupation_column]
        )
        
        # Pivot to create risk matrix
        risk_matrix = risk_data.pivot_table(
            index='occupation_id',
            columns='risk_category',
            values='adjusted_risk_score',
            aggfunc='sum',
            fill_value=0
        )
        
        # Add total risk score column
        risk_matrix['total_risk'] = risk_matrix.sum(axis=1)
        
        # Sort by total risk
        risk_matrix = risk_matrix.sort_values('total_risk', ascending=False)
        
        return risk_matrix
    
    def generate_risk_summary(
        self, 
        data: pd.DataFrame,
        requirement_column: str = 'requirement_type',
        category_column: str = 'category',
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation',
        estimate_column: str = 'estimate',
        std_error_column: str = 'standard_error'
    ) -> Dict[str, Any]:
        """
        Generate comprehensive environmental risk summary.
        
        Args:
            data: DataFrame containing occupation data
            requirement_column: Name of the requirement type column
            category_column: Name of the category column
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            estimate_column: Name of the estimate column
            std_error_column: Name of the standard error column
            
        Returns:
            Dictionary containing comprehensive risk analysis
        """
        # Identify environmental conditions
        environmental_data = self.identify_environmental_conditions(
            data, requirement_column, category_column
        )
        
        if environmental_data.empty:
            return {
                'overview': {
                    'total_environmental_records': 0,
                    'environmental_occupations': 0,
                    'risk_categories_found': 0
                },
                'message': 'No environmental condition data found in dataset'
            }
        
        # Calculate risk scores
        risk_data = self.calculate_risk_scores(
            environmental_data, soc_code_column, occupation_column,
            estimate_column, std_error_column
        )
        
        # Rank occupations
        top_risk_occupations = self.rank_occupations_by_risk(
            risk_data, soc_code_column, occupation_column, top_n=20
        )
        
        # Analyze by category
        category_analysis = self.analyze_risk_by_category(
            risk_data, soc_code_column, occupation_column
        )
        
        # Create risk matrix
        risk_matrix = self.create_risk_matrix(
            risk_data, soc_code_column, occupation_column
        )
        
        # Overall statistics
        overview = {
            'total_environmental_records': len(environmental_data),
            'environmental_occupations': environmental_data[soc_code_column].nunique(),
            'risk_categories_found': len(environmental_data['risk_category'].unique()),
            'mean_risk_score': risk_data['adjusted_risk_score'].mean(),
            'max_risk_score': risk_data['adjusted_risk_score'].max(),
            'mean_reliability': risk_data['reliability_score'].mean(),
            'high_risk_occupations': len(risk_data[risk_data['adjusted_risk_score'] > 
                                                  risk_data['adjusted_risk_score'].quantile(0.9)])
        }
        
        return {
            'overview': overview,
            'top_risk_occupations': [
                {
                    'occupation': result.occupation_category,
                    'risk_score': result.value,
                    'confidence_interval': result.confidence_interval,
                    'reliability': result.reliability_score,
                    'risk_factors': result.footnote_context[0].replace('Risk conditions: ', '')
                }
                for result in top_risk_occupations
            ],
            'category_analysis': category_analysis,
            'risk_matrix_shape': {
                'occupations': len(risk_matrix),
                'risk_categories': len(risk_matrix.columns) - 1,  # Exclude total_risk column
                'highest_risk_occupation': risk_matrix.index[0] if not risk_matrix.empty else None,
                'highest_total_risk': risk_matrix['total_risk'].iloc[0] if not risk_matrix.empty else 0
            },
            'risk_categories': {
                name: info['description'] 
                for name, info in self.ENVIRONMENTAL_CONDITIONS.items()
            }
        }