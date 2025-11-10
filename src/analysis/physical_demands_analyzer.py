"""
Physical demands analysis functionality for occupation data reports.
Implements physical demand scoring, intensity matrices, and ergonomic assessments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings

from ..interfaces import AnalysisResult
from .statistical_analyzer import StatisticalAnalyzer


class PhysicalDemandsAnalyzer:
    """
    Analyzer for physical demands across occupations.
    Handles physical demand scoring, intensity matrices, and ergonomic assessments.
    """
    
    # Physical demand categories and their characteristics
    PHYSICAL_DEMANDS = {
        'lifting': {
            'keywords': ['lifting', 'lift', 'raising objects'],
            'weight_thresholds': {
                'light': (0, 20),      # 0-20 lbs
                'medium': (20, 50),    # 20-50 lbs
                'heavy': (50, 100),    # 50-100 lbs
                'very_heavy': (100, float('inf'))  # 100+ lbs
            },
            'risk_multiplier': 1.0,
            'description': 'Lifting and raising objects'
        },
        'carrying': {
            'keywords': ['carrying', 'carry', 'transporting objects'],
            'weight_thresholds': {
                'light': (0, 20),
                'medium': (20, 50),
                'heavy': (50, 100),
                'very_heavy': (100, float('inf'))
            },
            'risk_multiplier': 0.9,
            'description': 'Carrying and transporting objects'
        },
        'climbing': {
            'keywords': ['climbing', 'climb', 'ascending', 'ladder'],
            'frequency_thresholds': {
                'rarely': (0, 10),      # 0-10% of time
                'occasionally': (10, 33), # 10-33% of time
                'frequently': (33, 66),   # 33-66% of time
                'constantly': (66, 100)   # 66-100% of time
            },
            'risk_multiplier': 1.2,
            'description': 'Climbing ladders, stairs, or structures'
        },
        'standing': {
            'keywords': ['standing', 'stand', 'upright position'],
            'time_thresholds': {
                'minimal': (0, 25),      # 0-25% of time
                'moderate': (25, 50),    # 25-50% of time
                'frequent': (50, 75),    # 50-75% of time
                'constant': (75, 100)    # 75-100% of time
            },
            'risk_multiplier': 0.6,
            'description': 'Standing for extended periods'
        },
        'sitting': {
            'keywords': ['sitting', 'sit', 'seated position'],
            'time_thresholds': {
                'minimal': (0, 25),
                'moderate': (25, 50),
                'frequent': (50, 75),
                'constant': (75, 100)
            },
            'risk_multiplier': 0.3,
            'description': 'Sitting for extended periods'
        },
        'walking': {
            'keywords': ['walking', 'walk', 'moving on foot'],
            'time_thresholds': {
                'minimal': (0, 25),
                'moderate': (25, 50),
                'frequent': (50, 75),
                'constant': (75, 100)
            },
            'risk_multiplier': 0.4,
            'description': 'Walking and moving on foot'
        },
        'bending': {
            'keywords': ['bending', 'bend', 'stooping', 'flexing'],
            'frequency_thresholds': {
                'rarely': (0, 10),
                'occasionally': (10, 33),
                'frequently': (33, 66),
                'constantly': (66, 100)
            },
            'risk_multiplier': 1.1,
            'description': 'Bending, stooping, or flexing'
        },
        'reaching': {
            'keywords': ['reaching', 'reach', 'extending arms'],
            'frequency_thresholds': {
                'rarely': (0, 10),
                'occasionally': (10, 33),
                'frequently': (33, 66),
                'constantly': (66, 100)
            },
            'risk_multiplier': 0.7,
            'description': 'Reaching and extending arms'
        },
        'kneeling': {
            'keywords': ['kneeling', 'kneel', 'on knees'],
            'frequency_thresholds': {
                'rarely': (0, 10),
                'occasionally': (10, 33),
                'frequently': (33, 66),
                'constantly': (66, 100)
            },
            'risk_multiplier': 1.0,
            'description': 'Kneeling or working on knees'
        },
        'crawling': {
            'keywords': ['crawling', 'crawl', 'moving on hands and knees'],
            'frequency_thresholds': {
                'rarely': (0, 10),
                'occasionally': (10, 33),
                'frequently': (33, 66),
                'constantly': (66, 100)
            },
            'risk_multiplier': 1.3,
            'description': 'Crawling or moving on hands and knees'
        }
    }
    
    def __init__(self, statistical_analyzer: Optional[StatisticalAnalyzer] = None):
        """
        Initialize the physical demands analyzer.
        
        Args:
            statistical_analyzer: Optional StatisticalAnalyzer instance
        """
        self.stats_analyzer = statistical_analyzer or StatisticalAnalyzer()
    
    def identify_physical_demands(
        self, 
        data: pd.DataFrame,
        requirement_column: str = 'requirement_type',
        category_column: str = 'category'
    ) -> pd.DataFrame:
        """
        Identify physical demand records in the dataset.
        
        Args:
            data: DataFrame containing occupation data
            requirement_column: Name of the requirement type column
            category_column: Name of the category column
            
        Returns:
            DataFrame filtered to physical demand records
        """
        # Filter for physical demands
        physical_mask = data[requirement_column].str.contains(
            'physical|lifting|carrying|climbing|standing|sitting|walking|bending|reaching|kneeling|crawling',
            case=False, na=False
        )
        
        physical_data = data[physical_mask].copy()
        
        # Categorize physical demands
        physical_data['demand_category'] = physical_data.apply(
            lambda row: self._categorize_physical_demand(
                row[requirement_column], row[category_column]
            ), axis=1
        )
        
        # Filter out uncategorized demands
        physical_data = physical_data[
            physical_data['demand_category'] != 'other'
        ].copy()
        
        return physical_data
    
    def _categorize_physical_demand(
        self, 
        requirement_type: str, 
        category: str
    ) -> str:
        """
        Categorize physical demand based on requirement type and category.
        
        Args:
            requirement_type: The requirement type text
            category: The category text
            
        Returns:
            Physical demand category name
        """
        combined_text = f"{requirement_type} {category}".lower()
        
        for demand_name, demand_info in self.PHYSICAL_DEMANDS.items():
            for keyword in demand_info['keywords']:
                if keyword.lower() in combined_text:
                    return demand_name
        
        return 'other'
    
    def calculate_demand_scores(
        self, 
        physical_data: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation',
        estimate_column: str = 'estimate',
        std_error_column: str = 'standard_error'
    ) -> pd.DataFrame:
        """
        Calculate physical demand scores for occupations.
        
        Args:
            physical_data: DataFrame with physical demand data
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            estimate_column: Name of the estimate column
            std_error_column: Name of the standard error column
            
        Returns:
            DataFrame with demand scores by occupation
        """
        if physical_data.empty:
            return pd.DataFrame()
        
        # Calculate confidence intervals for estimates
        ci_data = self.stats_analyzer.calculate_confidence_intervals(
            physical_data[estimate_column],
            physical_data[std_error_column]
        )
        
        physical_data = physical_data.copy()
        physical_data['lower_ci'] = ci_data['lower_ci']
        physical_data['upper_ci'] = ci_data['upper_ci']
        physical_data['margin_error'] = ci_data['margin_error']
        
        # Calculate reliability scores
        physical_data['reliability_score'] = self.stats_analyzer.calculate_reliability_scores(
            physical_data[estimate_column],
            physical_data[std_error_column]
        )
        
        # Calculate intensity levels
        physical_data['intensity_level'] = physical_data.apply(
            lambda row: self._calculate_intensity_level(
                row['demand_category'], row[estimate_column]
            ), axis=1
        )
        
        # Calculate demand scores
        physical_data['risk_multiplier'] = physical_data['demand_category'].map(
            lambda x: self.PHYSICAL_DEMANDS.get(x, {}).get('risk_multiplier', 0.5)
        )
        
        # Demand score = estimate * risk_multiplier * reliability_score
        physical_data['demand_score'] = (
            physical_data[estimate_column] * 
            physical_data['risk_multiplier'] * 
            physical_data['reliability_score']
        )
        
        # Adjust demand score based on confidence interval width
        ci_width = physical_data['upper_ci'] - physical_data['lower_ci']
        ci_adjustment = 1 / (1 + ci_width / physical_data[estimate_column].abs())
        physical_data['adjusted_demand_score'] = (
            physical_data['demand_score'] * ci_adjustment
        )
        
        return physical_data
    
    def _calculate_intensity_level(self, demand_category: str, estimate_value: float) -> str:
        """
        Calculate intensity level for a physical demand.
        
        Args:
            demand_category: The physical demand category
            estimate_value: The estimate value (percentage, weight, etc.)
            
        Returns:
            Intensity level string
        """
        if demand_category not in self.PHYSICAL_DEMANDS:
            return 'unknown'
        
        demand_info = self.PHYSICAL_DEMANDS[demand_category]
        
        # Check for weight thresholds (lifting, carrying)
        if 'weight_thresholds' in demand_info:
            for level, (min_val, max_val) in demand_info['weight_thresholds'].items():
                if min_val <= estimate_value < max_val:
                    return level
        
        # Check for frequency/time thresholds
        elif 'frequency_thresholds' in demand_info:
            for level, (min_val, max_val) in demand_info['frequency_thresholds'].items():
                if min_val <= estimate_value < max_val:
                    return level
        
        elif 'time_thresholds' in demand_info:
            for level, (min_val, max_val) in demand_info['time_thresholds'].items():
                if min_val <= estimate_value < max_val:
                    return level
        
        return 'moderate'
    
    def create_intensity_matrix(
        self, 
        demand_data: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation'
    ) -> pd.DataFrame:
        """
        Create intensity matrix showing requirement levels across occupations.
        
        Args:
            demand_data: DataFrame with calculated demand scores
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            
        Returns:
            DataFrame with occupations as rows and demand categories as columns
        """
        if demand_data.empty:
            return pd.DataFrame()
        
        # Create occupation identifier
        demand_data = demand_data.copy()
        demand_data['occupation_id'] = (
            demand_data[soc_code_column] + " - " + demand_data[occupation_column]
        )
        
        # Pivot to create intensity matrix
        intensity_matrix = demand_data.pivot_table(
            index='occupation_id',
            columns='demand_category',
            values='adjusted_demand_score',
            aggfunc='sum',
            fill_value=0
        )
        
        # Add total demand score column
        intensity_matrix['total_physical_demand'] = intensity_matrix.sum(axis=1)
        
        # Sort by total demand
        intensity_matrix = intensity_matrix.sort_values('total_physical_demand', ascending=False)
        
        return intensity_matrix
    
    def calculate_ergonomic_assessments(
        self, 
        demand_data: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation'
    ) -> List[Dict[str, Any]]:
        """
        Calculate ergonomic assessments with accommodation recommendations.
        
        Args:
            demand_data: DataFrame with calculated demand scores
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            
        Returns:
            List of ergonomic assessment dictionaries
        """
        if demand_data.empty:
            return []
        
        assessments = []
        
        # Group by occupation
        for (soc_code, occupation), group in demand_data.groupby([soc_code_column, occupation_column]):
            
            # Calculate overall physical demand
            total_demand = group['adjusted_demand_score'].sum()
            mean_demand = group['adjusted_demand_score'].mean()
            max_demand = group['adjusted_demand_score'].max()
            
            # Identify high-risk demands
            high_risk_demands = group[
                group['adjusted_demand_score'] > group['adjusted_demand_score'].quantile(0.75)
            ]
            
            # Generate risk assessment
            risk_level = self._assess_ergonomic_risk(total_demand, mean_demand, max_demand)
            
            # Generate accommodation recommendations
            accommodations = self._generate_accommodations(high_risk_demands)
            
            assessment = {
                'soc_code': soc_code,
                'occupation': occupation,
                'total_physical_demand': total_demand,
                'mean_demand_score': mean_demand,
                'max_demand_score': max_demand,
                'risk_level': risk_level,
                'high_risk_demands': [
                    {
                        'category': row['demand_category'],
                        'score': row['adjusted_demand_score'],
                        'estimate': row['estimate'],
                        'intensity': row['intensity_level']
                    }
                    for _, row in high_risk_demands.iterrows()
                ],
                'accommodation_recommendations': accommodations,
                'demand_categories': list(group['demand_category'].unique()),
                'mean_reliability': group['reliability_score'].mean()
            }
            
            assessments.append(assessment)
        
        # Sort by total physical demand (descending)
        assessments.sort(key=lambda x: x['total_physical_demand'], reverse=True)
        
        return assessments
    
    def _assess_ergonomic_risk(
        self, 
        total_demand: float, 
        mean_demand: float, 
        max_demand: float
    ) -> str:
        """
        Assess ergonomic risk level based on demand scores.
        
        Args:
            total_demand: Total physical demand score
            mean_demand: Mean demand score
            max_demand: Maximum demand score
            
        Returns:
            Risk level string
        """
        # Risk thresholds (these can be adjusted based on domain expertise)
        if total_demand > 100 or max_demand > 50:
            return 'Very High'
        elif total_demand > 75 or max_demand > 35:
            return 'High'
        elif total_demand > 50 or max_demand > 25:
            return 'Moderate'
        elif total_demand > 25 or max_demand > 15:
            return 'Low'
        else:
            return 'Very Low'
    
    def _generate_accommodations(self, high_risk_demands: pd.DataFrame) -> List[str]:
        """
        Generate accommodation recommendations based on high-risk demands.
        
        Args:
            high_risk_demands: DataFrame with high-risk physical demands
            
        Returns:
            List of accommodation recommendation strings
        """
        accommodations = []
        
        for _, demand in high_risk_demands.iterrows():
            category = demand['demand_category']
            intensity = demand['intensity_level']
            
            if category == 'lifting':
                if intensity in ['heavy', 'very_heavy']:
                    accommodations.extend([
                        'Provide mechanical lifting aids (hoists, forklifts)',
                        'Implement team lifting procedures for heavy objects',
                        'Use adjustable height work surfaces'
                    ])
                else:
                    accommodations.append('Provide proper lifting training and techniques')
            
            elif category == 'carrying':
                if intensity in ['heavy', 'very_heavy']:
                    accommodations.extend([
                        'Provide wheeled carts or dollies',
                        'Use conveyor systems for material transport',
                        'Implement job rotation to reduce carrying load'
                    ])
            
            elif category == 'climbing':
                accommodations.extend([
                    'Ensure proper ladder safety and fall protection',
                    'Provide scaffolding or elevated platforms',
                    'Use mechanical lifts where possible'
                ])
            
            elif category == 'standing':
                if intensity in ['frequent', 'constant']:
                    accommodations.extend([
                        'Provide anti-fatigue mats',
                        'Allow for sit-stand workstations',
                        'Implement regular break schedules'
                    ])
            
            elif category == 'bending':
                accommodations.extend([
                    'Adjust work surface heights to reduce bending',
                    'Provide long-handled tools',
                    'Use mechanical aids for low-level tasks'
                ])
            
            elif category == 'reaching':
                accommodations.extend([
                    'Organize workspace to minimize reaching',
                    'Provide adjustable shelving and storage',
                    'Use extension tools for overhead tasks'
                ])
            
            elif category in ['kneeling', 'crawling']:
                accommodations.extend([
                    'Provide knee pads and protective equipment',
                    'Use creepers or rolling platforms',
                    'Modify workspace layout to reduce floor-level work'
                ])
        
        # Remove duplicates and return
        return list(set(accommodations))
    
    def rank_occupations_by_physical_demands(
        self, 
        demand_data: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation',
        top_n: Optional[int] = None
    ) -> List[AnalysisResult]:
        """
        Rank occupations by overall physical demand levels.
        
        Args:
            demand_data: DataFrame with calculated demand scores
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            top_n: Number of top demanding occupations to return (None for all)
            
        Returns:
            List of AnalysisResult objects ranked by physical demands
        """
        if demand_data.empty:
            return []
        
        # Aggregate demand scores by occupation
        occupation_demands = demand_data.groupby([soc_code_column, occupation_column]).agg({
            'adjusted_demand_score': ['sum', 'mean', 'count'],
            'reliability_score': 'mean',
            'demand_category': lambda x: list(x.unique()),
            'estimate': 'mean',
            'lower_ci': 'mean',
            'upper_ci': 'mean',
            'intensity_level': lambda x: list(x.unique())
        }).round(4)
        
        # Flatten column names
        occupation_demands.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] 
                                    for col in occupation_demands.columns]
        
        # Reset index
        occupation_demands = occupation_demands.reset_index()
        
        # Calculate overall demand score
        occupation_demands['overall_demand_score'] = occupation_demands['sum_adjusted_demand_score']
        
        # Sort by overall demand score
        occupation_demands = occupation_demands.sort_values(
            'overall_demand_score', ascending=False
        ).reset_index(drop=True)
        
        # Limit to top N if specified
        if top_n is not None:
            occupation_demands = occupation_demands.head(top_n)
        
        # Convert to AnalysisResult objects
        results = []
        for _, row in occupation_demands.iterrows():
            demand_categories = row['<lambda>_demand_category']
            intensity_levels = row['<lambda>_intensity_level']
            
            result = AnalysisResult(
                occupation_category=f"{row[soc_code_column]} - {row[occupation_column]}",
                metric_name="physical_demand_score",
                value=row['overall_demand_score'],
                confidence_interval=(row['mean_lower_ci'], row['mean_upper_ci']),
                reliability_score=row['mean_reliability_score'],
                footnote_context=[
                    f"Physical demands: {', '.join(demand_categories)}",
                    f"Number of demand factors: {row['count_adjusted_demand_score']}",
                    f"Average demand per factor: {row['mean_adjusted_demand_score']:.2f}",
                    f"Intensity levels: {', '.join(set(intensity_levels))}",
                    f"Mean estimate: {row['mean_estimate']:.2f}%",
                    f"Overall reliability: {row['mean_reliability_score']:.3f}"
                ]
            )
            results.append(result)
        
        return results
    
    def generate_physical_demands_summary(
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
        Generate comprehensive physical demands summary.
        
        Args:
            data: DataFrame containing occupation data
            requirement_column: Name of the requirement type column
            category_column: Name of the category column
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            estimate_column: Name of the estimate column
            std_error_column: Name of the standard error column
            
        Returns:
            Dictionary containing comprehensive physical demands analysis
        """
        # Identify physical demands
        physical_data = self.identify_physical_demands(
            data, requirement_column, category_column
        )
        
        if physical_data.empty:
            return {
                'overview': {
                    'total_physical_records': 0,
                    'physical_demand_occupations': 0,
                    'demand_categories_found': 0
                },
                'message': 'No physical demand data found in dataset'
            }
        
        # Calculate demand scores
        demand_data = self.calculate_demand_scores(
            physical_data, soc_code_column, occupation_column,
            estimate_column, std_error_column
        )
        
        # Rank occupations
        top_demand_occupations = self.rank_occupations_by_physical_demands(
            demand_data, soc_code_column, occupation_column, top_n=20
        )
        
        # Create intensity matrix
        intensity_matrix = self.create_intensity_matrix(
            demand_data, soc_code_column, occupation_column
        )
        
        # Calculate ergonomic assessments
        ergonomic_assessments = self.calculate_ergonomic_assessments(
            demand_data, soc_code_column, occupation_column
        )
        
        # Overall statistics
        overview = {
            'total_physical_records': len(physical_data),
            'physical_demand_occupations': physical_data[soc_code_column].nunique(),
            'demand_categories_found': len(physical_data['demand_category'].unique()),
            'mean_demand_score': demand_data['adjusted_demand_score'].mean(),
            'max_demand_score': demand_data['adjusted_demand_score'].max(),
            'mean_reliability': demand_data['reliability_score'].mean(),
            'high_demand_occupations': len(demand_data[demand_data['adjusted_demand_score'] > 
                                                     demand_data['adjusted_demand_score'].quantile(0.9)])
        }
        
        return {
            'overview': overview,
            'top_demand_occupations': [
                {
                    'occupation': result.occupation_category,
                    'demand_score': result.value,
                    'confidence_interval': result.confidence_interval,
                    'reliability': result.reliability_score,
                    'physical_demands': result.footnote_context[0].replace('Physical demands: ', '')
                }
                for result in top_demand_occupations
            ],
            'intensity_matrix_shape': {
                'occupations': len(intensity_matrix),
                'demand_categories': len(intensity_matrix.columns) - 1,  # Exclude total column
                'highest_demand_occupation': intensity_matrix.index[0] if not intensity_matrix.empty else None,
                'highest_total_demand': intensity_matrix['total_physical_demand'].iloc[0] if not intensity_matrix.empty else 0
            },
            'ergonomic_assessments': {
                'total_assessments': len(ergonomic_assessments),
                'high_risk_occupations': len([a for a in ergonomic_assessments if a['risk_level'] in ['High', 'Very High']]),
                'common_accommodations': self._get_common_accommodations(ergonomic_assessments)
            },
            'demand_categories': {
                name: info['description'] 
                for name, info in self.PHYSICAL_DEMANDS.items()
            }
        }
    
    def _get_common_accommodations(self, assessments: List[Dict[str, Any]]) -> List[str]:
        """Get most common accommodation recommendations."""
        all_accommodations = []
        for assessment in assessments:
            all_accommodations.extend(assessment['accommodation_recommendations'])
        
        # Count occurrences
        from collections import Counter
        accommodation_counts = Counter(all_accommodations)
        
        # Return top 10 most common
        return [acc for acc, count in accommodation_counts.most_common(10)]