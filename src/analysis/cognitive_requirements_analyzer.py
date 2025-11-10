"""
Cognitive requirements analysis functionality for occupation data reports.
Implements cognitive demand categorization, skill similarity identification, and educational pathway planning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Optional sklearn imports
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some advanced clustering features will be disabled.")

from ..interfaces import AnalysisResult
from .statistical_analyzer import StatisticalAnalyzer


class CognitiveRequirementsAnalyzer:
    """
    Analyzer for cognitive and mental requirements across occupations.
    Handles cognitive demand categorization, skill similarity, and educational planning.
    """
    
    # Cognitive requirement categories and their characteristics
    COGNITIVE_REQUIREMENTS = {
        'problem_solving': {
            'keywords': ['problem solving', 'analytical thinking', 'troubleshooting', 'critical thinking'],
            'complexity_levels': {
                'basic': (0, 25),        # 0-25% complexity
                'intermediate': (25, 50), # 25-50% complexity
                'advanced': (50, 75),    # 50-75% complexity
                'expert': (75, 100)      # 75-100% complexity
            },
            'weight': 1.0,
            'description': 'Problem-solving and analytical thinking requirements'
        },
        'decision_making': {
            'keywords': ['decision making', 'judgment', 'choices', 'evaluation'],
            'complexity_levels': {
                'routine': (0, 25),
                'guided': (25, 50),
                'independent': (50, 75),
                'strategic': (75, 100)
            },
            'weight': 0.9,
            'description': 'Decision-making and judgment requirements'
        },
        'literacy': {
            'keywords': ['reading', 'writing', 'comprehension', 'documentation'],
            'skill_levels': {
                'basic': (0, 25),
                'functional': (25, 50),
                'proficient': (50, 75),
                'advanced': (75, 100)
            },
            'weight': 0.8,
            'description': 'Reading, writing, and literacy requirements'
        },
        'numeracy': {
            'keywords': ['mathematics', 'calculations', 'numerical', 'quantitative'],
            'skill_levels': {
                'basic': (0, 25),
                'intermediate': (25, 50),
                'advanced': (50, 75),
                'expert': (75, 100)
            },
            'weight': 0.8,
            'description': 'Mathematical and numerical skill requirements'
        },
        'memory': {
            'keywords': ['memory', 'recall', 'retention', 'memorization'],
            'demand_levels': {
                'minimal': (0, 25),
                'moderate': (25, 50),
                'high': (50, 75),
                'intensive': (75, 100)
            },
            'weight': 0.7,
            'description': 'Memory and information retention requirements'
        },
        'attention': {
            'keywords': ['attention', 'concentration', 'focus', 'vigilance'],
            'demand_levels': {
                'basic': (0, 25),
                'sustained': (25, 50),
                'selective': (50, 75),
                'divided': (75, 100)
            },
            'weight': 0.8,
            'description': 'Attention and concentration requirements'
        },
        'learning': {
            'keywords': ['learning', 'training', 'adaptation', 'skill acquisition'],
            'complexity_levels': {
                'routine': (0, 25),
                'moderate': (25, 50),
                'complex': (50, 75),
                'continuous': (75, 100)
            },
            'weight': 0.7,
            'description': 'Learning and adaptation requirements'
        },
        'communication': {
            'keywords': ['communication', 'verbal', 'presentation', 'instruction'],
            'skill_levels': {
                'basic': (0, 25),
                'interpersonal': (25, 50),
                'professional': (50, 75),
                'expert': (75, 100)
            },
            'weight': 0.8,
            'description': 'Communication and interpersonal skill requirements'
        }
    }
    
    def __init__(self, statistical_analyzer: Optional[StatisticalAnalyzer] = None):
        """
        Initialize the cognitive requirements analyzer.
        
        Args:
            statistical_analyzer: Optional StatisticalAnalyzer instance
        """
        self.stats_analyzer = statistical_analyzer or StatisticalAnalyzer()
    
    def identify_cognitive_requirements(
        self, 
        data: pd.DataFrame,
        requirement_column: str = 'requirement_type',
        category_column: str = 'category'
    ) -> pd.DataFrame:
        """
        Identify cognitive requirement records in the dataset.
        
        Args:
            data: DataFrame containing occupation data
            requirement_column: Name of the requirement type column
            category_column: Name of the category column
            
        Returns:
            DataFrame filtered to cognitive requirement records
        """
        # Filter for cognitive and mental requirements
        cognitive_mask = data[requirement_column].str.contains(
            'cognitive|mental|problem|decision|literacy|reading|writing|mathematics|memory|attention|learning|communication',
            case=False, na=False
        )
        
        cognitive_data = data[cognitive_mask].copy()
        
        # Categorize cognitive requirements
        cognitive_data['cognitive_category'] = cognitive_data.apply(
            lambda row: self._categorize_cognitive_requirement(
                row[requirement_column], row[category_column]
            ), axis=1
        )
        
        # Filter out uncategorized requirements
        cognitive_data = cognitive_data[
            cognitive_data['cognitive_category'] != 'other'
        ].copy()
        
        return cognitive_data
    
    def _categorize_cognitive_requirement(
        self, 
        requirement_type: str, 
        category: str
    ) -> str:
        """
        Categorize cognitive requirement based on requirement type and category.
        
        Args:
            requirement_type: The requirement type text
            category: The category text
            
        Returns:
            Cognitive requirement category name
        """
        combined_text = f"{requirement_type} {category}".lower()
        
        for req_name, req_info in self.COGNITIVE_REQUIREMENTS.items():
            for keyword in req_info['keywords']:
                if keyword.lower() in combined_text:
                    return req_name
        
        return 'other'
    
    def calculate_cognitive_demand_scores(
        self, 
        cognitive_data: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation',
        estimate_column: str = 'estimate',
        std_error_column: str = 'standard_error'
    ) -> pd.DataFrame:
        """
        Calculate cognitive demand scores for occupations.
        
        Args:
            cognitive_data: DataFrame with cognitive requirement data
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            estimate_column: Name of the estimate column
            std_error_column: Name of the standard error column
            
        Returns:
            DataFrame with cognitive demand scores by occupation
        """
        if cognitive_data.empty:
            return pd.DataFrame()
        
        # Calculate confidence intervals for estimates
        ci_data = self.stats_analyzer.calculate_confidence_intervals(
            cognitive_data[estimate_column],
            cognitive_data[std_error_column]
        )
        
        cognitive_data = cognitive_data.copy()
        cognitive_data['lower_ci'] = ci_data['lower_ci']
        cognitive_data['upper_ci'] = ci_data['upper_ci']
        cognitive_data['margin_error'] = ci_data['margin_error']
        
        # Calculate reliability scores
        cognitive_data['reliability_score'] = self.stats_analyzer.calculate_reliability_scores(
            cognitive_data[estimate_column],
            cognitive_data[std_error_column]
        )
        
        # Calculate complexity/skill levels
        cognitive_data['skill_level'] = cognitive_data.apply(
            lambda row: self._calculate_skill_level(
                row['cognitive_category'], row[estimate_column]
            ), axis=1
        )
        
        # Calculate cognitive demand scores
        cognitive_data['weight'] = cognitive_data['cognitive_category'].map(
            lambda x: self.COGNITIVE_REQUIREMENTS.get(x, {}).get('weight', 0.5)
        )
        
        # Demand score = estimate * weight * reliability_score
        cognitive_data['cognitive_score'] = (
            cognitive_data[estimate_column] * 
            cognitive_data['weight'] * 
            cognitive_data['reliability_score']
        )
        
        # Adjust cognitive score based on confidence interval width
        ci_width = cognitive_data['upper_ci'] - cognitive_data['lower_ci']
        ci_adjustment = 1 / (1 + ci_width / cognitive_data[estimate_column].abs())
        cognitive_data['adjusted_cognitive_score'] = (
            cognitive_data['cognitive_score'] * ci_adjustment
        )
        
        return cognitive_data
    
    def _calculate_skill_level(self, cognitive_category: str, estimate_value: float) -> str:
        """
        Calculate skill level for a cognitive requirement.
        
        Args:
            cognitive_category: The cognitive requirement category
            estimate_value: The estimate value (percentage, frequency, etc.)
            
        Returns:
            Skill level string
        """
        if cognitive_category not in self.COGNITIVE_REQUIREMENTS:
            return 'unknown'
        
        req_info = self.COGNITIVE_REQUIREMENTS[cognitive_category]
        
        # Check for different level types
        for level_type in ['complexity_levels', 'skill_levels', 'demand_levels']:
            if level_type in req_info:
                for level, (min_val, max_val) in req_info[level_type].items():
                    if min_val <= estimate_value < max_val:
                        return level
        
        return 'moderate'
    
    def categorize_occupations_by_cognitive_demands(
        self, 
        cognitive_data: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation'
    ) -> pd.DataFrame:
        """
        Categorize occupations by cognitive demand levels.
        
        Args:
            cognitive_data: DataFrame with calculated cognitive scores
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            
        Returns:
            DataFrame with occupation cognitive demand categories
        """
        if cognitive_data.empty:
            return pd.DataFrame()
        
        # Aggregate cognitive scores by occupation
        occupation_cognitive = cognitive_data.groupby([soc_code_column, occupation_column]).agg({
            'adjusted_cognitive_score': ['sum', 'mean', 'count'],
            'reliability_score': 'mean',
            'cognitive_category': lambda x: list(x.unique()),
            'skill_level': lambda x: list(x.unique()),
            'estimate': 'mean'
        }).round(4)
        
        # Flatten column names
        occupation_cognitive.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] 
                                      for col in occupation_cognitive.columns]
        
        # Reset index
        occupation_cognitive = occupation_cognitive.reset_index()
        
        # Calculate overall cognitive demand level
        occupation_cognitive['total_cognitive_score'] = occupation_cognitive['sum_adjusted_cognitive_score']
        
        # Categorize cognitive demand levels
        cognitive_scores = occupation_cognitive['total_cognitive_score']
        occupation_cognitive['cognitive_demand_level'] = pd.cut(
            cognitive_scores,
            bins=[0, cognitive_scores.quantile(0.25), cognitive_scores.quantile(0.5), 
                  cognitive_scores.quantile(0.75), cognitive_scores.max()],
            labels=['Low', 'Moderate', 'High', 'Very High'],
            include_lowest=True
        )
        
        return occupation_cognitive.sort_values('total_cognitive_score', ascending=False)
    
    def identify_skill_similarity(
        self, 
        cognitive_data: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation',
        similarity_threshold: float = 0.8
    ) -> pd.DataFrame:
        """
        Identify occupations with similar cognitive skill requirements.
        
        Args:
            cognitive_data: DataFrame with cognitive requirement data
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            similarity_threshold: Minimum similarity score to include
            
        Returns:
            DataFrame with occupation similarity pairs for educational planning
        """
        if cognitive_data.empty:
            return pd.DataFrame()
        
        # Create occupation-cognitive requirement matrix
        pivot_data = cognitive_data.pivot_table(
            index=[soc_code_column, occupation_column],
            columns='cognitive_category',
            values='adjusted_cognitive_score',
            aggfunc='sum',
            fill_value=0
        )
        
        # Calculate cosine similarity matrix
        if SKLEARN_AVAILABLE:
            similarity_matrix = cosine_similarity(pivot_data.values)
            similarity_df = pd.DataFrame(
                similarity_matrix,
                index=pivot_data.index,
                columns=pivot_data.index
            )
        else:
            # Fallback to correlation-based similarity
            similarity_df = pivot_data.T.corr()
            warnings.warn("Using correlation-based similarity instead of cosine similarity (sklearn not available)")
        
        # Extract similarity pairs above threshold
        similarity_pairs = []
        occupations = list(pivot_data.index)
        
        for i, occ1 in enumerate(occupations):
            for j, occ2 in enumerate(occupations[i+1:], i+1):
                similarity = similarity_df.loc[occ1, occ2]
                if similarity >= similarity_threshold:
                    similarity_pairs.append({
                        'occupation_1_soc': occ1[0],
                        'occupation_1_name': occ1[1],
                        'occupation_2_soc': occ2[0],
                        'occupation_2_name': occ2[1],
                        'similarity_score': similarity,
                        'similarity_category': self._categorize_similarity(similarity),
                        'shared_cognitive_skills': self._identify_shared_skills(
                            pivot_data.loc[occ1], pivot_data.loc[occ2]
                        )
                    })
        
        # Sort by similarity score
        similarity_pairs_df = pd.DataFrame(similarity_pairs)
        if not similarity_pairs_df.empty:
            similarity_pairs_df = similarity_pairs_df.sort_values('similarity_score', ascending=False)
        
        return similarity_pairs_df
    
    def _categorize_similarity(self, similarity_score: float) -> str:
        """Categorize similarity scores into descriptive labels."""
        if similarity_score >= 0.95:
            return "Nearly Identical"
        elif similarity_score >= 0.9:
            return "Very High"
        elif similarity_score >= 0.8:
            return "High"
        elif similarity_score >= 0.7:
            return "Moderate"
        else:
            return "Low"
    
    def _identify_shared_skills(self, profile1: pd.Series, profile2: pd.Series) -> List[str]:
        """Identify shared cognitive skills between two occupation profiles."""
        shared_skills = []
        
        for skill in profile1.index:
            if profile1[skill] > 0 and profile2[skill] > 0:
                # Both occupations have this cognitive requirement
                avg_score = (profile1[skill] + profile2[skill]) / 2
                if avg_score > profile1.mean():  # Above average requirement
                    shared_skills.append(skill)
        
        return shared_skills
    
    def create_educational_pathways(
        self, 
        similarity_data: pd.DataFrame,
        cognitive_data: pd.DataFrame,
        soc_code_column: str = 'soc_code'
    ) -> Dict[str, Any]:
        """
        Create educational pathway recommendations based on skill similarities.
        
        Args:
            similarity_data: DataFrame with occupation similarities
            cognitive_data: DataFrame with cognitive requirement data
            soc_code_column: Name of the SOC code column
            
        Returns:
            Dictionary with educational pathway recommendations
        """
        if similarity_data.empty or cognitive_data.empty:
            return {}
        
        pathways = {}
        
        # Group similar occupations into career clusters
        clusters = self._create_career_clusters(similarity_data)
        
        for cluster_id, cluster_occupations in clusters.items():
            # Get cognitive requirements for this cluster
            cluster_cognitive_data = cognitive_data[
                cognitive_data[soc_code_column].isin([occ[0] for occ in cluster_occupations])
            ]
            
            # Analyze common cognitive requirements
            common_requirements = self._analyze_cluster_requirements(cluster_cognitive_data)
            
            # Generate educational recommendations
            educational_recommendations = self._generate_educational_recommendations(
                common_requirements
            )
            
            pathways[f"cluster_{cluster_id}"] = {
                'occupations': [
                    {'soc_code': occ[0], 'name': occ[1]} 
                    for occ in cluster_occupations
                ],
                'common_cognitive_requirements': common_requirements,
                'educational_recommendations': educational_recommendations,
                'transferable_skills': self._identify_transferable_skills(common_requirements),
                'skill_gaps': self._identify_skill_gaps(cluster_cognitive_data)
            }
        
        return pathways
    
    def _create_career_clusters(self, similarity_data: pd.DataFrame) -> Dict[int, List[Tuple[str, str]]]:
        """Create career clusters based on similarity data."""
        clusters = {}
        cluster_id = 0
        processed_occupations = set()
        
        for _, row in similarity_data.iterrows():
            occ1 = (row['occupation_1_soc'], row['occupation_1_name'])
            occ2 = (row['occupation_2_soc'], row['occupation_2_name'])
            
            if occ1 not in processed_occupations and occ2 not in processed_occupations:
                # Create new cluster
                clusters[cluster_id] = [occ1, occ2]
                processed_occupations.add(occ1)
                processed_occupations.add(occ2)
                cluster_id += 1
            elif occ1 in processed_occupations and occ2 not in processed_occupations:
                # Add occ2 to existing cluster containing occ1
                for cid, cluster in clusters.items():
                    if occ1 in cluster:
                        cluster.append(occ2)
                        processed_occupations.add(occ2)
                        break
            elif occ2 in processed_occupations and occ1 not in processed_occupations:
                # Add occ1 to existing cluster containing occ2
                for cid, cluster in clusters.items():
                    if occ2 in cluster:
                        cluster.append(occ1)
                        processed_occupations.add(occ1)
                        break
        
        return clusters
    
    def _analyze_cluster_requirements(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze common cognitive requirements within a cluster."""
        if cluster_data.empty:
            return {}
        
        requirements = {}
        
        for category in cluster_data['cognitive_category'].unique():
            category_data = cluster_data[cluster_data['cognitive_category'] == category]
            
            requirements[category] = {
                'mean_score': category_data['adjusted_cognitive_score'].mean(),
                'frequency': len(category_data),
                'skill_levels': list(category_data['skill_level'].unique()),
                'importance': category_data['weight'].iloc[0] if not category_data.empty else 0
            }
        
        return requirements
    
    def _generate_educational_recommendations(self, requirements: Dict[str, Any]) -> List[str]:
        """Generate educational recommendations based on cognitive requirements."""
        recommendations = []
        
        for category, info in requirements.items():
            mean_score = info['mean_score']
            skill_levels = info['skill_levels']
            
            if category == 'problem_solving':
                if 'advanced' in skill_levels or 'expert' in skill_levels:
                    recommendations.extend([
                        'Advanced problem-solving and critical thinking courses',
                        'Case study analysis and simulation training',
                        'Systems thinking and analytical methods'
                    ])
                else:
                    recommendations.append('Basic problem-solving and analytical thinking training')
            
            elif category == 'decision_making':
                if 'strategic' in skill_levels or 'independent' in skill_levels:
                    recommendations.extend([
                        'Strategic decision-making and leadership courses',
                        'Risk assessment and management training',
                        'Business judgment and evaluation methods'
                    ])
                else:
                    recommendations.append('Decision-making fundamentals and judgment training')
            
            elif category == 'literacy':
                if 'advanced' in skill_levels or 'proficient' in skill_levels:
                    recommendations.extend([
                        'Advanced technical writing and communication',
                        'Professional documentation and reporting',
                        'Information analysis and synthesis'
                    ])
                else:
                    recommendations.append('Basic literacy and communication skills')
            
            elif category == 'numeracy':
                if 'advanced' in skill_levels or 'expert' in skill_levels:
                    recommendations.extend([
                        'Advanced mathematics and statistical analysis',
                        'Quantitative methods and data analysis',
                        'Financial modeling and numerical reasoning'
                    ])
                else:
                    recommendations.append('Basic mathematics and numerical skills')
            
            elif category == 'communication':
                if 'expert' in skill_levels or 'professional' in skill_levels:
                    recommendations.extend([
                        'Advanced presentation and public speaking',
                        'Professional communication and interpersonal skills',
                        'Leadership communication and team management'
                    ])
                else:
                    recommendations.append('Basic communication and interpersonal skills')
        
        return list(set(recommendations))  # Remove duplicates
    
    def _identify_transferable_skills(self, requirements: Dict[str, Any]) -> List[str]:
        """Identify transferable skills within a career cluster."""
        transferable = []
        
        high_importance_skills = [
            category for category, info in requirements.items()
            if info['importance'] >= 0.8 and info['mean_score'] > 50
        ]
        
        for skill in high_importance_skills:
            if skill in self.COGNITIVE_REQUIREMENTS:
                transferable.append(self.COGNITIVE_REQUIREMENTS[skill]['description'])
        
        return transferable
    
    def _identify_skill_gaps(self, cluster_data: pd.DataFrame) -> List[str]:
        """Identify potential skill gaps within a career cluster."""
        gaps = []
        
        # Identify cognitive categories with high variability
        for category in cluster_data['cognitive_category'].unique():
            category_data = cluster_data[cluster_data['cognitive_category'] == category]
            score_std = category_data['adjusted_cognitive_score'].std()
            score_mean = category_data['adjusted_cognitive_score'].mean()
            
            # High variability suggests skill gaps
            if score_std > score_mean * 0.5:
                gaps.append(f"Inconsistent {category} requirements across cluster occupations")
        
        return gaps
    
    def generate_cognitive_requirements_summary(
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
        Generate comprehensive cognitive requirements summary.
        
        Args:
            data: DataFrame containing occupation data
            requirement_column: Name of the requirement type column
            category_column: Name of the category column
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            estimate_column: Name of the estimate column
            std_error_column: Name of the standard error column
            
        Returns:
            Dictionary containing comprehensive cognitive requirements analysis
        """
        # Identify cognitive requirements
        cognitive_data = self.identify_cognitive_requirements(
            data, requirement_column, category_column
        )
        
        if cognitive_data.empty:
            return {
                'overview': {
                    'total_cognitive_records': 0,
                    'cognitive_occupations': 0,
                    'cognitive_categories_found': 0
                },
                'message': 'No cognitive requirement data found in dataset'
            }
        
        # Calculate cognitive demand scores
        demand_data = self.calculate_cognitive_demand_scores(
            cognitive_data, soc_code_column, occupation_column,
            estimate_column, std_error_column
        )
        
        # Categorize occupations by cognitive demands
        occupation_categories = self.categorize_occupations_by_cognitive_demands(
            demand_data, soc_code_column, occupation_column
        )
        
        # Identify skill similarities
        skill_similarities = self.identify_skill_similarity(
            demand_data, soc_code_column, occupation_column
        )
        
        # Create educational pathways
        educational_pathways = self.create_educational_pathways(
            skill_similarities, demand_data, soc_code_column
        )
        
        # Overall statistics
        overview = {
            'total_cognitive_records': len(cognitive_data),
            'cognitive_occupations': cognitive_data[soc_code_column].nunique(),
            'cognitive_categories_found': len(cognitive_data['cognitive_category'].unique()),
            'mean_cognitive_score': demand_data['adjusted_cognitive_score'].mean(),
            'max_cognitive_score': demand_data['adjusted_cognitive_score'].max(),
            'mean_reliability': demand_data['reliability_score'].mean(),
            'high_cognitive_occupations': len(occupation_categories[
                occupation_categories['cognitive_demand_level'].isin(['High', 'Very High'])
            ]) if not occupation_categories.empty else 0
        }
        
        return {
            'overview': overview,
            'cognitive_demand_distribution': {
                level: len(occupation_categories[occupation_categories['cognitive_demand_level'] == level])
                for level in ['Low', 'Moderate', 'High', 'Very High']
            } if not occupation_categories.empty else {},
            'top_cognitive_occupations': [
                {
                    'soc_code': row[soc_code_column],
                    'occupation': row[occupation_column],
                    'cognitive_score': row['total_cognitive_score'],
                    'demand_level': row['cognitive_demand_level'],
                    'cognitive_categories': row['<lambda>_cognitive_category']
                }
                for _, row in occupation_categories.head(10).iterrows()
            ] if not occupation_categories.empty else [],
            'skill_similarity_analysis': {
                'total_similar_pairs': len(skill_similarities),
                'high_similarity_pairs': len(skill_similarities[skill_similarities['similarity_score'] >= 0.9]) if not skill_similarities.empty else 0,
                'career_clusters': len(educational_pathways)
            },
            'educational_pathways': educational_pathways,
            'cognitive_categories': {
                name: info['description'] 
                for name, info in self.COGNITIVE_REQUIREMENTS.items()
            }
        }