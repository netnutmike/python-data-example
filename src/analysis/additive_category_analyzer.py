"""
Additive category analysis functionality for occupation data reports.
Implements additive estimate grouping, comprehensive requirement profiles, and job family classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Optional sklearn imports
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some advanced clustering features will be disabled.")

from ..interfaces import AnalysisResult
from .statistical_analyzer import StatisticalAnalyzer


class AdditiveCategoryAnalyzer:
    """
    Analyzer for additive category relationships across occupations.
    Handles additive estimate grouping, requirement profiles, and job family classification.
    """
    
    def __init__(self, statistical_analyzer: Optional[StatisticalAnalyzer] = None):
        """
        Initialize the additive category analyzer.
        
        Args:
            statistical_analyzer: Optional StatisticalAnalyzer instance
        """
        self.stats_analyzer = statistical_analyzer or StatisticalAnalyzer()
    
    def identify_additive_estimates(
        self, 
        data: pd.DataFrame,
        additive_code_column: str = 'additive_code',
        additive_column: str = 'additive'
    ) -> pd.DataFrame:
        """
        Identify records with additive estimates that can be grouped together.
        
        Args:
            data: DataFrame containing occupation data
            additive_code_column: Name of the additive code column
            additive_column: Name of the additive description column
            
        Returns:
            DataFrame filtered to additive estimate records
        """
        # Filter for records with additive codes (non-null and non-empty)
        additive_mask = (
            data[additive_code_column].notna() & 
            (data[additive_code_column] != '') &
            (data[additive_code_column] != '0')
        )
        
        additive_data = data[additive_mask].copy()
        
        # Clean and standardize additive codes
        additive_data['additive_code_clean'] = additive_data[additive_code_column].astype(str).str.strip()
        
        # Group additive estimates by code
        additive_data['additive_group'] = additive_data['additive_code_clean']
        
        return additive_data
    
    def group_additive_estimates(
        self, 
        additive_data: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation',
        estimate_column: str = 'estimate',
        std_error_column: str = 'standard_error',
        additive_group_column: str = 'additive_group'
    ) -> pd.DataFrame:
        """
        Group additive estimates by occupation and additive code.
        
        Args:
            additive_data: DataFrame with additive estimate data
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            estimate_column: Name of the estimate column
            std_error_column: Name of the standard error column
            additive_group_column: Name of the additive group column
            
        Returns:
            DataFrame with grouped additive estimates
        """
        if additive_data.empty:
            return pd.DataFrame()
        
        # Group by occupation and additive group
        grouped_estimates = additive_data.groupby([
            soc_code_column, occupation_column, additive_group_column
        ]).agg({
            estimate_column: ['sum', 'count', 'mean', 'std'],
            std_error_column: ['mean', 'std'],
            'requirement_type': lambda x: list(x.unique()),
            'category': lambda x: list(x.unique())
        }).round(4)
        
        # Flatten column names
        grouped_estimates.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] 
                                   for col in grouped_estimates.columns]
        
        # Reset index
        grouped_estimates = grouped_estimates.reset_index()
        
        # Calculate combined standard errors for summed estimates
        # Using root sum of squares method for independent estimates
        grouped_estimates['combined_std_error'] = np.sqrt(
            grouped_estimates['count_estimate'] * (grouped_estimates['mean_standard_error'] ** 2)
        )
        
        # Calculate confidence intervals for summed estimates
        ci_data = self.stats_analyzer.calculate_confidence_intervals(
            grouped_estimates['sum_estimate'],
            grouped_estimates['combined_std_error']
        )
        
        grouped_estimates['lower_ci'] = ci_data['lower_ci']
        grouped_estimates['upper_ci'] = ci_data['upper_ci']
        grouped_estimates['margin_error'] = ci_data['margin_error']
        
        # Calculate reliability scores
        grouped_estimates['reliability_score'] = self.stats_analyzer.calculate_reliability_scores(
            grouped_estimates['sum_estimate'],
            grouped_estimates['combined_std_error']
        )
        
        return grouped_estimates
    
    def calculate_comprehensive_requirement_profiles(
        self, 
        grouped_data: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation'
    ) -> pd.DataFrame:
        """
        Calculate comprehensive requirement profiles by combining additive estimates.
        
        Args:
            grouped_data: DataFrame with grouped additive estimates
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            
        Returns:
            DataFrame with comprehensive requirement profiles
        """
        if grouped_data.empty:
            return pd.DataFrame()
        
        # Create requirement profiles by occupation
        profiles = []
        
        for (soc_code, occupation), group in grouped_data.groupby([soc_code_column, occupation_column]):
            
            # Calculate total additive score
            total_additive_score = group['sum_estimate'].sum()
            
            # Calculate weighted average reliability
            weights = group['sum_estimate'] / total_additive_score if total_additive_score > 0 else np.ones(len(group)) / len(group)
            weighted_reliability = (group['reliability_score'] * weights).sum()
            
            # Identify requirement composition
            requirement_types = []
            for req_list in group['<lambda>_requirement_type']:
                requirement_types.extend(req_list)
            requirement_composition = list(set(requirement_types))
            
            # Calculate profile statistics
            profile = {
                soc_code_column: soc_code,
                occupation_column: occupation,
                'total_additive_score': total_additive_score,
                'num_additive_groups': len(group),
                'mean_group_score': group['sum_estimate'].mean(),
                'max_group_score': group['sum_estimate'].max(),
                'min_group_score': group['sum_estimate'].min(),
                'score_std': group['sum_estimate'].std(),
                'weighted_reliability': weighted_reliability,
                'requirement_composition': requirement_composition,
                'num_requirement_types': len(requirement_composition),
                'additive_groups': list(group['additive_group'].unique()),
                'profile_complexity': len(group) * len(requirement_composition),  # Complexity metric
                'score_distribution': {
                    'q25': group['sum_estimate'].quantile(0.25),
                    'q50': group['sum_estimate'].quantile(0.50),
                    'q75': group['sum_estimate'].quantile(0.75)
                }
            }
            
            profiles.append(profile)
        
        profiles_df = pd.DataFrame(profiles)
        
        # Sort by total additive score
        profiles_df = profiles_df.sort_values('total_additive_score', ascending=False)
        
        return profiles_df
    
    def classify_job_families(
        self, 
        requirement_profiles: pd.DataFrame,
        n_clusters: int = 8,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation'
    ) -> Dict[str, Any]:
        """
        Classify occupations into job families based on additive requirement patterns.
        
        Args:
            requirement_profiles: DataFrame with comprehensive requirement profiles
            n_clusters: Number of job family clusters to create
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            
        Returns:
            Dictionary with job family classifications
        """
        if requirement_profiles.empty or len(requirement_profiles) < n_clusters:
            return {}
        
        if not SKLEARN_AVAILABLE:
            warnings.warn("scikit-learn not available. Using simple quantile-based clustering instead.")
            return self._simple_quantile_clustering(requirement_profiles, n_clusters, soc_code_column, occupation_column)
        
        # Prepare features for clustering
        feature_columns = [
            'total_additive_score', 'num_additive_groups', 'mean_group_score',
            'score_std', 'weighted_reliability', 'num_requirement_types',
            'profile_complexity'
        ]
        
        # Handle missing values
        features = requirement_profiles[feature_columns].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to profiles
        requirement_profiles = requirement_profiles.copy()
        requirement_profiles['job_family_cluster'] = cluster_labels
        
        # Analyze clusters
        job_families = {}
        
        for cluster_id in range(n_clusters):
            cluster_data = requirement_profiles[requirement_profiles['job_family_cluster'] == cluster_id]
            
            # Calculate cluster characteristics
            cluster_stats = {
                'cluster_id': cluster_id,
                'num_occupations': len(cluster_data),
                'occupations': [
                    {
                        'soc_code': row[soc_code_column],
                        'occupation': row[occupation_column],
                        'total_score': row['total_additive_score'],
                        'complexity': row['profile_complexity']
                    }
                    for _, row in cluster_data.iterrows()
                ],
                'cluster_characteristics': {
                    'mean_total_score': cluster_data['total_additive_score'].mean(),
                    'mean_num_groups': cluster_data['num_additive_groups'].mean(),
                    'mean_complexity': cluster_data['profile_complexity'].mean(),
                    'mean_reliability': cluster_data['weighted_reliability'].mean(),
                    'score_range': {
                        'min': cluster_data['total_additive_score'].min(),
                        'max': cluster_data['total_additive_score'].max(),
                        'std': cluster_data['total_additive_score'].std()
                    }
                },
                'common_requirements': self._identify_common_requirements(cluster_data),
                'cluster_profile': self._generate_cluster_profile(cluster_data)
            }
            
            job_families[f"family_{cluster_id}"] = cluster_stats
        
        # Add overall clustering metrics
        clustering_metrics = {
            'total_occupations_clustered': len(requirement_profiles),
            'num_job_families': n_clusters,
            'clustering_features': feature_columns,
            'cluster_sizes': [len(requirement_profiles[requirement_profiles['job_family_cluster'] == i]) 
                            for i in range(n_clusters)],
            'silhouette_analysis': self._calculate_silhouette_analysis(features_scaled, cluster_labels)
        }
        
        return {
            'job_families': job_families,
            'clustering_metrics': clustering_metrics,
            'clustered_profiles': requirement_profiles
        }
    
    def _identify_common_requirements(self, cluster_data: pd.DataFrame) -> List[str]:
        """Identify common requirement types within a job family cluster."""
        all_requirements = []
        for req_list in cluster_data['requirement_composition']:
            all_requirements.extend(req_list)
        
        # Count frequency of each requirement type
        from collections import Counter
        req_counts = Counter(all_requirements)
        
        # Return requirements that appear in at least 50% of occupations in the cluster
        threshold = len(cluster_data) * 0.5
        common_requirements = [req for req, count in req_counts.items() if count >= threshold]
        
        return common_requirements
    
    def _generate_cluster_profile(self, cluster_data: pd.DataFrame) -> str:
        """Generate a descriptive profile for a job family cluster."""
        mean_score = cluster_data['total_additive_score'].mean()
        mean_complexity = cluster_data['profile_complexity'].mean()
        mean_groups = cluster_data['num_additive_groups'].mean()
        
        if mean_score > cluster_data['total_additive_score'].quantile(0.75):
            score_level = "High"
        elif mean_score > cluster_data['total_additive_score'].quantile(0.5):
            score_level = "Moderate"
        else:
            score_level = "Low"
        
        if mean_complexity > cluster_data['profile_complexity'].quantile(0.75):
            complexity_level = "High"
        elif mean_complexity > cluster_data['profile_complexity'].quantile(0.5):
            complexity_level = "Moderate"
        else:
            complexity_level = "Low"
        
        return f"{score_level} requirement intensity, {complexity_level} complexity profile"
    
    def _simple_quantile_clustering(
        self, 
        requirement_profiles: pd.DataFrame, 
        n_clusters: int,
        soc_code_column: str,
        occupation_column: str
    ) -> Dict[str, Any]:
        """Simple quantile-based clustering when sklearn is not available."""
        # Use total_additive_score for simple clustering
        scores = requirement_profiles['total_additive_score']
        
        # Create quantile-based clusters
        quantiles = np.linspace(0, 1, n_clusters + 1)
        cluster_labels = pd.cut(scores, bins=scores.quantile(quantiles), labels=False, include_lowest=True)
        
        # Handle NaN labels (can occur with duplicate values)
        cluster_labels = cluster_labels.fillna(0).astype(int)
        
        requirement_profiles = requirement_profiles.copy()
        requirement_profiles['job_family_cluster'] = cluster_labels
        
        # Analyze clusters (simplified version)
        job_families = {}
        
        for cluster_id in range(n_clusters):
            cluster_data = requirement_profiles[requirement_profiles['job_family_cluster'] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
                
            cluster_stats = {
                'cluster_id': cluster_id,
                'num_occupations': len(cluster_data),
                'occupations': [
                    {
                        'soc_code': row[soc_code_column],
                        'occupation': row[occupation_column],
                        'total_score': row['total_additive_score'],
                        'complexity': row['profile_complexity']
                    }
                    for _, row in cluster_data.iterrows()
                ],
                'cluster_characteristics': {
                    'mean_total_score': cluster_data['total_additive_score'].mean(),
                    'mean_num_groups': cluster_data['num_additive_groups'].mean(),
                    'mean_complexity': cluster_data['profile_complexity'].mean(),
                    'mean_reliability': cluster_data['weighted_reliability'].mean(),
                    'score_range': {
                        'min': cluster_data['total_additive_score'].min(),
                        'max': cluster_data['total_additive_score'].max(),
                        'std': cluster_data['total_additive_score'].std()
                    }
                },
                'common_requirements': self._identify_common_requirements(cluster_data),
                'cluster_profile': self._generate_cluster_profile(cluster_data)
            }
            
            job_families[f"family_{cluster_id}"] = cluster_stats
        
        clustering_metrics = {
            'total_occupations_clustered': len(requirement_profiles),
            'num_job_families': len(job_families),
            'clustering_method': 'quantile-based',
            'cluster_sizes': [len(requirement_profiles[requirement_profiles['job_family_cluster'] == i]) 
                            for i in range(n_clusters)],
            'note': 'Simple quantile-based clustering used (sklearn not available)'
        }
        
        return {
            'job_families': job_families,
            'clustering_metrics': clustering_metrics,
            'clustered_profiles': requirement_profiles
        }

    def _calculate_silhouette_analysis(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate silhouette analysis metrics for clustering quality."""
        if not SKLEARN_AVAILABLE:
            return {'average_silhouette_score': 0.0, 'note': 'Silhouette analysis requires scikit-learn'}
            
        try:
            from sklearn.metrics import silhouette_score, silhouette_samples
            
            silhouette_avg = silhouette_score(features, labels)
            silhouette_samples_scores = silhouette_samples(features, labels)
            
            return {
                'average_silhouette_score': silhouette_avg,
                'min_silhouette_score': silhouette_samples_scores.min(),
                'max_silhouette_score': silhouette_samples_scores.max(),
                'std_silhouette_score': silhouette_samples_scores.std()
            }
        except ImportError:
            return {'average_silhouette_score': 0.0, 'note': 'Silhouette analysis requires scikit-learn'}
    
    def analyze_additive_patterns(
        self, 
        grouped_data: pd.DataFrame,
        additive_group_column: str = 'additive_group'
    ) -> Dict[str, Any]:
        """
        Analyze patterns in additive estimate groupings.
        
        Args:
            grouped_data: DataFrame with grouped additive estimates
            additive_group_column: Name of the additive group column
            
        Returns:
            Dictionary with additive pattern analysis
        """
        if grouped_data.empty:
            return {}
        
        patterns = {}
        
        # Analyze each additive group
        for group_code in grouped_data[additive_group_column].unique():
            group_data = grouped_data[grouped_data[additive_group_column] == group_code]
            
            patterns[group_code] = {
                'num_occupations': len(group_data),
                'mean_estimate': group_data['sum_estimate'].mean(),
                'median_estimate': group_data['sum_estimate'].median(),
                'std_estimate': group_data['sum_estimate'].std(),
                'min_estimate': group_data['sum_estimate'].min(),
                'max_estimate': group_data['sum_estimate'].max(),
                'mean_reliability': group_data['reliability_score'].mean(),
                'requirement_types': self._get_unique_requirements(group_data),
                'top_occupations': [
                    {
                        'soc_code': row['soc_code'],
                        'occupation': row['occupation'],
                        'estimate': row['sum_estimate'],
                        'reliability': row['reliability_score']
                    }
                    for _, row in group_data.nlargest(5, 'sum_estimate').iterrows()
                ]
            }
        
        # Overall pattern statistics
        overall_stats = {
            'total_additive_groups': len(patterns),
            'mean_occupations_per_group': np.mean([p['num_occupations'] for p in patterns.values()]),
            'most_common_group': max(patterns.keys(), key=lambda x: patterns[x]['num_occupations']),
            'highest_estimate_group': max(patterns.keys(), key=lambda x: patterns[x]['mean_estimate']),
            'group_size_distribution': {
                'small_groups': len([p for p in patterns.values() if p['num_occupations'] <= 5]),
                'medium_groups': len([p for p in patterns.values() if 5 < p['num_occupations'] <= 20]),
                'large_groups': len([p for p in patterns.values() if p['num_occupations'] > 20])
            }
        }
        
        return {
            'additive_group_patterns': patterns,
            'overall_statistics': overall_stats
        }
    
    def _get_unique_requirements(self, group_data: pd.DataFrame) -> List[str]:
        """Get unique requirement types for an additive group."""
        all_requirements = []
        for req_list in group_data['<lambda>_requirement_type']:
            all_requirements.extend(req_list)
        return list(set(all_requirements))
    
    def create_requirement_composition_matrix(
        self, 
        requirement_profiles: pd.DataFrame,
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation'
    ) -> pd.DataFrame:
        """
        Create a matrix showing requirement composition across occupations.
        
        Args:
            requirement_profiles: DataFrame with comprehensive requirement profiles
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            
        Returns:
            DataFrame matrix with occupations as rows and requirement types as columns
        """
        if requirement_profiles.empty:
            return pd.DataFrame()
        
        # Get all unique requirement types
        all_requirements = set()
        for req_list in requirement_profiles['requirement_composition']:
            all_requirements.update(req_list)
        
        all_requirements = sorted(list(all_requirements))
        
        # Create matrix
        matrix_data = []
        
        for _, row in requirement_profiles.iterrows():
            occupation_id = f"{row[soc_code_column]} - {row[occupation_column]}"
            occupation_requirements = row['requirement_composition']
            
            # Create binary vector for requirement presence
            requirement_vector = [1 if req in occupation_requirements else 0 for req in all_requirements]
            
            matrix_row = [occupation_id] + requirement_vector + [
                row['total_additive_score'],
                row['profile_complexity'],
                row['weighted_reliability']
            ]
            
            matrix_data.append(matrix_row)
        
        # Create DataFrame
        columns = ['occupation'] + all_requirements + ['total_score', 'complexity', 'reliability']
        composition_matrix = pd.DataFrame(matrix_data, columns=columns)
        
        return composition_matrix
    
    def generate_additive_analysis_summary(
        self, 
        data: pd.DataFrame,
        additive_code_column: str = 'additive_code',
        additive_column: str = 'additive',
        soc_code_column: str = 'soc_code',
        occupation_column: str = 'occupation',
        estimate_column: str = 'estimate',
        std_error_column: str = 'standard_error'
    ) -> Dict[str, Any]:
        """
        Generate comprehensive additive category analysis summary.
        
        Args:
            data: DataFrame containing occupation data
            additive_code_column: Name of the additive code column
            additive_column: Name of the additive description column
            soc_code_column: Name of the SOC code column
            occupation_column: Name of the occupation column
            estimate_column: Name of the estimate column
            std_error_column: Name of the standard error column
            
        Returns:
            Dictionary containing comprehensive additive analysis
        """
        # Identify additive estimates
        additive_data = self.identify_additive_estimates(
            data, additive_code_column, additive_column
        )
        
        if additive_data.empty:
            return {
                'overview': {
                    'total_additive_records': 0,
                    'additive_occupations': 0,
                    'additive_groups_found': 0
                },
                'message': 'No additive estimate data found in dataset'
            }
        
        # Group additive estimates
        grouped_data = self.group_additive_estimates(
            additive_data, soc_code_column, occupation_column,
            estimate_column, std_error_column
        )
        
        # Calculate comprehensive requirement profiles
        requirement_profiles = self.calculate_comprehensive_requirement_profiles(
            grouped_data, soc_code_column, occupation_column
        )
        
        # Classify job families
        job_family_classification = self.classify_job_families(
            requirement_profiles, soc_code_column=soc_code_column, 
            occupation_column=occupation_column
        )
        
        # Analyze additive patterns
        additive_patterns = self.analyze_additive_patterns(grouped_data)
        
        # Create requirement composition matrix
        composition_matrix = self.create_requirement_composition_matrix(
            requirement_profiles, soc_code_column, occupation_column
        )
        
        # Overall statistics
        overview = {
            'total_additive_records': len(additive_data),
            'additive_occupations': additive_data[soc_code_column].nunique(),
            'additive_groups_found': additive_data['additive_group'].nunique(),
            'mean_additive_score': grouped_data['sum_estimate'].mean() if not grouped_data.empty else 0,
            'max_additive_score': grouped_data['sum_estimate'].max() if not grouped_data.empty else 0,
            'mean_reliability': grouped_data['reliability_score'].mean() if not grouped_data.empty else 0,
            'occupations_with_profiles': len(requirement_profiles)
        }
        
        return {
            'overview': overview,
            'requirement_profiles': {
                'total_profiles': len(requirement_profiles),
                'top_complex_occupations': [
                    {
                        'soc_code': row[soc_code_column],
                        'occupation': row[occupation_column],
                        'total_score': row['total_additive_score'],
                        'complexity': row['profile_complexity'],
                        'num_groups': row['num_additive_groups']
                    }
                    for _, row in requirement_profiles.head(10).iterrows()
                ] if not requirement_profiles.empty else [],
                'complexity_distribution': {
                    'mean_complexity': requirement_profiles['profile_complexity'].mean() if not requirement_profiles.empty else 0,
                    'max_complexity': requirement_profiles['profile_complexity'].max() if not requirement_profiles.empty else 0,
                    'complexity_quartiles': {
                        'q25': requirement_profiles['profile_complexity'].quantile(0.25) if not requirement_profiles.empty else 0,
                        'q50': requirement_profiles['profile_complexity'].quantile(0.50) if not requirement_profiles.empty else 0,
                        'q75': requirement_profiles['profile_complexity'].quantile(0.75) if not requirement_profiles.empty else 0
                    }
                }
            },
            'job_family_classification': job_family_classification,
            'additive_patterns': additive_patterns,
            'composition_matrix_shape': {
                'occupations': len(composition_matrix),
                'requirement_types': len(composition_matrix.columns) - 4 if not composition_matrix.empty else 0  # Exclude occupation, total_score, complexity, reliability
            }
        }