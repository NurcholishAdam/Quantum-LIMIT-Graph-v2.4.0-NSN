# -*- coding: utf-8 -*-
"""
Contributor-Aware Rank Feedback Loop
Recommend optimal ranks based on contributor history and efficiency

Based on:
    Zhang, Y., et al. (2024). "Deep Hierarchical Learning with Nested Subspace Networks."
    arXiv preprint. NSN framework for hierarchical representation learning.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SubmissionRecord:
    """Record of a contributor submission"""
    contributor_id: str
    language: str
    rank: int
    accuracy: float
    flops: float
    uncertainty: float
    timestamp: str
    efficiency: float  # accuracy / flops


@dataclass
class RankRecommendation:
    """Rank recommendation for contributor"""
    contributor_id: str
    recommended_rank: int
    confidence: float
    rationale: str
    unexplored_pairs: List[Tuple[int, str]]  # (rank, language) pairs
    efficiency_prediction: float
    personalized_badge: str


class RankFeedbackGenerator:
    """
    Recommend optimal ranks based on contributor history and efficiency.
    
    Leaderboard Extension:
    - Personalized rank badges
    - Suggestion panel for unexplored rank-language pairs
    """
    
    def __init__(self):
        self.submission_history: Dict[str, List[SubmissionRecord]] = {}
        self.rank_options = [8, 16, 32, 64, 128, 256]
        self.language_options = [
            'english', 'chinese', 'spanish', 'french', 'german',
            'russian', 'arabic', 'japanese', 'korean', 'portuguese',
            'indonesian', 'vietnamese', 'thai', 'swahili', 'yoruba'
        ]
        
    def record_submission(
        self,
        contributor_id: str,
        language: str,
        rank: int,
        accuracy: float,
        flops: float,
        uncertainty: float,
        timestamp: str = None
    ):
        """Record a contributor submission"""
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
        
        efficiency = accuracy / flops if flops > 0 else 0.0
        
        record = SubmissionRecord(
            contributor_id=contributor_id,
            language=language,
            rank=rank,
            accuracy=accuracy,
            flops=flops,
            uncertainty=uncertainty,
            timestamp=timestamp,
            efficiency=efficiency
        )
        
        if contributor_id not in self.submission_history:
            self.submission_history[contributor_id] = []
        
        self.submission_history[contributor_id].append(record)
        logger.info(
            f"Recorded submission: {contributor_id} - {language} @ rank {rank} "
            f"(accuracy: {accuracy:.3f}, efficiency: {efficiency:.2e})"
        )
    
    def recommend_rank(
        self,
        contributor_id: str,
        target_language: Optional[str] = None
    ) -> RankRecommendation:
        """
        Recommend optimal rank based on contributor history.
        
        Args:
            contributor_id: Contributor identifier
            target_language: Optional target language for recommendation
            
        Returns:
            RankRecommendation with personalized suggestions
        """
        submissions = self.submission_history.get(contributor_id, [])
        
        if not submissions:
            # New contributor: recommend starting rank
            return RankRecommendation(
                contributor_id=contributor_id,
                recommended_rank=32,
                confidence=0.5,
                rationale="Starting recommendation for new contributor",
                unexplored_pairs=self._get_unexplored_pairs(contributor_id),
                efficiency_prediction=0.0,
                personalized_badge="🌟 Newcomer"
            )
        
        # Analyze submission history
        if target_language:
            # Language-specific recommendation
            lang_submissions = [s for s in submissions if s.language == target_language]
            if lang_submissions:
                return self._recommend_from_history(
                    contributor_id, lang_submissions, target_language
                )
        
        # General recommendation based on all submissions
        return self._recommend_from_history(contributor_id, submissions)
    
    def _recommend_from_history(
        self,
        contributor_id: str,
        submissions: List[SubmissionRecord],
        target_language: Optional[str] = None
    ) -> RankRecommendation:
        """Generate recommendation from submission history"""
        # Find best efficiency rank
        best_submission = max(submissions, key=lambda s: s.efficiency)
        
        # Analyze rank performance
        rank_performance = self._analyze_rank_performance(submissions)
        
        # Find optimal rank
        recommended_rank = self._select_optimal_rank(rank_performance)
        
        # Compute confidence
        confidence = self._compute_recommendation_confidence(
            submissions, recommended_rank
        )
        
        # Generate rationale
        rationale = self._generate_rationale(
            submissions, recommended_rank, best_submission
        )
        
        # Find unexplored pairs
        unexplored = self._get_unexplored_pairs(contributor_id)
        
        # Predict efficiency
        efficiency_prediction = self._predict_efficiency(
            submissions, recommended_rank
        )
        
        # Assign badge
        badge = self._assign_badge(submissions)
        
        return RankRecommendation(
            contributor_id=contributor_id,
            recommended_rank=recommended_rank,
            confidence=confidence,
            rationale=rationale,
            unexplored_pairs=unexplored[:5],  # Top 5 suggestions
            efficiency_prediction=efficiency_prediction,
            personalized_badge=badge
        )
    
    def _analyze_rank_performance(
        self, submissions: List[SubmissionRecord]
    ) -> Dict[int, Dict[str, float]]:
        """Analyze performance at each rank"""
        rank_stats = {}
        
        for rank in self.rank_options:
            rank_subs = [s for s in submissions if s.rank == rank]
            
            if rank_subs:
                rank_stats[rank] = {
                    'avg_accuracy': np.mean([s.accuracy for s in rank_subs]),
                    'avg_efficiency': np.mean([s.efficiency for s in rank_subs]),
                    'avg_uncertainty': np.mean([s.uncertainty for s in rank_subs]),
                    'count': len(rank_subs)
                }
            else:
                rank_stats[rank] = {
                    'avg_accuracy': 0.0,
                    'avg_efficiency': 0.0,
                    'avg_uncertainty': 1.0,
                    'count': 0
                }
        
        return rank_stats
    
    def _select_optimal_rank(
        self, rank_performance: Dict[int, Dict[str, float]]
    ) -> int:
        """Select optimal rank based on performance"""
        # Score each rank by efficiency and accuracy
        scores = {}
        
        for rank, stats in rank_performance.items():
            if stats['count'] == 0:
                scores[rank] = 0.0
            else:
                # Weighted score: 60% efficiency, 40% accuracy
                scores[rank] = (
                    0.6 * stats['avg_efficiency'] * 1e8 +  # Scale efficiency
                    0.4 * stats['avg_accuracy']
                )
        
        # Return rank with highest score
        if not scores or max(scores.values()) == 0:
            return 32  # Default
        
        return max(scores, key=scores.get)
    
    def _compute_recommendation_confidence(
        self, submissions: List[SubmissionRecord], recommended_rank: int
    ) -> float:
        """Compute confidence in recommendation"""
        # Confidence based on:
        # - Number of submissions at recommended rank
        # - Consistency of performance
        # - Total submission count
        
        rank_subs = [s for s in submissions if s.rank == recommended_rank]
        
        if not rank_subs:
            return 0.3  # Low confidence for untested rank
        
        # Sample size factor
        sample_factor = min(len(rank_subs) / 10.0, 1.0)
        
        # Consistency factor (low variance in efficiency)
        efficiencies = [s.efficiency for s in rank_subs]
        if len(efficiencies) > 1:
            consistency = 1.0 - min(np.std(efficiencies) / np.mean(efficiencies), 1.0)
        else:
            consistency = 0.5
        
        # Experience factor
        experience = min(len(submissions) / 20.0, 1.0)
        
        confidence = 0.4 * sample_factor + 0.3 * consistency + 0.3 * experience
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _generate_rationale(
        self,
        submissions: List[SubmissionRecord],
        recommended_rank: int,
        best_submission: SubmissionRecord
    ) -> str:
        """Generate human-readable rationale"""
        rank_subs = [s for s in submissions if s.rank == recommended_rank]
        
        if not rank_subs:
            return (
                f"Rank {recommended_rank} recommended based on interpolation "
                f"from your best performance at rank {best_submission.rank} "
                f"(efficiency: {best_submission.efficiency:.2e})"
            )
        
        avg_accuracy = np.mean([s.accuracy for s in rank_subs])
        avg_efficiency = np.mean([s.efficiency for s in rank_subs])
        
        return (
            f"Rank {recommended_rank} shows best efficiency ({avg_efficiency:.2e}) "
            f"with {len(rank_subs)} submissions averaging {avg_accuracy:.3f} accuracy. "
            f"This balances compute cost and performance for your editing style."
        )
    
    def _get_unexplored_pairs(
        self, contributor_id: str
    ) -> List[Tuple[int, str]]:
        """Get unexplored rank-language pairs"""
        submissions = self.submission_history.get(contributor_id, [])
        
        explored = set((s.rank, s.language) for s in submissions)
        
        all_pairs = [
            (rank, lang)
            for rank in self.rank_options
            for lang in self.language_options
        ]
        
        unexplored = [pair for pair in all_pairs if pair not in explored]
        
        # Prioritize by potential value
        # Prefer: medium ranks, diverse languages
        def priority_score(pair):
            rank, lang = pair
            rank_score = 1.0 - abs(rank - 64) / 128.0  # Prefer rank 64
            
            # Prefer low-resource languages (more impact)
            low_resource = ['indonesian', 'vietnamese', 'thai', 'swahili', 'yoruba']
            lang_score = 1.5 if lang in low_resource else 1.0
            
            return rank_score * lang_score
        
        unexplored.sort(key=priority_score, reverse=True)
        
        return unexplored
    
    def _predict_efficiency(
        self, submissions: List[SubmissionRecord], rank: int
    ) -> float:
        """Predict efficiency at given rank"""
        # Simple linear interpolation from existing data
        rank_subs = [s for s in submissions if s.rank == rank]
        
        if rank_subs:
            return np.mean([s.efficiency for s in rank_subs])
        
        # Interpolate from nearby ranks
        nearby_ranks = sorted([s.rank for s in submissions])
        
        if not nearby_ranks:
            return 0.0
        
        # Find closest ranks
        lower = [r for r in nearby_ranks if r < rank]
        upper = [r for r in nearby_ranks if r > rank]
        
        if lower and upper:
            lower_rank = max(lower)
            upper_rank = min(upper)
            
            lower_eff = np.mean([
                s.efficiency for s in submissions if s.rank == lower_rank
            ])
            upper_eff = np.mean([
                s.efficiency for s in submissions if s.rank == upper_rank
            ])
            
            # Linear interpolation
            weight = (rank - lower_rank) / (upper_rank - lower_rank)
            return lower_eff * (1 - weight) + upper_eff * weight
        
        # Use closest available rank
        closest_rank = min(nearby_ranks, key=lambda r: abs(r - rank))
        return np.mean([s.efficiency for s in submissions if s.rank == closest_rank])
    
    def _assign_badge(self, submissions: List[SubmissionRecord]) -> str:
        """Assign personalized badge based on performance"""
        if not submissions:
            return "🌟 Newcomer"
        
        # Analyze submission characteristics
        total_subs = len(submissions)
        unique_langs = len(set(s.language for s in submissions))
        unique_ranks = len(set(s.rank for s in submissions))
        avg_accuracy = np.mean([s.accuracy for s in submissions])
        avg_efficiency = np.mean([s.efficiency for s in submissions])
        
        # Badge criteria
        if total_subs >= 50 and unique_langs >= 10:
            return "🏆 Master Contributor"
        elif avg_efficiency > 1e-7:
            return "⚡ Efficiency Expert"
        elif avg_accuracy > 0.95:
            return "🎯 Accuracy Champion"
        elif unique_ranks >= 5:
            return "🔬 Rank Explorer"
        elif unique_langs >= 8:
            return "🌍 Multilingual Specialist"
        elif total_subs >= 20:
            return "💪 Active Contributor"
        elif total_subs >= 10:
            return "📈 Rising Star"
        else:
            return "🚀 Getting Started"
    
    def generate_feedback_panel(
        self, contributor_id: str
    ) -> Dict[str, any]:
        """
        Generate comprehensive feedback panel for dashboard.
        
        Returns:
            Dict with recommendations, stats, and suggestions
        """
        submissions = self.submission_history.get(contributor_id, [])
        recommendation = self.recommend_rank(contributor_id)
        
        if not submissions:
            return {
                'recommendation': recommendation,
                'stats': {},
                'suggestions': [
                    "Start with rank 32 for balanced performance",
                    "Try high-resource languages (English, Chinese) first",
                    "Focus on accuracy before optimizing efficiency"
                ]
            }
        
        # Compute statistics
        stats = {
            'total_submissions': len(submissions),
            'unique_languages': len(set(s.language for s in submissions)),
            'unique_ranks': len(set(s.rank for s in submissions)),
            'avg_accuracy': float(np.mean([s.accuracy for s in submissions])),
            'avg_efficiency': float(np.mean([s.efficiency for s in submissions])),
            'best_accuracy': float(max(s.accuracy for s in submissions)),
            'best_efficiency': float(max(s.efficiency for s in submissions))
        }
        
        # Generate suggestions
        suggestions = self._generate_suggestions(submissions, recommendation)
        
        return {
            'recommendation': recommendation,
            'stats': stats,
            'suggestions': suggestions
        }

    
    def _generate_suggestions(
        self,
        submissions: List[SubmissionRecord],
        recommendation: RankRecommendation
    ) -> List[str]:
        """Generate actionable suggestions"""
        suggestions = []
        
        # Analyze gaps
        tested_ranks = set(s.rank for s in submissions)
        tested_langs = set(s.language for s in submissions)
        
        # Rank diversity
        if len(tested_ranks) < 3:
            suggestions.append(
                f"Try exploring more ranks - you've only tested {len(tested_ranks)} so far"
            )
        
        # Language diversity
        low_resource = ['indonesian', 'vietnamese', 'thai', 'swahili', 'yoruba']
        tested_low_resource = [l for l in tested_langs if l in low_resource]
        
        if len(tested_low_resource) < 2:
            suggestions.append(
                "Consider testing low-resource languages for higher impact"
            )
        
        # Efficiency optimization
        avg_efficiency = np.mean([s.efficiency for s in submissions])
        if avg_efficiency < 5e-8:
            suggestions.append(
                "Focus on efficiency - try lower ranks to reduce FLOPs"
            )
        
        # Accuracy improvement
        avg_accuracy = np.mean([s.accuracy for s in submissions])
        if avg_accuracy < 0.85:
            suggestions.append(
                "Accuracy could be improved - try higher ranks or refine your edits"
            )
        
        # Unexplored pairs
        if recommendation.unexplored_pairs:
            top_pair = recommendation.unexplored_pairs[0]
            suggestions.append(
                f"High-value opportunity: Try rank {top_pair[0]} with {top_pair[1]}"
            )
        
        return suggestions[:5]  # Top 5 suggestions
