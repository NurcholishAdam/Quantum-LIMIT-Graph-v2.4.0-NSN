# -*- coding: utf-8 -*-
"""
Cross-Lingual Edit Propagation via Subspace Containment
Transfer high-resource corrections to low-resource languages using containment scores

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
class ContainmentScore:
    """Subspace containment analysis result"""
    source_lang: str
    target_lang: str
    rank: int
    containment_score: float  # 0-1, how much target is contained in source
    overlap_dimension: int  # Dimension of overlap
    confidence: float
    propagation_recommended: bool


@dataclass
class PropagationResult:
    """Result of edit propagation"""
    source_lang: str
    target_lang: str
    rank: int
    edit_vector: np.ndarray
    propagated_vector: np.ndarray
    containment_score: float
    success: bool
    quality_score: float  # Predicted quality after propagation
    propagation_path: List[str]  # Languages in propagation chain


class EditPropagationEngine:
    """
    Transfer edits from high-resource to low-resource languages using
    subspace containment analysis.
    
    Dashboard Extension:
    - Heatmap of containment scores across language pairs
    - Flow arrows showing edit propagation paths
    """
    
    def __init__(self):
        self.language_embeddings = self._initialize_language_embeddings()
        self.containment_cache: Dict[Tuple[str, str, int], ContainmentScore] = {}
        self.propagation_history: List[PropagationResult] = []
        
    def _initialize_language_embeddings(self) -> Dict[str, np.ndarray]:
        """Initialize language subspace embeddings"""
        # Simulated language embeddings (in practice, learned from data)
        np.random.seed(42)
        
        languages = {
            # High-resource languages (larger subspaces)
            'english': np.random.randn(256),
            'chinese': np.random.randn(256),
            'spanish': np.random.randn(256),
            'french': np.random.randn(256),
            'german': np.random.randn(256),
            
            # Medium-resource languages
            'russian': np.random.randn(256),
            'arabic': np.random.randn(256),
            'japanese': np.random.randn(256),
            'korean': np.random.randn(256),
            'portuguese': np.random.randn(256),
            
            # Low-resource languages (smaller subspaces)
            'indonesian': np.random.randn(256),
            'vietnamese': np.random.randn(256),
            'thai': np.random.randn(256),
            'swahili': np.random.randn(256),
            'yoruba': np.random.randn(256)
        }
        
        # Normalize embeddings
        for lang in languages:
            languages[lang] = languages[lang] / np.linalg.norm(languages[lang])
        
        return languages
    
    def evaluate_subspace_containment(
        self,
        source_lang: str,
        target_lang: str,
        rank: int
    ) -> ContainmentScore:
        """
        Evaluate how much target language subspace is contained in source.
        
        Args:
            source_lang: High-resource source language
            target_lang: Low-resource target language
            rank: NSN rank for analysis
            
        Returns:
            ContainmentScore with containment metrics
        """
        cache_key = (source_lang, target_lang, rank)
        if cache_key in self.containment_cache:
            return self.containment_cache[cache_key]
        
        # Get language embeddings
        source_emb = self.language_embeddings.get(source_lang)
        target_emb = self.language_embeddings.get(target_lang)
        
        if source_emb is None or target_emb is None:
            logger.warning(f"Unknown language: {source_lang} or {target_lang}")
            return ContainmentScore(
                source_lang=source_lang,
                target_lang=target_lang,
                rank=rank,
                containment_score=0.0,
                overlap_dimension=0,
                confidence=0.0,
                propagation_recommended=False
            )
        
        # Compute containment via projection
        # Truncate to rank dimension
        source_subspace = source_emb[:rank]
        target_subspace = target_emb[:rank]
        
        # Containment score: cosine similarity in rank-dimensional subspace
        containment = float(np.dot(source_subspace, target_subspace))
        containment = (containment + 1.0) / 2.0  # Normalize to [0, 1]
        
        # Overlap dimension: effective rank of shared subspace
        overlap_dim = int(rank * containment)
        
        # Confidence based on rank and language resource levels
        confidence = self._compute_containment_confidence(
            source_lang, target_lang, rank, containment
        )
        
        # Recommend propagation if containment > 0.75 and confidence > 0.7
        propagation_recommended = containment > 0.75 and confidence > 0.7
        
        result = ContainmentScore(
            source_lang=source_lang,
            target_lang=target_lang,
            rank=rank,
            containment_score=containment,
            overlap_dimension=overlap_dim,
            confidence=confidence,
            propagation_recommended=propagation_recommended
        )
        
        self.containment_cache[cache_key] = result
        return result
    
    def _compute_containment_confidence(
        self,
        source_lang: str,
        target_lang: str,
        rank: int,
        containment: float
    ) -> float:
        """Compute confidence in containment score"""
        # Higher confidence for:
        # - Higher ranks (more dimensions to analyze)
        # - Higher containment scores
        # - Related language families
        
        rank_factor = min(rank / 128.0, 1.0)
        containment_factor = containment
        
        # Language family bonus (simplified)
        family_bonus = 0.0
        if (source_lang in ['english', 'german', 'french', 'spanish'] and
            target_lang in ['english', 'german', 'french', 'spanish']):
            family_bonus = 0.1
        
        confidence = 0.5 * rank_factor + 0.4 * containment_factor + family_bonus
        return float(np.clip(confidence, 0.0, 1.0))
    
    def propagate_edit(
        self,
        source_lang: str,
        target_lang: str,
        rank: int,
        edit_vector: np.ndarray
    ) -> PropagationResult:
        """
        Propagate edit from source to target language.
        
        Args:
            source_lang: Source language
            target_lang: Target language
            rank: NSN rank
            edit_vector: Edit vector in source language
            
        Returns:
            PropagationResult with propagated edit
        """
        # Evaluate containment
        containment = self.evaluate_subspace_containment(
            source_lang, target_lang, rank
        )
        
        if not containment.propagation_recommended:
            logger.warning(
                f"Propagation not recommended: {source_lang} → {target_lang} "
                f"(containment: {containment.containment_score:.3f})"
            )
            
            result = PropagationResult(
                source_lang=source_lang,
                target_lang=target_lang,
                rank=rank,
                edit_vector=edit_vector,
                propagated_vector=np.zeros_like(edit_vector),
                containment_score=containment.containment_score,
                success=False,
                quality_score=0.0,
                propagation_path=[source_lang, target_lang]
            )
            
            self.propagation_history.append(result)
            return result
        
        # Propagate edit via subspace projection
        propagated_vector = self._transfer_edit(
            edit_vector, source_lang, target_lang, rank
        )
        
        # Compute quality score
        quality_score = self._compute_propagation_quality(
            edit_vector, propagated_vector, containment.containment_score
        )
        
        result = PropagationResult(
            source_lang=source_lang,
            target_lang=target_lang,
            rank=rank,
            edit_vector=edit_vector,
            propagated_vector=propagated_vector,
            containment_score=containment.containment_score,
            success=True,
            quality_score=quality_score,
            propagation_path=[source_lang, target_lang]
        )
        
        self.propagation_history.append(result)
        logger.info(
            f"Propagated edit: {source_lang} → {target_lang} "
            f"(quality: {quality_score:.3f})"
        )
        
        return result
    
    def _transfer_edit(
        self,
        edit_vector: np.ndarray,
        source_lang: str,
        target_lang: str,
        rank: int
    ) -> np.ndarray:
        """Transfer edit vector from source to target language"""
        # Get language embeddings
        source_emb = self.language_embeddings[source_lang]
        target_emb = self.language_embeddings[target_lang]
        
        # Project edit onto shared subspace
        # Simplified: weighted combination based on containment
        source_subspace = source_emb[:rank]
        target_subspace = target_emb[:rank]
        
        # Compute transfer matrix (simplified)
        transfer_weight = np.dot(source_subspace, target_subspace)
        
        # Apply transfer
        propagated = edit_vector * transfer_weight
        
        return propagated
    
    def _compute_propagation_quality(
        self,
        original: np.ndarray,
        propagated: np.ndarray,
        containment: float
    ) -> float:
        """Compute quality of propagated edit"""
        # Quality based on:
        # - Containment score
        # - Vector similarity
        # - Magnitude preservation
        
        if np.linalg.norm(propagated) < 1e-6:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(original, propagated) / (
            np.linalg.norm(original) * np.linalg.norm(propagated)
        )
        similarity = (similarity + 1.0) / 2.0  # Normalize to [0, 1]
        
        # Magnitude preservation
        mag_ratio = np.linalg.norm(propagated) / np.linalg.norm(original)
        mag_score = 1.0 - abs(1.0 - mag_ratio)
        
        # Combined quality
        quality = 0.5 * containment + 0.3 * similarity + 0.2 * mag_score
        
        return float(np.clip(quality, 0.0, 1.0))
    
    def compute_containment_heatmap(
        self,
        languages: List[str],
        rank: int
    ) -> np.ndarray:
        """
        Compute containment heatmap for dashboard visualization.
        
        Args:
            languages: List of languages to analyze
            rank: NSN rank
            
        Returns:
            Heatmap matrix (languages x languages)
        """
        n = len(languages)
        heatmap = np.zeros((n, n))
        
        for i, source in enumerate(languages):
            for j, target in enumerate(languages):
                if i == j:
                    heatmap[i, j] = 1.0
                else:
                    containment = self.evaluate_subspace_containment(
                        source, target, rank
                    )
                    heatmap[i, j] = containment.containment_score
        
        return heatmap
    
    def find_propagation_paths(
        self,
        source_lang: str,
        target_langs: List[str],
        rank: int,
        min_containment: float = 0.75
    ) -> Dict[str, List[str]]:
        """
        Find optimal propagation paths from source to multiple targets.
        
        Returns:
            Dict mapping target language to propagation path
        """
        paths = {}
        
        for target in target_langs:
            # Direct path
            direct_containment = self.evaluate_subspace_containment(
                source_lang, target, rank
            )
            
            if direct_containment.containment_score >= min_containment:
                paths[target] = [source_lang, target]
            else:
                # Try indirect path through intermediate language
                best_path = None
                best_score = 0.0
                
                for intermediate in self.language_embeddings.keys():
                    if intermediate in [source_lang, target]:
                        continue
                    
                    c1 = self.evaluate_subspace_containment(
                        source_lang, intermediate, rank
                    )
                    c2 = self.evaluate_subspace_containment(
                        intermediate, target, rank
                    )
                    
                    combined_score = c1.containment_score * c2.containment_score
                    
                    if combined_score > best_score and combined_score >= min_containment:
                        best_score = combined_score
                        best_path = [source_lang, intermediate, target]
                
                if best_path:
                    paths[target] = best_path
                else:
                    paths[target] = []  # No viable path
        
        return paths
