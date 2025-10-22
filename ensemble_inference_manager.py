# -*- coding: utf-8 -*-
"""
Ensemble Inference Across Backends
Run edits across multiple backends and compute agreement scores

"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BackendResult:
    """Result from a single backend"""
    backend_id: str
    edit_vector: np.ndarray
    output: np.ndarray
    confidence: float
    latency: float  # seconds
    success: bool
    error_message: Optional[str] = None


@dataclass
class EnsembleResult:
    """Result from ensemble inference"""
    edit_vector: np.ndarray
    backend_results: List[BackendResult]
    consensus_output: np.ndarray
    agreement_score: float
    reliability_boost: float
    agreement_matrix: np.ndarray
    best_backend: str
    ensemble_confidence: float


class EnsembleInferenceManager:
    """
    Run edits across multiple quantum backends and compute agreement scores.
    
    Dashboard Extension:
    - Agreement matrix across backends
    - Reliability boost from ensemble consensus
    """
    
    def __init__(self):
        self.backend_configs = self._initialize_backend_configs()
        self.inference_history: List[EnsembleResult] = []
        
    def _initialize_backend_configs(self) -> Dict[str, Dict]:
        """Initialize backend configurations"""
        return {
            'ibm_manila': {
                'qubits': 5,
                'error_rate': 0.08,
                'gate_fidelity': 0.92,
                'coherence_time': 30.0,
                'base_latency': 0.05
            },
            'ibm_washington': {
                'qubits': 127,
                'error_rate': 0.02,
                'gate_fidelity': 0.98,
                'coherence_time': 120.0,
                'base_latency': 0.15
            },
            'russian_simulator': {
                'qubits': 256,
                'error_rate': 0.001,
                'gate_fidelity': 0.999,
                'coherence_time': 1000.0,
                'base_latency': 0.30
            },
            'ibm_kyoto': {
                'qubits': 127,
                'error_rate': 0.025,
                'gate_fidelity': 0.975,
                'coherence_time': 100.0,
                'base_latency': 0.12
            },
            'google_sycamore': {
                'qubits': 53,
                'error_rate': 0.015,
                'gate_fidelity': 0.985,
                'coherence_time': 80.0,
                'base_latency': 0.08
            }
        }
    
    def run_ensemble_inference(
        self,
        edit_vector: np.ndarray,
        backend_list: List[str]
    ) -> EnsembleResult:
        """
        Run inference across multiple backends and compute ensemble result.
        
        Args:
            edit_vector: Edit vector to apply
            backend_list: List of backend IDs (e.g., ['ibm_manila', 'ibm_washington'])
            
        Returns:
            EnsembleResult with consensus and agreement metrics
        """
        # Run inference on each backend
        backend_results = []
        
        for backend_id in backend_list:
            result = self._run_single_backend(backend_id, edit_vector)
            backend_results.append(result)
        
        # Compute agreement matrix
        agreement_matrix = self._compute_agreement_matrix(backend_results)
        
        # Compute consensus output
        consensus_output = self._compute_consensus(backend_results)
        
        # Compute overall agreement score
        agreement_score = self._compute_overall_agreement(agreement_matrix)
        
        # Compute reliability boost
        reliability_boost = self._compute_reliability_boost(
            backend_results, agreement_score
        )
        
        # Find best backend
        best_backend = self._select_best_backend(backend_results)
        
        # Compute ensemble confidence
        ensemble_confidence = self._compute_ensemble_confidence(
            backend_results, agreement_score
        )
        
        result = EnsembleResult(
            edit_vector=edit_vector,
            backend_results=backend_results,
            consensus_output=consensus_output,
            agreement_score=agreement_score,
            reliability_boost=reliability_boost,
            agreement_matrix=agreement_matrix,
            best_backend=best_backend,
            ensemble_confidence=ensemble_confidence
        )
        
        self.inference_history.append(result)
        
        logger.info(
            f"Ensemble inference complete: {len(backend_list)} backends, "
            f"agreement: {agreement_score:.3f}, boost: {reliability_boost:.3f}"
        )
        
        return result
    
    def _run_single_backend(
        self, backend_id: str, edit_vector: np.ndarray
    ) -> BackendResult:
        """Run inference on a single backend"""
        config = self.backend_configs.get(backend_id)
        
        if config is None:
            logger.warning(f"Unknown backend: {backend_id}")
            return BackendResult(
                backend_id=backend_id,
                edit_vector=edit_vector,
                output=np.zeros_like(edit_vector),
                confidence=0.0,
                latency=0.0,
                success=False,
                error_message=f"Unknown backend: {backend_id}"
            )
        
        # Simulate inference with backend-specific noise
        noise_level = config['error_rate']
        noise = np.random.randn(*edit_vector.shape) * noise_level
        
        output = edit_vector + noise
        
        # Confidence based on gate fidelity
        confidence = config['gate_fidelity']
        
        # Latency based on backend and vector size
        latency = config['base_latency'] * (1 + len(edit_vector) / 1000.0)
        
        return BackendResult(
            backend_id=backend_id,
            edit_vector=edit_vector,
            output=output,
            confidence=confidence,
            latency=latency,
            success=True
        )
    
    def _compute_agreement_matrix(
        self, results: List[BackendResult]
    ) -> np.ndarray:
        """Compute pairwise agreement matrix between backends"""
        n = len(results)
        agreement_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    # Cosine similarity between outputs
                    output_i = results[i].output
                    output_j = results[j].output
                    
                    if np.linalg.norm(output_i) < 1e-6 or np.linalg.norm(output_j) < 1e-6:
                        agreement_matrix[i, j] = 0.0
                    else:
                        similarity = np.dot(output_i, output_j) / (
                            np.linalg.norm(output_i) * np.linalg.norm(output_j)
                        )
                        # Normalize to [0, 1]
                        agreement_matrix[i, j] = (similarity + 1.0) / 2.0
        
        return agreement_matrix
    
    def _compute_consensus(
        self, results: List[BackendResult]
    ) -> np.ndarray:
        """Compute consensus output from all backends"""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return np.zeros_like(results[0].edit_vector)
        
        # Weighted average by confidence
        total_confidence = sum(r.confidence for r in successful_results)
        
        if total_confidence < 1e-6:
            # Unweighted average
            outputs = [r.output for r in successful_results]
            return np.mean(outputs, axis=0)
        
        # Confidence-weighted average
        consensus = np.zeros_like(successful_results[0].output)
        
        for result in successful_results:
            weight = result.confidence / total_confidence
            consensus += weight * result.output
        
        return consensus
    
    def _compute_overall_agreement(self, agreement_matrix: np.ndarray) -> float:
        """Compute overall agreement score from matrix"""
        # Average of off-diagonal elements
        n = agreement_matrix.shape[0]
        
        if n <= 1:
            return 1.0
        
        # Sum off-diagonal elements
        total = 0.0
        count = 0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    total += agreement_matrix[i, j]
                    count += 1
        
        return total / count if count > 0 else 0.0
    
    def _compute_reliability_boost(
        self, results: List[BackendResult], agreement_score: float
    ) -> float:
        """
        Compute reliability boost from ensemble consensus.
        
        Boost is higher when:
        - More backends agree
        - Individual backends have high confidence
        - Agreement score is high
        """
        if not results:
            return 0.0
        
        # Average individual confidence
        avg_confidence = np.mean([r.confidence for r in results if r.success])
        
        # Ensemble size factor
        ensemble_factor = min(len(results) / 5.0, 1.0)
        
        # Boost formula
        boost = (
            0.4 * agreement_score +
            0.3 * avg_confidence +
            0.3 * ensemble_factor
        )
        
        return float(np.clip(boost, 0.0, 1.0))
    
    def _select_best_backend(self, results: List[BackendResult]) -> str:
        """Select best backend based on confidence and success"""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return results[0].backend_id if results else "none"
        
        # Score by confidence and inverse latency
        scores = {}
        
        for result in successful_results:
            scores[result.backend_id] = (
                0.7 * result.confidence +
                0.3 * (1.0 / (1.0 + result.latency))
            )
        
        return max(scores, key=scores.get)
    
    def _compute_ensemble_confidence(
        self, results: List[BackendResult], agreement_score: float
    ) -> float:
        """Compute overall ensemble confidence"""
        if not results:
            return 0.0
        
        # Combine individual confidences with agreement
        avg_confidence = np.mean([r.confidence for r in results if r.success])
        
        # Ensemble confidence is boosted by agreement
        ensemble_confidence = 0.6 * avg_confidence + 0.4 * agreement_score
        
        return float(np.clip(ensemble_confidence, 0.0, 1.0))
    
    def compare_backends(
        self, edit_vectors: List[np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare all backends across multiple edit vectors.
        
        Returns:
            Dict mapping backend_id to performance metrics
        """
        backend_stats = {
            backend_id: {
                'avg_confidence': [],
                'avg_latency': [],
                'success_rate': []
            }
            for backend_id in self.backend_configs.keys()
        }
        
        for edit_vector in edit_vectors:
            for backend_id in self.backend_configs.keys():
                result = self._run_single_backend(backend_id, edit_vector)
                
                backend_stats[backend_id]['avg_confidence'].append(result.confidence)
                backend_stats[backend_id]['avg_latency'].append(result.latency)
                backend_stats[backend_id]['success_rate'].append(1.0 if result.success else 0.0)
        
        # Compute averages
        comparison = {}
        
        for backend_id, stats in backend_stats.items():
            comparison[backend_id] = {
                'avg_confidence': float(np.mean(stats['avg_confidence'])),
                'avg_latency': float(np.mean(stats['avg_latency'])),
                'success_rate': float(np.mean(stats['success_rate']))
            }
        
        return comparison
    
    def get_agreement_heatmap(
        self, backend_list: List[str], edit_vector: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get agreement heatmap for visualization.
        
        Returns:
            Tuple of (agreement_matrix, backend_labels)
        """
        result = self.run_ensemble_inference(edit_vector, backend_list)
        return result.agreement_matrix, backend_list
    
    def compute_reliability_metrics(self) -> Dict[str, float]:
        """Compute overall reliability metrics from history"""
        if not self.inference_history:
            return {
                'avg_agreement': 0.0,
                'avg_reliability_boost': 0.0,
                'avg_ensemble_confidence': 0.0
            }
        
        return {
            'avg_agreement': float(np.mean([
                r.agreement_score for r in self.inference_history
            ])),
            'avg_reliability_boost': float(np.mean([
                r.reliability_boost for r in self.inference_history
            ])),
            'avg_ensemble_confidence': float(np.mean([
                r.ensemble_confidence for r in self.inference_history
            ]))
        }
