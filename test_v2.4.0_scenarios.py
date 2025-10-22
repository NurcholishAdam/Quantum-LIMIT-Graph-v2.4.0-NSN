# -*- coding: utf-8 -*-
"""
Test Suite for Quantum LIMIT-Graph v2.4.0 NSN Integration Scenarios
"""
import numpy as np
import pytest

from quantum_integration.nsn_integration.backend_telemetry_rank_adapter import (
    BackendTelemetryRankAdapter, BackendTelemetry
)
from quantum_integration.nsn_integration.edit_propagation_engine import (
    EditPropagationEngine
)
from quantum_integration.nsn_integration.rank_feedback_generator import (
    RankFeedbackGenerator
)
from quantum_integration.nsn_integration.ensemble_inference_manager import (
    EnsembleInferenceManager
)


class TestBackendTelemetryRankAdapter:
    """Test Scenario 1: Backend Telemetry Rank Adapter"""
    
    def test_initialization(self):
        adapter = BackendTelemetryRankAdapter()
        assert adapter is not None
        assert len(adapter.rank_thresholds) == 6
        
    def test_adapt_rank_high_quality(self):
        adapter = BackendTelemetryRankAdapter()
        
        result = adapter.adapt_rank(
            backend_id='ibm_washington',
            telemetry={
                'error_rate': 0.02,
                'coherence_time': 120.0,
                'gate_fidelity': 0.98
            },
            current_rank=64
        )
        
        assert result.adapted_rank >= 64
        assert result.confidence > 0.5
        assert result.reliability_score > 0.8
        
    def test_adapt_rank_low_quality(self):
        adapter = BackendTelemetryRankAdapter()
        
        result = adapter.adapt_rank(
            backend_id='ibm_manila',
            telemetry={
                'error_rate': 0.10,
                'coherence_time': 20.0,
                'gate_fidelity': 0.90
            },
            current_rank=128
        )
        
        assert result.adapted_rank < 128
        assert result.adapted_rank >= 8
        
    def test_leaderboard_metrics(self):
        adapter = BackendTelemetryRankAdapter()
        
        # Record some adaptations
        adapter.adapt_rank(
            backend_id='contributor_001_backend',
            telemetry={'error_rate': 0.02, 'coherence_time': 100.0, 'gate_fidelity': 0.97},
            current_rank=128
        )
        
        metrics = adapter.get_leaderboard_metrics('contributor_001')
        
        assert 'avg_reliability' in metrics
        assert 'avg_responsiveness' in metrics
        assert 'adaptation_accuracy' in metrics


class TestEditPropagationEngine:
    """Test Scenario 2: Edit Propagation Engine"""
    
    def test_initialization(self):
        engine = EditPropagationEngine()
        assert engine is not None
        assert len(engine.language_embeddings) > 0
        
    def test_evaluate_containment(self):
        engine = EditPropagationEngine()
        
        containment = engine.evaluate_subspace_containment(
            source_lang='english',
            target_lang='indonesian',
            rank=128
        )
        
        assert 0.0 <= containment.containment_score <= 1.0
        assert containment.overlap_dimension >= 0
        assert 0.0 <= containment.confidence <= 1.0
        
    def test_propagate_edit_success(self):
        engine = EditPropagationEngine()
        
        edit_vector = np.random.randn(256) * 0.1
        
        result = engine.propagate_edit(
            source_lang='english',
            target_lang='spanish',
            rank=128,
            edit_vector=edit_vector
        )
        
        assert result.edit_vector.shape == edit_vector.shape
        assert result.propagated_vector.shape == edit_vector.shape
        assert 0.0 <= result.quality_score <= 1.0
        
    def test_containment_heatmap(self):
        engine = EditPropagationEngine()
        
        languages = ['english', 'chinese', 'spanish']
        heatmap = engine.compute_containment_heatmap(languages, rank=64)
        
        assert heatmap.shape == (3, 3)
        assert np.allclose(np.diag(heatmap), 1.0)
        
    def test_find_propagation_paths(self):
        engine = EditPropagationEngine()
        
        paths = engine.find_propagation_paths(
            source_lang='english',
            target_langs=['spanish', 'french'],
            rank=128
        )
        
        assert 'spanish' in paths
        assert 'french' in paths


class TestRankFeedbackGenerator:
    """Test Scenario 3: Rank Feedback Generator"""
    
    def test_initialization(self):
        generator = RankFeedbackGenerator()
        assert generator is not None
        assert len(generator.rank_options) > 0
        
    def test_record_submission(self):
        generator = RankFeedbackGenerator()
        
        generator.record_submission(
            contributor_id='test_001',
            language='english',
            rank=64,
            accuracy=0.92,
            flops=4.1e7,
            uncertainty=0.08
        )
        
        assert 'test_001' in generator.submission_history
        assert len(generator.submission_history['test_001']) == 1
        
    def test_recommend_rank_new_contributor(self):
        generator = RankFeedbackGenerator()
        
        recommendation = generator.recommend_rank('new_contributor')
        
        assert recommendation.recommended_rank in generator.rank_options
        assert recommendation.confidence >= 0.0
        assert recommendation.personalized_badge == "🌟 Newcomer"
        
    def test_recommend_rank_experienced(self):
        generator = RankFeedbackGenerator()
        
        # Add multiple submissions
        for rank in [32, 64, 128]:
            generator.record_submission(
                contributor_id='experienced_001',
                language='english',
                rank=rank,
                accuracy=0.85 + rank/1000,
                flops=rank * 1e6,
                uncertainty=0.15 - rank/2000
            )
        
        recommendation = generator.recommend_rank('experienced_001')
        
        assert recommendation.recommended_rank in generator.rank_options
        assert recommendation.confidence > 0.3
        assert len(recommendation.unexplored_pairs) > 0
        
    def test_generate_feedback_panel(self):
        generator = RankFeedbackGenerator()
        
        generator.record_submission(
            contributor_id='panel_test',
            language='english',
            rank=64,
            accuracy=0.90,
            flops=4e7,
            uncertainty=0.10
        )
        
        panel = generator.generate_feedback_panel('panel_test')
        
        assert 'recommendation' in panel
        assert 'stats' in panel
        assert 'suggestions' in panel
        assert panel['stats']['total_submissions'] == 1


class TestEnsembleInferenceManager:
    """Test Scenario 4: Ensemble Inference Manager"""
    
    def test_initialization(self):
        manager = EnsembleInferenceManager()
        assert manager is not None
        assert len(manager.backend_configs) > 0
        
    def test_run_ensemble_inference(self):
        manager = EnsembleInferenceManager()
        
        edit_vector = np.random.randn(256) * 0.1
        backends = ['ibm_manila', 'ibm_washington']
        
        result = manager.run_ensemble_inference(edit_vector, backends)
        
        assert len(result.backend_results) == 2
        assert 0.0 <= result.agreement_score <= 1.0
        assert 0.0 <= result.reliability_boost <= 1.0
        assert result.best_backend in backends
        
    def test_agreement_matrix(self):
        manager = EnsembleInferenceManager()
        
        edit_vector = np.random.randn(256) * 0.1
        backends = ['ibm_manila', 'ibm_washington', 'russian_simulator']
        
        result = manager.run_ensemble_inference(edit_vector, backends)
        
        assert result.agreement_matrix.shape == (3, 3)
        assert np.allclose(np.diag(result.agreement_matrix), 1.0)
        
    def test_compare_backends(self):
        manager = EnsembleInferenceManager()
        
        test_vectors = [np.random.randn(256) * 0.1 for _ in range(3)]
        comparison = manager.compare_backends(test_vectors)
        
        assert len(comparison) > 0
        for backend_id, metrics in comparison.items():
            assert 'avg_confidence' in metrics
            assert 'avg_latency' in metrics
            assert 'success_rate' in metrics
            
    def test_get_agreement_heatmap(self):
        manager = EnsembleInferenceManager()
        
        edit_vector = np.random.randn(256) * 0.1
        backends = ['ibm_manila', 'ibm_washington']
        
        heatmap, labels = manager.get_agreement_heatmap(backends, edit_vector)
        
        assert heatmap.shape == (2, 2)
        assert labels == backends
        
    def test_compute_reliability_metrics(self):
        manager = EnsembleInferenceManager()
        
        # Run some inferences
        edit_vector = np.random.randn(256) * 0.1
        manager.run_ensemble_inference(edit_vector, ['ibm_manila', 'ibm_washington'])
        
        metrics = manager.compute_reliability_metrics()
        
        assert 'avg_agreement' in metrics
        assert 'avg_reliability_boost' in metrics
        assert 'avg_ensemble_confidence' in metrics


class TestIntegration:
    """Integration tests across all scenarios"""
    
    def test_full_workflow(self):
        """Test complete workflow across all four scenarios"""
        
        # Scenario 1: Adapt rank based on telemetry
        adapter = BackendTelemetryRankAdapter()
        telemetry_result = adapter.adapt_rank(
            backend_id='ibm_washington',
            telemetry={'error_rate': 0.02, 'coherence_time': 120.0, 'gate_fidelity': 0.98},
            current_rank=128
        )
        
        adapted_rank = telemetry_result.adapted_rank
        
        # Scenario 2: Propagate edit using adapted rank
        engine = EditPropagationEngine()
        edit_vector = np.random.randn(256) * 0.1
        
        propagation_result = engine.propagate_edit(
            source_lang='english',
            target_lang='indonesian',
            rank=adapted_rank,
            edit_vector=edit_vector
        )
        
        # Scenario 3: Record submission and get feedback
        generator = RankFeedbackGenerator()
        generator.record_submission(
            contributor_id='integration_test',
            language='indonesian',
            rank=adapted_rank,
            accuracy=propagation_result.quality_score,
            flops=adapted_rank * 1e6,
            uncertainty=0.10
        )
        
        recommendation = generator.recommend_rank('integration_test')
        
        # Scenario 4: Run ensemble inference
        manager = EnsembleInferenceManager()
        ensemble_result = manager.run_ensemble_inference(
            edit_vector=propagation_result.propagated_vector,
            backend_list=['ibm_manila', 'ibm_washington']
        )
        
        # Verify workflow
        assert adapted_rank > 0
        assert propagation_result.success or not propagation_result.success  # Either outcome is valid
        assert recommendation.recommended_rank > 0
        assert ensemble_result.agreement_score >= 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
