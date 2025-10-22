# -*- coding: utf-8 -*-
"""
Demo: Quantum LIMIT-Graph v2.4.0 NSN Integration Scenarios

Demonstrates all four modular components:
1. Backend Telemetry Rank Adapter
2. Edit Propagation Engine
3. Rank Feedback Generator
4. Ensemble Inference Manager
"""
import numpy as np
import json
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend_telemetry_rank_adapter import BackendTelemetryRankAdapter
from edit_propagation_engine import EditPropagationEngine
from rank_feedback_generator import RankFeedbackGenerator
from ensemble_inference_manager import EnsembleInferenceManager


def demo_scenario_1_telemetry_adaptation():
    """Scenario 1: Real-Time Backend-Aware Rank Adaptation"""
    print("\n" + "="*80)
    print("SCENARIO 1: Real-Time Backend-Aware Rank Adaptation")
    print("="*80)
    
    adapter = BackendTelemetryRankAdapter()
    
    # Test different backend conditions
    test_cases = [
        {
            'backend_id': 'ibm_washington',
            'telemetry': {
                'error_rate': 0.02,
                'coherence_time': 120.0,
                'gate_fidelity': 0.98
            },
            'current_rank': 128
        },
        {
            'backend_id': 'ibm_manila',
            'telemetry': {
                'error_rate': 0.09,
                'coherence_time': 25.0,
                'gate_fidelity': 0.91
            },
            'current_rank': 128
        },
        {
            'backend_id': 'russian_simulator',
            'telemetry': {
                'error_rate': 0.001,
                'coherence_time': 500.0,
                'gate_fidelity': 0.999
            },
            'current_rank': 64
        }
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n📊 Testing {case['backend_id']}:")
        print(f"   Error Rate: {case['telemetry']['error_rate']:.3f}")
        print(f"   Coherence Time: {case['telemetry']['coherence_time']:.1f}μs")
        print(f"   Gate Fidelity: {case['telemetry']['gate_fidelity']:.3f}")
        
        result = adapter.adapt_rank(
            backend_id=case['backend_id'],
            telemetry=case['telemetry'],
            current_rank=case['current_rank']
        )
        
        print(f"\n   ✅ Adaptation Result:")
        print(f"      Original Rank: {result.original_rank}")
        print(f"      Adapted Rank: {result.adapted_rank}")
        print(f"      Confidence: {result.confidence:.3f}")
        print(f"      Reliability: {result.reliability_score:.3f}")
        print(f"      Responsiveness: {result.responsiveness_score:.1f}")
        print(f"      Rationale: {result.rationale}")
        
        results.append(result)
    
    # Export for leaderboard
    adapter.export_telemetry_edits('telemetry_edits_v2.4.0.json')
    print(f"\n💾 Exported telemetry edits to telemetry_edits_v2.4.0.json")
    
    return results


def demo_scenario_2_edit_propagation():
    """Scenario 2: Cross-Lingual Edit Propagation"""
    print("\n" + "="*80)
    print("SCENARIO 2: Cross-Lingual Edit Propagation via Subspace Containment")
    print("="*80)
    
    engine = EditPropagationEngine()
    
    # Test propagation paths
    test_cases = [
        ('english', 'indonesian', 128),
        ('chinese', 'vietnamese', 64),
        ('spanish', 'portuguese', 32),
        ('english', 'swahili', 128),
        ('french', 'yoruba', 64)
    ]
    
    print("\n📈 Containment Analysis:")
    
    for source, target, rank in test_cases:
        containment = engine.evaluate_subspace_containment(source, target, rank)
        
        print(f"\n   {source.capitalize()} → {target.capitalize()} @ rank {rank}:")
        print(f"      Containment Score: {containment.containment_score:.3f}")
        print(f"      Overlap Dimension: {containment.overlap_dimension}")
        print(f"      Confidence: {containment.confidence:.3f}")
        print(f"      Propagation Recommended: {'✅ Yes' if containment.propagation_recommended else '❌ No'}")
    
    # Test actual propagation
    print("\n\n🔄 Edit Propagation:")
    
    edit_vector = np.random.randn(256) * 0.1
    
    propagation_result = engine.propagate_edit(
        source_lang='english',
        target_lang='indonesian',
        rank=128,
        edit_vector=edit_vector
    )
    
    print(f"\n   English → Indonesian:")
    print(f"      Success: {'✅' if propagation_result.success else '❌'}")
    print(f"      Quality Score: {propagation_result.quality_score:.3f}")
    print(f"      Containment: {propagation_result.containment_score:.3f}")
    print(f"      Path: {' → '.join(propagation_result.propagation_path)}")
    
    # Compute containment heatmap
    languages = ['english', 'chinese', 'spanish', 'indonesian', 'swahili']
    heatmap = engine.compute_containment_heatmap(languages, rank=128)
    
    print(f"\n\n📊 Containment Heatmap (rank 128):")
    print(f"   Languages: {languages}")
    print(f"   Heatmap shape: {heatmap.shape}")
    print(f"   Average containment: {np.mean(heatmap[np.triu_indices_from(heatmap, k=1)]):.3f}")
    
    # Find propagation paths
    paths = engine.find_propagation_paths(
        source_lang='english',
        target_langs=['indonesian', 'swahili', 'vietnamese'],
        rank=128
    )
    
    print(f"\n\n🗺️  Propagation Paths from English:")
    for target, path in paths.items():
        if path:
            print(f"      → {target.capitalize()}: {' → '.join(path)}")
        else:
            print(f"      → {target.capitalize()}: No viable path")
    
    return engine


def demo_scenario_3_rank_feedback():
    """Scenario 3: Contributor-Aware Rank Feedback Loop"""
    print("\n" + "="*80)
    print("SCENARIO 3: Contributor-Aware Rank Feedback Loop")
    print("="*80)
    
    generator = RankFeedbackGenerator()
    
    # Simulate contributor submissions
    contributors = {
        'contributor_001': [
            {'language': 'english', 'rank': 32, 'accuracy': 0.88, 'flops': 1.02e7, 'uncertainty': 0.12},
            {'language': 'english', 'rank': 64, 'accuracy': 0.92, 'flops': 4.1e7, 'uncertainty': 0.08},
            {'language': 'english', 'rank': 128, 'accuracy': 0.95, 'flops': 1.64e8, 'uncertainty': 0.05},
            {'language': 'chinese', 'rank': 64, 'accuracy': 0.90, 'flops': 4.1e7, 'uncertainty': 0.09},
            {'language': 'indonesian', 'rank': 32, 'accuracy': 0.75, 'flops': 1.02e7, 'uncertainty': 0.20}
        ],
        'contributor_002': [
            {'language': 'spanish', 'rank': 16, 'accuracy': 0.82, 'flops': 2.56e6, 'uncertainty': 0.15},
            {'language': 'spanish', 'rank': 32, 'accuracy': 0.87, 'flops': 1.02e7, 'uncertainty': 0.11},
            {'language': 'french', 'rank': 32, 'accuracy': 0.86, 'flops': 1.02e7, 'uncertainty': 0.12}
        ]
    }
    
    # Record submissions
    for contributor_id, submissions in contributors.items():
        print(f"\n👤 Recording submissions for {contributor_id}:")
        for sub in submissions:
            generator.record_submission(
                contributor_id=contributor_id,
                language=sub['language'],
                rank=sub['rank'],
                accuracy=sub['accuracy'],
                flops=sub['flops'],
                uncertainty=sub['uncertainty']
            )
            print(f"   ✓ {sub['language']} @ rank {sub['rank']}: "
                  f"accuracy={sub['accuracy']:.3f}, FLOPs={sub['flops']:.2e}")
    
    # Generate recommendations
    print("\n\n🎯 Rank Recommendations:")
    
    for contributor_id in contributors.keys():
        recommendation = generator.recommend_rank(contributor_id)
        
        print(f"\n   {contributor_id}:")
        print(f"      Badge: {recommendation.personalized_badge}")
        print(f"      Recommended Rank: {recommendation.recommended_rank}")
        print(f"      Confidence: {recommendation.confidence:.3f}")
        print(f"      Predicted Efficiency: {recommendation.efficiency_prediction:.2e}")
        print(f"      Rationale: {recommendation.rationale}")
        
        if recommendation.unexplored_pairs:
            print(f"\n      🔍 Top Unexplored Opportunities:")
            for rank, lang in recommendation.unexplored_pairs[:3]:
                print(f"         • Rank {rank} with {lang}")
    
    # Generate feedback panel
    print("\n\n📋 Feedback Panel for contributor_001:")
    panel = generator.generate_feedback_panel('contributor_001')
    
    print(f"\n   Statistics:")
    for key, value in panel['stats'].items():
        if isinstance(value, float):
            print(f"      {key}: {value:.3f}")
        else:
            print(f"      {key}: {value}")
    
    print(f"\n   Suggestions:")
    for i, suggestion in enumerate(panel['suggestions'], 1):
        print(f"      {i}. {suggestion}")
    
    return generator


def demo_scenario_4_ensemble_inference():
    """Scenario 4: Ensemble Inference Across Backends"""
    print("\n" + "="*80)
    print("SCENARIO 4: Ensemble Inference Across Backends")
    print("="*80)
    
    manager = EnsembleInferenceManager()
    
    # Test edit vector
    edit_vector = np.random.randn(256) * 0.1
    
    # Test with different backend combinations
    backend_combinations = [
        ['ibm_manila', 'ibm_washington'],
        ['ibm_washington', 'russian_simulator'],
        ['ibm_manila', 'ibm_washington', 'russian_simulator'],
        ['ibm_washington', 'ibm_kyoto', 'google_sycamore']
    ]
    
    print("\n🔬 Ensemble Inference Tests:")
    
    for backends in backend_combinations:
        print(f"\n   Testing: {', '.join(backends)}")
        
        result = manager.run_ensemble_inference(edit_vector, backends)
        
        print(f"\n   📊 Results:")
        print(f"      Agreement Score: {result.agreement_score:.3f}")
        print(f"      Reliability Boost: {result.reliability_boost:.3f}")
        print(f"      Ensemble Confidence: {result.ensemble_confidence:.3f}")
        print(f"      Best Backend: {result.best_backend}")
        
        print(f"\n      Individual Backend Results:")
        for backend_result in result.backend_results:
            print(f"         • {backend_result.backend_id}:")
            print(f"           Confidence: {backend_result.confidence:.3f}")
            print(f"           Latency: {backend_result.latency:.3f}s")
            print(f"           Success: {'✅' if backend_result.success else '❌'}")
    
    # Backend comparison
    print("\n\n📈 Backend Comparison:")
    
    test_vectors = [np.random.randn(256) * 0.1 for _ in range(5)]
    comparison = manager.compare_backends(test_vectors)
    
    print(f"\n   Across {len(test_vectors)} test vectors:")
    for backend_id, metrics in comparison.items():
        print(f"\n      {backend_id}:")
        print(f"         Avg Confidence: {metrics['avg_confidence']:.3f}")
        print(f"         Avg Latency: {metrics['avg_latency']:.3f}s")
        print(f"         Success Rate: {metrics['success_rate']:.1%}")
    
    # Agreement heatmap
    print("\n\n🗺️  Agreement Matrix:")
    
    all_backends = ['ibm_manila', 'ibm_washington', 'russian_simulator']
    agreement_matrix, labels = manager.get_agreement_heatmap(all_backends, edit_vector)
    
    print(f"\n   Backends: {labels}")
    print(f"   Matrix shape: {agreement_matrix.shape}")
    print(f"   Average pairwise agreement: {np.mean(agreement_matrix[np.triu_indices_from(agreement_matrix, k=1)]):.3f}")
    
    # Overall reliability metrics
    metrics = manager.compute_reliability_metrics()
    
    print(f"\n\n📊 Overall Reliability Metrics:")
    print(f"      Avg Agreement: {metrics['avg_agreement']:.3f}")
    print(f"      Avg Reliability Boost: {metrics['avg_reliability_boost']:.3f}")
    print(f"      Avg Ensemble Confidence: {metrics['avg_ensemble_confidence']:.3f}")
    
    return manager


def main():
    """Run all scenario demos"""
    print("\n" + "="*80)
    print("Quantum LIMIT-Graph v2.4.0 - NSN Integration Scenarios Demo")
    print("="*80)
    print("\nDemonstrating four modular components for NSN integration:")
    print("1. Backend Telemetry Rank Adapter")
    print("2. Edit Propagation Engine")
    print("3. Rank Feedback Generator")
    print("4. Ensemble Inference Manager")
    
    # Run all scenarios
    telemetry_results = demo_scenario_1_telemetry_adaptation()
    propagation_engine = demo_scenario_2_edit_propagation()
    feedback_generator = demo_scenario_3_rank_feedback()
    ensemble_manager = demo_scenario_4_ensemble_inference()
    
    # Summary
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\n✅ All four scenarios demonstrated successfully!")
    print("\n📁 Generated Files:")
    print("   • telemetry_edits_v2.4.0.json - Telemetry-aware rank adaptations")
    print("\n🎯 Key Capabilities:")
    print("   • Real-time rank adaptation based on backend health")
    print("   • Cross-lingual edit propagation via subspace containment")
    print("   • Personalized rank recommendations for contributors")
    print("   • Ensemble inference with agreement scoring")
    print("\n🚀 Ready for integration with Quantum LIMIT-Graph v2.4.0!")


if __name__ == '__main__':
    main()
