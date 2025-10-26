# NSN Integration with LIMIT-Graph and REPAIR

Comprehensive integration of **Nested Subspace Networks (NSNs)** with LIMIT-Graph and REPAIR to enhance quantum benchmarking and multilingual edit reliability.

## 🚀 **NEW in v2.4.0: Contribution-Ready Modules + Hugging Face Dashboard**

**Four modular scenarios** are now ready for community contributions with an **interactive Hugging Face Spaces dashboard**!

### Quick Links
- 🎯 **[Contributor Guide](CONTRIBUTOR_GUIDE.md)** - Start contributing now!
- 🎨 **[Live Dashboard](https://huggingface.co/spaces/AIResAgTeam/quantum-nsn-integration)** - Try it online
- 📚 **[Complete Documentation](INDEX.md)** - Full navigation index
- 🚀 **[Deployment Guide](HUGGINGFACE_DEPLOYMENT.md)** - Deploy your own

### v2.4.0 Contribution Scenarios

| Scenario | Module | Dashboard Panel | Reward |
|----------|--------|-----------------|--------|
| **1. Backend Telemetry** | [backend_telemetry_rank_adapter.py](backend_telemetry_rank_adapter.py) | FLOPs vs Reliability | Responsiveness Badge |
| **2. Edit Propagation** | [edit_propagation_engine.py](edit_propagation_engine.py) | Containment Heatmap | Quality Score |
| **3. Rank Feedback** | [rank_feedback_generator.py](rank_feedback_generator.py) | Personalized Feedback | Efficiency Badge |
| **4. Ensemble Inference** | [ensemble_inference_manager.py](ensemble_inference_manager.py) | Agreement Matrix | Reliability Boost |

---

## Overview

This integration implements both **original NSN features** and **new v2.4.0 contribution-ready scenarios**:

### Original Features (v2.3.0)
1. **Backend-Aware Rank Selection**: Dynamically adjust model rank based on quantum backend constraints
2. **Multilingual Edit Reliability**: Evaluate how rank affects correction accuracy across languages
3. **Contributor Challenges**: Design leaderboard tasks with rank-aware evaluation and compute-performance frontiers

### New v2.4.0 Features
4. **Real-Time Backend Telemetry Adaptation**: Dynamic rank shifts based on live backend health
5. **Cross-Lingual Edit Propagation**: Transfer edits via subspace containment analysis
6. **Contributor-Aware Rank Feedback**: Personalized recommendations with badge system
7. **Ensemble Inference Across Backends**: Multi-backend consensus with agreement scoring

## Architecture

```
nsn_integration/
├── Core Modules (Original v2.3.0)
│   ├── backend_aware_rank_selector.py       # Backend-aware rank selection
│   ├── multilingual_nsn_evaluator.py        # Multilingual evaluation
│   ├── nsn_leaderboard.py                   # Contributor challenges
│   ├── nsn_dashboard.py                     # Visualization dashboard
│   └── limit_graph_nsn_integration.py       # LIMIT-Graph integration
│
├── v2.4.0 Contribution Modules (NEW!)
│   ├── backend_telemetry_rank_adapter.py    # Real-time rank adaptation
│   ├── edit_propagation_engine.py           # Cross-lingual propagation
│   ├── rank_feedback_generator.py           # Personalized feedback
│   └── ensemble_inference_manager.py        # Multi-backend consensus
│
├── Hugging Face Dashboard (NEW!)
│   ├── huggingface_dashboard.py             # 6-panel interactive UI
│   ├── app.py                               # Spaces entry point
│   └── requirements_dashboard.txt           # Dashboard dependencies
│
├── Documentation
│   ├── README.md                            # This file
│   ├── CONTRIBUTOR_GUIDE.md                 # How to contribute
│   ├── HUGGINGFACE_DEPLOYMENT.md            # Deployment guide
│   ├── INDEX.md                             # Navigation index
│   └── [10+ other docs]                     # Complete documentation
│
└── Testing & Demo
    ├── test_v2.4.0_scenarios.py             # Test suite
    ├── demo_v2.4.0_scenarios.py             # Demo script
    └── demo_complete_nsn_integration.py     # Original demo
```

## 🎨 Hugging Face Spaces Dashboard

**NEW in v2.4.0**: Interactive 6-panel dashboard for community contributions!

### Dashboard Panels

1. **📊 Backend Telemetry** - Real-time rank adaptation visualization
   - Line chart: FLOPs vs Reliability across backend states
   - Metric: Responsiveness vs Reliability trade-off

2. **🌍 Multilingual Accuracy** - Language × Rank performance heatmap
   - Heatmap: Accuracy across 15+ languages and 6 ranks
   - Metric: Average accuracy matrix

3. **🔗 Edit Propagation** - Cross-lingual transfer visualization
   - Containment heatmap with flow arrows
   - Metric: Quality score of propagated edits

4. **⚡ Pareto Frontier** - Efficiency vs Expressiveness trade-offs
   - Scatter plot with Pareto optimal line
   - Metric: Position on efficiency frontier

5. **🏆 Contributor Leaderboard** - Personalized feedback panel
   - Badge display and performance statistics
   - Metric: Efficiency badge (accuracy/FLOPs)

6. **🔬 Ensemble Inference** - Multi-backend agreement matrix
   - Backend consensus heatmap
   - Metric: Agreement score + Reliability boost

### Try It Now

```bash
# Run locally
pip install -r requirements_dashboard.txt
python app.py

# Or visit live dashboard
# https://huggingface.co/spaces/AIResAgTeam/quantum-nsn-integration
```

### Deploy Your Own

```bash
export HF_USERNAME="nurcholish"
./deploy_to_spaces.sh
```

See [HUGGINGFACE_DEPLOYMENT.md](HUGGINGFACE_DEPLOYMENT.md) for complete instructions.

## Stage 1: Backend-Aware Rank Selection

### Features

- **Dynamic Rank Adjustment**: Automatically select optimal NSN rank based on quantum backend characteristics
- **Backend Support**:
  - IBM Manila (5 qubits, noisy) → Low-rank inference (r=8)
  - IBM Washington (127 qubits, high-fidelity) → High-rank inference (r=128-256)
  - Russian Simulators (stable) → Maximum-rank inference (r=256)
- **FLOPs vs Reliability Visualization**: Plot compute-performance curves for each backend

### Usage

```python
from Quantum-LIMIT-Graph-v2.4.0-NSN import BackendAwareRankSelector, BackendType

# Create selector
selector = BackendAwareRankSelector()

# Get rank recommendation
recommendation = selector.get_rank_recommendation(
    backend_type=BackendType.IBM_WASHINGTON,
    compute_budget=1e8,
    min_reliability=0.85
)

print(f"Recommended Rank: {recommendation['recommended_rank']}")
print(f"Expected Reliability: {recommendation['expected_reliability']:.3f}")
print(f"Rationale: {recommendation['rationale']}")

# Compute FLOPs vs reliability curve
curve = selector.compute_flops_vs_reliability(BackendType.IBM_WASHINGTON)
```

## Stage 2: Multilingual Edit Reliability

### Features

- **Cross-Language Evaluation**: Assess edit accuracy across 15+ languages
- **Resource-Aware Training**: Uncertainty-weighted training for low/medium/high-resource languages
- **Subspace Containment Analysis**: Visualize how low-resource language edits nest within high-resource language subspaces
- **Optimal Rank Selection**: Find best rank per language given accuracy and compute constraints

### Language Support

- **High-Resource**: English, Chinese, Spanish, French, German
- **Medium-Resource**: Russian, Arabic, Japanese, Korean, Portuguese
- **Low-Resource**: Indonesian, Vietnamese, Thai, Swahili, Yoruba

### Usage

```python
from Quantum-LIMIT-Graph-v2.4.0-NSN import MultilingualNSNEvaluator

# Create evaluator
evaluator = MultilingualNSNEvaluator()

# Evaluate single language
result = evaluator.evaluate_language_edit(
    language='indonesian',
    rank=64
)

print(f"Accuracy: {result.edit_accuracy:.3f}")
print(f"Uncertainty: {result.uncertainty:.3f}")

# Comprehensive analysis
languages = ['english', 'chinese', 'indonesian', 'swahili']
analysis = evaluator.analyze_rank_language_matrix(languages)

# Get uncertainty weights for balanced training
weights = evaluator.compute_uncertainty_weights(languages)

# Analyze subspace containment
containment = evaluator.evaluate_subspace_containment(
    source_lang='indonesian',
    target_lang='english',
    rank=64
)

print(f"Containment Score: {containment.containment_score:.3f}")
```

## Stage 3: Contributor Challenges

### Features

- **Leaderboard System**: Track contributor submissions across multiple ranks
- **Pareto Frontier**: Visualize compute-performance trade-offs
- **Rank-Specific Feedback**: Provide detailed feedback on expressiveness, efficiency, and uncertainty
- **Challenge Management**: Create and manage multilingual editing challenges

### Usage

```python
from Quantum-LIMIT-Graph-v2.4.0-NSN import NSNLeaderboard

# Create leaderboard
leaderboard = NSNLeaderboard()

# Create challenge
challenge = leaderboard.create_challenge(
    challenge_id="multilingual_edit_2025",
    title="Multilingual Model Editing Challenge",
    description="Optimize edit accuracy across languages and ranks",
    languages=['english', 'chinese', 'indonesian'],
    ranks=[8, 16, 32, 64, 128, 256]
)

# Submit edit
rank_results = {
    8: {'accuracy': 0.75, 'uncertainty': 0.20, 'flops': 6.4e5, 'efficiency': 0.012},
    32: {'accuracy': 0.88, 'uncertainty': 0.12, 'flops': 1.02e7, 'efficiency': 0.009},
    128: {'accuracy': 0.95, 'uncertainty': 0.05, 'flops': 1.64e8, 'efficiency': 0.006}
}

submission = leaderboard.submit_edit(
    challenge_id="multilingual_edit_2025",
    contributor_id="contributor_001",
    language="english",
    edit_description="Optimized factual correction",
    rank_results=rank_results
)

# Get leaderboard
rankings = leaderboard.get_leaderboard("multilingual_edit_2025")

# Compute Pareto frontier
frontier = leaderboard.compute_pareto_frontier("multilingual_edit_2025")

# Generate feedback
feedback = leaderboard.generate_feedback(submission.submission_id)
```

## Dashboard Visualizations

### Available Plots

1. **FLOPs vs Reliability**: Backend performance curves
2. **Multilingual Heatmap**: Accuracy matrix across languages and ranks
3. **Subspace Containment**: Nested subspace analysis
4. **Pareto Frontier**: Compute-performance trade-offs
5. **Leaderboard Rankings**: Top contributor visualization
6. **Uncertainty Analysis**: Uncertainty reduction across ranks
7. **Comprehensive Dashboard**: Multi-panel overview

### Usage

```python
from Quantum-LIMIT-Graph-v2.4.0-NSN import NSNDashboard

# Create dashboard
dashboard = NSNDashboard()

# Plot FLOPs vs Reliability
dashboard.plot_flops_vs_reliability(
    backend_curves=backend_curves,
    save_path='flops_vs_reliability.png'
)

# Plot multilingual heatmap
dashboard.plot_multilingual_heatmap(
    accuracy_matrix=accuracy_matrix,
    save_path='multilingual_heatmap.png'
)

# Plot Pareto frontier
dashboard.plot_pareto_frontier(
    frontier_data=frontier_data,
    save_path='pareto_frontier.png'
)

# Create comprehensive dashboard
dashboard.create_comprehensive_dashboard(
    backend_curves=backend_curves,
    accuracy_matrix=accuracy_matrix,
    containment_data=containment_data,
    frontier_data=frontier_data,
    leaderboard=rankings,
    save_path='comprehensive_dashboard.png'
)
```

## LIMIT-Graph Integration

### Benchmarking Harness

The NSN integration is embedded into the LIMIT-Graph benchmarking harness for seamless evaluation:

```python
from Quantum-LIMIT-Graph-v2.4.0-NSN import (
    LIMITGraphNSNBenchmark,
    BenchmarkConfig
)

# Create configuration
config = BenchmarkConfig(
    backend_type=BackendType.IBM_WASHINGTON,
    languages=['english', 'chinese', 'indonesian'],
    target_reliability=0.85,
    compute_budget=1e8
)

# Create benchmark
benchmark = LIMITGraphNSNBenchmark(config)

# Run benchmark
test_cases = [
    {'language': 'english', 'text': 'The capital of France is Paris'},
    {'language': 'chinese', 'text': '北京是中国的首都'},
    {'language': 'indonesian', 'text': 'Jakarta adalah ibu kota Indonesia'}
]

results = benchmark.run_benchmark(test_cases)

# Visualize results
benchmark.visualize_benchmark_results(results, save_path='benchmark_results.png')

# Compare backends
comparison = benchmark.compare_backends(test_cases)
```

## Running the Complete Demo

```bash
# Run complete NSN integration demo
python demo_complete_nsn_integration.py

# Run LIMIT-Graph integration demo
python limit_graph_nsn_integration.py
```

### Demo Output

The demo will:
1. Test backend-aware rank selection for IBM Manila, IBM Washington, and Russian Simulator
2. Evaluate multilingual edit reliability across 9 languages
3. Create contributor challenges and generate leaderboard
4. Generate comprehensive visualizations
5. Export results to JSON

### Generated Files

- `nsn_flops_vs_reliability.png`: Backend performance curves
- `nsn_multilingual_heatmap.png`: Language-rank accuracy matrix
- `nsn_subspace_containment.png`: Subspace nesting visualization
- `nsn_pareto_frontier.png`: Compute-performance frontier
- `nsn_leaderboard_rankings.png`: Top contributor rankings
- `nsn_uncertainty_analysis.png`: Uncertainty reduction analysis
- `nsn_comprehensive_dashboard.png`: Multi-panel dashboard
- `limit_graph_nsn_results.json`: Benchmark results

## Key Concepts

### Nested Subspace Networks (NSNs)

NSNs represent model parameters in nested subspaces of increasing rank:
- **Low Rank (r=8-16)**: Fast inference, lower accuracy, suitable for noisy backends
- **Medium Rank (r=32-64)**: Balanced performance
- **High Rank (r=128-256)**: Maximum accuracy, high compute, requires stable backends

### Backend-Aware Selection

Quantum backend characteristics determine optimal rank:
- **Qubit Count**: More qubits → higher rank capacity
- **Error Rate**: Lower error → higher rank feasibility
- **Gate Fidelity**: Higher fidelity → better high-rank performance
- **Coherence Time**: Longer coherence → supports complex circuits

### Multilingual Subspace Containment

Low-resource language edits often nest within high-resource language subspaces:
- **Indonesian → English**: ~85% containment at rank 128
- **Swahili → English**: ~80% containment at rank 128
- **Vietnamese → Chinese**: ~75% containment at rank 64

This enables transfer learning and cross-lingual edit propagation.

## Integration with Existing Components

### REPAIR Integration

```python
from quantum_integration.social_science_extensions import REPAIRInferenceWrapper
from quantum_integration.nsn_integration import BackendAwareRankSelector

# Select rank based on backend
selector = BackendAwareRankSelector()
rank_config = selector.select_rank(BackendType.IBM_WASHINGTON)

# Use rank in REPAIR inference
# (REPAIR wrapper can be extended to accept rank parameter)
```

### Quantum Health Monitoring

```python
from quantum_integration import quantum_health_checker
from Quantum-LIMIT-Graph-v2.4.0-NSN import BackendAwareRankSelector

# Check backend health
health = quantum_health_checker.check_backend_health('ibm_washington')

# Adjust rank based on health
if health['status'] == 'degraded':
    # Use lower rank for stability
    rank = 32
else:
    # Use optimal rank
    rank = selector.select_rank(BackendType.IBM_WASHINGTON).rank
```

## Performance Metrics

### Benchmark Results (Example)

| Backend | Rank | Accuracy | Uncertainty | FLOPs | Inference Time |
|---------|------|----------|-------------|-------|----------------|
| IBM Manila | 8 | 0.76 | 0.18 | 6.4e5 | 10ms |
| IBM Washington | 128 | 0.95 | 0.05 | 1.6e8 | 160ms |
| Russian Simulator | 256 | 0.97 | 0.03 | 6.6e8 | 320ms |

### Multilingual Performance

| Language | Resource Level | Rank 8 | Rank 32 | Rank 128 |
|----------|---------------|--------|---------|----------|
| English | High | 0.90 | 0.93 | 0.96 |
| Chinese | High | 0.89 | 0.92 | 0.95 |
| Russian | Medium | 0.78 | 0.85 | 0.91 |
| Indonesian | Low | 0.65 | 0.75 | 0.85 |
| Swahili | Low | 0.62 | 0.72 | 0.83 |

## Contributing

To contribute to NSN integration:

1. **Submit Edits**: Use the leaderboard system to submit your edits
2. **Evaluate Across Ranks**: Test your edits at multiple NSN ranks
3. **Optimize Efficiency**: Aim for the Pareto frontier (high accuracy, low FLOPs)
4. **Document Results**: Share your findings and techniques

## Citation

This integration is based on the Nested Subspace Networks (NSN) framework from:

```bibtex
@article{zhang2024deep,
  title={Deep Hierarchical Learning with Nested Subspace Networks},
  author={Zhang, Yifan and others},
  journal={arXiv preprint},
  year={2024},
  note={NSN framework for hierarchical representation learning with nested subspaces}
}
```

If you use this NSN integration in your research, please cite both the original NSN paper and this implementation:

```bibtex
@software{Quantum-LIMIT-Graph-v2.4.0-NSN,
  title={NSN Integration with LIMIT-Graph and REPAIR for Quantum Benchmarking},
  author={AI Research Agent Team},
  year={2025},
  url={https://github.com/NurcholishAdam/Quantum-LIMIT-Graph-v2.4.0-NSN},
  note={Integration of Nested Subspace Networks with quantum computing backends and multilingual model editing}
}
```

### Acknowledgments

We acknowledge the original NSN framework authors for their foundational work on hierarchical representation learning with nested subspaces, which enabled this integration with quantum benchmarking and multilingual edit reliability.

## License

This integration is part of the LIMIT-Graph project and follows the same license terms.

## Support

For questions or issues:
- Open an issue on GitHub
- Check the demo scripts for usage examples
- Review the comprehensive documentation in each module

## v2.4.0 New Scenarios

### Scenario 1: Real-Time Backend-Aware Rank Adaptation

**Module**: `backend_telemetry_rank_adapter.py`

Dynamically adjusts NSN ranks based on real-time backend health metrics.

**Inputs**:
- `backend_id`: e.g., "ibm_washington"
- `telemetry`: Dict with `error_rate`, `coherence_time`, `gate_fidelity`

**Challenge Extension**:
- Contributors submit telemetry-aware edits
- Leaderboard ranks by reliability vs responsiveness

**Usage**:
```python
from Quantum-LIMIT-Graph-v2.4.0-NSN import BackendTelemetryRankAdapter

adapter = BackendTelemetryRankAdapter()

result = adapter.adapt_rank(
    backend_id='ibm_washington',
    telemetry={
        'error_rate': 0.02,
        'coherence_time': 120.0,
        'gate_fidelity': 0.98
    },
    current_rank=128
)

print(f"Adapted Rank: {result.adapted_rank}")
print(f"Reliability: {result.reliability_score:.3f}")
print(f"Rationale: {result.rationale}")
```

### Scenario 2: Cross-Lingual Edit Propagation via Subspace Containment

**Module**: `edit_propagation_engine.py`

Transfers high-resource corrections to low-resource languages using containment scores.

**Inputs**:
- `source_lang`: High-resource language
- `target_lang`: Low-resource language
- `rank`: NSN rank
- `edit_vector`: Edit to propagate

**Dashboard Extension**:
- Heatmap of containment scores
- Flow arrows showing edit propagation paths

**Usage**:
```python
from quantum_integration.nsn_integration import EditPropagationEngine
import numpy as np

engine = EditPropagationEngine()

# Evaluate containment
containment = engine.evaluate_subspace_containment(
    source_lang='english',
    target_lang='indonesian',
    rank=128
)

print(f"Containment Score: {containment.containment_score:.3f}")

# Propagate edit
edit_vector = np.random.randn(256) * 0.1
result = engine.propagate_edit(
    source_lang='english',
    target_lang='indonesian',
    rank=128,
    edit_vector=edit_vector
)

print(f"Quality Score: {result.quality_score:.3f}")
```

### Scenario 3: Contributor-Aware Rank Feedback Loop

**Module**: `rank_feedback_generator.py`

Recommends optimal ranks based on contributor history and efficiency.

**Inputs**:
- `contributor_id`: Contributor identifier
- `past_submissions`: List with `accuracy`, `flops`, `uncertainty`

**Leaderboard Extension**:
- Personalized rank badges
- Suggestion panel for unexplored rank-language pairs

**Usage**:
```python
from quantum_integration.nsn_integration import RankFeedbackGenerator

generator = RankFeedbackGenerator()

# Record submissions
generator.record_submission(
    contributor_id='contributor_001',
    language='english',
    rank=64,
    accuracy=0.92,
    flops=4.1e7,
    uncertainty=0.08
)

# Get recommendation
recommendation = generator.recommend_rank('contributor_001')

print(f"Badge: {recommendation.personalized_badge}")
print(f"Recommended Rank: {recommendation.recommended_rank}")
print(f"Rationale: {recommendation.rationale}")

# Get feedback panel
panel = generator.generate_feedback_panel('contributor_001')
print(f"Suggestions: {panel['suggestions']}")
```

### Scenario 4: Ensemble Inference Across Backends

**Module**: `ensemble_inference_manager.py`

Runs edits across multiple backends and computes agreement scores.

**Inputs**:
- `edit_vector`: Edit to apply
- `backend_list`: e.g., `['ibm_manila', 'ibm_washington', 'russian_simulator']`

**Dashboard Extension**:
- Agreement matrix across backends
- Reliability boost from ensemble consensus

**Usage**:
```python
from quantum_integration.nsn_integration import EnsembleInferenceManager
import numpy as np

manager = EnsembleInferenceManager()

edit_vector = np.random.randn(256) * 0.1

result = manager.run_ensemble_inference(
    edit_vector=edit_vector,
    backend_list=['ibm_manila', 'ibm_washington', 'russian_simulator']
)

print(f"Agreement Score: {result.agreement_score:.3f}")
print(f"Reliability Boost: {result.reliability_boost:.3f}")
print(f"Best Backend: {result.best_backend}")

# Get agreement matrix for visualization
agreement_matrix, labels = manager.get_agreement_heatmap(
    backend_list=['ibm_manila', 'ibm_washington', 'russian_simulator'],
    edit_vector=edit_vector
)
```

## Running v2.4.0 Scenarios Demo

```bash
# Run complete v2.4.0 scenarios demo
python demo_v2.4.0_scenarios.py
```

### Demo Output

The demo will:
1. Test real-time rank adaptation across different backend conditions
2. Evaluate cross-lingual edit propagation with containment analysis
3. Generate personalized rank recommendations for contributors
4. Run ensemble inference across multiple backends
5. Export telemetry edits and generate visualizations

### Generated Files

- `telemetry_edits_v2.4.0.json`: Telemetry-aware rank adaptations for leaderboard

## Roadmap

- [x] Real-time rank adaptation based on backend telemetry ✅ v2.4.0
- [x] Multi-backend ensemble inference ✅ v2.4.0
- [x] Cross-lingual edit propagation ✅ v2.4.0
- [x] Contributor-aware feedback system ✅ v2.4.0
- [ ] Automated hyperparameter tuning for rank selection
- [ ] Extended language support (50+ languages)
- [ ] Integration with Hugging Face Spaces for public leaderboard
- [ ] Quantum circuit optimization for rank-specific operations



