# -*- coding: utf-8 -*-
"""
Nested Subspace Networks (NSN) Integration for LIMIT-Graph and REPAIR
Enhances quantum benchmarking and multilingual edit reliability

v2.4.0 New Scenarios:
- Backend Telemetry Rank Adapter: Real-time rank adaptation
- Edit Propagation Engine: Cross-lingual edit transfer
- Rank Feedback Generator: Contributor-aware recommendations
- Ensemble Inference Manager: Multi-backend consensus
"""

from .backend_aware_rank_selector import BackendAwareRankSelector, BackendType, RankConfig
from .multilingual_nsn_evaluator import MultilingualNSNEvaluator
from .nsn_leaderboard import NSNLeaderboard, ContributorChallenge
from .nsn_dashboard import NSNDashboard

# v2.4.0 New Components
from .backend_telemetry_rank_adapter import (
    BackendTelemetryRankAdapter,
    BackendTelemetry,
    AdaptationResult
)
from .edit_propagation_engine import (
    EditPropagationEngine,
    ContainmentScore,
    PropagationResult
)
from .rank_feedback_generator import (
    RankFeedbackGenerator,
    SubmissionRecord,
    RankRecommendation
)
from .ensemble_inference_manager import (
    EnsembleInferenceManager,
    BackendResult,
    EnsembleResult
)

__all__ = [
    # Original components
    'BackendAwareRankSelector',
    'BackendType',
    'RankConfig',
    'MultilingualNSNEvaluator',
    'NSNLeaderboard',
    'ContributorChallenge',
    'NSNDashboard',
    
    # v2.4.0 New Components
    'BackendTelemetryRankAdapter',
    'BackendTelemetry',
    'AdaptationResult',
    'EditPropagationEngine',
    'ContainmentScore',
    'PropagationResult',
    'RankFeedbackGenerator',
    'SubmissionRecord',
    'RankRecommendation',
    'EnsembleInferenceManager',
    'BackendResult',
    'EnsembleResult'
]

__version__ = '2.4.0'

# Contribution-ready modules with Hugging Face dashboard
__contribution_ready__ = True
__dashboard_url__ = 'https://huggingface.co/spaces/your-org/nsn-integration-dashboard'
