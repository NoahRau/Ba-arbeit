"""
Central Configuration for Hybrid Ontology Matching Pipeline.

This file contains all important parameters for the pipeline:
- Matcher weights
- Confidence thresholds
- Top-K settings
- Model configurations
- Paths
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
CACHE_DIR = PROJECT_ROOT / 'cache'
RESULTS_DIR = DATA_DIR / 'results'

# Data subdirectories
S1000D_DATA_DIR = DATA_DIR / 's1000d'
ONTOLOGY_DIR = DATA_DIR / 'ontologies'
EMBEDDINGS_CACHE_DIR = CACHE_DIR / 'embeddings'
LOGS_DIR = CACHE_DIR / 'logs'

# Ensure directories exist
EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MATCHER CONFIGURATIONS
# ============================================================================

# Matcher Weights (must sum to ~1.0)
# Optimized based on evaluation results:
# - KROMA: High weight (DMC-based heuristics are very precise for S1000D)
# - DeepOnto: Medium-high (BERT semantic understanding is crucial)
# - String: Medium (good baseline, helps when others fail)
MATCHER_WEIGHTS = {
    'kroma': 0.45,      # Increased from 0.4 (DMC heuristics are very reliable)
    'deeponto': 0.35,   # Keep (semantic understanding critical)
    'string': 0.20      # Decreased from 0.25 (less reliable alone)
}

# KROMA Matcher Configuration
KROMA_CONFIG = {
    # DMC chapter mappings (chapter code -> keywords)
    # These map S1000D DMC chapters to ontology concepts
    'dmc_chapter_keywords': {
        'D00': ['bicycle', 'bike', 'general', 'system'],
        'DA0': ['wheel', 'tire', 'hub', 'spoke', 'rim'],
        'DA1': ['steering', 'handlebar', 'stem', 'fork'],
        'DA2': ['braking', 'brake', 'caliper', 'disc', 'pad'],
        'DA3': ['transmission', 'drivetrain', 'chain', 'gear', 'cassette', 'derailleur'],
        'DA4': ['seat', 'saddle', 'seatpost'],
        'DA5': ['frame', 'frameset', 'structure'],
    },

    # Score weights for KROMA components
    'score_weights': {
        'chapter_match': 0.40,      # Increased from 0.35 (chapter is very important)
        'label_overlap': 0.25,      # Keep
        'context_keywords': 0.20,   # Keep
        'hierarchy_bonus': 0.15     # Keep
    },

    # Thresholds
    'min_score': 0.3,  # Minimum score to consider a match
}

# DeepOnto Matcher Configuration
DEEPONTO_CONFIG = {
    # BERT Model
    'model_name': 'answerdotai/ModernBERT-base',
    'max_seq_length': 8192,
    'batch_size': 8,

    # Embeddings cache
    'cache_file': EMBEDDINGS_CACHE_DIR / 'deeponto_embeddings_cache.pkl',

    # Similarity thresholds
    'min_similarity': 0.5,  # Minimum cosine similarity

    # Ontology reasoning penalties
    'subsumption_penalty': 0.5,  # Reduce score by 50% for parent-child relationships
    'sibling_penalty': 0.7,      # Reduce score by 30% for sibling relationships

    # Context integration
    'label_weight': 0.6,    # Weight for label similarity
    'context_weight': 0.4,  # Weight for context similarity
}

# String Matcher Configuration
STRING_CONFIG = {
    # Score weights
    'jaccard_weight': 0.40,      # Token overlap (Jaccard)
    'sequence_weight': 0.35,     # Edit distance
    'context_weight': 0.25,      # Context similarity

    # Thresholds
    'exact_match_score': 1.0,    # Exact string match
    'substring_score': 0.85,     # Substring match
    'min_score': 0.3,            # Minimum score to consider
}

# ============================================================================
# AGGREGATION CONFIGURATION
# ============================================================================

AGGREGATION_CONFIG = {
    # Method: 'rank_fusion' or 'weighted_sum'
    'method': 'rank_fusion',

    # Rank Fusion parameters
    'rrf_k': 60,  # RRF constant (standard value)

    # Top-K candidates to keep after aggregation
    'top_k': 60,  # Keep all for neural reranker filtering
}

# ============================================================================
# NEURAL RERANKER CONFIGURATION
# ============================================================================

NEURAL_RERANKER_CONFIG = {
    # Model
    'model_name': 'BAAI/bge-reranker-v2-m3',

    # Device (None = auto-detect)
    'device': None,  # 'cuda', 'cpu', or None

    # Top-K candidates to keep after reranking
    'top_k': 7,  # Filter from 60 aggregated candidates to 7 for LLM

    # Score combination weights
    'reranker_weight': 0.7,  # Weight for neural reranker score
    'aggregation_weight': 0.3,  # Weight for aggregated score

    # Enable neural reranking
    'enabled': True,
}

# ============================================================================
# LLM RERANKING CONFIGURATION
# ============================================================================

LLM_CONFIG = {
    # Model
    'model': 'claude-sonnet-4-5-20250929',

    # Confidence threshold
    # Higher = more conservative (fewer matches, higher precision)
    # Lower = more liberal (more matches, potentially lower precision)
    'confidence_threshold': 0.90,  # Lowered from 0.95 to improve recall

    # Temperature (for creative tasks, but we want deterministic)
    'temperature': 0.0,

    # Max tokens
    'max_tokens': 1024,

    # Enable LLM reranking by default
    'enabled': True,
}

# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

PIPELINE_CONFIG = {
    # Top-K candidates per matcher
    'matcher_top_k': 60,  # Each matcher returns top 60 candidates

    # Top-K candidates after aggregation (before neural reranking)
    'aggregation_top_k': 60,  # Keep all 60 for neural reranker

    # Top-K candidates after neural reranking (sent to LLM)
    'reranker_top_k': 7,  # Neural reranker filters to 7 best candidates

    # Use neural reranking
    'use_neural_reranker': True,

    # Use LLM reranking
    'use_llm': True,

    # Aggregation method
    'aggregation_method': 'rank_fusion',

    # Fallback strategy if LLM fails
    'fallback_to_aggregation': True,
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

EVALUATION_CONFIG = {
    # Metrics to compute
    'metrics': [
        'precision',
        'recall',
        'f1_score',
        'accuracy',
        'mcc',  # Matthews Correlation Coefficient
    ],

    # Confusion matrix
    'plot_confusion_matrix': True,

    # Error analysis
    'max_errors_to_show': 10,  # Max false positives/negatives to display
}

# ============================================================================
# GOLD STANDARD CONFIGURATION
# ============================================================================

GOLD_STANDARD_CONFIG = {
    # Sample size for manual annotation
    'sample_size': 50,

    # Top-K candidates to show
    'top_k': 5,

    # Output file
    'output_file': RESULTS_DIR / 'gold_standard.json',
}

# ============================================================================
# ONTOLOGY URLS
# ============================================================================

ONTOLOGY_URLS = {
    'tbox': 'https://giuliamenna.github.io/BikeOntology/final_data/tbox_bikeo.owl',
    'abox': 'https://giuliamenna.github.io/BikeOntology/final_data/abox_bikeo.ttl',
    'final': 'https://giuliamenna.github.io/BikeOntology/final_data/final_bikeo.owl'
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': LOGS_DIR / 'pipeline.log',
}

# ============================================================================
# PERFORMANCE TUNING
# ============================================================================

PERFORMANCE_CONFIG = {
    # Batch processing
    'batch_size': 8,

    # Parallel processing (for matchers)
    'n_jobs': -1,  # -1 = use all CPU cores

    # Cache settings
    'use_cache': True,
    'cache_embeddings': True,
}

# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

# Print configuration summary
def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 80)
    print("HYBRID PIPELINE CONFIGURATION (v4 - Neural Reranker)")
    print("=" * 80)
    print("\nMatcher Weights:")
    for matcher, weight in MATCHER_WEIGHTS.items():
        print(f"  {matcher:12s}: {weight:.2f}")

    print(f"\nPipeline Flow:")
    print(f"  1. Matchers: Each returns top-{PIPELINE_CONFIG['matcher_top_k']} candidates")
    print(f"  2. Aggregation: Combines to top-{PIPELINE_CONFIG['aggregation_top_k']} ({AGGREGATION_CONFIG['method']})")
    print(f"  3. Neural Reranker: Filters to top-{PIPELINE_CONFIG['reranker_top_k']} (BGE)")
    print(f"  4. LLM: Final decision on {PIPELINE_CONFIG['reranker_top_k']} candidates")

    print(f"\nNeural Reranker:")
    print(f"  Enabled: {NEURAL_RERANKER_CONFIG['enabled']}")
    print(f"  Model: {NEURAL_RERANKER_CONFIG['model_name']}")
    print(f"  Top-K: {NEURAL_RERANKER_CONFIG['top_k']}")

    print(f"\nLLM Reranking:")
    print(f"  Enabled: {LLM_CONFIG['enabled']}")
    print(f"  Model: {LLM_CONFIG['model']}")
    print(f"  Confidence Threshold: {LLM_CONFIG['confidence_threshold']:.2f}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    print_config_summary()
