# Pipeline v7 Quick Reference

**One-page reference for common operations**

## Setup (Choose One)

### Option A: Neo4j (Recommended)
```bash
# 1. Start Neo4j
docker run -d -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j

# 2. Migrate data
python scripts/migrate_to_neo4j.py

# 3. Run
python scripts/run_pipeline_v7.py --mode graph
```

### Option B: Semantic Blocking
```bash
# 1. Train blocker
python scripts/train_blocker.py

# 2. Run
python scripts/run_pipeline_v7.py --mode blocking
```

## Command Line

```bash
# Run with Neo4j
python scripts/run_pipeline_v7.py --mode graph

# Run with blocking
python scripts/run_pipeline_v7.py --mode blocking

# Auto mode (try graph, fallback to blocking)
python scripts/run_pipeline_v7.py --mode auto

# Enable LLM
python scripts/run_pipeline_v7.py --mode graph --use-llm

# Test with limited concepts
python scripts/run_pipeline_v7.py --mode graph --limit 10

# Custom output
python scripts/run_pipeline_v7.py --mode graph --output results/my_results.csv

# Run demos
python scripts/demo_pipeline_v7.py
```

## Python API

### Basic Usage

```python
from pipeline.hybrid_pipeline_v7 import HybridPipelineV7

# Initialize
pipeline = HybridPipelineV7(
    use_graph_store=True,
    use_blocking=True,
    neo4j_uri="bolt://localhost:7687"
)

# Match concept
result = pipeline.match_concept(source_concept)

# Result
print(result['selected_uri'])
print(result['confidence'])
```

### With Neo4j

```python
pipeline = HybridPipelineV7(
    use_graph_store=True,
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)
```

### With Blocking

```python
pipeline = HybridPipelineV7(
    source_df=source_df,
    target_df=target_df,
    use_graph_store=False,
    use_blocking=True
)
```

### Fallback Mode

```python
pipeline = HybridPipelineV7(
    source_df=source_df,
    target_df=target_df,
    use_graph_store=False,
    use_blocking=False
)
```

## Neo4j Operations

### Connect and Retrieve

```python
from storage.graph_store import KnowledgeGraphStore

# Connect
store = KnowledgeGraphStore(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# Retrieve with reasoning
candidates = store.retrieve_candidates_with_reasoning(
    query_embedding=embedding,
    source_uri=uri,
    top_k=50,
    reasoning_hops=2
)

# Close
store.close()
```

### Ingest Data

```python
# Initialize schema
store.initialize_schema(embedding_dim=768)

# Ingest DataFrames
store.ingest_dataframe(s1000d_df, node_label='S1000DConcept')
store.ingest_dataframe(ontology_df, node_label='OntologyClass')

# Create relationships
store.create_hierarchical_relationships(ontology_df, node_label='OntologyClass')
```

### Multi-Hop Reasoning

```python
# Get reasoning paths
context = store.get_reasoning_context(
    uri="http://example.org/Wheel",
    hops=2,
    max_paths=10
)

# Returns formatted paths:
# [PATH-1] Wheel -> PART_OF -> WheelSystem
# [PATH-2] Wheel -> HAS_PART -> Spoke
```

## Semantic Blocking

### Train Blocker

```python
from blocking.blocker import SemanticBlocker

# Train
blocker = SemanticBlocker(n_clusters=10)
blocker.fit(target_df, embedding_column='embedding')

# Save
blocker.save('semantic_blocker.pkl')
```

### Use Blocker

```python
# Load
blocker = SemanticBlocker.load('semantic_blocker.pkl')

# Get candidates
candidates = blocker.get_blocked_candidates(
    query_embedding,
    top_k_clusters=1,
    include_neighbors=True
)
```

### Adaptive K-Selection

```python
from blocking.blocker import AdaptiveBlocker

# Auto-select k
blocker = AdaptiveBlocker(min_clusters=5, max_clusters=20)
blocker.fit_adaptive(target_df, embedding_column='embedding')

print(f"Optimal k: {blocker.n_clusters}")
```

## Configuration

### Pipeline Parameters

```python
HybridPipelineV7(
    # Storage
    use_graph_store=True,
    use_blocking=True,
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",

    # Stages
    use_neural_reranker=True,
    use_llm=False,

    # Tuning
    matcher_top_k=50,
    aggregation_top_k=60,
    reranker_top_k=20,
    aggregation_method='rank_fusion'
)
```

### Neo4j Configuration

```python
KnowledgeGraphStore(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)
```

### Blocking Configuration

```python
SemanticBlocker(
    n_clusters=10,           # Number of clusters
    use_minibatch=False,     # MiniBatch K-Means
    random_state=42          # Reproducibility
)
```

## Performance Tuning

### Increase Recall (Blocking)

```python
# More clusters (smaller blocks)
blocker = SemanticBlocker(n_clusters=20)

# Include neighbor clusters
candidates = blocker.get_blocked_candidates(
    query_embedding,
    top_k_clusters=1,
    include_neighbors=True
)

# Use multiple clusters
candidates = blocker.get_blocked_candidates(
    query_embedding,
    top_k_clusters=2
)
```

### Increase Speed

```python
# Fewer clusters (larger blocks)
blocker = SemanticBlocker(n_clusters=5)

# Use MiniBatch for large datasets
blocker = SemanticBlocker(n_clusters=10, use_minibatch=True)

# Don't include neighbors
candidates = blocker.get_blocked_candidates(
    query_embedding,
    include_neighbors=False
)
```

### Graph Reasoning Tuning

```python
# Fewer hops (faster)
candidates = store.retrieve_candidates_with_reasoning(
    query_embedding,
    reasoning_hops=1
)

# More hops (better context)
candidates = store.retrieve_candidates_with_reasoning(
    query_embedding,
    reasoning_hops=3
)
```

## Troubleshooting

### Neo4j Connection Failed

```bash
# Check if running
docker ps | grep neo4j

# Start Neo4j
docker run -d -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j

# Use fallback
python scripts/run_pipeline_v7.py --mode blocking
```

### Blocker Not Found

```bash
# Train blocker
python scripts/train_blocker.py

# Check cache
ls cache/semantic_blocker.pkl

# Use Neo4j instead
python scripts/run_pipeline_v7.py --mode graph
```

### Low Performance

```bash
# Make sure you're using optimization
python scripts/run_pipeline_v7.py --mode auto  # Not fallback!

# Check embeddings are cached
# Don't regenerate on every run

# Use Neo4j for best performance
python scripts/run_pipeline_v7.py --mode graph
```

## Common Tasks

### Run Quick Test

```bash
python scripts/run_pipeline_v7.py --mode auto --limit 10
```

### Full Production Run

```bash
python scripts/run_pipeline_v7.py --mode graph --use-llm
```

### Retrain Blocker

```bash
# After updating embeddings
python scripts/train_blocker.py
```

### Update Neo4j Data

```bash
# Re-run migration
python scripts/migrate_to_neo4j.py
```

### Compare Modes

```python
# Run demo comparisons
python scripts/demo_pipeline_v7.py
```

## File Locations

```
Project Structure:
├── src/
│   ├── storage/
│   │   ├── graph_store.py          # Neo4j backend
│   │   └── __init__.py
│   ├── blocking/
│   │   ├── blocker.py              # Semantic blocking
│   │   └── __init__.py
│   └── pipeline/
│       ├── hybrid_pipeline_v7.py   # Main pipeline
│       └── hybrid_pipeline.py      # Old v4
├── scripts/
│   ├── run_pipeline_v7.py          # Production runner
│   ├── demo_pipeline_v7.py         # Demos
│   ├── train_blocker.py            # Train blocker
│   └── migrate_to_neo4j.py         # Migrate to Neo4j
├── docs/
│   ├── PIPELINE_V7.md              # Complete guide
│   ├── NEO4J_BACKEND.md            # Neo4j docs
│   ├── SEMANTIC_BLOCKING.md        # Blocking docs
│   └── V7_QUICK_REFERENCE.md       # This file
└── cache/
    └── semantic_blocker.pkl        # Trained blocker
```

## Expected Performance

| Metric | v4 | v7 (Graph) | v7 (Blocking) |
|--------|-----|------------|---------------|
| **Time** | 5-10 min | 30-60 sec | 45-90 sec |
| **Speedup** | 1x | 10x | 7x |
| **F1** | 41.0% | 41-42% | 40-41% |

## Key Concepts

### Retrieval Modes

- **Graph**: Neo4j vector + graph (26x reduction, +1% F1)
- **Blocking**: K-Means clustering (10x reduction, -1% F1)
- **Fallback**: No optimization (1x, baseline quality)
- **Auto**: Try graph → blocking → fallback

### Multi-Hop Reasoning

Extracts graph paths:
```
Wheel -> PART_OF -> WheelSystem
Wheel -> HAS_PART -> Spoke
```

Adds to neural reranker input for better context.

### Semantic Blocking

1. Cluster targets with K-Means
2. Route query to nearest cluster
3. Match only within cluster
4. 10x speedup, <1% F1 loss

## Resources

- **Full Guide**: `docs/PIPELINE_V7.md`
- **Neo4j Setup**: `docs/NEO4J_BACKEND.md`
- **Blocking Guide**: `docs/SEMANTIC_BLOCKING.md`
- **Reasoning**: `docs/MULTIHOP_REASONING.md`
- **Implementation Summary**: `docs/V7_IMPLEMENTATION_SUMMARY.md`

---

**Version**: 7.0.0
**Last Updated**: 2025-01-12
**Status**: Production Ready
