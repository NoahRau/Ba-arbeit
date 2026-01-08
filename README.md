# State-of-the-Art Hybrid Ontology Matching Pipeline

A modern, modular implementation of ontology matching combining multiple complementary strategies for maximum precision and recall.

## 🎯 Overview

This project implements a 4-stage hybrid pipeline for ontology alignment between S1000D technical documentation and OWL ontologies:

1. **Candidate Generation**: KROMA (DMC-based) + DeepOnto (BERT semantic) + String matching
2. **Aggregation**: Weighted rank fusion combining complementary signals
3. **LLM Reranking**: Claude Sonnet 4.5 for intelligent final selection
4. **Validation**: Optional post-processing and validation

## 📁 Project Structure

```
BA-arbeit/
├── src/                          # Source code modules
│   ├── matchers/                 # Matching algorithms
│   │   ├── base_matcher.py      # Abstract matcher interface
│   │   ├── kroma_matcher.py     # DMC-based heuristic matcher
│   │   ├── deeponto_matcher.py  # BERT semantic matcher
│   │   ├── string_matcher.py    # String similarity baseline
│   │   └── aml_matcher.py       # AML wrapper (experimental)
│   ├── aggregation/             # Score aggregation
│   │   └── weighted_aggregator.py
│   ├── reranking/               # LLM reranking
│   │   └── llm_reranker.py
│   ├── pipeline/                # Main pipeline orchestration
│   │   └── hybrid_pipeline.py
│   ├── evaluation/              # Evaluation utilities
│   │   └── kroma_evaluation.py
│   ├── validation/              # Validation logic (placeholder)
│   └── data_loader.py           # Data loading with hierarchical context
│
├── scripts/                      # Executable scripts
│   ├── generate_matches_for_annotation.py
│   ├── evaluate_annotated_matches.py
│   ├── create_gold_standard.py
│   └── run_benchmark.py
│
├── data/                         # Data files
│   ├── s1000d/                  # S1000D XML files
│   ├── ontologies/              # OWL ontology files
│   └── results/                 # Generated matches and evaluations
│
├── cache/                        # Cache files
│   ├── embeddings/              # BERT embeddings cache
│   └── logs/                    # Execution logs
│
├── docs/                         # Documentation
│   ├── ANNOTATION_GUIDE.md      # Manual annotation instructions
│   ├── DEMO_EVALUATION_REPORT.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── QUICK_START.md
│   └── REFACTORING_PLAN.md
│
├── legacy/                       # Old implementation (archived)
├── tools/                        # External tools (AML)
├── app.py                        # Main application entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BA-arbeit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Claude API key (for LLM reranking):
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Generate Matches for Annotation

Run the hybrid pipeline on all S1000D concepts:

```bash
python scripts/generate_matches_for_annotation.py
```

This generates `data/results/hybrid_pipeline_matches.csv` with:
- Pipeline's selected matches
- Top-5 candidates for each concept
- Confidence scores and methods used

### Manual Annotation

1. Open `data/results/hybrid_pipeline_matches.csv`
2. Fill in the `is_match_manual` column (TRUE/FALSE)
3. See `docs/ANNOTATION_GUIDE.md` for detailed instructions
4. Save as `data/results/hybrid_pipeline_matches_ANNOTATED.csv`

### Evaluate Results

After manual annotation, compute metrics:

```bash
python scripts/evaluate_annotated_matches.py data/results/hybrid_pipeline_matches_ANNOTATED.csv --plot
```

This generates:
- Evaluation report (Markdown) with Precision, Recall, F1, MCC
- Confusion matrix visualization
- Error analysis (false positives/negatives)
- Metrics JSON file

## 🧩 Core Components

### 1. Matchers

#### KROMA Matcher (`src/matchers/kroma_matcher.py`)
- Exploits S1000D DMC structure (chapter codes)
- Hierarchical component matching
- Domain-specific heuristics
- **Weight**: 0.4 (highest)

#### DeepOnto Matcher (`src/matchers/deeponto_matcher.py`)
- ModernBERT embeddings (8192 token context)
- Subsumption filtering (parent-child not a match)
- Sibling detection
- Hierarchical context integration
- **Weight**: 0.35

#### String Matcher (`src/matchers/string_matcher.py`)
- Jaccard similarity (token overlap)
- Edit distance (sequence similarity)
- Context-aware scoring
- **Weight**: 0.25

### 2. Aggregation (`src/aggregation/weighted_aggregator.py`)

Combines matcher outputs using:
- **Rank Fusion (RRF)**: Reciprocal Rank Fusion for robust aggregation
- **Weighted Sum**: Direct score combination (alternative)

### 3. LLM Reranking (`src/reranking/llm_reranker.py`)

- Model: Claude Sonnet 4.5
- Listwise evaluation of top-5 candidates
- German language prompts
- Conservative threshold (0.95 confidence)
- NULL support for "no good match"

### 4. Pipeline (`src/pipeline/hybrid_pipeline.py`)

Orchestrates all stages:
```python
from src.pipeline.hybrid_pipeline import HybridPipeline
from src.data_loader import load_all_concepts

# Load data
df = load_all_concepts()
s1000d_df = df[df['source'] == 's1000d']
ontology_df = df[df['source'] == 'bike_ontology']

# Initialize pipeline
pipeline = HybridPipeline(
    s1000d_df,
    ontology_df,
    use_llm=True,
    aggregation_method='rank_fusion'
)

# Match a concept
result = pipeline.match_concept(source_concept, top_k=5)
```

## 📊 Performance

**Current results** (on 56 S1000D concepts):
- **Precision**: 88.89% (8 TP, 1 FP)
- **Recall**: 61.54% (8 TP, 5 FN)
- **F1-Score**: 72.73%
- **MCC**: 0.690

The pipeline prioritizes **precision over recall** - better to miss a match than create a wrong one.

## 🔧 Configuration

### Matcher Weights

Edit `src/aggregation/weighted_aggregator.py`:
```python
self.weights = {
    'kroma': 0.4,      # DMC-based heuristics
    'deeponto': 0.35,  # BERT semantic
    'string': 0.25     # String similarity
}
```

### LLM Confidence Threshold

Edit `src/reranking/llm_reranker.py`:
```python
self.confidence_threshold = 0.95  # Lower = more matches
```

### Top-K Candidates

Edit pipeline calls:
```python
pipeline.match_concept(concept, top_k=5)  # Number of candidates
```

## 📚 Documentation

- **[Quick Start](docs/QUICK_START.md)**: Get started quickly
- **[Annotation Guide](docs/ANNOTATION_GUIDE.md)**: Manual annotation instructions
- **[Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)**: Technical details
- **[Refactoring Plan](docs/REFACTORING_PLAN.md)**: Architecture decisions

## 🧪 Testing

Run evaluation on KROMA matcher:
```bash
python src/evaluation/kroma_evaluation.py
```

Run full benchmark:
```bash
python scripts/run_benchmark.py
```

## 🔍 Data Format

### S1000D XML Structure
```xml
<dmodule>
  <identAndStatusSection>
    <dmAddress>
      <dmIdent>
        <dmCode>DMC-S1000DBIKE-AAA-DA0-10-10-00AA-921A-A</dmCode>
      </dmIdent>
    </dmAddress>
  </identAndStatusSection>
  <content>...</content>
</dmodule>
```

### OWL Ontology
BikeOntology from: https://giuliamenna.github.io/BikeOntology/

Classes with hierarchical structure:
- `owl:Class` with `rdfs:subClassOf` relationships
- Named individuals with `rdf:type` assertions

## 🛠️ Development

### Adding a New Matcher

1. Create `src/matchers/my_matcher.py`:
```python
from src.matchers.base_matcher import BaseMatcher

class MyMatcher(BaseMatcher):
    def find_candidates(self, source_concept, top_k=10):
        # Your matching logic
        return [(uri, score), ...]

    def batch_match(self, source_concepts, top_k=10):
        # Batch implementation
        return {uri: candidates, ...}
```

2. Register in `src/pipeline/hybrid_pipeline.py`
3. Add weight in `src/aggregation/weighted_aggregator.py`

### Running Tests

```bash
# Test data loader
python -c "from src.data_loader import load_all_concepts; print(load_all_concepts())"

# Test pipeline
python -c "from src.pipeline.hybrid_pipeline import HybridPipeline; help(HybridPipeline)"
```

## 📦 Dependencies

Key packages:
- `deeponto>=0.9.0` - DeepOnto framework
- `sentence-transformers` - ModernBERT embeddings
- `anthropic>=0.40.0` - Claude API
- `owlready2>=0.46` - OWL processing
- `scikit-learn` - Evaluation metrics
- `pandas`, `numpy` - Data manipulation

See `requirements.txt` for full list.

## 🤝 Contributing

This is a research project for ontology matching. For improvements:

1. Create a new branch
2. Make your changes
3. Test thoroughly
4. Document changes in `docs/`

## 📄 License

Academic research project - see LICENSE file.

## 🙏 Acknowledgments

- **DeepOnto Framework**: Oxford/Manchester (https://github.com/KRR-Oxford/DeepOnto)
- **BikeOntology**: Giulia Menna (https://giuliamenna.github.io/BikeOntology/)
- **S1000D Standard**: ASD specification for technical documentation
- **Claude API**: Anthropic for LLM capabilities

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Last Updated**: 2026-01-08
