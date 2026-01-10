"""
DeepOnto-Inspired Semantic Matcher.

Combines BERT embeddings with ontology reasoning:
- Hierarchical context embeddings
- Subsumption-aware matching (parent-child not a match)
- Sibling detection
- Semantic similarity with structural constraints

Based on principles from DeepOnto framework (Oxford/Manchester):
https://github.com/KRR-Oxford/DeepOnto
"""

import re
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    from .base_matcher import BaseMatcher
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from matchers.base_matcher import BaseMatcher

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / 'cache' / 'embeddings'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class DeepOntoMatcher(BaseMatcher):
    """
    Semantic matcher using BERT embeddings + ontology reasoning.

    Features:
    1. ModernBERT embeddings for semantic similarity
    2. Hierarchical context integration
    3. Subsumption filtering (parent-child are NOT matches)
    4. Sibling detection (siblings are NOT matches)
    5. Label + context fusion
    """

    def __init__(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        model_name: str = 'answerdotai/ModernBERT-base',
        cache_file: str = None
    ):
        super().__init__(source_df, target_df)

        self.model_name = model_name
        if cache_file is None:
            self.cache_file = CACHE_DIR / 'deeponto_embeddings_cache.pkl'
        else:
            self.cache_file = Path(cache_file)

        # Load BERT model
        print(f"  Loading {model_name}...")
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True
        )
        print(f"  ✓ Model loaded (max length: {self.model.max_seq_length})")

        # Build or load embeddings
        self.target_embeddings = self._build_embeddings()

    def _build_embeddings(self) -> np.ndarray:
        """
        Build or load BERT embeddings for target concepts.

        Returns:
            Array of embeddings (n_concepts, embedding_dim)
        """
        # Try to load from cache
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)

                if cache['count'] == len(self.target_df):
                    print(f"  ✓ Loaded cached embeddings ({cache['count']} concepts)")
                    return cache['embeddings']
            except Exception as e:
                print(f"  Warning: Failed to load cache: {e}")

        # Build embeddings
        print(f"  Building embeddings for {len(self.target_df)} concepts...")

        texts = []
        for idx, row in self.target_df.iterrows():
            text = self._create_embedding_text(row)
            texts.append(text)

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=8,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Cache embeddings
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'count': len(self.target_df),
                    'embeddings': embeddings
                }, f)
            print(f"  ✓ Cached embeddings to {self.cache_file}")
        except Exception as e:
            print(f"  Warning: Failed to cache embeddings: {e}")

        return embeddings

    def _create_embedding_text(self, row: pd.Series) -> str:
        """
        Create text for embedding from concept data.

        Uses hierarchical context + label for better semantic representation.

        Args:
            row: DataFrame row with concept data

        Returns:
            Text string for embedding
        """
        label = row.get('label', '')
        context = row.get('context_text', '')

        # For hierarchical concepts, use full context
        # For flat concepts, just use label
        if context and len(context) > len(label):
            # Limit to 500 chars (ModernBERT can handle 8K, but we want efficiency)
            if len(context) > 500:
                # Take hierarchy + beginning of description
                return context[:500]
            return context
        else:
            return label

    def find_candidates(
        self,
        source_concept: Dict[str, Any],
        top_k: int = 10,
        use_mmr: bool = True,
        lambda_param: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Find top-k candidates using semantic similarity + ontology reasoning + MMR diversity.

        Args:
            source_concept: Source concept dict with 'uri', 'label', 'context_text'
            top_k: Number of candidates to return
            use_mmr: Use Maximal Marginal Relevance for diversity (default: True)
            lambda_param: MMR balance parameter (0=max diversity, 1=max relevance)

        Returns:
            List of (target_uri, score) tuples
        """
        # Create embedding for source
        source_text = self._create_embedding_text(pd.Series(source_concept))
        source_embedding = self.model.encode([source_text], convert_to_numpy=True)[0]

        # Compute cosine similarities
        similarities = self._cosine_similarity(source_embedding, self.target_embeddings)

        if use_mmr:
            # Use MMR for diverse candidate selection
            candidates = self._mmr_candidate_selection(
                source_concept,
                source_embedding,
                similarities,
                top_k=top_k,
                lambda_param=lambda_param
            )
        else:
            # Original greedy selection
            # Get top candidates (before filtering)
            top_indices = np.argsort(similarities)[::-1][:top_k * 3]  # Get more, then filter

            # Apply ontology reasoning filters
            candidates = []
            for idx in top_indices:
                target_row = self.target_df.iloc[idx]
                score = float(similarities[idx])

                # Filter 1: Check if subsumption relationship (parent-child)
                if self._is_subsumption(source_concept, target_row):
                    score *= 0.5  # Penalize subsumption matches

                # Filter 2: Check if siblings (same parent, different concepts)
                if self._are_siblings(source_concept, target_row):
                    score *= 0.6  # Penalize sibling matches

                # Filter 3: Structural compatibility
                if not self._are_structurally_compatible(source_concept, target_row):
                    score *= 0.7  # Penalize structurally incompatible

                candidates.append((target_row['uri'], score))

            # Re-sort after filtering and return top_k
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:top_k]

        return candidates

    def _mmr_candidate_selection(
        self,
        source_concept: Dict[str, Any],
        source_embedding: np.ndarray,
        similarities: np.ndarray,
        top_k: int = 10,
        lambda_param: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Maximal Marginal Relevance (MMR) for diverse candidate selection.

        MMR = argmax[λ * Sim(Di, Q) - (1-λ) * max(Sim(Di, Dj))]
        where:
        - Sim(Di, Q) = Similarity to query (source concept)
        - Sim(Di, Dj) = Similarity to already selected candidates
        - λ = Balance between relevance and diversity

        Args:
            source_concept: Source concept
            source_embedding: Source embedding
            similarities: Precomputed similarities to all targets
            top_k: Number of diverse candidates to select
            lambda_param: Balance parameter (0=max diversity, 1=max relevance)

        Returns:
            List of diverse (target_uri, score) tuples
        """
        # Start with top candidates pool (top 50 most similar)
        candidate_pool_size = min(50, len(self.target_df))
        top_indices = np.argsort(similarities)[::-1][:candidate_pool_size]

        selected_indices = []
        selected_embeddings = []

        for _ in range(min(top_k, len(top_indices))):
            mmr_scores = []

            for idx in top_indices:
                if idx in selected_indices:
                    continue

                # Relevance to query
                relevance = similarities[idx]

                # Apply ontology reasoning penalties to relevance
                target_row = self.target_df.iloc[idx]
                if self._is_subsumption(source_concept, target_row):
                    relevance *= 0.5
                if self._are_siblings(source_concept, target_row):
                    relevance *= 0.6
                if not self._are_structurally_compatible(source_concept, target_row):
                    relevance *= 0.7

                # Diversity: max similarity to already selected
                if len(selected_embeddings) > 0:
                    candidate_emb = self.target_embeddings[idx]
                    # Compute similarity to all selected
                    diversities = [
                        np.dot(candidate_emb / np.linalg.norm(candidate_emb),
                               sel_emb / np.linalg.norm(sel_emb))
                        for sel_emb in selected_embeddings
                    ]
                    max_similarity_to_selected = max(diversities)
                else:
                    max_similarity_to_selected = 0.0

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity_to_selected

                mmr_scores.append((idx, mmr_score, relevance))

            if not mmr_scores:
                break

            # Select best MMR score
            best_idx, best_mmr, best_relevance = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            selected_embeddings.append(self.target_embeddings[best_idx])

        # Build results with original relevance scores (not MMR scores)
        results = []
        for idx in selected_indices:
            target_row = self.target_df.iloc[idx]
            score = float(similarities[idx])

            # Apply ontology reasoning penalties
            if self._is_subsumption(source_concept, target_row):
                score *= 0.5
            if self._are_siblings(source_concept, target_row):
                score *= 0.6
            if not self._are_structurally_compatible(source_concept, target_row):
                score *= 0.7

            results.append((target_row['uri'], score))

        return results

    def batch_match(
        self,
        source_concepts: List[Dict[str, Any]],
        top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Batch matching for multiple source concepts.

        Args:
            source_concepts: List of source concept dicts
            top_k: Number of candidates per source

        Returns:
            Dictionary mapping source_uri to candidates
        """
        results = {}
        for concept in source_concepts:
            source_uri = concept['uri']
            candidates = self.find_candidates(concept, top_k)
            results[source_uri] = candidates

        return results

    def _is_subsumption(self, concept_a: Dict, concept_b: pd.Series) -> bool:
        """
        Check if one concept is a parent/child of the other (subsumption).

        Uses hierarchical context to detect parent-child relationships.

        Args:
            concept_a: Source concept
            concept_b: Target concept

        Returns:
            True if subsumption relationship detected
        """
        context_a = concept_a.get('context_text', '')
        context_b = concept_b.get('context_text', '')

        if not context_a or not context_b:
            return False

        # Extract hierarchy levels
        levels_a = [s.strip() for s in context_a.split('>') if s.strip()]
        levels_b = [s.strip() for s in context_b.split('>') if s.strip()]

        if not levels_a or not levels_b:
            return False

        # Check if one hierarchy is a prefix of the other
        # e.g., "BikeComponent > Wheel" vs "BikeComponent > Wheel > Hub"
        min_len = min(len(levels_a), len(levels_b))

        # If same prefix but different lengths -> parent-child
        if levels_a[:min_len] == levels_b[:min_len] and len(levels_a) != len(levels_b):
            return True

        return False

    def _are_siblings(self, concept_a: Dict, concept_b: pd.Series) -> bool:
        """
        Check if concepts are siblings (same parent, different children).

        Args:
            concept_a: Source concept
            concept_b: Target concept

        Returns:
            True if siblings detected
        """
        context_a = concept_a.get('context_text', '')
        context_b = concept_b.get('context_text', '')

        if not context_a or not context_b:
            return False

        levels_a = [s.strip() for s in context_a.split('>') if s.strip()]
        levels_b = [s.strip() for s in context_b.split('>') if s.strip()]

        if len(levels_a) < 2 or len(levels_b) < 2:
            return False

        # Same parent but different final level
        if (levels_a[:-1] == levels_b[:-1] and
            levels_a[-1].lower() != levels_b[-1].lower()):
            return True

        return False

    def _are_structurally_compatible(self, concept_a: Dict, concept_b: pd.Series) -> bool:
        """
        Check if concepts are structurally compatible for matching.

        Heuristics:
        - Similar hierarchy depth
        - Not wildly different label lengths
        - Compatible types (if available)

        Args:
            concept_a: Source concept
            concept_b: Target concept

        Returns:
            True if compatible
        """
        # Check hierarchy depth similarity
        context_a = concept_a.get('context_text', '')
        context_b = concept_b.get('context_text', '')

        levels_a = len([s for s in context_a.split('>') if s.strip()])
        levels_b = len([s for s in context_b.split('>') if s.strip()])

        # Allow depth difference of max 1
        if abs(levels_a - levels_b) > 1:
            return False

        # Check label length compatibility (very crude heuristic)
        label_a = concept_a.get('label', '')
        label_b = concept_b.get('label', '')

        if label_a and label_b:
            len_ratio = max(len(label_a), len(label_b)) / (min(len(label_a), len(label_b)) + 1)
            # If one label is 3x longer than the other, probably incompatible
            if len_ratio > 3.0:
                return False

        return True

    @staticmethod
    def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between vector and matrix.

        Args:
            vec: Single vector (1D)
            matrix: Matrix of vectors (2D)

        Returns:
            Array of similarities
        """
        vec_norm = vec / np.linalg.norm(vec)
        matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.dot(matrix_norm, vec_norm)


def main():
    """
    Test DeepOnto Matcher.
    """
    from data_loader import load_all_concepts

    print("=" * 70)
    print("DEEPONTO MATCHER TEST")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading data...")
    df = load_all_concepts('bike')

    s1000d_df = df[df['source'] == 's1000d'].reset_index(drop=True)
    ontology_df = df[df['source'] == 'bike_ontology'].reset_index(drop=True)

    print(f"  S1000D concepts: {len(s1000d_df)}")
    print(f"  Ontology concepts: {len(ontology_df)}")

    # Initialize DeepOnto Matcher
    print("\n[2/3] Initializing DeepOnto Matcher...")
    deeponto = DeepOntoMatcher(s1000d_df, ontology_df)

    # Test matching
    print("\n[3/3] Testing semantic matching...")
    print("=" * 70)

    # Test cases
    test_concepts = [
        'S1000DBIKE-AAA-DA0',  # Wheel
        'S1000DBIKE-AAA-DA1',  # Brake
        'S1000DBIKE-AAA-DA2',  # Drivetrain
    ]

    for pattern in test_concepts:
        matching = s1000d_df[s1000d_df['uri'].str.contains(pattern, na=False)]

        if matching.empty:
            continue

        concept = matching.iloc[0].to_dict()

        print(f"\n--- Test: {concept['label']} ---")
        print(f"URI: {concept['uri']}")

        # Find candidates
        candidates = deeponto.find_candidates(concept, top_k=5)

        print(f"\nTop 5 Semantic Candidates:")
        for i, (target_uri, score) in enumerate(candidates, 1):
            target = ontology_df[ontology_df['uri'] == target_uri].iloc[0]
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   Label: {target['label']}")
            print(f"   URI: {target_uri}")

    print("\n" + "=" * 70)
    print("DeepOnto Matcher test completed!")


if __name__ == '__main__':
    main()
