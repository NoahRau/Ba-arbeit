"""
Semantic Blocking für Ontology Matching.

Reduziert den Suchraum durch Clustering der Kandidaten
basierend auf Embeddings, sodass nur relevante Cluster
für das Matching betrachtet werden müssen.
"""

import logging
import pickle
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from pathlib import Path

logger = logging.getLogger(__name__)


class SemanticBlocker:
    """
    Semantic Blocking via K-Means Clustering.

    Reduziert Kandidaten-Raum durch:
    1. Clustering der Ontologie-Konzepte (offline)
    2. Zuordnung von Query-Konzepten zu Clustern (online)
    3. Retrieval nur aus relevanten Clustern

    Performance-Verbesserung: O(n*m) → O(n*m/k) wobei k = Anzahl Cluster
    """

    def __init__(
        self,
        n_clusters: int = 10,
        use_minibatch: bool = False,
        random_state: int = 42
    ):
        """
        Initialize semantic blocker.

        Args:
            n_clusters: Number of clusters to create
            use_minibatch: Use MiniBatchKMeans for large datasets
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.use_minibatch = use_minibatch
        self.random_state = random_state

        # K-Means model
        if use_minibatch:
            self.clusterer = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                batch_size=1000
            )
        else:
            self.clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10
            )

        # Data storage
        self.concepts: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_map: Dict[int, List[int]] = {}  # cluster_id -> concept_indices
        self.is_fitted: bool = False

    def fit(
        self,
        concepts_df: pd.DataFrame,
        embedding_column: str = 'embedding'
    ):
        """
        Fit blocker on target concepts (ontology).

        Args:
            concepts_df: DataFrame with concepts and embeddings
            embedding_column: Name of column containing embeddings
        """
        logger.info(f"Fitting SemanticBlocker on {len(concepts_df)} concepts...")

        # Store concepts
        self.concepts = concepts_df.copy()

        # Extract embeddings
        self.embeddings = self._extract_embeddings(concepts_df, embedding_column)

        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("No valid embeddings found in concepts_df")

        # Fit K-Means
        logger.info(f"Running K-Means clustering (k={self.n_clusters})...")
        self.cluster_labels = self.clusterer.fit_predict(self.embeddings)

        # Build cluster map: cluster_id -> [concept_indices]
        self._build_cluster_map()

        # Calculate quality metrics
        self._calculate_metrics()

        self.is_fitted = True
        logger.info(f"✓ Blocker fitted successfully")

    def _extract_embeddings(
        self,
        df: pd.DataFrame,
        embedding_column: str
    ) -> np.ndarray:
        """Extract embeddings from DataFrame."""
        embeddings = []

        for idx, row in df.iterrows():
            emb = row.get(embedding_column)

            if emb is None:
                # Generate random embedding as fallback
                emb = np.random.rand(768)
                logger.warning(f"No embedding for {row.get('label', idx)}, using random")

            # Convert to numpy array
            if isinstance(emb, list):
                emb = np.array(emb)
            elif isinstance(emb, np.ndarray):
                pass
            else:
                emb = np.random.rand(768)

            embeddings.append(emb)

        return np.array(embeddings)

    def _build_cluster_map(self):
        """Build mapping from cluster_id to concept indices."""
        self.cluster_map = {}

        for idx, cluster_id in enumerate(self.cluster_labels):
            cluster_id = int(cluster_id)
            if cluster_id not in self.cluster_map:
                self.cluster_map[cluster_id] = []
            self.cluster_map[cluster_id].append(idx)

        # Log statistics
        cluster_sizes = {cid: len(indices) for cid, indices in self.cluster_map.items()}
        avg_size = np.mean(list(cluster_sizes.values()))
        logger.info(f"Cluster sizes: min={min(cluster_sizes.values())}, "
                   f"max={max(cluster_sizes.values())}, "
                   f"avg={avg_size:.1f}")

    def _calculate_metrics(self):
        """Calculate clustering quality metrics."""
        if len(self.embeddings) < self.n_clusters:
            logger.warning("Too few samples for silhouette score")
            return

        try:
            silhouette = silhouette_score(
                self.embeddings,
                self.cluster_labels,
                sample_size=min(10000, len(self.embeddings))
            )
            logger.info(f"Silhouette Score: {silhouette:.3f}")
        except Exception as e:
            logger.warning(f"Could not calculate silhouette score: {e}")

    def get_relevant_clusters(
        self,
        query_embedding: np.ndarray,
        top_k: int = 1,
        include_neighbors: bool = True
    ) -> List[int]:
        """
        Get relevant cluster IDs for a query.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of closest clusters to retrieve
            include_neighbors: Include neighboring clusters for recall

        Returns:
            List of cluster IDs
        """
        if not self.is_fitted:
            raise ValueError("Blocker not fitted. Call fit() first.")

        # Ensure proper shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Find closest clusters by distance to centroids
        centroids = self.clusterer.cluster_centers_
        distances = np.linalg.norm(centroids - query_embedding, axis=1)

        # Get top-k closest clusters
        top_cluster_ids = np.argsort(distances)[:top_k].tolist()

        # Optionally include neighbor clusters
        if include_neighbors and top_k == 1:
            # Add second-closest cluster for recall
            if len(distances) > 1:
                second_closest = np.argsort(distances)[1]
                top_cluster_ids.append(int(second_closest))

        return top_cluster_ids

    def get_blocked_candidates(
        self,
        query_embedding: np.ndarray,
        top_k_clusters: int = 1,
        include_neighbors: bool = True
    ) -> pd.DataFrame:
        """
        Get candidate concepts from relevant clusters.

        Args:
            query_embedding: Query embedding vector
            top_k_clusters: Number of clusters to search
            include_neighbors: Include neighboring clusters

        Returns:
            DataFrame with candidate concepts
        """
        if not self.is_fitted:
            raise ValueError("Blocker not fitted. Call fit() first.")

        # Get relevant clusters
        cluster_ids = self.get_relevant_clusters(
            query_embedding,
            top_k=top_k_clusters,
            include_neighbors=include_neighbors
        )

        # Collect candidate indices from these clusters
        candidate_indices = []
        for cluster_id in cluster_ids:
            if cluster_id in self.cluster_map:
                candidate_indices.extend(self.cluster_map[cluster_id])

        # Return candidate concepts
        if not candidate_indices:
            logger.warning(f"No candidates found in clusters {cluster_ids}")
            return pd.DataFrame()

        candidates = self.concepts.iloc[candidate_indices].copy()
        candidates['cluster_id'] = [
            self.cluster_labels[idx] for idx in candidate_indices
        ]

        return candidates

    def get_statistics(self) -> Dict[str, Any]:
        """Get blocking statistics."""
        if not self.is_fitted:
            return {'fitted': False}

        cluster_sizes = [len(indices) for indices in self.cluster_map.values()]

        return {
            'fitted': True,
            'n_concepts': len(self.concepts),
            'n_clusters': self.n_clusters,
            'cluster_sizes': {
                'min': min(cluster_sizes),
                'max': max(cluster_sizes),
                'avg': np.mean(cluster_sizes),
                'std': np.std(cluster_sizes)
            },
            'reduction_factor': len(self.concepts) / np.mean(cluster_sizes)
        }

    def get_cluster_labels_for_concepts(self) -> np.ndarray:
        """Get cluster labels for all fitted concepts."""
        if not self.is_fitted:
            raise ValueError("Blocker not fitted")
        return self.cluster_labels

    def get_cluster_distribution(self) -> Dict[int, int]:
        """Get distribution of concepts across clusters."""
        if not self.is_fitted:
            raise ValueError("Blocker not fitted")

        return {cid: len(indices) for cid, indices in self.cluster_map.items()}

    def save(self, filepath: Path):
        """Save blocker to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted blocker")

        data = {
            'n_clusters': self.n_clusters,
            'use_minibatch': self.use_minibatch,
            'random_state': self.random_state,
            'clusterer': self.clusterer,
            'concepts': self.concepts,
            'embeddings': self.embeddings,
            'cluster_labels': self.cluster_labels,
            'cluster_map': self.cluster_map,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Blocker saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'SemanticBlocker':
        """Load blocker from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        blocker = cls(
            n_clusters=data['n_clusters'],
            use_minibatch=data['use_minibatch'],
            random_state=data['random_state']
        )

        blocker.clusterer = data['clusterer']
        blocker.concepts = data['concepts']
        blocker.embeddings = data['embeddings']
        blocker.cluster_labels = data['cluster_labels']
        blocker.cluster_map = data['cluster_map']
        blocker.is_fitted = data['is_fitted']

        logger.info(f"Blocker loaded from {filepath}")
        return blocker


class AdaptiveBlocker(SemanticBlocker):
    """
    Adaptive semantic blocker that automatically determines optimal k.

    Uses elbow method to find optimal number of clusters.
    """

    def __init__(
        self,
        max_clusters: int = 20,
        min_clusters: int = 5,
        use_minibatch: bool = False,
        random_state: int = 42
    ):
        """
        Initialize adaptive blocker.

        Args:
            max_clusters: Maximum number of clusters to try
            min_clusters: Minimum number of clusters
            use_minibatch: Use MiniBatchKMeans
            random_state: Random state
        """
        self.max_clusters = max_clusters
        self.min_clusters = min_clusters

        # Start with mid-range
        n_clusters = (max_clusters + min_clusters) // 2

        super().__init__(
            n_clusters=n_clusters,
            use_minibatch=use_minibatch,
            random_state=random_state
        )

    def fit_adaptive(
        self,
        concepts_df: pd.DataFrame,
        embedding_column: str = 'embedding'
    ):
        """
        Fit blocker with automatic k selection.

        Uses elbow method on inertia to find optimal k.

        Args:
            concepts_df: DataFrame with concepts and embeddings
            embedding_column: Name of embedding column
        """
        logger.info(f"Fitting AdaptiveBlocker (k={self.min_clusters}-{self.max_clusters})...")

        # Extract embeddings
        self.concepts = concepts_df.copy()
        self.embeddings = self._extract_embeddings(concepts_df, embedding_column)

        # Try different k values
        inertias = []
        k_values = range(self.min_clusters, self.max_clusters + 1)

        for k in k_values:
            if self.use_minibatch:
                clusterer = MiniBatchKMeans(
                    n_clusters=k,
                    random_state=self.random_state,
                    batch_size=1000
                )
            else:
                clusterer = KMeans(
                    n_clusters=k,
                    random_state=self.random_state,
                    n_init=5  # Fewer inits for speed
                )

            clusterer.fit(self.embeddings)
            inertias.append(clusterer.inertia_)

            logger.info(f"k={k}: inertia={clusterer.inertia_:.2f}")

        # Find elbow (simple method: largest decrease)
        deltas = np.diff(inertias)
        optimal_k = k_values[np.argmin(deltas)]

        logger.info(f"Optimal k (elbow method): {optimal_k}")

        # Fit with optimal k
        self.n_clusters = optimal_k

        if self.use_minibatch:
            self.clusterer = MiniBatchKMeans(
                n_clusters=optimal_k,
                random_state=self.random_state,
                batch_size=1000
            )
        else:
            self.clusterer = KMeans(
                n_clusters=optimal_k,
                random_state=self.random_state,
                n_init=10
            )

        self.cluster_labels = self.clusterer.fit_predict(self.embeddings)
        self._build_cluster_map()
        self._calculate_metrics()

        self.is_fitted = True
        logger.info(f"✓ Adaptive blocker fitted with k={optimal_k}")


def test_semantic_blocker():
    """Test semantic blocker."""
    print("=" * 80)
    print("SEMANTIC BLOCKER TEST")
    print("=" * 80)

    # Create test data
    np.random.seed(42)

    # Simulate 3 clusters in 768-dim space
    cluster1 = np.random.randn(300, 768) + [1, 0, 0] * 256  # Brake concepts
    cluster2 = np.random.randn(400, 768) + [0, 1, 0] * 256  # Wheel concepts
    cluster3 = np.random.randn(200, 768) + [0, 0, 1] * 256  # Chain concepts

    embeddings = np.vstack([cluster1, cluster2, cluster3])

    # Create DataFrame
    concepts = []
    labels = ['Brake'] * 300 + ['Wheel'] * 400 + ['Chain'] * 200

    for i, emb in enumerate(embeddings):
        concepts.append({
            'uri': f'concept:{i}',
            'label': labels[i],
            'embedding': emb
        })

    concepts_df = pd.DataFrame(concepts)

    # Test blocker
    print("\n[1/3] Fitting blocker...")
    blocker = SemanticBlocker(n_clusters=3)
    blocker.fit(concepts_df)

    stats = blocker.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test blocking
    print("\n[2/3] Testing blocking...")
    query_emb = np.random.randn(768) + [1, 0, 0] * 256  # Should match cluster 1

    candidates = blocker.get_blocked_candidates(query_emb, top_k_clusters=1)
    print(f"\nQuery (Brake-like): Retrieved {len(candidates)} candidates")
    print(f"Labels: {candidates['label'].value_counts().to_dict()}")

    # Calculate reduction
    reduction = len(concepts_df) / len(candidates)
    print(f"\nSearch space reduction: {reduction:.1f}x")

    # Test adaptive blocker
    print("\n[3/3] Testing adaptive blocker...")
    adaptive = AdaptiveBlocker(max_clusters=5, min_clusters=2)
    adaptive.fit_adaptive(concepts_df)

    print(f"Optimal k: {adaptive.n_clusters}")

    print("\n" + "=" * 80)
    print("✓ Test complete")


if __name__ == '__main__':
    test_semantic_blocker()
