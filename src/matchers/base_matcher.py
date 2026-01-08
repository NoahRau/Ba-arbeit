"""
Base Matcher Interface for all ontology matchers.
Defines common interface that KROMA, DeepOnto, and AML must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import pandas as pd


class BaseMatcher(ABC):
    """
    Abstract base class for ontology matchers.
    All matchers (KROMA, DeepOnto, AML) must inherit from this.
    """

    def __init__(self, source_df: pd.DataFrame, target_df: pd.DataFrame):
        """
        Initialize matcher with source and target ontologies.

        Args:
            source_df: DataFrame with S1000D concepts
                Required columns: ['uri', 'label', 'context_text', 'source']
            target_df: DataFrame with target ontology concepts
                Required columns: ['uri', 'label', 'context_text', 'source']
        """
        self.source_df = source_df
        self.target_df = target_df
        self.name = self.__class__.__name__

    @abstractmethod
    def find_candidates(
        self,
        source_concept: Dict[str, Any],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find top-k candidate matches for a source concept.

        Args:
            source_concept: Dictionary with keys:
                - 'uri': Concept URI
                - 'label': Concept label
                - 'context_text': Hierarchical context
            top_k: Number of candidates to return

        Returns:
            List of (target_uri, score) tuples, sorted by score descending
            Example: [
                ('http://ontology.com#Wheel', 0.92),
                ('http://ontology.com#WheelAssembly', 0.85),
                ...
            ]
        """
        pass

    @abstractmethod
    def batch_match(
        self,
        source_concepts: List[Dict[str, Any]],
        top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find candidates for multiple source concepts (batch processing).

        Args:
            source_concepts: List of source concept dictionaries
            top_k: Number of candidates per source concept

        Returns:
            Dictionary mapping source_uri to list of (target_uri, score) tuples
            Example: {
                'source_uri_1': [('target_uri_1', 0.9), ('target_uri_2', 0.8)],
                'source_uri_2': [('target_uri_3', 0.85), ...]
            }
        """
        pass

    def get_name(self) -> str:
        """Get matcher name (e.g., 'KROMAMatcher', 'DeepOntoMatcher')."""
        return self.name

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get matcher statistics.

        Returns:
            Dictionary with statistics like:
            {
                'matcher_name': str,
                'source_count': int,
                'target_count': int,
                'total_comparisons': int
            }
        """
        return {
            'matcher_name': self.name,
            'source_count': len(self.source_df),
            'target_count': len(self.target_df),
            'total_comparisons': len(self.source_df) * len(self.target_df)
        }


class MatcherResult:
    """
    Standardized result format for matcher outputs.
    Used by aggregation layer.
    """

    def __init__(
        self,
        source_uri: str,
        target_uri: str,
        score: float,
        matcher_name: str,
        metadata: Dict[str, Any] = None
    ):
        """
        Args:
            source_uri: Source concept URI
            target_uri: Target concept URI
            score: Confidence score (0.0 to 1.0)
            matcher_name: Name of the matcher that produced this result
            metadata: Optional additional information
        """
        self.source_uri = source_uri
        self.target_uri = target_uri
        self.score = score
        self.matcher_name = matcher_name
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source_uri': self.source_uri,
            'target_uri': self.target_uri,
            'score': self.score,
            'matcher_name': self.matcher_name,
            'metadata': self.metadata
        }

    def __repr__(self):
        return (
            f"MatcherResult(source={self.source_uri}, "
            f"target={self.target_uri}, score={self.score:.3f}, "
            f"matcher={self.matcher_name})"
        )
