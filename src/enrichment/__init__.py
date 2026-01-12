"""
Enrichment Module for Context-Aware Matching.

This module provides feature extraction and context enrichment
for technical documentation matching.
"""

from .rich_document import RichDocument
from .feature_extractor import FeatureExtractor

__all__ = ['RichDocument', 'FeatureExtractor']
