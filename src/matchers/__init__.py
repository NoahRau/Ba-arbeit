"""
Ontology Matchers Package.

Contains all matcher implementations:
- KROMA: S1000D DMC-based heuristic matcher
- DeepOnto: BERTMap/BERTSubs semantic matcher
- AML: AgreementMakerLight string-based matcher
"""

from .base_matcher import BaseMatcher, MatcherResult

__all__ = ['BaseMatcher', 'MatcherResult']
