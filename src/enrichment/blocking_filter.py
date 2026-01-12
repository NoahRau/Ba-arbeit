"""
Feature-Based Blocking Filter.

Phase A of contextual matching: Filter candidates based on hard structural
and semantic features before expensive similarity computation.
"""

from typing import List, Set, Dict, Any, Optional
from .rich_document import RichDocument


class BlockingFilter:
    """
    Filters candidates using structural and semantic features.

    Blocks (rejects) candidates that:
    - Come from incompatible hierarchical levels
    - Have no overlapping domain/keywords
    - Are from completely different technical domains
    """

    def __init__(
        self,
        max_depth_difference: int = 2,
        min_keyword_overlap: int = 1,
        require_domain_match: bool = False,
        require_entity_overlap: bool = False
    ):
        """
        Initialize blocking filter.

        Args:
            max_depth_difference: Maximum allowed hierarchy depth difference
            min_keyword_overlap: Minimum number of overlapping keywords required
            require_domain_match: If True, require technical domain to match
            require_entity_overlap: If True, require at least one entity overlap
        """
        self.max_depth_difference = max_depth_difference
        self.min_keyword_overlap = min_keyword_overlap
        self.require_domain_match = require_domain_match
        self.require_entity_overlap = require_entity_overlap

    def filter_candidates(
        self,
        source_doc: RichDocument,
        candidate_docs: List[RichDocument]
    ) -> List[RichDocument]:
        """
        Filter candidates based on structural and semantic features.

        Args:
            source_doc: Source document
            candidate_docs: List of candidate documents

        Returns:
            Filtered list of candidates that pass blocking criteria
        """
        filtered = []

        for candidate in candidate_docs:
            if self._should_keep_candidate(source_doc, candidate):
                filtered.append(candidate)

        return filtered

    def _should_keep_candidate(
        self,
        source: RichDocument,
        candidate: RichDocument
    ) -> bool:
        """
        Determine if candidate should be kept based on blocking rules.

        Args:
            source: Source document
            candidate: Candidate document

        Returns:
            True if candidate passes all blocking criteria
        """
        # Rule 1: Check hierarchy depth compatibility
        if not self._check_depth_compatibility(source, candidate):
            return False

        # Rule 2: Check keyword overlap
        if not self._check_keyword_overlap(source, candidate):
            return False

        # Rule 3: Check domain compatibility (if required)
        if self.require_domain_match:
            if not self._check_domain_match(source, candidate):
                return False

        # Rule 4: Check entity overlap (if required)
        if self.require_entity_overlap:
            if not self._check_entity_overlap(source, candidate):
                return False

        # Rule 5: Check hierarchy compatibility
        if not self._check_hierarchy_compatibility(source, candidate):
            return False

        return True

    def _check_depth_compatibility(
        self,
        source: RichDocument,
        candidate: RichDocument
    ) -> bool:
        """
        Check if documents are at compatible hierarchy depths.

        Args:
            source: Source document
            candidate: Candidate document

        Returns:
            True if depth difference is acceptable
        """
        depth_diff = abs(source.depth_level - candidate.depth_level)
        return depth_diff <= self.max_depth_difference

    def _check_keyword_overlap(
        self,
        source: RichDocument,
        candidate: RichDocument
    ) -> bool:
        """
        Check if documents have sufficient keyword overlap.

        Args:
            source: Source document
            candidate: Candidate document

        Returns:
            True if keyword overlap meets minimum
        """
        if not source.keywords or not candidate.keywords:
            return True  # Don't filter if keywords not available

        source_keywords = set(kw.lower() for kw in source.keywords)
        candidate_keywords = set(kw.lower() for kw in candidate.keywords)

        overlap = len(source_keywords & candidate_keywords)

        return overlap >= self.min_keyword_overlap

    def _check_domain_match(
        self,
        source: RichDocument,
        candidate: RichDocument
    ) -> bool:
        """
        Check if documents are from the same technical domain.

        Args:
            source: Source document
            candidate: Candidate document

        Returns:
            True if domains match or are compatible
        """
        if not source.technical_domain or not candidate.technical_domain:
            return True  # Don't filter if domain not classified

        # Exact match
        if source.technical_domain == candidate.technical_domain:
            return True

        # Check for domain compatibility (e.g., 'wheel' compatible with 'wheel_system')
        if self._are_domains_compatible(source.technical_domain, candidate.technical_domain):
            return True

        return False

    def _are_domains_compatible(self, domain1: str, domain2: str) -> bool:
        """
        Check if two domains are compatible (one is subset of other).

        Args:
            domain1: First domain
            domain2: Second domain

        Returns:
            True if compatible
        """
        # Simple substring check
        return domain1 in domain2 or domain2 in domain1

    def _check_entity_overlap(
        self,
        source: RichDocument,
        candidate: RichDocument
    ) -> bool:
        """
        Check if documents have overlapping named entities.

        Args:
            source: Source document
            candidate: Candidate document

        Returns:
            True if at least one entity overlaps
        """
        if not source.entities or not candidate.entities:
            return True  # Don't filter if entities not available

        source_entities = set(e.lower() for e in source.entities)
        candidate_entities = set(e.lower() for e in candidate.entities)

        return len(source_entities & candidate_entities) > 0

    def _check_hierarchy_compatibility(
        self,
        source: RichDocument,
        candidate: RichDocument
    ) -> bool:
        """
        Check if hierarchy paths are compatible.

        Rejects if one hierarchy is completely unrelated to the other.

        Args:
            source: Source document
            candidate: Candidate document

        Returns:
            True if hierarchies are compatible
        """
        if not source.hierarchy_path or not candidate.hierarchy_path:
            return True  # Don't filter if hierarchy not available

        # If hierarchies share any common ancestors, they're compatible
        source_path_lower = [p.lower() for p in source.hierarchy_path]
        candidate_path_lower = [p.lower() for p in candidate.hierarchy_path]

        # Check for overlap
        source_set = set(source_path_lower)
        candidate_set = set(candidate_path_lower)

        # If they share any hierarchical elements, compatible
        if source_set & candidate_set:
            return True

        # Check if root elements are semantically similar
        # (e.g., "BikeComponent" vs "BicyclePart")
        if source_path_lower and candidate_path_lower:
            source_root = source_path_lower[0]
            candidate_root = candidate_path_lower[0]

            # Check if roots have common substring
            if len(source_root) > 4 and len(candidate_root) > 4:
                # Simple similarity: share first 4 chars
                if source_root[:4] == candidate_root[:4]:
                    return True

        # If no hierarchy overlap and not similar roots, still allow
        # (we don't want to be too restrictive in blocking)
        return True


class ContextualBlockingFilter(BlockingFilter):
    """
    Enhanced blocking filter that considers full context, not just features.
    """

    def __init__(
        self,
        max_depth_difference: int = 3,  # Increased from 2 - more lenient
        min_keyword_overlap: int = 1,  # Decreased from 2 - more lenient
        require_domain_match: bool = False,
        require_entity_overlap: bool = False
    ):
        """Initialize with lenient contextual settings for better recall."""
        super().__init__(
            max_depth_difference=max_depth_difference,
            min_keyword_overlap=min_keyword_overlap,
            require_domain_match=require_domain_match,
            require_entity_overlap=require_entity_overlap
        )

    def _should_keep_candidate(
        self,
        source: RichDocument,
        candidate: RichDocument
    ) -> bool:
        """
        Enhanced filtering with context awareness.

        Args:
            source: Source document
            candidate: Candidate document

        Returns:
            True if candidate passes all criteria
        """
        # First apply base blocking rules
        if not super()._should_keep_candidate(source, candidate):
            return False

        # Additional contextual check: parent context similarity
        if source.parent_context and candidate.parent_context:
            if not self._check_parent_similarity(source, candidate):
                # Don't hard block on parent mismatch, just note it
                pass

        return True

    def _check_parent_similarity(
        self,
        source: RichDocument,
        candidate: RichDocument
    ) -> bool:
        """
        Check if parent contexts are similar.

        Args:
            source: Source document
            candidate: Candidate document

        Returns:
            True if parents are similar
        """
        if not source.parent_context or not candidate.parent_context:
            return True

        # Simple word overlap in parent context
        source_words = set(source.parent_context.lower().split())
        candidate_words = set(candidate.parent_context.lower().split())

        overlap = len(source_words & candidate_words)

        return overlap >= 2


def main():
    """Test blocking filter."""
    print("=" * 70)
    print("BLOCKING FILTER TEST")
    print("=" * 70)

    # Create test documents
    source = RichDocument(
        uri="test:source:1",
        label="Wheel Assembly Procedure",
        raw_content="Install the wheel assembly on the front fork...",
        hierarchy_path=["Bicycle", "FrontAssembly", "Wheel"],
        depth_level=3,
        technical_domain="wheel_system",
        entities=["wheel", "fork", "install"],
        keywords=["wheel", "assembly", "install", "front", "fork"],
        source="s1000d"
    )

    candidates = [
        # Good match - same domain, similar depth
        RichDocument(
            uri="onto:wheel:1",
            label="Wheel",
            raw_content="A wheel component of a bicycle...",
            hierarchy_path=["BikeComponent", "Wheel"],
            depth_level=2,
            technical_domain="wheel_system",
            entities=["wheel", "component"],
            keywords=["wheel", "component", "bicycle"],
            source="ontology"
        ),
        # Bad match - wrong domain, no overlap
        RichDocument(
            uri="onto:hydraulic:1",
            label="Hydraulic System",
            raw_content="The hydraulic brake system...",
            hierarchy_path=["BikeComponent", "BrakeSystem", "Hydraulic"],
            depth_level=3,
            technical_domain="brake_system",
            entities=["hydraulic", "brake"],
            keywords=["hydraulic", "brake", "system"],
            source="ontology"
        ),
        # Medium match - related domain
        RichDocument(
            uri="onto:tire:1",
            label="Tire",
            raw_content="Tire mounted on wheel rim...",
            hierarchy_path=["BikeComponent", "Wheel", "Tire"],
            depth_level=3,
            technical_domain="wheel_system",
            entities=["tire", "wheel", "rim"],
            keywords=["tire", "wheel", "rim", "mounted"],
            source="ontology"
        ),
    ]

    # Test filter
    print("\nTesting BlockingFilter...")
    filter_strict = BlockingFilter(
        max_depth_difference=1,
        min_keyword_overlap=2,
        require_domain_match=True,
        require_entity_overlap=True
    )

    filtered_strict = filter_strict.filter_candidates(source, candidates)

    print(f"\nSource: {source.label}")
    print(f"  Domain: {source.technical_domain}")
    print(f"  Depth: {source.depth_level}")
    print(f"  Keywords: {source.keywords[:5]}")

    print(f"\nCandidates: {len(candidates)}")
    print(f"Filtered (strict): {len(filtered_strict)}")

    for cand in filtered_strict:
        print(f"  ✓ {cand.label} (domain: {cand.technical_domain}, depth: {cand.depth_level})")

    # Test with lenient filter
    print("\nTesting ContextualBlockingFilter (lenient)...")
    filter_lenient = ContextualBlockingFilter(
        max_depth_difference=2,
        min_keyword_overlap=1
    )

    filtered_lenient = filter_lenient.filter_candidates(source, candidates)
    print(f"Filtered (lenient): {len(filtered_lenient)}")

    for cand in filtered_lenient:
        print(f"  ✓ {cand.label}")

    print("\n" + "=" * 70)
    print("✓ Blocking filter test complete!")


if __name__ == '__main__':
    main()
