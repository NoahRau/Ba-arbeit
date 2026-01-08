"""
KROMA (Knowledge-Rich Ontology Matching Approach) Matcher.
Specialized matcher for S1000D DMC/SNS code structures.

Uses S1000D-specific heuristics:
- DMC chapter code analysis (e.g., D00, DA0, DA1)
- SNS (System Numbering System) code matching
- Hierarchical path similarity
- Technical terminology mapping
"""

import re
from typing import List, Tuple, Dict, Any
import pandas as pd
from collections import defaultdict

try:
    from .base_matcher import BaseMatcher
except ImportError:
    # For direct script execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from matchers.base_matcher import BaseMatcher


class KROMAMatcher(BaseMatcher):
    """
    Knowledge-Rich Ontology Matching Approach for S1000D.

    Exploits S1000D DMC structure:
    - Model Identification Code
    - System/Subsystem Code (Chapter)
    - Section/Subsection Code
    - Information Code

    Example DMC: S1000DBIKE-AAA-D00-00-00-00AA-041A-A
                 System: BIKE
                 Model: AAA
                 Chapter: D00 (Main Assembly)
                 Section: 041 (Description)
    """

    def __init__(self, source_df: pd.DataFrame, target_df: pd.DataFrame):
        super().__init__(source_df, target_df)

        # DMC Chapter to Ontology Concept Mapping
        # This can be learned from training data or manually curated
        self.dmc_chapter_mappings = self._build_dmc_mappings()

        # Terminology mappings for common S1000D terms
        self.terminology_map = self._build_terminology_map()

    def _build_dmc_mappings(self) -> Dict[str, List[str]]:
        """
        Build mapping from DMC chapter codes to ontology concept keywords.

        Returns:
            Dictionary mapping chapter codes to relevant keywords
        """
        return {
            # Main Systems (Chapter D)
            'D00': ['bicycle', 'bike', 'main', 'assembly', 'general', 'structure'],
            'D05': ['equipment', 'auxiliary', 'accessory'],

            # Wheel System (Chapter DA0)
            'DA0': ['wheel', 'wheelset', 'hub', 'spoke', 'rim', 'tire', 'tyre'],

            # Brake System (Chapter DA1)
            'DA1': ['brake', 'braking', 'calliper', 'caliper', 'disc', 'pad'],

            # Drivetrain (Chapter DA2)
            'DA2': ['drivetrain', 'chain', 'cassette', 'sprocket', 'gear', 'crank', 'chainring'],

            # Frame System (Chapter DA3)
            'DA3': ['frame', 'fork', 'structure', 'chassis'],

            # Steering System (Chapter DA4)
            'DA4': ['steering', 'handlebar', 'stem', 'headset', 'grip'],

            # Seat System (Chapter DA5)
            'DA5': ['seat', 'saddle', 'seatpost', 'post'],

            # Lighting System (Special LIGHTING)
            'LIGHTING': ['light', 'lighting', 'lamp', 'illumination', 'bulb'],
        }

    def _build_terminology_map(self) -> Dict[str, List[str]]:
        """
        Build S1000D to standard terminology mapping.

        Returns:
            Dictionary mapping S1000D terms to standard terms
        """
        return {
            # S1000D specific abbreviations
            'ldg': ['landing'],
            'gr': ['gear'],
            'assy': ['assembly'],
            'sys': ['system'],
            'comp': ['component'],
            'proc': ['procedure'],
            'desc': ['description'],
            'maint': ['maintenance'],
            'rem': ['remove', 'removal'],
            'inst': ['install', 'installation'],
            'insp': ['inspect', 'inspection'],

            # Common variations
            'calliper': ['caliper'],
            'tyre': ['tire'],
            'colour': ['color'],
        }

    def find_candidates(
        self,
        source_concept: Dict[str, Any],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find top-k candidate matches using KROMA heuristics.

        Args:
            source_concept: S1000D concept with 'uri', 'label', 'context_text'
            top_k: Number of candidates to return

        Returns:
            List of (target_uri, score) tuples sorted by score descending
        """
        source_uri = source_concept['uri']
        source_label = source_concept.get('label', '')
        source_context = source_concept.get('context_text', '')

        # Extract DMC codes and hierarchy
        dmc_info = self._parse_dmc_uri(source_uri)

        # Calculate scores for all target concepts
        candidates = []
        for idx, target_row in self.target_df.iterrows():
            target_uri = target_row['uri']
            target_label = target_row.get('label', '')
            target_context = target_row.get('context_text', '')

            # Compute KROMA score
            score = self._compute_kroma_score(
                dmc_info,
                source_label,
                source_context,
                target_label,
                target_context
            )

            if score > 0.0:  # Only include non-zero scores
                candidates.append((target_uri, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:top_k]

    def batch_match(
        self,
        source_concepts: List[Dict[str, Any]],
        top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Batch matching for multiple source concepts.

        Args:
            source_concepts: List of S1000D concepts
            top_k: Number of candidates per source

        Returns:
            Dictionary mapping source_uri to list of (target_uri, score)
        """
        results = {}
        for concept in source_concepts:
            source_uri = concept['uri']
            candidates = self.find_candidates(concept, top_k)
            results[source_uri] = candidates

        return results

    def _parse_dmc_uri(self, uri: str) -> Dict[str, str]:
        """
        Parse DMC code from URI.

        Example:
            s1000d:S1000DBIKE-AAA-DA0-10-20-00AA-921A-A
            -> {
                'system': 'BIKE',
                'model': 'AAA',
                'chapter': 'DA0',
                'section': '10',
                'subsection': '20',
                'info_code': '921'
            }

        Args:
            uri: S1000D URI

        Returns:
            Dictionary with parsed DMC components
        """
        # Extract DMC from URI (after last / or :)
        dmc_code = uri.split('/')[-1].split(':')[-1]

        # Parse DMC structure: S1000D<SYS>-<MODEL>-<CHAP>-<SEC>-<SUBSEC>-...
        parts = dmc_code.split('-')

        result = {
            'system': '',
            'model': '',
            'chapter': '',
            'section': '',
            'subsection': '',
            'info_code': '',
            'full_code': dmc_code
        }

        if len(parts) >= 3:
            # Extract system from first part (e.g., S1000DBIKE -> BIKE)
            system_part = parts[0]
            if 'S1000D' in system_part:
                result['system'] = system_part.replace('S1000D', '')

            result['model'] = parts[1] if len(parts) > 1 else ''
            result['chapter'] = parts[2] if len(parts) > 2 else ''
            result['section'] = parts[3] if len(parts) > 3 else ''
            result['subsection'] = parts[4] if len(parts) > 4 else ''

            # Info code is typically 3 digits (e.g., 041, 921)
            if len(parts) > 6:
                info_part = parts[6]
                # Extract numeric part (e.g., 041A -> 041)
                match = re.match(r'(\d{3})', info_part)
                if match:
                    result['info_code'] = match.group(1)

        return result

    def _compute_kroma_score(
        self,
        dmc_info: Dict[str, str],
        source_label: str,
        source_context: str,
        target_label: str,
        target_context: str
    ) -> float:
        """
        Compute KROMA similarity score using multiple heuristics.

        Args:
            dmc_info: Parsed DMC information
            source_label: S1000D label
            source_context: S1000D context
            target_label: Ontology label
            target_context: Ontology context

        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        weights = {
            'chapter_match': 0.35,      # DMC chapter is very reliable
            'label_overlap': 0.25,      # Direct label similarity
            'context_keywords': 0.20,   # Context contains chapter keywords
            'terminology': 0.15,        # S1000D terminology mapping
            'hierarchy_similarity': 0.05 # Hierarchical path similarity
        }

        # 1. DMC Chapter Match
        chapter = dmc_info.get('chapter', '')
        if chapter:
            chapter_score = self._score_chapter_match(chapter, target_label, target_context)
            score += weights['chapter_match'] * chapter_score

        # 2. Label Overlap
        label_score = self._score_label_overlap(source_label, target_label)
        score += weights['label_overlap'] * label_score

        # 3. Context Keywords
        if chapter:
            context_score = self._score_context_keywords(chapter, target_context)
            score += weights['context_keywords'] * context_score

        # 4. Terminology Mapping
        term_score = self._score_terminology(source_label, target_label)
        score += weights['terminology'] * term_score

        # 5. Hierarchy Similarity
        hier_score = self._score_hierarchy(source_context, target_context)
        score += weights['hierarchy_similarity'] * hier_score

        return min(score, 1.0)  # Cap at 1.0

    def _score_chapter_match(self, chapter: str, target_label: str, target_context: str) -> float:
        """
        Score based on DMC chapter to ontology concept mapping.

        Args:
            chapter: DMC chapter code (e.g., 'DA0', 'DA1')
            target_label: Target concept label
            target_context: Target concept context

        Returns:
            Score 0.0-1.0
        """
        # Get keywords for this chapter
        keywords = self.dmc_chapter_mappings.get(chapter, [])
        if not keywords:
            # Try parent chapter (e.g., DA0 -> D00)
            if len(chapter) > 1:
                parent_chapter = chapter[0] + '00'
                keywords = self.dmc_chapter_mappings.get(parent_chapter, [])

        if not keywords:
            return 0.0

        # Check if any keyword appears in label or context
        target_text = (target_label + ' ' + target_context).lower()

        matches = sum(1 for kw in keywords if kw.lower() in target_text)

        if matches == 0:
            return 0.0
        elif matches == 1:
            return 0.6
        elif matches >= 2:
            return 0.9

        return 0.0

    def _score_label_overlap(self, source_label: str, target_label: str) -> float:
        """
        Score based on word overlap in labels.

        Args:
            source_label: Source label
            target_label: Target label

        Returns:
            Score 0.0-1.0
        """
        # Tokenize and normalize
        source_words = set(re.findall(r'\w+', source_label.lower()))
        target_words = set(re.findall(r'\w+', target_label.lower()))

        # Remove stop words
        stop_words = {'the', 'a', 'an', 'of', 'to', 'in', 'for', 'and', 'or'}
        source_words -= stop_words
        target_words -= stop_words

        if not source_words or not target_words:
            return 0.0

        # Jaccard similarity
        intersection = source_words & target_words
        union = source_words | target_words

        return len(intersection) / len(union) if union else 0.0

    def _score_context_keywords(self, chapter: str, target_context: str) -> float:
        """
        Score based on chapter keywords appearing in target context.

        Args:
            chapter: DMC chapter code
            target_context: Target concept context

        Returns:
            Score 0.0-1.0
        """
        keywords = self.dmc_chapter_mappings.get(chapter, [])
        if not keywords:
            return 0.0

        context_lower = target_context.lower()
        matches = sum(1 for kw in keywords if kw.lower() in context_lower)

        # Normalize by number of keywords
        return min(matches / len(keywords), 1.0)

    def _score_terminology(self, source_label: str, target_label: str) -> float:
        """
        Score based on S1000D terminology mapping.

        Args:
            source_label: Source label
            target_label: Target label

        Returns:
            Score 0.0-1.0
        """
        source_lower = source_label.lower()
        target_lower = target_label.lower()

        score = 0.0
        matches = 0

        for s1000d_term, standard_terms in self.terminology_map.items():
            if s1000d_term in source_lower:
                for std_term in standard_terms:
                    if std_term in target_lower:
                        matches += 1
                        break

        # Normalize (assume max 3 terminology matches)
        return min(matches / 3.0, 1.0)

    def _score_hierarchy(self, source_context: str, target_context: str) -> float:
        """
        Score based on hierarchical path similarity.

        Args:
            source_context: Source hierarchical context
            target_context: Target hierarchical context

        Returns:
            Score 0.0-1.0
        """
        # Extract hierarchy levels (split by '>')
        source_levels = [s.strip() for s in source_context.split('>') if s.strip()]
        target_levels = [s.strip() for s in target_context.split('>') if s.strip()]

        if not source_levels or not target_levels:
            return 0.0

        # Compare hierarchy depth similarity
        depth_diff = abs(len(source_levels) - len(target_levels))
        depth_score = 1.0 / (1.0 + depth_diff)

        return depth_score


def main():
    """
    Test KROMA Matcher on bike dataset.
    """
    from data_loader import load_all_concepts

    print("=" * 70)
    print("KROMA MATCHER TEST")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    df = load_all_concepts('bike')

    s1000d_df = df[df['source'] == 's1000d'].reset_index(drop=True)
    ontology_df = df[df['source'] == 'bike_ontology'].reset_index(drop=True)

    print(f"   S1000D concepts: {len(s1000d_df)}")
    print(f"   Ontology concepts: {len(ontology_df)}")

    # Initialize KROMA
    print("\n2. Initializing KROMA Matcher...")
    kroma = KROMAMatcher(s1000d_df, ontology_df)

    # Test on sample concepts
    print("\n3. Testing KROMA matching...")
    print("=" * 70)

    test_cases = [
        'S1000DBIKE-AAA-DA0',  # Wheel system
        'S1000DBIKE-AAA-DA1',  # Brake system
        'S1000DBIKE-AAA-DA2',  # Drivetrain
        'S1000DLIGHTING',      # Lighting
    ]

    for test_pattern in test_cases:
        # Find matching S1000D concepts
        matching = s1000d_df[s1000d_df['uri'].str.contains(test_pattern, na=False)]

        if matching.empty:
            continue

        concept = matching.iloc[0].to_dict()

        print(f"\n--- Test: {concept['label']} ---")
        print(f"URI: {concept['uri']}")

        # Find candidates
        candidates = kroma.find_candidates(concept, top_k=5)

        print(f"\nTop 5 Candidates:")
        for i, (target_uri, score) in enumerate(candidates, 1):
            target = ontology_df[ontology_df['uri'] == target_uri].iloc[0]
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   Label: {target['label']}")
            print(f"   URI: {target_uri}")

    print("\n" + "=" * 70)
    print("KROMA Matcher test completed!")


if __name__ == '__main__':
    main()
