"""
Enriched Data Loader for Context-Aware Pipeline.

Loads data and creates RichDocument objects with full context,
features, and domain knowledge enrichment.
"""

import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all_concepts
from enrichment.rich_document import RichDocument
from enrichment.feature_extractor import FeatureExtractor
from enrichment.domain_knowledge import DomainKnowledgeManager
from enrichment.query_expander import QueryExpander

try:
    from config import ONTOLOGY_DIR, CACHE_DIR
except ImportError:
    # Fallback
    ONTOLOGY_DIR = Path(__file__).parent.parent / 'data' / 'ontologies'
    CACHE_DIR = Path(__file__).parent.parent / 'cache'


class EnrichedDataLoader:
    """
    Loads and enriches data for context-aware matching.

    Creates RichDocument objects with:
    - Hierarchical context
    - Neighbor context
    - Extracted features (entities, keywords)
    - Domain knowledge expansion
    """

    def __init__(
        self,
        load_domain_knowledge: bool = True,
        context_window_size: int = 2
    ):
        """
        Initialize enriched data loader.

        Args:
            load_domain_knowledge: Whether to load domain knowledge for expansion
            context_window_size: Neighbor context window size
        """
        self.context_window_size = context_window_size

        # Domain knowledge (optional)
        self.domain_knowledge = None
        self.query_expander = None

        if load_domain_knowledge:
            self._load_domain_knowledge()

        # Feature extractor
        domain_keywords = {
            'wheel_system': ['wheel', 'tire', 'hub', 'spoke', 'rim'],
            'brake_system': ['brake', 'caliper', 'disc', 'pad', 'rotor'],
            'drivetrain': ['chain', 'gear', 'cassette', 'derailleur', 'crank'],
            'frame_structure': ['frame', 'fork', 'handlebar', 'stem'],
        }

        self.feature_extractor = FeatureExtractor(
            context_window_size=context_window_size,
            domain_keywords=domain_keywords
        )

    def _load_domain_knowledge(self):
        """Load domain knowledge from ontology."""
        print("  Loading domain knowledge...")

        try:
            tbox_file = ONTOLOGY_DIR / 'tbox.owl'

            if tbox_file.exists():
                manager = DomainKnowledgeManager(cache_dir=CACHE_DIR / 'domain_knowledge')
                self.domain_knowledge = manager.load_from_owl(tbox_file)

                # Create query expander
                self.query_expander = QueryExpander(
                    self.domain_knowledge,
                    max_synonyms=3,
                    max_hypernyms=2,
                    max_related=2
                )

                print("  ✓ Domain knowledge loaded")
            else:
                print(f"  Warning: {tbox_file} not found, skipping domain knowledge")

        except Exception as e:
            print(f"  Warning: Failed to load domain knowledge: {e}")

    def load_enriched_concepts(
        self,
        include_ontologies: bool = True
    ) -> tuple[List[RichDocument], List[RichDocument]]:
        """
        Load and enrich concepts from all sources.

        Args:
            include_ontologies: Whether to include ontology data

        Returns:
            (source_docs, target_docs) tuple of RichDocument lists
        """
        print("\n" + "=" * 70)
        print("LOADING ENRICHED CONCEPTS")
        print("=" * 70)

        # Load raw data using existing loader
        print("\n[1/3] Loading raw data...")
        df = load_all_concepts(include_ontologies=include_ontologies)

        # Split into source and target
        s1000d_df = df[df['source'] == 's1000d'].reset_index(drop=True)
        ontology_df = df[df['source'] == 'bike_ontology'].reset_index(drop=True)

        print(f"\n  S1000D concepts: {len(s1000d_df)}")
        print(f"  Ontology concepts: {len(ontology_df)}")

        # Enrich S1000D concepts
        print("\n[2/3] Enriching S1000D concepts...")
        source_docs = self._enrich_dataframe(s1000d_df, source_type='s1000d')

        # Enrich ontology concepts
        print("\n[3/3] Enriching ontology concepts...")
        target_docs = self._enrich_dataframe(ontology_df, source_type='bike_ontology')

        print("\n" + "=" * 70)
        print(f"✓ ENRICHMENT COMPLETE")
        print(f"  Source documents: {len(source_docs)}")
        print(f"  Target documents: {len(target_docs)}")
        print("=" * 70)

        return source_docs, target_docs

    def _enrich_dataframe(
        self,
        df: pd.DataFrame,
        source_type: str
    ) -> List[RichDocument]:
        """
        Enrich DataFrame rows into RichDocuments.

        Args:
            df: DataFrame with concept data
            source_type: Source identifier

        Returns:
            List of RichDocument objects
        """
        rich_docs = []

        for idx, row in df.iterrows():
            # Create RichDocument using feature extractor
            rich_doc = self.feature_extractor.enrich_from_dataframe_row(row, source=source_type)

            # Apply domain knowledge expansion (if available)
            if self.query_expander:
                self.query_expander.expand_rich_document(rich_doc, update_in_place=True)

                # Add domain tags
                rich_doc.domain_tags = self.query_expander.find_domain_tags(rich_doc.raw_content)

            rich_docs.append(rich_doc)

        return rich_docs

    def convert_to_dataframe(self, rich_docs: List[RichDocument]) -> pd.DataFrame:
        """
        Convert RichDocuments back to DataFrame (for compatibility).

        Args:
            rich_docs: List of RichDocuments

        Returns:
            DataFrame
        """
        data = [doc.to_dict() for doc in rich_docs]
        return pd.DataFrame(data)


def load_enriched_data(
    include_ontologies: bool = True,
    use_domain_knowledge: bool = True
) -> tuple[List[RichDocument], List[RichDocument]]:
    """
    Helper function to load enriched data.

    Args:
        include_ontologies: Include ontology data
        use_domain_knowledge: Use domain knowledge for expansion

    Returns:
        (source_docs, target_docs) tuple
    """
    loader = EnrichedDataLoader(load_domain_knowledge=use_domain_knowledge)
    return loader.load_enriched_concepts(include_ontologies=include_ontologies)


def main():
    """Test enriched data loader."""
    print("=" * 70)
    print("ENRICHED DATA LOADER TEST")
    print("=" * 70)

    # Load enriched data
    source_docs, target_docs = load_enriched_data(
        include_ontologies=True,
        use_domain_knowledge=True
    )

    # Show sample
    print("\n" + "=" * 70)
    print("SAMPLE ENRICHED DOCUMENTS")
    print("=" * 70)

    print("\nSource Document (S1000D):")
    if source_docs:
        doc = source_docs[0]
        print(f"  URI: {doc.uri}")
        print(f"  Label: {doc.label}")
        print(f"  Hierarchy: {' > '.join(doc.hierarchy_path) if doc.hierarchy_path else 'N/A'}")
        print(f"  Domain: {doc.technical_domain}")
        print(f"  Entities: {doc.entities[:5]}")
        print(f"  Keywords: {doc.keywords[:5]}")
        print(f"  Expanded terms: {doc.expanded_terms[:5]}")
        print(f"  Domain tags: {doc.domain_tags[:3]}")

    print("\nTarget Document (Ontology):")
    if target_docs:
        doc = target_docs[0]
        print(f"  URI: {doc.uri}")
        print(f"  Label: {doc.label}")
        print(f"  Hierarchy: {' > '.join(doc.hierarchy_path) if doc.hierarchy_path else 'N/A'}")
        print(f"  Domain: {doc.technical_domain}")
        print(f"  Entities: {doc.entities[:5]}")
        print(f"  Keywords: {doc.keywords[:5]}")
        print(f"  Expanded terms: {doc.expanded_terms[:5]}")

    print("\n" + "=" * 70)
    print("✓ Enriched data loader test complete!")


if __name__ == '__main__':
    main()
