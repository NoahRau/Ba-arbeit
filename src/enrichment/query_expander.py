"""
Query Expansion using Domain Knowledge.

Expands queries with synonyms, related terms, and hierarchical context
to improve matching recall.
"""

from typing import List, Set, Optional
from .domain_knowledge import DomainKnowledge
from .rich_document import RichDocument


class QueryExpander:
    """
    Expands queries using domain knowledge.

    Expansion strategies:
    1. Synonym expansion (sameAs, equivalentClass)
    2. Hypernym expansion (parent classes)
    3. Related term expansion
    4. Label variation expansion
    """

    def __init__(
        self,
        domain_knowledge: DomainKnowledge,
        max_synonyms: int = 3,
        max_hypernyms: int = 2,
        max_related: int = 2
    ):
        """
        Initialize query expander.

        Args:
            domain_knowledge: Domain knowledge object
            max_synonyms: Maximum synonyms to add per term
            max_hypernyms: Maximum hypernyms to add per term
            max_related: Maximum related terms to add
        """
        self.knowledge = domain_knowledge
        self.max_synonyms = max_synonyms
        self.max_hypernyms = max_hypernyms
        self.max_related = max_related

    def expand_text(self, text: str) -> str:
        """
        Expand text with synonyms and related terms.

        Args:
            text: Input text

        Returns:
            Expanded text with additional terms
        """
        # Split into words
        words = text.lower().split()
        expanded_set = set(words)  # Start with original words

        for word in words:
            # Look up in domain knowledge
            if word in self.knowledge.label_to_uri:
                uris = list(self.knowledge.label_to_uri[word])

                for uri in uris[:1]:  # Process first matching concept
                    # Add synonyms
                    synonyms = self.knowledge.synonyms.get(uri, set())
                    for syn in list(synonyms)[:self.max_synonyms]:
                        if syn and syn.lower() != word:
                            expanded_set.add(syn.lower())

                    # Add parent labels (hypernyms)
                    parents = self.knowledge.parents.get(uri, set())
                    for parent_uri in list(parents)[:self.max_hypernyms]:
                        parent_labels = self.knowledge.labels.get(parent_uri, [])
                        if parent_labels:
                            expanded_set.add(parent_labels[0].lower())

                    # Add related terms
                    related = self.knowledge.related.get(uri, set())
                    for rel_uri in list(related)[:self.max_related]:
                        rel_labels = self.knowledge.labels.get(rel_uri, [])
                        if rel_labels:
                            expanded_set.add(rel_labels[0].lower())

        return ' '.join(sorted(expanded_set))

    def expand_rich_document(
        self,
        doc: RichDocument,
        update_in_place: bool = True
    ) -> RichDocument:
        """
        Expand a RichDocument with domain knowledge.

        Args:
            doc: RichDocument to expand
            update_in_place: If True, update doc.expanded_terms

        Returns:
            RichDocument (updated if update_in_place=True)
        """
        expanded_terms = set()

        # Expand entities
        for entity in doc.entities:
            if entity.lower() in self.knowledge.label_to_uri:
                uris = list(self.knowledge.label_to_uri[entity.lower()])
                for uri in uris[:1]:
                    # Add synonyms
                    synonyms = self.knowledge.synonyms.get(uri, set())
                    expanded_terms.update(list(synonyms)[:self.max_synonyms])

                    # Add parent labels
                    parents = self.knowledge.parents.get(uri, set())
                    for parent_uri in list(parents)[:self.max_hypernyms]:
                        parent_labels = self.knowledge.labels.get(parent_uri, [])
                        if parent_labels:
                            expanded_terms.add(parent_labels[0])

        # Expand keywords
        for keyword in doc.keywords:
            if keyword.lower() in self.knowledge.label_to_uri:
                uris = list(self.knowledge.label_to_uri[keyword.lower()])
                for uri in uris[:1]:
                    synonyms = self.knowledge.synonyms.get(uri, set())
                    expanded_terms.update(list(synonyms)[:2])

        # Update document
        if update_in_place:
            doc.expanded_terms = list(expanded_terms)

        return doc

    def expand_keywords_list(self, keywords: List[str]) -> List[str]:
        """
        Expand a list of keywords with synonyms.

        Args:
            keywords: List of keywords

        Returns:
            Expanded list with original + synonyms
        """
        expanded = set(keywords)

        for keyword in keywords:
            if keyword.lower() in self.knowledge.label_to_uri:
                uris = list(self.knowledge.label_to_uri[keyword.lower()])

                for uri in uris[:1]:
                    synonyms = self.knowledge.synonyms.get(uri, set())
                    expanded.update(list(synonyms)[:self.max_synonyms])

        return list(expanded)

    def find_domain_tags(self, text: str) -> List[str]:
        """
        Find domain-specific tags/concepts mentioned in text.

        Args:
            text: Input text

        Returns:
            List of domain tags (concept URIs or labels)
        """
        text_lower = text.lower()
        tags = []

        # Check all known labels
        for label, uris in self.knowledge.label_to_uri.items():
            if label in text_lower:
                # Get primary labels for the URIs
                for uri in list(uris)[:1]:
                    primary_labels = self.knowledge.labels.get(uri, [])
                    if primary_labels:
                        tags.append(primary_labels[0])

        return list(set(tags))


class ContextualQueryExpander(QueryExpander):
    """
    Enhanced query expander that considers hierarchical context.
    """

    def expand_with_context(
        self,
        text: str,
        hierarchy_context: Optional[List[str]] = None
    ) -> str:
        """
        Expand query considering hierarchical context.

        Args:
            text: Input text
            hierarchy_context: Hierarchical path context

        Returns:
            Expanded text
        """
        # Base expansion
        expanded = self.expand_text(text)

        # Add context-based expansion
        if hierarchy_context:
            context_terms = set()

            for context_level in hierarchy_context:
                # Look up context term
                if context_level.lower() in self.knowledge.label_to_uri:
                    uris = list(self.knowledge.label_to_uri[context_level.lower()])

                    for uri in uris[:1]:
                        # Add related terms from context
                        related = self.knowledge.related.get(uri, set())
                        for rel_uri in list(related)[:2]:
                            rel_labels = self.knowledge.labels.get(rel_uri, [])
                            if rel_labels:
                                context_terms.add(rel_labels[0].lower())

            if context_terms:
                expanded += ' ' + ' '.join(context_terms)

        return expanded


def main():
    """Test query expander."""
    print("=" * 70)
    print("QUERY EXPANDER TEST")
    print("=" * 70)

    from pathlib import Path
    from config import ONTOLOGY_DIR, CACHE_DIR
    from .domain_knowledge import DomainKnowledgeManager

    print("\n[1/3] Loading domain knowledge...")
    manager = DomainKnowledgeManager(cache_dir=CACHE_DIR / 'domain_knowledge')

    tbox_file = ONTOLOGY_DIR / 'tbox.owl'
    if not tbox_file.exists():
        print(f"  Error: {tbox_file} not found")
        return

    knowledge = manager.load_from_owl(tbox_file)

    print("\n[2/3] Testing query expansion...")
    expander = QueryExpander(knowledge, max_synonyms=3)

    test_queries = [
        "wheel assembly procedure",
        "brake maintenance",
        "bicycle frame inspection"
    ]

    for query in test_queries:
        expanded = expander.expand_text(query)
        print(f"\n  Original: {query}")
        print(f"  Expanded: {expanded}")

    print("\n[3/3] Testing RichDocument expansion...")
    test_doc = RichDocument(
        uri="test:1",
        label="Wheel Installation",
        raw_content="Install the front wheel on the bicycle fork",
        entities=["wheel", "fork", "bicycle"],
        keywords=["install", "front", "wheel", "fork"],
        source="test"
    )

    expander.expand_rich_document(test_doc, update_in_place=True)

    print(f"\n  Document: {test_doc.label}")
    print(f"  Original entities: {test_doc.entities}")
    print(f"  Expanded terms: {test_doc.expanded_terms[:10]}")

    print("\n" + "=" * 70)
    print("✓ Query expander test complete!")


if __name__ == '__main__':
    main()
