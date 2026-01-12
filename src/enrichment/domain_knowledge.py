"""
Domain Knowledge Manager.

Extracts and manages domain knowledge from ontologies (tbox.owl)
for synonym expansion, query expansion, and semantic enrichment.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import rdflib
from rdflib import Graph, Namespace, URIRef, Literal


class DomainKnowledge:
    """
    Domain knowledge extracted from ontologies.

    Contains:
    - Synonyms (sameAs, equivalentClass)
    - Hypernyms (parent classes)
    - Hyponyms (child classes)
    - Related terms
    - Labels and descriptions
    """

    def __init__(self):
        """Initialize domain knowledge store."""
        # Concept URI -> labels
        self.labels: Dict[str, List[str]] = defaultdict(list)

        # Concept URI -> synonyms (from sameAs, equivalentClass)
        self.synonyms: Dict[str, Set[str]] = defaultdict(set)

        # Concept URI -> parent concepts
        self.parents: Dict[str, Set[str]] = defaultdict(set)

        # Concept URI -> child concepts
        self.children: Dict[str, Set[str]] = defaultdict(set)

        # Concept URI -> related concepts
        self.related: Dict[str, Set[str]] = defaultdict(set)

        # Concept URI -> description
        self.descriptions: Dict[str, str] = {}

        # Label -> concept URIs (reverse lookup)
        self.label_to_uri: Dict[str, Set[str]] = defaultdict(set)

    def add_label(self, uri: str, label: str):
        """Add label for a concept."""
        if label and label not in self.labels[uri]:
            self.labels[uri].append(label)
            self.label_to_uri[label.lower()].add(uri)

    def add_synonym(self, uri: str, synonym: str):
        """Add synonym for a concept."""
        if synonym:
            self.synonyms[uri].add(synonym)

    def add_parent(self, uri: str, parent_uri: str):
        """Add parent relationship."""
        self.parents[uri].add(parent_uri)
        self.children[parent_uri].add(uri)

    def add_related(self, uri: str, related_uri: str):
        """Add related concept."""
        self.related[uri].add(related_uri)

    def add_description(self, uri: str, description: str):
        """Add description for a concept."""
        if description:
            self.descriptions[uri] = description

    def get_all_labels(self, uri: str) -> List[str]:
        """Get all labels (including synonyms) for a concept."""
        labels = self.labels.get(uri, []).copy()
        labels.extend(self.synonyms.get(uri, []))
        return list(set(labels))

    def expand_query(self, text: str) -> str:
        """
        Expand query with synonyms and related terms.

        Args:
            text: Input text

        Returns:
            Expanded text with synonyms
        """
        words = text.lower().split()
        expanded_words = []

        for word in words:
            expanded_words.append(word)

            # Check if word matches a label
            if word in self.label_to_uri:
                uris = self.label_to_uri[word]
                for uri in uris:
                    # Add synonyms
                    synonyms = self.synonyms.get(uri, set())
                    for syn in list(synonyms)[:2]:  # Limit to 2 synonyms
                        if syn.lower() != word:
                            expanded_words.append(syn.lower())

        return ' '.join(expanded_words)

    def get_concept_context(self, uri: str) -> Dict[str, any]:
        """
        Get full context for a concept (parents, children, related).

        Args:
            uri: Concept URI

        Returns:
            Dictionary with context information
        """
        return {
            'labels': self.labels.get(uri, []),
            'synonyms': list(self.synonyms.get(uri, set())),
            'parents': list(self.parents.get(uri, set())),
            'children': list(self.children.get(uri, set())),
            'related': list(self.related.get(uri, set())),
            'description': self.descriptions.get(uri, ''),
        }


class DomainKnowledgeManager:
    """
    Manages domain knowledge extraction from OWL ontologies.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize domain knowledge manager.

        Args:
            cache_dir: Directory to cache extracted knowledge
        """
        self.cache_dir = cache_dir
        self.knowledge = DomainKnowledge()

        # RDF namespaces
        self.OWL = Namespace("http://www.w3.org/2002/07/owl#")
        self.RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
        self.RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        self.SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

    def load_from_owl(self, owl_path: Path) -> DomainKnowledge:
        """
        Load domain knowledge from OWL file.

        Args:
            owl_path: Path to OWL/RDF file

        Returns:
            DomainKnowledge object
        """
        print(f"  Loading domain knowledge from {owl_path.name}...")

        # Try to load from cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{owl_path.stem}_knowledge.json"
            if cache_file.exists():
                print(f"  ✓ Loaded from cache")
                return self._load_from_cache(cache_file)

        # Parse OWL file
        g = Graph()
        g.parse(str(owl_path))

        print(f"  Loaded {len(g)} triples")

        # Extract knowledge
        self._extract_labels(g)
        self._extract_synonyms(g)
        self._extract_hierarchy(g)
        self._extract_descriptions(g)
        self._extract_related_concepts(g)

        # Cache if directory provided
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{owl_path.stem}_knowledge.json"
            self._save_to_cache(cache_file)

        print(f"  ✓ Extracted knowledge: {len(self.knowledge.labels)} concepts")

        return self.knowledge

    def _extract_labels(self, g: Graph):
        """Extract labels from ontology."""
        # rdfs:label
        for subj, obj in g.subject_objects(self.RDFS.label):
            uri = str(subj)
            label = str(obj)
            self.knowledge.add_label(uri, label)

        # skos:prefLabel
        for subj, obj in g.subject_objects(self.SKOS.prefLabel):
            uri = str(subj)
            label = str(obj)
            self.knowledge.add_label(uri, label)

    def _extract_synonyms(self, g: Graph):
        """Extract synonyms from ontology."""
        # owl:sameAs
        for subj, obj in g.subject_objects(self.OWL.sameAs):
            uri = str(subj)
            synonym_uri = str(obj)

            # Get label of synonym
            synonym_labels = list(g.objects(obj, self.RDFS.label))
            if synonym_labels:
                synonym = str(synonym_labels[0])
                self.knowledge.add_synonym(uri, synonym)

        # owl:equivalentClass
        for subj, obj in g.subject_objects(self.OWL.equivalentClass):
            uri = str(subj)
            equiv_uri = str(obj)

            equiv_labels = list(g.objects(obj, self.RDFS.label))
            if equiv_labels:
                equiv = str(equiv_labels[0])
                self.knowledge.add_synonym(uri, equiv)

        # skos:altLabel (alternative labels)
        for subj, obj in g.subject_objects(self.SKOS.altLabel):
            uri = str(subj)
            alt_label = str(obj)
            self.knowledge.add_synonym(uri, alt_label)

    def _extract_hierarchy(self, g: Graph):
        """Extract class hierarchy."""
        # rdfs:subClassOf
        for subj, obj in g.subject_objects(self.RDFS.subClassOf):
            child_uri = str(subj)
            parent_uri = str(obj)

            # Skip blank nodes
            if not parent_uri.startswith('http'):
                continue

            self.knowledge.add_parent(child_uri, parent_uri)

    def _extract_descriptions(self, g: Graph):
        """Extract descriptions/comments."""
        # rdfs:comment
        for subj, obj in g.subject_objects(self.RDFS.comment):
            uri = str(subj)
            description = str(obj)
            self.knowledge.add_description(uri, description)

        # skos:definition
        for subj, obj in g.subject_objects(self.SKOS.definition):
            uri = str(subj)
            definition = str(obj)
            self.knowledge.add_description(uri, definition)

    def _extract_related_concepts(self, g: Graph):
        """Extract related concepts."""
        # skos:related
        for subj, obj in g.subject_objects(self.SKOS.related):
            uri = str(subj)
            related_uri = str(obj)
            self.knowledge.add_related(uri, related_uri)

        # owl:ObjectProperty relations (generic)
        for subj, pred, obj in g.triples((None, None, None)):
            # Skip if not object property
            if not isinstance(obj, URIRef):
                continue

            pred_str = str(pred)

            # Skip basic RDF/RDFS/OWL predicates
            if any(ns in pred_str for ns in ['rdf-syntax', 'rdf-schema', 'owl#']):
                continue

            # This is a domain-specific relation
            uri = str(subj)
            related_uri = str(obj)

            self.knowledge.add_related(uri, related_uri)

    def _save_to_cache(self, cache_file: Path):
        """Save knowledge to JSON cache."""
        cache_data = {
            'labels': {k: v for k, v in self.knowledge.labels.items()},
            'synonyms': {k: list(v) for k, v in self.knowledge.synonyms.items()},
            'parents': {k: list(v) for k, v in self.knowledge.parents.items()},
            'children': {k: list(v) for k, v in self.knowledge.children.items()},
            'related': {k: list(v) for k, v in self.knowledge.related.items()},
            'descriptions': self.knowledge.descriptions,
        }

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

    def _load_from_cache(self, cache_file: Path) -> DomainKnowledge:
        """Load knowledge from JSON cache."""
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)

        knowledge = DomainKnowledge()

        # Load labels
        for uri, labels in cache_data.get('labels', {}).items():
            for label in labels:
                knowledge.add_label(uri, label)

        # Load synonyms
        for uri, synonyms in cache_data.get('synonyms', {}).items():
            for syn in synonyms:
                knowledge.add_synonym(uri, syn)

        # Load parents
        for uri, parents in cache_data.get('parents', {}).items():
            for parent in parents:
                knowledge.add_parent(uri, parent)

        # Load related
        for uri, related in cache_data.get('related', {}).items():
            for rel in related:
                knowledge.add_related(uri, rel)

        # Load descriptions
        for uri, desc in cache_data.get('descriptions', {}).items():
            knowledge.add_description(uri, desc)

        self.knowledge = knowledge
        return knowledge


def main():
    """Test domain knowledge manager."""
    print("=" * 70)
    print("DOMAIN KNOWLEDGE MANAGER TEST")
    print("=" * 70)

    from config import ONTOLOGY_DIR, CACHE_DIR

    print("\n[1/2] Loading domain knowledge from tbox.owl...")
    manager = DomainKnowledgeManager(cache_dir=CACHE_DIR / 'domain_knowledge')

    tbox_file = ONTOLOGY_DIR / 'tbox.owl'
    if not tbox_file.exists():
        print(f"  Error: {tbox_file} not found")
        return

    knowledge = manager.load_from_owl(tbox_file)

    print("\n[2/2] Testing knowledge extraction...")

    # Show some concepts
    print(f"\nTotal concepts with labels: {len(knowledge.labels)}")

    # Sample concepts
    sample_uris = list(knowledge.labels.keys())[:5]
    for uri in sample_uris:
        labels = knowledge.labels[uri]
        synonyms = knowledge.synonyms.get(uri, set())
        parents = knowledge.parents.get(uri, set())

        print(f"\n  Concept: {uri.split('#')[-1] if '#' in uri else uri.split('/')[-1]}")
        print(f"    Labels: {labels}")
        if synonyms:
            print(f"    Synonyms: {list(synonyms)[:3]}")
        if parents:
            parent_labels = [p.split('#')[-1] if '#' in p else p.split('/')[-1] for p in list(parents)[:3]]
            print(f"    Parents: {parent_labels}")

    # Test query expansion
    print("\n  Testing query expansion:")
    test_queries = ["wheel", "brake", "bike"]
    for query in test_queries:
        expanded = knowledge.expand_query(query)
        if expanded != query:
            print(f"    '{query}' -> '{expanded}'")

    print("\n" + "=" * 70)
    print("✓ Domain knowledge test complete!")


if __name__ == '__main__':
    main()
