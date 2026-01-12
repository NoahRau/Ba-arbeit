"""
Feature & Context Extractor for Technical Documentation.

Extracts structural features, hierarchical context, and neighbor information
from XML documents and ontologies.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from xml.etree import ElementTree as ET

from .rich_document import RichDocument


class FeatureExtractor:
    """
    Extracts features and context from technical documentation.

    Features extracted:
    - XML hierarchical paths
    - Structural scope (element type)
    - Neighbor context (previous/next siblings)
    - Parent context
    - Named entities (technical terms)
    - Keywords
    """

    def __init__(
        self,
        context_window_size: int = 2,
        domain_keywords: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize feature extractor.

        Args:
            context_window_size: Number of siblings to include in context (±N)
            domain_keywords: Domain-specific keywords for classification
        """
        self.context_window_size = context_window_size
        self.domain_keywords = domain_keywords or {}

        # Technical entity patterns (simple NER)
        self.entity_patterns = [
            r'\b(?:wheel|tire|frame|brake|chain|gear|hub|spoke|rim|fork|handlebar)\b',
            r'\b(?:hydraulic|pneumatic|mechanical|electrical|electronic)\b',
            r'\b(?:install|remove|replace|clean|lubricate|inspect|adjust|tighten)\b',
        ]

    def extract_from_xml(self, xml_path: Path) -> List[RichDocument]:
        """
        Extract rich documents from XML file (S1000D).

        Args:
            xml_path: Path to XML file

        Returns:
            List of RichDocument objects
        """
        documents = []

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract DMC code from filename or root
            dmc = self._extract_dmc_code(xml_path, root)

            # Traverse XML and extract elements
            elements = self._traverse_xml(root, parent_path=[])

            # Build RichDocuments with context
            for i, elem_data in enumerate(elements):
                doc = self._build_rich_document(
                    elem_data,
                    dmc,
                    elements,
                    current_index=i
                )
                if doc:
                    documents.append(doc)

        except Exception as e:
            print(f"Warning: Failed to parse {xml_path}: {e}")

        return documents

    def _extract_dmc_code(self, xml_path: Path, root: ET.Element) -> str:
        """
        Extract DMC code from filename or XML root.

        Args:
            xml_path: Path to XML file
            root: XML root element

        Returns:
            DMC code string
        """
        # Try to extract from filename
        filename = xml_path.stem
        if 'DMC-' in filename or 'S1000D' in filename:
            return filename

        # Try to find in XML structure
        dmc_elements = ['dmCode', 'dmc', 'DMC']
        for tag in dmc_elements:
            elem = root.find(f".//{tag}")
            if elem is not None:
                return elem.get('code', filename)

        return filename

    def _traverse_xml(
        self,
        element: ET.Element,
        parent_path: List[str],
        current_path: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Recursively traverse XML and extract elements with context.

        Args:
            element: Current XML element
            parent_path: Path to parent element
            current_path: Current hierarchical path

        Returns:
            List of element data dictionaries
        """
        if current_path is None:
            current_path = []

        elements = []

        # Get element text
        text = self._get_element_text(element)

        # Only process if element has meaningful text
        if text and len(text.strip()) > 5:
            elem_data = {
                'tag': element.tag,
                'text': text,
                'hierarchy_path': current_path.copy(),
                'parent_path': parent_path.copy(),
                'attributes': dict(element.attrib),
                'element': element,
            }
            elements.append(elem_data)

        # Recurse into children
        new_path = current_path + [element.tag]
        for child in element:
            child_elements = self._traverse_xml(child, current_path, new_path)
            elements.extend(child_elements)

        return elements

    def _get_element_text(self, element: ET.Element) -> str:
        """
        Get clean text from XML element.

        Args:
            element: XML element

        Returns:
            Clean text string
        """
        # Get all text content including children
        text_parts = []

        if element.text:
            text_parts.append(element.text.strip())

        for child in element:
            child_text = self._get_element_text(child)
            if child_text:
                text_parts.append(child_text)

            if child.tail:
                text_parts.append(child.tail.strip())

        text = ' '.join(text_parts)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _build_rich_document(
        self,
        elem_data: Dict[str, Any],
        dmc: str,
        all_elements: List[Dict[str, Any]],
        current_index: int
    ) -> Optional[RichDocument]:
        """
        Build RichDocument from element data with full context.

        Args:
            elem_data: Element data dictionary
            dmc: DMC code
            all_elements: All extracted elements (for neighbors)
            current_index: Current element index

        Returns:
            RichDocument or None
        """
        text = elem_data['text']
        if not text or len(text.strip()) < 5:
            return None

        # Generate URI
        uri = f"s1000d:{dmc}:{elem_data['tag']}:{current_index}"

        # Extract label (use tag or first part of text)
        label = elem_data['tag']
        if len(text) < 100:
            label = text[:50]
        else:
            # Try to find title-like text
            first_sentence = text.split('.')[0]
            label = first_sentence[:50] if len(first_sentence) < 100 else elem_data['tag']

        # Extract entities
        entities = self._extract_entities(text)

        # Extract keywords
        keywords = self._extract_keywords(text)

        # Classify technical domain
        technical_domain = self._classify_domain(text, keywords)

        # Get parent context
        parent_context = None
        if elem_data['parent_path']:
            parent_tag = elem_data['parent_path'][-1]
            parent_context = f"Parent: {parent_tag}"

        # Get neighbor context
        previous_sibling = None
        next_sibling = None
        context_window = []

        # Previous siblings
        for i in range(1, self.context_window_size + 1):
            prev_idx = current_index - i
            if prev_idx >= 0:
                prev_text = all_elements[prev_idx]['text']
                if i == 1:
                    previous_sibling = prev_text[:200]  # Limit length
                context_window.insert(0, prev_text[:100])

        # Next siblings
        for i in range(1, self.context_window_size + 1):
            next_idx = current_index + i
            if next_idx < len(all_elements):
                next_text = all_elements[next_idx]['text']
                if i == 1:
                    next_sibling = next_text[:200]
                context_window.append(next_text[:100])

        # Create RichDocument
        doc = RichDocument(
            uri=uri,
            label=label,
            raw_content=text,
            hierarchy_path=elem_data['hierarchy_path'],
            structural_scope=elem_data['tag'],
            depth_level=len(elem_data['hierarchy_path']),
            previous_sibling=previous_sibling,
            next_sibling=next_sibling,
            parent_context=parent_context,
            context_window=context_window,
            technical_domain=technical_domain,
            entities=entities,
            keywords=keywords,
            source='s1000d',
            metadata={
                'dmc': dmc,
                'attributes': elem_data['attributes'],
            }
        )

        return doc

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities (simple pattern-based NER).

        Args:
            text: Input text

        Returns:
            List of entities
        """
        entities = []
        text_lower = text.lower()

        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            entities.extend(matches)

        # Deduplicate and limit
        return list(set(entities))[:20]

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text.

        Args:
            text: Input text

        Returns:
            List of keywords
        """
        # Simple keyword extraction: words longer than 4 chars, not stopwords
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'will', 'their', 'about', 'which'}

        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        keywords = [w for w in words if w not in stopwords]

        # Count frequency and return top keywords
        from collections import Counter
        counter = Counter(keywords)
        return [word for word, count in counter.most_common(15)]

    def _classify_domain(self, text: str, keywords: List[str]) -> Optional[str]:
        """
        Classify technical domain based on keywords.

        Args:
            text: Input text
            keywords: Extracted keywords

        Returns:
            Domain classification or None
        """
        if not self.domain_keywords:
            return None

        text_lower = text.lower()
        domain_scores = {}

        for domain, domain_terms in self.domain_keywords.items():
            score = 0
            for term in domain_terms:
                if term.lower() in text_lower:
                    score += 1
                if term.lower() in keywords:
                    score += 2  # Keywords weighted higher

            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]

        return None

    def enrich_from_dataframe_row(self, row: Any, source: str = 'ontology') -> RichDocument:
        """
        Create RichDocument from pandas DataFrame row (for ontology data).

        Args:
            row: DataFrame row
            source: Data source identifier

        Returns:
            RichDocument
        """
        uri = row.get('uri', '')
        label = row.get('label', '')
        content = row.get('context_text', label)

        # Parse hierarchy from context_text if available
        hierarchy_path = []
        if 'context_text' in row and row['context_text']:
            # context_text format: "Parent > Child > Concept"
            parts = [p.strip() for p in str(row['context_text']).split('>')]
            hierarchy_path = parts[:-1]  # All except last (which is the label itself)

        # Extract features
        entities = self._extract_entities(content)
        keywords = self._extract_keywords(content)
        technical_domain = self._classify_domain(content, keywords)

        return RichDocument(
            uri=uri,
            label=label,
            raw_content=content,
            hierarchy_path=hierarchy_path,
            depth_level=len(hierarchy_path),
            technical_domain=technical_domain,
            entities=entities,
            keywords=keywords,
            source=source,
            metadata=dict(row) if hasattr(row, 'to_dict') else {}
        )


def main():
    """Test feature extractor."""
    print("=" * 70)
    print("FEATURE EXTRACTOR TEST")
    print("=" * 70)

    # Test with sample data
    from src.data_loader import load_all_concepts

    print("\n[1/2] Loading concepts...")
    df = load_all_concepts(include_ontologies=True)

    print("\n[2/2] Testing feature extraction...")
    extractor = FeatureExtractor(
        context_window_size=2,
        domain_keywords={
            'wheel_system': ['wheel', 'tire', 'hub', 'spoke', 'rim'],
            'brake_system': ['brake', 'caliper', 'disc', 'pad'],
            'drivetrain': ['chain', 'gear', 'cassette', 'derailleur'],
        }
    )

    # Test on ontology data
    sample_row = df[df['source'] == 'bike_ontology'].iloc[0]
    rich_doc = extractor.enrich_from_dataframe_row(sample_row, source='bike_ontology')

    print(f"\nRich Document Created:")
    print(f"  URI: {rich_doc.uri}")
    print(f"  Label: {rich_doc.label}")
    print(f"  Hierarchy: {' > '.join(rich_doc.hierarchy_path)}")
    print(f"  Domain: {rich_doc.technical_domain}")
    print(f"  Entities: {rich_doc.entities[:5]}")
    print(f"  Keywords: {rich_doc.keywords[:5]}")

    print(f"\nFull Context:")
    print(rich_doc.get_full_context()[:500])

    print("\n" + "=" * 70)
    print("✓ Feature extractor test complete!")


if __name__ == '__main__':
    main()
