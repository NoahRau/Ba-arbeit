"""
Rich Document Object for Context-Aware Matching.

Contains not just raw text but structural and semantic metadata
for improved matching quality.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class RichDocument:
    """
    Rich Document Object with context and metadata.

    Attributes:
        uri: Unique identifier
        label: Primary label/title
        raw_content: The actual text content

        # Structural Context
        hierarchy_path: List of parent nodes (e.g., ['System', 'Fahrwerk', 'Wartung'])
        structural_scope: Type of element (e.g., 'title', 'warning', 'procedure_step')
        depth_level: How deep in the hierarchy (0 = root)

        # Neighbor Context
        previous_sibling: Text of previous element (or None)
        next_sibling: Text of next element (or None)
        parent_context: Text summary of parent section
        context_window: List of surrounding sentences/elements

        # Semantic Context
        semantic_summary: LLM-generated summary of containing module
        technical_domain: Domain classification (e.g., 'hydraulics', 'avionics')

        # Extracted Features
        entities: Named entities (components, tools, actions)
        keywords: Important technical keywords
        domain_tags: Domain-specific tags from ontology

        # Original metadata
        source: Data source ('s1000d', 'bike_ontology', etc.)
        metadata: Additional custom metadata
    """

    # Core fields
    uri: str
    label: str
    raw_content: str

    # Structural context
    hierarchy_path: List[str] = field(default_factory=list)
    structural_scope: Optional[str] = None
    depth_level: int = 0

    # Neighbor context
    previous_sibling: Optional[str] = None
    next_sibling: Optional[str] = None
    parent_context: Optional[str] = None
    context_window: List[str] = field(default_factory=list)

    # Semantic context
    semantic_summary: Optional[str] = None
    technical_domain: Optional[str] = None

    # Extracted features
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    domain_tags: List[str] = field(default_factory=list)

    # Query expansion (from domain knowledge)
    expanded_terms: List[str] = field(default_factory=list)  # Synonyms and related terms

    # Original metadata
    source: str = 'unknown'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_full_context(self) -> str:
        """
        Get the full context including hierarchy, neighbors, and content.

        Returns:
            Formatted context string
        """
        parts = []

        # Add hierarchy path
        if self.hierarchy_path:
            hierarchy_str = ' > '.join(self.hierarchy_path)
            parts.append(f"[HIERARCHY] {hierarchy_str}")

        # Add structural scope
        if self.structural_scope:
            parts.append(f"[SCOPE] {self.structural_scope}")

        # Add parent context
        if self.parent_context:
            parts.append(f"[PARENT] {self.parent_context}")

        # Add previous sibling
        if self.previous_sibling:
            parts.append(f"[PREVIOUS] {self.previous_sibling}")

        # Add main content
        parts.append(f"[CONTENT] {self.label}: {self.raw_content}")

        # Add next sibling
        if self.next_sibling:
            parts.append(f"[NEXT] {self.next_sibling}")

        # Add entities if available
        if self.entities:
            entities_str = ', '.join(self.entities[:5])  # Top 5
            parts.append(f"[ENTITIES] {entities_str}")

        return '\n'.join(parts)

    def get_contextual_embedding_text(self, mode: str = 'full') -> str:
        """
        Get text for embedding generation with configurable context level.

        Args:
            mode: 'minimal', 'medium', or 'full'

        Returns:
            Text string optimized for embedding
        """
        if mode == 'minimal':
            # Just label and content
            return f"{self.label}: {self.raw_content}"

        elif mode == 'medium':
            # Label + content + hierarchy
            parts = [f"{self.label}: {self.raw_content}"]
            if self.hierarchy_path:
                hierarchy = ' > '.join(self.hierarchy_path)
                parts.insert(0, hierarchy)
            return ' | '.join(parts)

        else:  # 'full'
            # Everything
            return self.get_full_context()

    def get_feature_vector(self) -> Dict[str, Any]:
        """
        Get structured feature vector for blocking/filtering.

        Returns:
            Dictionary of features
        """
        return {
            'hierarchy_path': self.hierarchy_path,
            'depth_level': self.depth_level,
            'structural_scope': self.structural_scope,
            'technical_domain': self.technical_domain,
            'entities': self.entities,
            'keywords': self.keywords,
            'domain_tags': self.domain_tags,
            'has_parent_context': self.parent_context is not None,
            'has_siblings': (self.previous_sibling is not None) or (self.next_sibling is not None),
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            'uri': self.uri,
            'label': self.label,
            'raw_content': self.raw_content,
            'hierarchy_path': self.hierarchy_path,
            'structural_scope': self.structural_scope,
            'depth_level': self.depth_level,
            'previous_sibling': self.previous_sibling,
            'next_sibling': self.next_sibling,
            'parent_context': self.parent_context,
            'context_window': self.context_window,
            'semantic_summary': self.semantic_summary,
            'technical_domain': self.technical_domain,
            'entities': self.entities,
            'keywords': self.keywords,
            'domain_tags': self.domain_tags,
            'source': self.source,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RichDocument':
        """
        Create RichDocument from dictionary.

        Args:
            data: Dictionary with document data

        Returns:
            RichDocument instance
        """
        return cls(**data)

    def __repr__(self) -> str:
        """String representation."""
        hierarchy = ' > '.join(self.hierarchy_path) if self.hierarchy_path else 'N/A'
        return f"RichDocument(uri={self.uri}, hierarchy={hierarchy}, entities={len(self.entities)})"


def create_rich_document_from_simple(
    uri: str,
    label: str,
    content: str,
    source: str = 'unknown',
    **kwargs
) -> RichDocument:
    """
    Helper function to create RichDocument from simple parameters.

    Args:
        uri: Document URI
        label: Document label
        content: Raw content
        source: Data source
        **kwargs: Additional metadata

    Returns:
        RichDocument instance
    """
    return RichDocument(
        uri=uri,
        label=label,
        raw_content=content,
        source=source,
        metadata=kwargs
    )
