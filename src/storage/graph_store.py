"""
Neo4j Graph Store for Ontology Matching.

Provides graph-based storage and retrieval for ontology concepts
with vector search and graph traversal capabilities.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from neo4j import GraphDatabase, Session
from neo4j.exceptions import ServiceUnavailable, AuthError
import pandas as pd

logger = logging.getLogger(__name__)


class KnowledgeGraphStore:
    """
    Neo4j-backed knowledge graph store for ontology matching.

    Features:
    - Vector-indexed embeddings for similarity search
    - Hierarchical relationships (PARENT_OF, RELATED_TO)
    - Graph-aware retrieval combining vector search + traversal
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        """
        Initialize connection to Neo4j.

        Args:
            uri: Neo4j connection URI
            user: Username
            password: Password
            database: Database name
        """
        self.uri = uri
        self.database = database

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info(f"✓ Connected to Neo4j at {uri}")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def initialize_schema(self, embedding_dim: int = 768):
        """
        Initialize Neo4j schema with indexes and constraints.

        Args:
            embedding_dim: Dimensionality of embedding vectors
        """
        with self.driver.session(database=self.database) as session:
            logger.info("Initializing Neo4j schema...")

            # Create constraints for unique URIs
            session.run("""
                CREATE CONSTRAINT s1000d_uri IF NOT EXISTS
                FOR (n:S1000DConcept) REQUIRE n.uri IS UNIQUE
            """)

            session.run("""
                CREATE CONSTRAINT ontology_uri IF NOT EXISTS
                FOR (n:OntologyClass) REQUIRE n.uri IS UNIQUE
            """)

            # Create indexes for fast lookup
            session.run("""
                CREATE INDEX s1000d_label IF NOT EXISTS
                FOR (n:S1000DConcept) ON (n.label)
            """)

            session.run("""
                CREATE INDEX ontology_label IF NOT EXISTS
                FOR (n:OntologyClass) ON (n.label)
            """)

            # Create vector index for S1000D concepts
            try:
                session.run(f"""
                    CREATE VECTOR INDEX s1000d_embeddings IF NOT EXISTS
                    FOR (n:S1000DConcept) ON (n.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {embedding_dim},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                logger.info(f"✓ Created vector index for S1000DConcept (dim={embedding_dim})")
            except Exception as e:
                logger.warning(f"Vector index creation skipped or failed: {e}")

            # Create vector index for Ontology classes
            try:
                session.run(f"""
                    CREATE VECTOR INDEX ontology_embeddings IF NOT EXISTS
                    FOR (n:OntologyClass) ON (n.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {embedding_dim},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                logger.info(f"✓ Created vector index for OntologyClass (dim={embedding_dim})")
            except Exception as e:
                logger.warning(f"Vector index creation skipped or failed: {e}")

            logger.info("✓ Schema initialization complete")

    def clear_all_data(self):
        """Delete all nodes and relationships. Use with caution!"""
        with self.driver.session(database=self.database) as session:
            logger.warning("Deleting all data from Neo4j...")
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("✓ All data cleared")

    def ingest_dataframe(
        self,
        df: pd.DataFrame,
        node_label: str,
        embedding_col: str = 'embedding',
        batch_size: int = 500
    ):
        """
        Ingest a DataFrame into Neo4j as nodes.

        Args:
            df: DataFrame with columns: uri, label, definition, context_text, embedding, etc.
            node_label: Neo4j label (S1000DConcept or OntologyClass)
            embedding_col: Column name containing embeddings
            batch_size: Number of nodes to create per batch
        """
        logger.info(f"Ingesting {len(df)} nodes as {node_label}...")

        with self.driver.session(database=self.database) as session:
            # Prepare data
            records = []
            for idx, row in df.iterrows():
                # Extract embedding if present
                embedding = None
                if embedding_col in row and row[embedding_col] is not None:
                    emb = row[embedding_col]
                    if isinstance(emb, np.ndarray):
                        embedding = emb.tolist()
                    elif isinstance(emb, list):
                        embedding = emb

                record = {
                    'uri': row.get('uri', f'unknown:{idx}'),
                    'label': row.get('label', 'Unknown'),
                    'definition': row.get('definition', row.get('raw_content', '')),
                    'context_text': row.get('context_text', ''),
                    'source': row.get('source', ''),
                    'embedding': embedding
                }
                records.append(record)

                # Batch insert
                if len(records) >= batch_size:
                    self._insert_node_batch(session, records, node_label)
                    records = []

            # Insert remaining
            if records:
                self._insert_node_batch(session, records, node_label)

        logger.info(f"✓ Ingested {len(df)} {node_label} nodes")

    def _insert_node_batch(self, session: Session, records: List[Dict], node_label: str):
        """Insert a batch of nodes."""
        query = f"""
            UNWIND $records AS record
            MERGE (n:{node_label} {{uri: record.uri}})
            SET n.label = record.label,
                n.definition = record.definition,
                n.context_text = record.context_text,
                n.source = record.source,
                n.embedding = record.embedding
        """
        session.run(query, records=records)

    def create_hierarchical_relationships(self, df: pd.DataFrame, node_label: str):
        """
        Create PARENT_OF relationships based on hierarchy in context_text.

        Parses context_text like "System > Component > Part" and creates edges.

        Args:
            df: DataFrame with uri and context_text columns
            node_label: Node label to match
        """
        logger.info(f"Creating hierarchical relationships for {node_label}...")

        relationships = []

        for _, row in df.iterrows():
            uri = row.get('uri')
            context = row.get('context_text', '')

            # Parse hierarchy from context
            # Expected format: contains hierarchy like "Bicycle > Wheel > Spoke"
            if '>' in context:
                # Try to extract hierarchy path
                parts = [p.strip() for p in context.split('>')]
                if len(parts) >= 2:
                    # For now, create relationship to immediate parent
                    # You can extend this to parse actual URIs if available
                    pass

            # Alternative: Use explicit parent_uri or hierarchy_path if available
            if 'parent_uri' in row and row['parent_uri']:
                relationships.append({
                    'parent_uri': row['parent_uri'],
                    'child_uri': uri
                })

        # Create relationships in batches
        if relationships:
            with self.driver.session(database=self.database) as session:
                query = f"""
                    UNWIND $rels AS rel
                    MATCH (parent:{node_label} {{uri: rel.parent_uri}})
                    MATCH (child:{node_label} {{uri: rel.child_uri}})
                    MERGE (parent)-[:PARENT_OF]->(child)
                """
                session.run(query, rels=relationships)
                logger.info(f"✓ Created {len(relationships)} PARENT_OF relationships")

    def retrieve_candidates(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        target_label: str = "OntologyClass",
        include_neighbors: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Graph-aware retrieval: Vector search + graph traversal.

        Steps:
        1. Vector similarity search to find top-k most similar nodes
        2. For each result, retrieve graph neighbors (parents, children, siblings)
        3. Return enriched candidates with contextual information

        Args:
            query_embedding: Query embedding vector
            top_k: Number of initial candidates from vector search
            target_label: Node label to search (OntologyClass or S1000DConcept)
            include_neighbors: Whether to include graph neighbors

        Returns:
            List of candidate dicts with uri, label, score, neighbors, etc.
        """
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        with self.driver.session(database=self.database) as session:
            # Step 1: Vector similarity search
            vector_search_query = f"""
                CALL db.index.vector.queryNodes('{target_label.lower()}_embeddings', $top_k, $query_embedding)
                YIELD node, score
                RETURN node.uri AS uri,
                       node.label AS label,
                       node.definition AS definition,
                       node.context_text AS context_text,
                       score
                ORDER BY score DESC
            """

            try:
                result = session.run(
                    vector_search_query,
                    top_k=top_k,
                    query_embedding=query_embedding
                )

                candidates = []
                for record in result:
                    candidate = {
                        'uri': record['uri'],
                        'label': record['label'],
                        'definition': record.get('definition', ''),
                        'context_text': record.get('context_text', ''),
                        'similarity_score': float(record['score'])
                    }
                    candidates.append(candidate)

            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                # Fallback: return random sample
                fallback_query = f"""
                    MATCH (n:{target_label})
                    RETURN n.uri AS uri,
                           n.label AS label,
                           n.definition AS definition,
                           n.context_text AS context_text
                    LIMIT $top_k
                """
                result = session.run(fallback_query, top_k=top_k)
                candidates = []
                for record in result:
                    candidate = {
                        'uri': record['uri'],
                        'label': record['label'],
                        'definition': record.get('definition', ''),
                        'context_text': record.get('context_text', ''),
                        'similarity_score': 0.5
                    }
                    candidates.append(candidate)

            # Step 2: Graph traversal for neighbors (if enabled)
            if include_neighbors and candidates:
                candidates = self._enrich_with_neighbors(session, candidates, target_label)

        logger.info(f"Retrieved {len(candidates)} candidates with graph context")
        return candidates

    def _enrich_with_neighbors(
        self,
        session: Session,
        candidates: List[Dict[str, Any]],
        node_label: str
    ) -> List[Dict[str, Any]]:
        """
        Enrich candidates with graph neighbors.

        For each candidate, find:
        - Parents (1 level up)
        - Children (1 level down)
        - Siblings (share same parent)
        """
        for candidate in candidates:
            uri = candidate['uri']

            # Query for neighbors
            neighbor_query = f"""
                MATCH (n:{node_label} {{uri: $uri}})
                OPTIONAL MATCH (parent)-[:PARENT_OF]->(n)
                OPTIONAL MATCH (n)-[:PARENT_OF]->(child)
                OPTIONAL MATCH (parent)-[:PARENT_OF]->(sibling)
                WHERE sibling.uri <> n.uri
                RETURN
                    collect(DISTINCT parent.label) AS parents,
                    collect(DISTINCT child.label) AS children,
                    collect(DISTINCT sibling.label)[..5] AS siblings
            """

            result = session.run(neighbor_query, uri=uri)
            record = result.single()

            if record:
                candidate['parents'] = record['parents'] or []
                candidate['children'] = record['children'] or []
                candidate['siblings'] = record['siblings'] or []

                # Create enriched context text
                context_parts = []
                if candidate['parents']:
                    context_parts.append(f"Parents: {', '.join(candidate['parents'])}")
                if candidate['children']:
                    context_parts.append(f"Children: {', '.join(candidate['children'][:3])}")

                candidate['graph_context'] = ' | '.join(context_parts) if context_parts else ''

        return candidates

    def get_reasoning_context(
        self,
        uri: str,
        hops: int = 2,
        max_paths: int = 10,
        relationship_types: List[str] = None
    ) -> str:
        """
        Get multi-hop reasoning context for a concept.

        Retrieves graph paths from the concept to related concepts
        up to N hops away, formatted as readable reasoning chains.

        Args:
            uri: URI of the concept
            hops: Number of hops (1-3 recommended)
            max_paths: Maximum number of paths to return
            relationship_types: List of relationship types to traverse
                               (default: PARENT_OF, PART_OF, RELATED_TO)

        Returns:
            Formatted string with reasoning paths

        Example output:
            [PATH] Wheel -> PART_OF -> LandingGear -> RELATED_TO -> Hydraulics
            [PATH] Wheel -> PARENT_OF -> Hub
        """
        if relationship_types is None:
            relationship_types = ['PARENT_OF', 'PART_OF', 'RELATED_TO']

        # Limit hops to reasonable range
        hops = max(1, min(hops, 3))

        with self.driver.session(database=self.database) as session:
            # Multi-hop path query
            query = f"""
                MATCH path = (start {{uri: $uri}})-[r*1..{hops}]-(connected)
                WHERE ALL(rel in relationships(path)
                    WHERE type(rel) IN $relationship_types)
                WITH path, length(path) as path_length
                ORDER BY path_length, connected.label
                LIMIT $max_paths
                RETURN
                    [node in nodes(path) | node.label] AS node_labels,
                    [rel in relationships(path) | type(rel)] AS rel_types,
                    length(path) as path_length
            """

            try:
                result = session.run(
                    query,
                    uri=uri,
                    relationship_types=relationship_types,
                    max_paths=max_paths
                )

                paths = []
                for record in result:
                    node_labels = record['node_labels']
                    rel_types = record['rel_types']
                    path_length = record['path_length']

                    # Build path string: Node1 -> REL -> Node2 -> REL -> Node3
                    path_parts = []
                    for i in range(len(node_labels)):
                        path_parts.append(node_labels[i])
                        if i < len(rel_types):
                            path_parts.append(f" -> {rel_types[i]} -> ")

                    path_str = ''.join(path_parts)
                    paths.append(f"[PATH-{path_length}] {path_str}")

                if not paths:
                    return f"[NO PATHS] {uri} (isolated node)"

                # Format as multi-line context
                context = "\n".join(paths)
                return context

            except Exception as e:
                logger.error(f"Failed to get reasoning context for {uri}: {e}")
                return f"[ERROR] Could not retrieve paths for {uri}"

    def get_reasoning_context_batch(
        self,
        uris: List[str],
        hops: int = 2,
        max_paths_per_uri: int = 5
    ) -> Dict[str, str]:
        """
        Get reasoning contexts for multiple URIs efficiently.

        Args:
            uris: List of concept URIs
            hops: Number of hops
            max_paths_per_uri: Max paths per concept

        Returns:
            Dictionary mapping uri -> reasoning_context
        """
        contexts = {}
        for uri in uris:
            contexts[uri] = self.get_reasoning_context(
                uri,
                hops=hops,
                max_paths=max_paths_per_uri
            )
        return contexts

    def retrieve_candidates_with_reasoning(
        self,
        query_embedding: np.ndarray,
        source_uri: str = None,
        top_k: int = 20,
        target_label: str = "OntologyClass",
        reasoning_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Enhanced retrieval with multi-hop reasoning context.

        This combines:
        1. Vector similarity search
        2. Graph neighbor enrichment
        3. Multi-hop reasoning paths

        Perfect for neural reranker input!

        Args:
            query_embedding: Query embedding vector
            source_uri: Optional URI of source concept (for source reasoning)
            top_k: Number of candidates
            target_label: Node label to search
            reasoning_hops: Number of hops for reasoning paths

        Returns:
            List of candidates with reasoning_context field
        """
        # Step 1: Standard retrieval with neighbors
        candidates = self.retrieve_candidates(
            query_embedding,
            top_k=top_k,
            target_label=target_label,
            include_neighbors=True
        )

        # Step 2: Add multi-hop reasoning for each candidate
        for candidate in candidates:
            candidate_uri = candidate['uri']

            # Get reasoning paths for this candidate
            reasoning = self.get_reasoning_context(
                candidate_uri,
                hops=reasoning_hops,
                max_paths=5
            )

            candidate['reasoning_context'] = reasoning

            # Create combined context for reranker
            context_parts = [
                f"[LABEL] {candidate['label']}",
                f"[DEFINITION] {candidate.get('definition', 'N/A')[:200]}"
            ]

            if candidate.get('graph_context'):
                context_parts.append(f"[NEIGHBORS] {candidate['graph_context']}")

            if reasoning and not reasoning.startswith('[NO PATHS]'):
                context_parts.append(f"[REASONING]\n{reasoning}")

            candidate['full_context'] = '\n'.join(context_parts)

        # Step 3: Optionally add source reasoning
        if source_uri:
            source_reasoning = self.get_reasoning_context(
                source_uri,
                hops=reasoning_hops,
                max_paths=5
            )

            # Add to all candidates for comparison
            for candidate in candidates:
                candidate['source_reasoning'] = source_reasoning

        logger.info(f"Retrieved {len(candidates)} candidates with multi-hop reasoning")
        return candidates

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the graph."""
        with self.driver.session(database=self.database) as session:
            # Count nodes by label
            s1000d_count = session.run("MATCH (n:S1000DConcept) RETURN count(n) AS count").single()['count']
            ontology_count = session.run("MATCH (n:OntologyClass) RETURN count(n) AS count").single()['count']

            # Count relationships
            parent_of_count = session.run("MATCH ()-[r:PARENT_OF]->() RETURN count(r) AS count").single()['count']
            related_to_count = session.run("MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS count").single()['count']

            return {
                's1000d_concepts': s1000d_count,
                'ontology_classes': ontology_count,
                'parent_of_relationships': parent_of_count,
                'related_to_relationships': related_to_count,
                'total_nodes': s1000d_count + ontology_count,
                'total_relationships': parent_of_count + related_to_count
            }


def test_graph_store():
    """Test the KnowledgeGraphStore."""
    print("=" * 80)
    print("NEO4J KNOWLEDGE GRAPH STORE TEST")
    print("=" * 80)

    # Initialize store
    store = KnowledgeGraphStore(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )

    # Initialize schema
    store.initialize_schema(embedding_dim=768)

    # Create test data
    test_df = pd.DataFrame([
        {
            'uri': 'test:concept:1',
            'label': 'Wheel',
            'definition': 'A circular component that rotates',
            'context_text': 'Bicycle > Components > Wheel',
            'source': 's1000d',
            'embedding': np.random.rand(768)
        },
        {
            'uri': 'test:concept:2',
            'label': 'Brake',
            'definition': 'A device for stopping motion',
            'context_text': 'Bicycle > Components > Brake',
            'source': 's1000d',
            'embedding': np.random.rand(768)
        }
    ])

    # Ingest data
    store.ingest_dataframe(test_df, node_label='S1000DConcept')

    # Test retrieval
    query_emb = np.random.rand(768)
    candidates = store.retrieve_candidates(query_emb, top_k=5, target_label='S1000DConcept')

    print(f"\nRetrieved {len(candidates)} candidates")
    for i, cand in enumerate(candidates[:3], 1):
        print(f"\n  {i}. {cand['label']}")
        print(f"     URI: {cand['uri']}")
        print(f"     Score: {cand['similarity_score']:.3f}")

    # Statistics
    stats = store.get_statistics()
    print("\n" + "=" * 80)
    print("GRAPH STATISTICS")
    print("=" * 80)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    store.close()


if __name__ == '__main__':
    test_graph_store()
