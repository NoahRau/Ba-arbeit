"""
Build Knowledge Base with ModernBERT embeddings.
Links S1000D concepts to BikeOntology using semantic similarity and LLM reasoning.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm

import pandas as pd
from sentence_transformers import SentenceTransformer
from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, Literal

from data_loader import load_all_concepts
from llm_reasoner import LLMReasoner


# Configuration
EMBEDDING_CACHE_FILE = Path('embeddings_cache.pkl')
OUTPUT_TTL_FILE = Path('new_linked_bike_network.ttl')

# Thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.85  # Direct owl:sameAs (no LLM needed)
LOW_CONFIDENCE_THRESHOLD = 0.60   # Needs LLM verification
CLAUDE_ACCEPTS_MATCH = True       # Claude says "is_match": true

# Namespaces
BIKE = Namespace("http://my-company.com/bike-ontology#")
S1000D = Namespace("http://my-company.com/s1000d/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")


class ModernBERTEmbedder:
    """
    Embedder using ModernBERT with support for long contexts (up to 8192 tokens).
    """

    def __init__(self, model_name: str = 'answerdotai/ModernBERT-base'):
        """
        Initialize ModernBERT embedder.

        Args:
            model_name: Model identifier for ModernBERT
        """
        print(f"Loading ModernBERT model: {model_name}")
        print("This may take a moment on first run...")

        try:
            # Load with trust_remote_code for ModernBERT
            self.model = SentenceTransformer(
                model_name,
                trust_remote_code=True
            )
            print(f"✓ Model loaded successfully")
            print(f"  Max sequence length: {self.model.max_seq_length}")

        except Exception as e:
            print(f"Error loading ModernBERT: {e}")
            print("Falling back to standard BERT model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Create embeddings for a batch of texts.
        ModernBERT can handle up to 8192 tokens, so we don't truncate aggressively.

        Args:
            texts: List of texts to embed
            show_progress: Show progress bar
            batch_size: Batch size for encoding

        Returns:
            Array of embeddings
        """
        # ModernBERT handles long contexts well, so we keep them
        # Only truncate if extremely long (>8000 chars ≈ 2000 tokens)
        processed_texts = []
        for text in texts:
            if len(text) > 8000:
                # Keep as much as possible while staying safe
                processed_texts.append(text[:8000] + "...")
            else:
                processed_texts.append(text)

        embeddings = self.model.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings


def cosine_similarity_batch(
    query_embedding: np.ndarray,
    embeddings: np.ndarray
) -> np.ndarray:
    """
    Calculate cosine similarity between query and all embeddings.

    Args:
        query_embedding: Single embedding vector
        embeddings: Matrix of embeddings

    Returns:
        Array of similarity scores
    """
    # Normalize
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute similarities
    similarities = np.dot(embeddings_norm, query_norm)

    return similarities


def build_embeddings(
    df: pd.DataFrame,
    embedder: ModernBERTEmbedder,
    force_rebuild: bool = False
) -> np.ndarray:
    """
    Build or load embeddings for all concepts.

    Args:
        df: DataFrame with concepts
        embedder: ModernBERT embedder
        force_rebuild: Force rebuilding even if cache exists

    Returns:
        Array of embeddings
    """
    if EMBEDDING_CACHE_FILE.exists() and not force_rebuild:
        print(f"\n✓ Loading cached embeddings from {EMBEDDING_CACHE_FILE}")
        with open(EMBEDDING_CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)

        # Verify cache matches current data
        if cache['count'] == len(df):
            print(f"  Cache valid: {cache['count']} embeddings")
            return cache['embeddings']
        else:
            print(f"  Cache outdated: {cache['count']} != {len(df)}")
            print("  Rebuilding embeddings...")

    print("\n" + "=" * 70)
    print("BUILDING EMBEDDINGS WITH ModernBERT")
    print("=" * 70)

    # Use context_text which already contains hierarchical information
    texts = []
    for _, row in df.iterrows():
        context_text = row.get('context_text', '')

        # context_text already contains: Hierarchy > Label | Description
        # For ModernBERT, we use this rich context directly
        if context_text:
            combined = context_text
        else:
            # Fallback to label if context_text is missing
            combined = row.get('label', 'No description available')

        texts.append(combined)

    print(f"\nGenerating embeddings for {len(texts)} concepts...")
    print("This may take several minutes with ModernBERT...")

    embeddings = embedder.embed_batch(texts, show_progress=True, batch_size=8)

    print(f"\n✓ Embeddings generated: shape {embeddings.shape}")

    # Cache embeddings
    print(f"Caching embeddings to {EMBEDDING_CACHE_FILE}...")
    with open(EMBEDDING_CACHE_FILE, 'wb') as f:
        pickle.dump({
            'count': len(df),
            'embeddings': embeddings,
            'model': 'ModernBERT-base'
        }, f)

    print("✓ Embeddings cached successfully")

    return embeddings


def find_matches(
    s1000d_df: pd.DataFrame,
    ontology_df: pd.DataFrame,
    s1000d_embeddings: np.ndarray,
    ontology_embeddings: np.ndarray,
    reasoner: LLMReasoner = None
) -> List[Dict[str, Any]]:
    """
    Find matches between S1000D concepts and ontology concepts.

    Matching strategy:
    - Score >= 0.85: Automatic owl:sameAs (high confidence)
    - Score 0.60-0.85: Claude verification needed
      - If Claude confirms: owl:sameAs
      - If Claude rejects: discard

    Args:
        s1000d_df: S1000D concepts DataFrame
        ontology_df: Ontology concepts DataFrame
        s1000d_embeddings: S1000D embeddings
        ontology_embeddings: Ontology embeddings
        reasoner: LLM reasoner for uncertain matches

    Returns:
        List of match dictionaries
    """
    print("\n" + "=" * 70)
    print("FINDING MATCHES WITH CLAUDE INTEGRATION")
    print("=" * 70)
    print(f"\nStrategy:")
    print(f"  Score >= 0.85: Automatic owl:sameAs")
    print(f"  Score 0.60-0.85: Claude verification")
    print(f"    ✓ Claude confirms → owl:sameAs")
    print(f"    ✗ Claude rejects → discard")

    matches = []
    stats = {
        'high_confidence': 0,
        'claude_accepted': 0,
        'claude_rejected': 0,
        'claude_calls': 0
    }

    print(f"\nComparing {len(s1000d_df)} S1000D concepts with {len(ontology_df)} ontology concepts...")

    for idx in tqdm(range(len(s1000d_df)), desc="Processing S1000D concepts", unit="concept"):
        s1000d_concept = s1000d_df.iloc[idx]
        s1000d_embedding = s1000d_embeddings[idx]

        # Calculate similarities with all ontology concepts
        similarities = cosine_similarity_batch(s1000d_embedding, ontology_embeddings)

        # Get top candidates
        top_indices = np.argsort(similarities)[::-1][:5]  # Top 5

        for onto_idx in top_indices:
            score = float(similarities[onto_idx])

            # Skip low scores (< 0.60)
            if score < LOW_CONFIDENCE_THRESHOLD:
                continue

            ontology_concept = ontology_df.iloc[onto_idx]

            # HIGH CONFIDENCE (>= 0.85): Direct owl:sameAs
            if score >= HIGH_CONFIDENCE_THRESHOLD:
                matches.append({
                    's1000d_uri': s1000d_concept['uri'],
                    's1000d_label': s1000d_concept['label'],
                    'ontology_uri': ontology_concept['uri'],
                    'ontology_label': ontology_concept['label'],
                    'similarity_score': score,
                    'link_type': 'owl:sameAs',
                    'method': 'embedding',
                    'verified': True
                })
                stats['high_confidence'] += 1

            # MEDIUM CONFIDENCE (0.60-0.85): Needs Claude verification
            elif reasoner is not None:
                try:
                    # Call Claude for verification
                    llm_result = reasoner.verify_match_with_claude(
                        s1000d_concept.to_dict(),
                        ontology_concept.to_dict()
                    )

                    stats['claude_calls'] += 1

                    # Claude says "Yes" (is_match = True) -> Accept as owl:sameAs
                    if llm_result['is_match']:
                        matches.append({
                            's1000d_uri': s1000d_concept['uri'],
                            's1000d_label': s1000d_concept['label'],
                            'ontology_uri': ontology_concept['uri'],
                            'ontology_label': ontology_concept['label'],
                            'similarity_score': score,
                            'link_type': 'owl:sameAs',
                            'llm_confidence': llm_result['confidence'],
                            'llm_reason': llm_result['reason'],
                            'method': 'embedding+llm',
                            'verified': True
                        })
                        stats['claude_accepted'] += 1
                    else:
                        # Claude says "No" -> Discard (don't add to matches)
                        stats['claude_rejected'] += 1

                except Exception as e:
                    print(f"\n⚠ Warning: LLM verification failed for {s1000d_concept['label']}: {e}")
                    continue

            # No reasoner available for medium confidence scores
            elif reasoner is None and LOW_CONFIDENCE_THRESHOLD <= score < HIGH_CONFIDENCE_THRESHOLD:
                # Skip this match (can't verify without Claude)
                continue

    print(f"\n✓ Matching complete")
    print(f"\n  Results:")
    print(f"    High confidence matches (≥0.85): {stats['high_confidence']}")
    print(f"    Claude API calls: {stats['claude_calls']}")
    print(f"      ✓ Accepted: {stats['claude_accepted']}")
    print(f"      ✗ Rejected: {stats['claude_rejected']}")
    print(f"    Total matches: {len(matches)}")

    return matches


def generate_linked_ontology(
    s1000d_df: pd.DataFrame,
    matches: List[Dict[str, Any]],
    output_file: Path = OUTPUT_TTL_FILE
) -> None:
    """
    Generate complete linked ontology with all S1000D concepts and their links.

    Args:
        s1000d_df: DataFrame with all S1000D concepts
        matches: List of match dictionaries
        output_file: Output TTL file path
    """
    print("\n" + "=" * 70)
    print("GENERATING LINKED ONTOLOGY")
    print("=" * 70)

    # Create RDF graph
    g = Graph()

    # Bind namespaces
    g.bind("bike", BIKE)
    g.bind("s1000d", S1000D)
    g.bind("skos", SKOS)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)

    # Step 1: Add all S1000D concepts as OWL Classes
    print(f"\nStep 1: Adding {len(s1000d_df)} S1000D concepts as OWL Classes...")

    for _, concept in tqdm(s1000d_df.iterrows(), total=len(s1000d_df), desc="Adding S1000D concepts"):
        s1000d_uri = URIRef(concept['uri'])

        # Declare as OWL Class (representing component/procedure types)
        g.add((s1000d_uri, RDF.type, OWL.Class))

        # Add label
        if concept['label']:
            g.add((s1000d_uri, RDFS.label, Literal(concept['label'], lang='en')))

        # Add context as comment
        if concept.get('context_text'):
            # Truncate very long contexts for TTL readability
            context_text = concept['context_text']
            if len(context_text) > 500:
                context_text = context_text[:500] + "..."
            g.add((s1000d_uri, RDFS.comment, Literal(context_text, lang='en')))

        # Add source annotation
        g.add((s1000d_uri, BIKE.source, Literal('S1000D')))

    print(f"✓ Added {len(s1000d_df)} S1000D concepts to ontology")

    # Step 2: Add links to BikeOntology concepts
    if matches:
        print(f"\nStep 2: Adding {len(matches)} links to BikeOntology...")

        same_as_count = 0
        close_match_count = 0

        for match in tqdm(matches, desc="Adding ontology links", unit="link"):
            s1000d_uri = URIRef(match['s1000d_uri'])
            ontology_uri = URIRef(match['ontology_uri'])
            score = match['similarity_score']
            link_type = match.get('link_type', 'owl:sameAs')

            # Add the appropriate relation based on link_type
            if link_type == 'owl:sameAs':
                g.add((s1000d_uri, OWL.sameAs, ontology_uri))
                same_as_count += 1
            else:
                # Fallback to skos:closeMatch
                g.add((s1000d_uri, SKOS.closeMatch, ontology_uri))
                close_match_count += 1

            # Add metadata about the match
            g.add((s1000d_uri, BIKE.similarityScore, Literal(score)))
            g.add((s1000d_uri, BIKE.matchingMethod, Literal(match['method'])))

            # Add LLM confidence if available
            if 'llm_confidence' in match:
                g.add((s1000d_uri, BIKE.llmConfidence, Literal(match['llm_confidence'])))
                if 'llm_reason' in match:
                    g.add((s1000d_uri, BIKE.llmReason, Literal(match['llm_reason'], lang='en')))

            # Add verification status
            g.add((s1000d_uri, BIKE.verified, Literal(match['verified'])))

        print(f"✓ Added links:")
        print(f"  owl:sameAs: {same_as_count}")
        if close_match_count > 0:
            print(f"  skos:closeMatch: {close_match_count}")
    else:
        print("\nStep 2: No matches to link (skipping)")

    # Step 3: Serialize to TTL
    print(f"\nStep 3: Serializing to {output_file}...")
    g.serialize(destination=str(output_file), format='turtle')

    print(f"\n✓ Linked ontology saved: {output_file}")
    print(f"  Total triples: {len(g)}")
    print(f"  S1000D concepts: {len(s1000d_df)}")
    print(f"  Links: {len(matches)}")


def main():
    """
    Main pipeline for building knowledge base.
    """
    print("=" * 70)
    print("MODERNBERT KNOWLEDGE BASE BUILDER")
    print("=" * 70)

    # Step 1: Load data
    print("\n[1/5] LOADING DATA")
    print("-" * 70)

    df = load_all_concepts(
        s1000d_folder='bike',
        include_ontologies=True
    )

    if df.empty:
        print("Error: No data loaded")
        return

    # Split by source
    s1000d_df = df[df['source'] == 's1000d'].reset_index(drop=True)
    ontology_df = df[df['source'] == 'bike_ontology'].reset_index(drop=True)

    print(f"\n✓ Data loaded:")
    print(f"  S1000D concepts: {len(s1000d_df)}")
    print(f"  Ontology concepts: {len(ontology_df)}")

    if len(ontology_df) == 0:
        print("\n⚠ Warning: No ontology concepts found!")
        print("The system will process S1000D data without ontology matching.")
        print("To include ontology matching in future runs:")
        print("  1. Configure ONTOLOGY_URLS in data_loader.py, OR")
        print("  2. Place .owl files in ontology_cache/ directory")
        print("\nContinuing with S1000D-only export...")

    # Step 2: Initialize ModernBERT
    print("\n[2/5] INITIALIZING ModernBERT")
    print("-" * 70)

    embedder = ModernBERTEmbedder()

    # Step 3: Build embeddings
    print("\n[3/5] BUILDING EMBEDDINGS")
    print("-" * 70)

    all_embeddings = build_embeddings(df, embedder, force_rebuild=False)

    # Split embeddings by source
    s1000d_indices = df[df['source'] == 's1000d'].index.tolist()
    ontology_indices = df[df['source'] == 'bike_ontology'].index.tolist()

    s1000d_embeddings = all_embeddings[s1000d_indices]
    ontology_embeddings = all_embeddings[ontology_indices] if ontology_indices else np.array([])

    print(f"\n✓ Embeddings ready:")
    print(f"  S1000D: {s1000d_embeddings.shape}")
    if len(ontology_embeddings) > 0:
        print(f"  Ontology: {ontology_embeddings.shape}")

    # Step 4: Find matches
    print("\n[4/5] FINDING MATCHES")
    print("-" * 70)

    if len(ontology_embeddings) == 0:
        print("Skipping matching (no ontology data)")
        matches = []
    else:
        # Initialize LLM reasoner
        try:
            reasoner = LLMReasoner()
            print("✓ Claude API initialized for verification")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize Claude API: {e}")
            print("  Proceeding without LLM verification")
            reasoner = None

        matches = find_matches(
            s1000d_df,
            ontology_df,
            s1000d_embeddings,
            ontology_embeddings,
            reasoner
        )

    # Step 5: Generate linked ontology
    print("\n[5/5] GENERATING LINKED ONTOLOGY")
    print("-" * 70)

    # Always generate ontology with all S1000D concepts
    # Links are added if matches exist
    generate_linked_ontology(s1000d_df, matches, OUTPUT_TTL_FILE)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Pipeline complete!")
    print(f"\n  Data processed:")
    print(f"    Total concepts: {len(df)}")
    print(f"    S1000D concepts: {len(s1000d_df)}")
    if len(ontology_df) > 0:
        print(f"    Ontology concepts: {len(ontology_df)}")

    print(f"\n  Matching results:")
    print(f"    Total matches: {len(matches)}")

    if matches:
        # Match method breakdown
        embedding_only = sum(1 for m in matches if m['method'] == 'embedding')
        embedding_llm = sum(1 for m in matches if m['method'] == 'embedding+llm')
        print(f"\n  Match methods:")
        print(f"    High confidence (≥0.85): {embedding_only}")
        print(f"    Claude verified (0.60-0.85): {embedding_llm}")

        # All matches are owl:sameAs in the new strategy
        print(f"\n  Link types:")
        print(f"    owl:sameAs: {len(matches)}")

    print(f"\n  Output:")
    print(f"    File: {OUTPUT_TTL_FILE}")
    print(f"    S1000D concepts exported: {len(s1000d_df)}")
    print(f"    Links created: {len(matches)}")


if __name__ == '__main__':
    main()
