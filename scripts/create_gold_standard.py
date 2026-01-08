"""
Gold Standard Creator for Ontology Matching Evaluation.
Manual annotation tool to create benchmark dataset.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_all_concepts
from build_knowledge_base import ModernBERTEmbedder, cosine_similarity_batch


# Configuration
GOLD_STANDARD_FILE = Path('gold_standard_metrics.json')
SAMPLE_SIZE = 50
TOP_K = 3


@st.cache_resource
def load_data_and_embeddings():
    """
    Load all data and build embeddings.
    Cached to avoid reloading.
    """
    # Load concepts
    df = load_all_concepts(
        s1000d_folder='bike',
        include_ontologies=True
    )

    if df.empty:
        st.error("No data loaded!")
        st.stop()

    # Split by source
    s1000d_df = df[df['source'] == 's1000d'].reset_index(drop=True)
    ontology_df = df[df['source'] == 'bike_ontology'].reset_index(drop=True)

    # Initialize embedder
    embedder = ModernBERTEmbedder()

    # Build embeddings
    with st.spinner("Building embeddings... This may take a few minutes..."):
        all_texts = []
        for _, row in df.iterrows():
            label = row.get('label', '')
            context = row.get('context', '')
            if label and context:
                combined = f"{label}. {context[:500]}"
            elif label:
                combined = label
            else:
                combined = context[:500] if context else ""
            all_texts.append(combined)

        embeddings = embedder.embed_batch(all_texts, show_progress=False, batch_size=8)

    # Split embeddings
    s1000d_indices = df[df['source'] == 's1000d'].index.tolist()
    ontology_indices = df[df['source'] == 'bike_ontology'].index.tolist()

    s1000d_embeddings = embeddings[s1000d_indices]
    ontology_embeddings = embeddings[ontology_indices]

    return s1000d_df, ontology_df, s1000d_embeddings, ontology_embeddings


def load_gold_standard():
    """
    Load existing gold standard annotations.
    """
    if GOLD_STANDARD_FILE.exists():
        with open(GOLD_STANDARD_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_gold_standard(annotations: List[Dict[str, Any]]):
    """
    Save gold standard annotations to JSON.
    """
    with open(GOLD_STANDARD_FILE, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)


def get_top_candidates(
    s1000d_concept: pd.Series,
    s1000d_embedding: np.ndarray,
    ontology_df: pd.DataFrame,
    ontology_embeddings: np.ndarray,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Get top K candidates from ontology for a given S1000D concept.
    """
    # Calculate similarities
    similarities = cosine_similarity_batch(s1000d_embedding, ontology_embeddings)

    # Get top indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    candidates = []
    for idx in top_indices:
        ontology_concept = ontology_df.iloc[idx]
        score = float(similarities[idx])

        candidates.append({
            'uri': ontology_concept['uri'],
            'label': ontology_concept['label'],
            'context': ontology_concept['context'],
            'similarity_score': score
        })

    return candidates


def main():
    """
    Main Streamlit application.
    """
    st.set_page_config(
        page_title="Gold Standard Creator",
        page_icon="⭐",
        layout="wide"
    )

    st.title("⭐ Gold Standard Creator")
    st.markdown("**Manual Annotation Tool for Ontology Matching Evaluation**")

    st.divider()

    # Load data
    s1000d_df, ontology_df, s1000d_embeddings, ontology_embeddings = load_data_and_embeddings()

    # Initialize session state
    if 'sample_indices' not in st.session_state:
        # Randomly sample S1000D concepts
        random.seed(42)  # Reproducible sampling
        all_indices = list(range(len(s1000d_df)))
        sample_size = min(SAMPLE_SIZE, len(s1000d_df))
        st.session_state.sample_indices = random.sample(all_indices, sample_size)
        st.session_state.current_idx = 0

    if 'annotations' not in st.session_state:
        st.session_state.annotations = load_gold_standard()

    # Sidebar: Progress and stats
    with st.sidebar:
        st.header("📊 Progress")

        current = st.session_state.current_idx + 1
        total = len(st.session_state.sample_indices)

        st.metric("Current Item", f"{current} / {total}")

        progress = current / total
        st.progress(progress)

        # Count annotations
        annotated_count = len(set(
            a['s1000d_uri']
            for a in st.session_state.annotations
        ))
        st.metric("Items Annotated", annotated_count)

        positive_count = sum(1 for a in st.session_state.annotations if a['is_match_manual'])
        st.metric("Positive Matches", positive_count)

        st.divider()

        st.header("ℹ️ Instructions")
        st.markdown("""
        1. Read the S1000D concept carefully
        2. For each BikeOntology candidate, decide:
           - ✅ Check if it's a **correct match**
           - ⬜ Leave unchecked if it's **not a match**
        3. Click **Save & Next** to continue
        4. Use **Previous** to review earlier items
        """)

        st.divider()

        if st.button("💾 Save All & Download"):
            save_gold_standard(st.session_state.annotations)
            st.success(f"Saved to {GOLD_STANDARD_FILE}")

            # Offer download
            json_str = json.dumps(st.session_state.annotations, indent=2, ensure_ascii=False)
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name="gold_standard_metrics.json",
                mime="application/json"
            )

    # Check if we're done
    if st.session_state.current_idx >= len(st.session_state.sample_indices):
        st.success("🎉 All items annotated!")
        st.balloons()

        save_gold_standard(st.session_state.annotations)

        st.info(f"Gold standard saved to: {GOLD_STANDARD_FILE}")

        # Show statistics
        st.header("📈 Annotation Statistics")
        df_stats = pd.DataFrame(st.session_state.annotations)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Annotations", len(df_stats))
        with col2:
            st.metric("Positive Matches", df_stats['is_match_manual'].sum())
        with col3:
            st.metric("Negative Matches", (~df_stats['is_match_manual']).sum())

        # Reset button
        if st.button("🔄 Start Over"):
            st.session_state.sample_indices = None
            st.session_state.current_idx = 0
            st.session_state.annotations = []
            st.rerun()

        return

    # Get current S1000D concept
    current_sample_idx = st.session_state.sample_indices[st.session_state.current_idx]
    s1000d_concept = s1000d_df.iloc[current_sample_idx]
    s1000d_embedding = s1000d_embeddings[current_sample_idx]

    # Get top candidates
    candidates = get_top_candidates(
        s1000d_concept,
        s1000d_embedding,
        ontology_df,
        ontology_embeddings,
        top_k=TOP_K
    )

    # Display S1000D concept
    st.header(f"📄 S1000D Concept #{st.session_state.current_idx + 1}")

    with st.container(border=True):
        st.subheader(s1000d_concept['label'])
        st.code(s1000d_concept['uri'], language=None)

        if s1000d_concept['context']:
            with st.expander("📖 Full Context", expanded=True):
                st.markdown(s1000d_concept['context'][:1000])
                if len(s1000d_concept['context']) > 1000:
                    st.caption(f"... ({len(s1000d_concept['context'])} chars total)")

    st.divider()

    # Display candidates for annotation
    st.header("🎯 Top 3 BikeOntology Candidates")
    st.markdown("**Select all candidates that are correct matches:**")

    # Store decisions
    decisions = {}

    for i, candidate in enumerate(candidates):
        with st.container(border=True):
            col1, col2 = st.columns([1, 5])

            with col1:
                # Checkbox for annotation
                key = f"match_{st.session_state.current_idx}_{i}"

                # Check if already annotated
                existing = next(
                    (a for a in st.session_state.annotations
                     if a['s1000d_uri'] == s1000d_concept['uri']
                     and a['bike_uri'] == candidate['uri']),
                    None
                )

                default_value = existing['is_match_manual'] if existing else False

                is_match = st.checkbox(
                    f"**Match #{i+1}**",
                    value=default_value,
                    key=key
                )

                decisions[candidate['uri']] = is_match

                # Show similarity score
                st.metric("Score", f"{candidate['similarity_score']:.3f}")

            with col2:
                st.markdown(f"### {candidate['label']}")
                st.code(candidate['uri'], language=None)

                if candidate['context']:
                    context_preview = candidate['context'][:300]
                    if len(candidate['context']) > 300:
                        context_preview += "..."
                    st.markdown(f"**Context:** {context_preview}")

    st.divider()

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.session_state.current_idx > 0:
            if st.button("⬅️ Previous", use_container_width=True):
                st.session_state.current_idx -= 1
                st.rerun()

    with col2:
        if st.button("⏭️ Skip", use_container_width=True):
            st.session_state.current_idx += 1
            st.rerun()

    with col3:
        if st.button("💾 Save & Next ➡️", type="primary", use_container_width=True):
            # Save annotations for current item
            for candidate in candidates:
                # Remove existing annotation if any
                st.session_state.annotations = [
                    a for a in st.session_state.annotations
                    if not (a['s1000d_uri'] == s1000d_concept['uri']
                           and a['bike_uri'] == candidate['uri'])
                ]

                # Add new annotation
                annotation = {
                    's1000d_uri': s1000d_concept['uri'],
                    's1000d_label': s1000d_concept['label'],
                    'bike_uri': candidate['uri'],
                    'bike_label': candidate['label'],
                    'similarity_score': candidate['similarity_score'],
                    'is_match_manual': decisions[candidate['uri']]
                }
                st.session_state.annotations.append(annotation)

            # Save to file
            save_gold_standard(st.session_state.annotations)

            # Move to next
            st.session_state.current_idx += 1
            st.rerun()


if __name__ == '__main__':
    main()
