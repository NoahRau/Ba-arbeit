"""
Gold Standard Creator for Ontology Matching Evaluation.
Manual annotation tool to create benchmark dataset.
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_all_concepts
from src.matchers.deeponto_matcher import DeepOntoMatcher


# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
GOLD_STANDARD_FILE = PROJECT_ROOT / 'data' / 'results' / 'gold_standard.json'
SAMPLE_SIZE = 50
TOP_K = 5


@st.cache_resource
def load_data_and_matcher():
    """
    Load all data and initialize matcher.
    Cached to avoid reloading.
    """
    # Load concepts
    df = load_all_concepts(include_ontologies=True)

    if df.empty:
        st.error("No data loaded!")
        st.stop()

    # Split by source
    s1000d_df = df[df['source'] == 's1000d'].reset_index(drop=True)
    ontology_df = df[df['source'] == 'bike_ontology'].reset_index(drop=True)

    # Initialize DeepOnto matcher (uses cached embeddings)
    with st.spinner("Loading BERT matcher... This may take a few minutes on first run..."):
        matcher = DeepOntoMatcher(s1000d_df, ontology_df)

    return s1000d_df, ontology_df, matcher


def get_top_candidates(
    source_concept: Dict[str, Any],
    matcher: DeepOntoMatcher,
    ontology_df: pd.DataFrame,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Get top K candidates for a source concept.

    Args:
        source_concept: Source concept dictionary
        matcher: DeepOnto matcher
        ontology_df: Target ontology DataFrame
        top_k: Number of candidates to return

    Returns:
        List of candidate dictionaries with similarity scores
    """
    # Get candidates from matcher
    candidates_raw = matcher.find_candidates(source_concept, top_k=top_k)

    # Build candidate list
    candidates = []
    for target_uri, score in candidates_raw:
        target_row = ontology_df[ontology_df['uri'] == target_uri]
        if not target_row.empty:
            target_row = target_row.iloc[0]
            candidates.append({
                'uri': target_uri,
                'label': target_row['label'],
                'context': target_row.get('context_text', ''),
                'score': float(score)
            })

    return candidates


def save_gold_standard(annotations: List[Dict[str, Any]]):
    """
    Save gold standard annotations to JSON.

    Args:
        annotations: List of annotation dictionaries
    """
    GOLD_STANDARD_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(GOLD_STANDARD_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'annotations': annotations,
            'total': len(annotations),
            'sample_size': SAMPLE_SIZE,
            'top_k': TOP_K
        }, f, indent=2, ensure_ascii=False)

    st.success(f"✅ Saved {len(annotations)} annotations to {GOLD_STANDARD_FILE}")


def load_gold_standard() -> List[Dict[str, Any]]:
    """
    Load existing gold standard annotations.

    Returns:
        List of annotation dictionaries
    """
    if GOLD_STANDARD_FILE.exists():
        with open(GOLD_STANDARD_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('annotations', [])
    return []


def main():
    """
    Main Streamlit application for gold standard creation.
    """
    st.set_page_config(
        page_title="Gold Standard Creator",
        page_icon="⭐",
        layout="wide"
    )

    st.title("⭐ Gold Standard Creator")
    st.markdown("Manual annotation tool for ontology matching evaluation")

    # Load data and matcher
    s1000d_df, ontology_df, matcher = load_data_and_matcher()

    # Display stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("S1000D Concepts", len(s1000d_df))
    with col2:
        st.metric("Ontology Concepts", len(ontology_df))
    with col3:
        existing_annotations = load_gold_standard()
        st.metric("Annotations", len(existing_annotations))

    st.divider()

    # Sample selection
    st.subheader("1. Sample Selection")

    if st.button("🎲 Generate Random Sample"):
        sample_indices = random.sample(range(len(s1000d_df)), min(SAMPLE_SIZE, len(s1000d_df)))
        st.session_state['sample_indices'] = sample_indices
        st.session_state['current_idx'] = 0
        st.session_state['annotations'] = []
        st.rerun()

    if 'sample_indices' not in st.session_state:
        st.info("Click 'Generate Random Sample' to start annotation")
        return

    # Get current sample
    sample_indices = st.session_state['sample_indices']
    current_idx = st.session_state.get('current_idx', 0)

    if current_idx >= len(sample_indices):
        st.success("🎉 All samples annotated!")
        if st.button("💾 Save Gold Standard"):
            save_gold_standard(st.session_state.get('annotations', []))
        return

    # Current concept
    concept_idx = sample_indices[current_idx]
    source_concept = s1000d_df.iloc[concept_idx].to_dict()

    # Progress
    st.progress((current_idx + 1) / len(sample_indices))
    st.markdown(f"**Sample {current_idx + 1} / {len(sample_indices)}**")

    st.divider()

    # Display source concept
    st.subheader("2. Source Concept (S1000D)")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**Label:** {source_concept['label']}")
        st.markdown(f"**URI:** `{source_concept['uri']}`")
    with col2:
        st.markdown(f"**Context:**")
        st.text_area("", source_concept.get('context_text', 'N/A'), height=100, disabled=True, key="src_context")

    st.divider()

    # Get candidates
    st.subheader("3. Top Candidates")

    with st.spinner("Finding top candidates..."):
        candidates = get_top_candidates(source_concept, matcher, ontology_df, TOP_K)

    # Display candidates
    for i, cand in enumerate(candidates):
        with st.expander(f"**Candidate {i+1}** - {cand['label']} (Score: {cand['score']:.3f})"):
            st.markdown(f"**URI:** `{cand['uri']}`")
            st.markdown(f"**Context:** {cand['context'][:300]}...")

    st.divider()

    # Annotation
    st.subheader("4. Annotation")

    col1, col2 = st.columns(2)

    with col1:
        match_exists = st.radio(
            "Does a correct match exist in the candidates?",
            ["Yes", "No (NULL)"],
            key=f"match_exists_{current_idx}"
        )

    with col2:
        if match_exists == "Yes":
            candidate_labels = [f"{i+1}. {c['label']} ({c['score']:.3f})" for i, c in enumerate(candidates)]
            selected = st.selectbox(
                "Select the correct match:",
                candidate_labels,
                key=f"selected_{current_idx}"
            )
            selected_idx = int(selected.split('.')[0]) - 1
            correct_uri = candidates[selected_idx]['uri']
        else:
            correct_uri = None

    # Notes
    notes = st.text_area("Notes (optional):", key=f"notes_{current_idx}")

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if current_idx > 0 and st.button("⬅️ Previous"):
            st.session_state['current_idx'] -= 1
            st.rerun()

    with col2:
        if st.button("💾 Save & Next ➡️", type="primary"):
            # Save annotation
            annotation = {
                'source_uri': source_concept['uri'],
                'source_label': source_concept['label'],
                'correct_match_uri': correct_uri,
                'is_match': correct_uri is not None,
                'candidates': [{'uri': c['uri'], 'label': c['label'], 'score': c['score']} for c in candidates],
                'notes': notes
            }

            if 'annotations' not in st.session_state:
                st.session_state['annotations'] = []

            # Update or append
            if current_idx < len(st.session_state['annotations']):
                st.session_state['annotations'][current_idx] = annotation
            else:
                st.session_state['annotations'].append(annotation)

            # Move to next
            st.session_state['current_idx'] += 1
            st.rerun()

    with col3:
        if st.button("⏭️ Skip"):
            st.session_state['current_idx'] += 1
            st.rerun()


if __name__ == '__main__':
    main()
