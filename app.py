"""
Streamlit App for Ontology Matching with KROMA-like approach.
Combines BERT vector search with Claude LLM verification.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import streamlit as st
import pandas as pd

from data_loader import load_s1000d_data
from bert_matcher import VectorIndex
from llm_reasoner import LLMReasoner


# Page configuration
st.set_page_config(
    page_title="S1000D Ontology Matcher",
    page_icon="🔍",
    layout="wide"
)


@st.cache_resource
def load_and_build_index():
    """
    Load S1000D data and build vector index.
    Cached to avoid reloading on every interaction.
    """
    with st.spinner("Loading S1000D data..."):
        df = load_s1000d_data('bike')

    if df.empty:
        st.error("No data loaded. Please check the 'bike' folder.")
        return None, None

    with st.spinner(f"Building vector index for {len(df)} concepts..."):
        index = VectorIndex(df)
        index.build_index()

    return df, index


@st.cache_resource
def initialize_llm_reasoner():
    """
    Initialize the LLM Reasoner.
    Cached to reuse the API client.
    """
    try:
        reasoner = LLMReasoner()
        return reasoner
    except ValueError as e:
        st.error(f"Failed to initialize Claude API: {e}")
        st.info("Please create a .env file with ANTHROPIC_API_KEY=your_key")
        return None


def save_feedback(
    concept_a: Dict[str, Any],
    concept_b: Dict[str, Any],
    user_decision: str,
    llm_result: Dict[str, Any] = None
):
    """
    Save user feedback to feedback.json for active learning.

    Args:
        concept_a: Source concept
        concept_b: Candidate concept
        user_decision: 'match' or 'reject'
        llm_result: Optional LLM verification result
    """
    feedback_file = Path("feedback.json")

    # Load existing feedback
    if feedback_file.exists():
        with open(feedback_file, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)
    else:
        feedback_data = []

    # Create feedback entry
    entry = {
        'timestamp': datetime.now().isoformat(),
        'concept_a_id': concept_a.get('id', ''),
        'concept_a_label': concept_a.get('label', ''),
        'concept_b_id': concept_b.get('id', ''),
        'concept_b_label': concept_b.get('label', ''),
        'user_decision': user_decision,
    }

    # Add LLM result if available
    if llm_result:
        entry['llm_prediction'] = llm_result.get('is_match', False)
        entry['llm_confidence'] = llm_result.get('confidence', 0.0)
        entry['llm_reason'] = llm_result.get('reason', '')

    feedback_data.append(entry)

    # Save feedback
    with open(feedback_file, 'w', encoding='utf-8') as f:
        json.dump(feedback_data, f, indent=2, ensure_ascii=False)


def display_concept_card(concept: Dict[str, Any], title: str):
    """
    Display a concept in a card format.
    """
    st.markdown(f"### {title}")
    st.markdown(f"**Label:** {concept.get('label', 'N/A')}")
    st.markdown(f"**ID:** `{concept.get('id', 'N/A')}`")

    context = concept.get('context', 'N/A')
    if len(context) > 300:
        context = context[:300] + "..."
    st.markdown(f"**Context:** {context}")


def main():
    """
    Main Streamlit application.
    """
    st.title("🔍 S1000D Ontology Matcher")
    st.markdown("**KROMA-inspired approach:** BERT Vector Search + Claude LLM Verification")

    st.divider()

    # Load data and build index
    df, index = load_and_build_index()

    if df is None or index is None:
        st.stop()

    # Initialize LLM reasoner
    reasoner = initialize_llm_reasoner()

    # Sidebar: Stats and info
    with st.sidebar:
        st.header("📊 Statistics")
        st.metric("Total Concepts", len(df))

        feedback_file = Path("feedback.json")
        if feedback_file.exists():
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
            st.metric("Feedback Entries", len(feedback_data))

        st.divider()

        st.header("ℹ️ About")
        st.markdown("""
        This tool matches S1000D concepts using:
        1. **BERT embeddings** for semantic similarity
        2. **Claude LLM** for intelligent verification
        3. **User feedback** for active learning
        """)

        if reasoner is None:
            st.warning("⚠️ Claude API not configured")

    # Main layout: Two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📋 Select Source Concept")

        # Create dropdown options
        concept_options = [
            f"{row['label']} ({row['id']})"
            for _, row in df.iterrows()
        ]

        selected_option = st.selectbox(
            "Choose a concept to find matches:",
            options=concept_options,
            key="source_concept"
        )

        # Get selected concept
        selected_idx = concept_options.index(selected_option)
        source_concept = df.iloc[selected_idx].to_dict()

        # Display source concept
        st.divider()
        display_concept_card(source_concept, "Source Concept")

    with col2:
        st.header("🎯 Top Candidates")

        # Find candidates using BERT
        query_text = f"{source_concept.get('label', '')}. {source_concept.get('context', '')[:200]}"
        candidates = index.find_candidates(query_text, top_k=6)

        # Remove the source concept itself from candidates
        candidates = [
            c for c in candidates
            if c['id'] != source_concept.get('id', '')
        ][:5]

        if not candidates:
            st.info("No candidates found.")
            st.stop()

        # Display candidates
        for i, candidate in enumerate(candidates):
            with st.expander(
                f"**Rank {candidate['rank']}** | {candidate['label']} | Score: {candidate['score']:.3f}",
                expanded=(i == 0)
            ):
                st.markdown(f"**ID:** `{candidate['id']}`")
                st.markdown(f"**Label:** {candidate['label']}")
                st.markdown(f"**Context:** {candidate['context']}")
                st.markdown(f"**Similarity Score:** {candidate['score']:.4f}")

                st.divider()

                # Verify button
                verify_key = f"verify_{i}"
                if st.button(f"🤖 Verify with Claude", key=verify_key):
                    if reasoner is None:
                        st.error("Claude API not configured. Add ANTHROPIC_API_KEY to .env")
                    else:
                        with st.spinner("Asking Claude..."):
                            # Store candidate in session state for feedback buttons
                            st.session_state[f'candidate_{i}'] = candidate

                            result = reasoner.verify_match_with_claude(
                                source_concept,
                                candidate
                            )

                            st.session_state[f'llm_result_{i}'] = result

                # Display LLM result if available
                if f'llm_result_{i}' in st.session_state:
                    result = st.session_state[f'llm_result_{i}']

                    # Color-coded result
                    if result['is_match']:
                        st.success(f"✅ **MATCH** (Confidence: {result['confidence']:.2f})")
                    else:
                        st.error(f"❌ **NO MATCH** (Confidence: {result['confidence']:.2f})")

                    st.markdown(f"**Reasoning:** {result['reason']}")

                    st.divider()

                    # Feedback buttons
                    col_confirm, col_reject = st.columns(2)

                    with col_confirm:
                        if st.button("✅ Confirm Match", key=f"confirm_{i}"):
                            save_feedback(
                                source_concept,
                                candidate,
                                'match',
                                result
                            )
                            st.success("Feedback saved: Match confirmed")

                    with col_reject:
                        if st.button("❌ Reject", key=f"reject_{i}"):
                            save_feedback(
                                source_concept,
                                candidate,
                                'reject',
                                result
                            )
                            st.success("Feedback saved: Match rejected")


if __name__ == '__main__':
    main()
