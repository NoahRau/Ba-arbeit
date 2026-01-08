"""
LLM-based Reasoning for Ontology Matching.
Uses Claude 3.5 Sonnet to verify if two concepts match.
"""

import json
import os
from typing import Dict, Any
from anthropic import Anthropic
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


class LLMReasoner:
    """
    Uses Claude API to reason about concept matches.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the LLM Reasoner.

        Args:
            api_key: Anthropic API key. If None, loads from ANTHROPIC_API_KEY env var
        """
        if api_key is None:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError(
                    "No API key provided. Set ANTHROPIC_API_KEY in .env file "
                    "or pass api_key parameter."
                )

        self.client = Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-5-20250929"

    def verify_match_with_claude(
        self,
        concept_a: Dict[str, Any],
        concept_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify if two concepts represent the same component or procedure.

        Args:
            concept_a: Dictionary with keys 'id', 'label', 'context'
            concept_b: Dictionary with keys 'id', 'label', 'context'

        Returns:
            Dictionary with:
                - is_match (bool): Whether concepts match
                - reason (str): Explanation for the decision
                - confidence (float): Confidence score 0-1
        """
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(concept_a, concept_b)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )

            # Extract response text
            response_text = response.content[0].text.strip()

            # Parse JSON response
            result = self._parse_response(response_text)

            return result

        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return {
                'is_match': False,
                'reason': f'Error during API call: {str(e)}',
                'confidence': 0.0
            }

    def _create_system_prompt(self) -> str:
        """
        Create the system prompt for Claude.
        Optimized for high precision and F1-score maximization.
        """
        return """You are an expert in S1000D technical documentation and ontology matching, participating in a benchmark evaluation process.

**YOUR MISSION**: Maximize the F1-Score by providing highly accurate match decisions. This requires balancing precision and recall, but PRIORITIZE PRECISION to avoid false positives.

**CONTEXT**: S1000D is an international specification for technical publications using a common source database. Data modules are identified by DMC codes and contain:
- Technical names (component/procedure names)
- Information names (specific aspect, e.g., "Description", "Maintenance")
- Context (detailed technical content)

**CRITICAL EVALUATION CRITERIA**:

You must be EXTREMELY CRITICAL when evaluating matches. Apply these strict rules:

1. **Functional Equivalence Required**:
   - "Similar" or "related" is NOT sufficient for a match
   - Only declare a match if the concepts represent THE SAME physical component, procedure, or functional entity
   - Example: "Brake pad" and "Brake system" are NOT matches (one is a part, the other is a system)

2. **Semantic Precision**:
   - S1000D data is technically precise and domain-specific
   - Generic similarities (e.g., both mention "bicycle") are insufficient
   - The technical function and scope must align exactly

3. **Hierarchical Relationships**:
   - Parent-child relationships are NOT matches (e.g., "Wheel" vs "Hub")
   - Sibling components are NOT matches (e.g., "Front brake" vs "Rear brake")
   - Only accept matches at the SAME hierarchical level representing the SAME entity

4. **Information Type Alignment**:
   - Both concepts should describe the same aspect of the same thing
   - A maintenance procedure for X is NOT the same as a description of X

5. **Context Verification**:
   - The detailed technical content must confirm they refer to the exact same thing
   - Look for technical specifications, part numbers, or functional descriptions
   - If contexts describe different purposes or scopes, it's NOT a match

**ERROR AVOIDANCE STRATEGY**:
- **Strongly avoid False Positives** (incorrectly declaring a match)
- A false positive is worse than a false negative in this context
- When in doubt, err on the side of caution: reject the match
- Only accept matches when you have HIGH CONFIDENCE (≥0.85) in functional equivalence

**CONFIDENCE SCORING** (be conservative):
- 0.95-1.0: Absolute certainty - identical concepts, possibly with different wording
- 0.85-0.95: Very high confidence - same component/procedure, minor differences
- 0.70-0.85: Moderate confidence - likely match but some ambiguity exists
- 0.50-0.70: Low confidence - related but not functionally equivalent
- 0.00-0.50: No match - different concepts or insufficient evidence

**OUTPUT FORMAT** - Respond with ONLY valid JSON:
{
  "is_match": true/false,
  "reason": "Technical explanation citing specific evidence for your decision",
  "confidence": 0.85
}

**IMPORTANT**:
- Set "is_match": true ONLY when confidence is ≥0.85 AND functional equivalence is established
- For confidence <0.85, set "is_match": false (better safe than sorry)
- Your reasoning should cite specific technical details from both concepts"""

    def _create_user_prompt(
        self,
        concept_a: Dict[str, Any],
        concept_b: Dict[str, Any]
    ) -> str:
        """
        Create the user prompt with both concepts.
        """
        # Extract fields with defaults
        a_id = concept_a.get('id', concept_a.get('uri', 'N/A'))
        a_label = concept_a.get('label', 'N/A')
        a_context = concept_a.get('context_text', concept_a.get('context', 'N/A'))

        b_id = concept_b.get('id', concept_b.get('uri', 'N/A'))
        b_label = concept_b.get('label', 'N/A')
        b_context = concept_b.get('context_text', concept_b.get('context', 'N/A'))

        # Truncate context to avoid token limits
        max_context_len = 500
        if len(a_context) > max_context_len:
            a_context = a_context[:max_context_len] + "..."
        if len(b_context) > max_context_len:
            b_context = b_context[:max_context_len] + "..."

        return f"""Compare these two S1000D concepts and determine if they represent the same component or procedure:

**Concept A:**
- ID: {a_id}
- Label: {a_label}
- Context: {a_context}

**Concept B:**
- ID: {b_id}
- Label: {b_label}
- Context: {b_context}

Analyze if these concepts match and respond with JSON only."""

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the JSON response from Claude.
        """
        try:
            # Try to extract JSON if there's extra text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)

            # Validate required fields
            if 'is_match' not in result:
                raise ValueError("Missing 'is_match' field")
            if 'reason' not in result:
                raise ValueError("Missing 'reason' field")
            if 'confidence' not in result:
                raise ValueError("Missing 'confidence' field")

            # Ensure types are correct
            result['is_match'] = bool(result['is_match'])
            result['reason'] = str(result['reason'])
            result['confidence'] = float(result['confidence'])

            return result

        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response text: {response_text}")
            return {
                'is_match': False,
                'reason': f'Failed to parse response: {str(e)}',
                'confidence': 0.0
            }


def main():
    """
    Test the LLM Reasoner with example concepts.
    """
    from data_loader import load_s1000d_data

    print("=" * 70)
    print("LLM Reasoner Test - Claude API Integration")
    print("=" * 70)

    # Load data
    print("\n1. Loading S1000D data...")
    df = load_s1000d_data('bike')

    if df.empty or len(df) < 2:
        print("Not enough data loaded. Exiting.")
        return

    print(f"   Loaded {len(df)} concepts")

    # Initialize reasoner
    print("\n2. Initializing Claude API...")
    try:
        reasoner = LLMReasoner()
        print("   API initialized successfully")
    except ValueError as e:
        print(f"   Error: {e}")
        print("\n   Please create a .env file with:")
        print("   ANTHROPIC_API_KEY=your_api_key_here")
        return

    # Test with different concept pairs
    print("\n3. Testing concept matching...")
    print("=" * 70)

    # Test 1: Similar concepts (should match)
    print("\n--- Test 1: Similar Concepts (Brake-related) ---")
    brake_concepts = df[df['label'].str.contains('Brake', case=False, na=False)]
    if len(brake_concepts) >= 2:
        concept_a = brake_concepts.iloc[0].to_dict()
        concept_b = brake_concepts.iloc[1].to_dict()

        print(f"\nConcept A: {concept_a['label']}")
        print(f"Concept B: {concept_b['label']}")

        result = reasoner.verify_match_with_claude(concept_a, concept_b)

        print(f"\nResult:")
        print(f"  Match: {result['is_match']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Reason: {result['reason']}")

    # Test 2: Different concepts (should not match)
    print("\n" + "=" * 70)
    print("--- Test 2: Different Concepts (Brake vs Frame) ---")
    brake_concept = df[df['label'].str.contains('Brake', case=False, na=False)]
    frame_concept = df[df['label'].str.contains('Frame', case=False, na=False)]

    if not brake_concept.empty and not frame_concept.empty:
        concept_a = brake_concept.iloc[0].to_dict()
        concept_b = frame_concept.iloc[0].to_dict()

        print(f"\nConcept A: {concept_a['label']}")
        print(f"Concept B: {concept_b['label']}")

        result = reasoner.verify_match_with_claude(concept_a, concept_b)

        print(f"\nResult:")
        print(f"  Match: {result['is_match']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Reason: {result['reason']}")

    print("\n" + "=" * 70)
    print("Test completed successfully!")


if __name__ == '__main__':
    main()
