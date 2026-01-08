"""
LLM Listwise Reranker using Claude.

Replaces pairwise verification with listwise ranking:
- Receives top-5 candidates from aggregation
- Performs comparative analysis
- Selects best match or NULL
- Returns reasoning for decision

Based on listwise learning-to-rank principles.
"""

import json
import os
from typing import List, Dict, Any, Optional
from anthropic import Anthropic
from dotenv import load_dotenv


load_dotenv()


class LLMReranker:
    """
    Listwise reranker using Claude Sonnet 4.5.

    Analyzes multiple candidates simultaneously and selects the best match.
    More effective than pairwise comparison.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize LLM Reranker.

        Args:
            api_key: Anthropic API key (default: from .env)
        """
        if api_key is None:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError(
                    "No API key provided. Set ANTHROPIC_API_KEY in .env"
                )

        self.client = Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-5-20250929"

    def rerank_candidates(
        self,
        source_concept: Dict[str, Any],
        candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Rerank candidates and select best match.

        Args:
            source_concept: Source concept dict with 'label', 'context_text', 'uri'
            candidates: List of candidate dicts with:
                - 'uri': Target URI
                - 'label': Target label
                - 'context_text': Target context
                - 'aggregated_score': Score from aggregation
                - 'matcher_details': Details from matchers

        Returns:
            Dictionary with:
            - 'selected_uri': Best match URI or None
            - 'selected_index': Index in candidates list (0-based) or None
            - 'reason': Reasoning from Claude
            - 'confidence': Confidence in decision

            Returns None if no good match found.
        """
        if not candidates:
            return None

        # Limit to top 5 candidates for Claude
        top_candidates = candidates[:5]

        # Create prompt
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(source_concept, top_candidates)

        try:
            # Call Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": user_prompt
                }]
            )

            # Parse response
            response_text = response.content[0].text.strip()
            result = self._parse_response(response_text)

            # Add selected URI if index is valid
            if result['selected_index'] is not None:
                idx = result['selected_index']
                if 0 <= idx < len(top_candidates):
                    result['selected_uri'] = top_candidates[idx]['uri']
                else:
                    result['selected_uri'] = None
                    result['selected_index'] = None
            else:
                result['selected_uri'] = None

            return result

        except Exception as e:
            print(f"Error calling Claude: {e}")
            return {
                'selected_uri': None,
                'selected_index': None,
                'reason': f'API Error: {str(e)}',
                'confidence': 0.0
            }

    def _create_system_prompt(self) -> str:
        """Create system prompt for listwise reranking."""
        return """Du bist ein Ontology-Alignment-Experte mit Fokus auf S1000D technische Dokumentation.

**DEINE AUFGABE:**
Du bekommst ein Quell-Konzept aus S1000D und eine Liste von 5 möglichen Kandidaten aus der Ziel-Ontologie.
Identifiziere den BESTEN Match oder antworte mit NULL, wenn keiner passt.

**KRITISCHE REGELN:**

1. **Funktionale Äquivalenz ist erforderlich:**
   - Nur Konzepte, die EXAKT DAS GLEICHE repräsentieren, sind Matches
   - "Ähnlich" oder "verwandt" ist NICHT ausreichend
   - Beispiel: "Brake pad" ≠ "Brake system" (Teil vs. System)

2. **Hierarchie beachten:**
   - Parent-Child sind KEINE Matches (z.B. "Wheel" ≠ "Hub")
   - Geschwister sind KEINE Matches (z.B. "Front Brake" ≠ "Rear Brake")
   - Nur gleiche hierarchische Ebene mit gleicher Entität

3. **Hierarchische Pfade nutzen:**
   - Die Hierarchiepfade geben wichtigen Kontext
   - Vergleiche die Pfade zur Disambiguierung
   - Beispiel: "BikeComponent > Wheel" vs "BikeComponent > Wheel > Hub"

4. **Strenge bei Unsicherheit:**
   - Im Zweifel → NULL wählen
   - False Positives sind schlimmer als False Negatives
   - Nur bei HOHER Sicherheit (≥85%) einen Match wählen

5. **Vergleichende Analyse:**
   - Analysiere ALLE 5 Kandidaten
   - Erkläre warum der gewählte besser ist als die anderen
   - Oder warum KEINER passt

**DENKPROZESS:**
1. Verstehe das Quell-Konzept (Hierarchie + Label + Kontext)
2. Für jeden Kandidaten:
   - Prüfe funktionale Äquivalenz
   - Prüfe hierarchische Kompatibilität
   - Prüfe technische Details
3. Vergleiche Kandidaten untereinander
4. Wähle den BESTEN oder NULL

**OUTPUT FORMAT:**
```json
{
  "selected_index": 0-4 oder null,
  "reason": "Schritt-für-Schritt Analyse mit spezifischen Details",
  "confidence": 0.0-1.0
}
```

**WICHTIG:** Antworte NUR mit gültigem JSON, kein zusätzlicher Text!"""

    def _create_user_prompt(
        self,
        source: Dict[str, Any],
        candidates: List[Dict[str, Any]]
    ) -> str:
        """
        Create user prompt with source and candidates.

        Args:
            source: Source concept
            candidates: List of candidate concepts

        Returns:
            Formatted prompt string
        """
        # Extract source info
        src_label = source.get('label', 'N/A')
        src_context = source.get('context_text', '')

        # Truncate context to avoid token limits
        src_context_short = src_context[:500] if len(src_context) > 500 else src_context

        prompt = f"""Vergleiche dieses Quell-Konzept mit 5 Kandidaten:

**QUELLE (SOURCE):**
Label: {src_label}
Kontext: {src_context_short}

**KANDIDATEN:**

"""

        # Add each candidate
        for i, cand in enumerate(candidates):
            cand_label = cand.get('label', 'N/A')
            cand_context = cand.get('context_text', '')
            cand_score = cand.get('aggregated_score', 0.0)

            # Truncate context
            cand_context_short = cand_context[:500] if len(cand_context) > 500 else cand_context

            prompt += f"""KANDIDAT {i} (Aggregierter Score: {cand_score:.3f}):
Label: {cand_label}
Kontext: {cand_context_short}

"""

        prompt += """**AUFGABE:**
1. Analysiere die Hierarchiepfade
2. Prüfe funktionale Äquivalenz für jeden Kandidaten
3. Wähle den BESTEN Kandidaten (0-4) oder NULL wenn keiner passt
4. Antworte mit JSON: {"selected_index": int|null, "reason": "str", "confidence": float}

Deine Antwort (NUR JSON):"""

        return prompt

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Claude's JSON response.

        Args:
            response_text: Raw response from Claude

        Returns:
            Parsed dictionary
        """
        try:
            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)

            # Validate and normalize
            selected_index = result.get('selected_index')
            if selected_index is not None:
                selected_index = int(selected_index)

            reason = result.get('reason', 'No reason provided')
            confidence = float(result.get('confidence', 0.0))

            return {
                'selected_index': selected_index,
                'reason': reason,
                'confidence': confidence
            }

        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response: {response_text}")
            return {
                'selected_index': None,
                'reason': f'Parse error: {str(e)}',
                'confidence': 0.0
            }


def main():
    """
    Test LLM Reranker with mock data.
    """
    print("=" * 70)
    print("LLM LISTWISE RERANKER TEST")
    print("=" * 70)

    # Initialize reranker
    print("\n[1/2] Initializing reranker...")
    try:
        reranker = LLMReranker()
        print("  ✓ Claude API initialized")
    except ValueError as e:
        print(f"  Error: {e}")
        return

    # Mock source concept
    source = {
        'label': 'Wheel - Description of how it is made',
        'context_text': 'S1000DBIKE > AAA > DA0 > 0 > 0 > 041 | Technical description of bicycle wheel assembly and components.',
        'uri': 'http://my-company.com/s1000d/S1000DBIKE-AAA-DA0-0-0-041'
    }

    # Mock candidates (from aggregation)
    candidates = [
        {
            'uri': 'http://purl.org/ontology/bikeo#Wheel',
            'label': 'Wheel',
            'context_text': 'BikeComponent > Wheel | The wheel assembly',
            'aggregated_score': 0.85
        },
        {
            'uri': 'http://purl.org/ontology/bikeo#Hub',
            'label': 'Hub',
            'context_text': 'BikeComponent > Wheel > Hub | The wheel hub',
            'aggregated_score': 0.72
        },
        {
            'uri': 'http://purl.org/ontology/bikeo#Tire',
            'label': 'Tire',
            'context_text': 'BikeComponent > Wheel > Tire | The tire',
            'aggregated_score': 0.68
        },
        {
            'uri': 'http://purl.org/ontology/bikeo#Frame',
            'label': 'Frame',
            'context_text': 'BikeComponent > Frame | The bicycle frame',
            'aggregated_score': 0.55
        },
        {
            'uri': 'http://purl.org/ontology/bikeo#Brake',
            'label': 'Brake',
            'context_text': 'BikeComponent > Brake | The brake system',
            'aggregated_score': 0.45
        }
    ]

    # Test reranking
    print("\n[2/2] Testing listwise reranking...")
    print("\nSource:", source['label'])
    print("\nCandidates:")
    for i, cand in enumerate(candidates):
        print(f"  {i}. {cand['label']} (score: {cand['aggregated_score']:.2f})")

    print("\nCalling Claude for reranking...")
    result = reranker.rerank_candidates(source, candidates)

    print("\n" + "=" * 70)
    print("RERANKING RESULT")
    print("=" * 70)

    if result:
        print(f"\nSelected Index: {result['selected_index']}")
        if result['selected_uri']:
            selected_cand = candidates[result['selected_index']]
            print(f"Selected URI: {result['selected_uri']}")
            print(f"Selected Label: {selected_cand['label']}")
        else:
            print("Selected: NULL (no good match)")

        print(f"\nConfidence: {result['confidence']:.2f}")
        print(f"\nReasoning:\n{result['reason']}")
    else:
        print("No result returned")

    print("\n" + "=" * 70)
    print("Test completed!")


if __name__ == '__main__':
    main()
