"""
AML (AgreementMakerLight) Matcher Wrapper.

Integrates AgreementMakerLight via subprocess for string-based matching.
AML is a robust baseline matcher specializing in:
- Exact string matching
- Token-based matching
- Edit distance
- Substring matching

Reference:
D. Faria et al., "The AgreementMakerLight Ontology Matching System",
ODBASE 2013.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pandas as pd
from rdflib import Graph, Namespace, RDF, Literal, URIRef
from xml.etree import ElementTree as ET

try:
    from .base_matcher import BaseMatcher
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from matchers.base_matcher import BaseMatcher


# Alignment API namespaces
ALIGN = Namespace("http://knowledgeweb.semanticweb.org/heterogeneity/alignment#")
RDF_NS = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")


class AMLMatcher(BaseMatcher):
    """
    Wrapper for AgreementMakerLight Java tool.

    Uses subprocess to run AML on temporary OWL files.
    Provides string-based matching baseline.
    """

    def __init__(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        aml_jar_path: str = "./AML_v3.2/AgreementMakerLight.jar"
    ):
        super().__init__(source_df, target_df)

        self.aml_jar = Path(aml_jar_path)
        if not self.aml_jar.exists():
            raise FileNotFoundError(
                f"AML jar not found: {self.aml_jar}\n"
                f"Expected at: {self.aml_jar.absolute()}"
            )

        # Run AML once to generate alignment
        print("  Running AML (this may take a moment)...")
        self.alignments = self._run_aml_matching()
        print(f"  ✓ AML found {len(self.alignments)} potential matches")

    def _create_simple_owl(self, df: pd.DataFrame, output_path: str):
        """
        Create a simple OWL file from DataFrame for AML.

        AML expects OWL/RDF format, so we create minimal ontologies.

        Args:
            df: DataFrame with concepts
            output_path: Path to save OWL file
        """
        g = Graph()

        # Define namespace
        NS = Namespace("http://temp.ontology.com/")
        g.bind("ns", NS)
        g.bind("rdf", RDF)
        g.bind("rdfs", Namespace("http://www.w3.org/2000/01/rdf-schema#"))

        OWL = Namespace("http://www.w3.org/2002/07/owl#")
        RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")

        # Add Ontology declaration
        ontology_uri = URIRef(str(NS) + "Ontology")
        g.add((ontology_uri, RDF.type, OWL.Ontology))

        # Add each concept as OWL Class
        for idx, row in df.iterrows():
            # Use original URI or create one
            uri_str = row.get('uri', '')
            if not uri_str:
                uri_str = str(NS) + f"Concept_{idx}"

            concept_uri = URIRef(uri_str)

            # Add as OWL Class
            g.add((concept_uri, RDF.type, OWL.Class))

            # Add label
            label = row.get('label', '')
            if label:
                g.add((concept_uri, RDFS.label, Literal(label)))

            # Add comment with context (limited to avoid huge files)
            context = row.get('context_text', '')
            if context:
                # Limit context to 200 chars for AML (it's primarily string-based)
                context_short = context[:200]
                g.add((concept_uri, RDFS.comment, Literal(context_short)))

        # Serialize to file
        g.serialize(destination=output_path, format='xml')

    def _run_aml_matching(self) -> List[Dict[str, Any]]:
        """
        Run AML matching on source and target ontologies.

        Returns:
            List of alignment dictionaries with:
            - source_uri: Source concept URI
            - target_uri: Target concept URI
            - confidence: Alignment confidence (0-1)
        """
        # Create temporary OWL files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.owl', delete=False) as src_file:
            src_path = src_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.owl', delete=False) as tgt_file:
            tgt_path = tgt_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.rdf', delete=False) as align_file:
            align_path = align_file.name

        try:
            # Create OWL files
            print("    Creating temporary OWL files...")
            self._create_simple_owl(self.source_df, src_path)
            self._create_simple_owl(self.target_df, tgt_path)

            # Run AML
            print("    Executing AML jar...")
            cmd = [
                "java",
                "-jar", str(self.aml_jar),
                "-s", src_path,
                "-t", tgt_path,
                "-o", align_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes max
            )

            if result.returncode != 0:
                print(f"    Warning: AML returned code {result.returncode}")
                print(f"    STDERR: {result.stderr[:500]}")
                return []

            # Parse alignment output
            print("    Parsing AML alignment output...")
            alignments = self._parse_aml_alignment(align_path)

            return alignments

        except subprocess.TimeoutExpired:
            print("    Warning: AML timed out after 5 minutes")
            return []
        except Exception as e:
            print(f"    Warning: AML execution failed: {e}")
            return []
        finally:
            # Cleanup temporary files
            for path in [src_path, tgt_path, align_path]:
                try:
                    Path(path).unlink()
                except:
                    pass

    def _parse_aml_alignment(self, alignment_path: str) -> List[Dict[str, Any]]:
        """
        Parse AML's alignment RDF output.

        AML uses the Alignment API format:
        <Alignment>
          <map>
            <Cell>
              <entity1 rdf:resource="..."/>
              <entity2 rdf:resource="..."/>
              <measure rdf:datatype="xsd:float">0.95</measure>
              <relation>=</relation>
            </Cell>
          </map>
        </Alignment>

        Args:
            alignment_path: Path to alignment RDF file

        Returns:
            List of alignment dictionaries
        """
        alignments = []

        try:
            tree = ET.parse(alignment_path)
            root = tree.getroot()

            # Find all Cell elements
            # Handle namespace prefixes
            for cell in root.iter():
                if cell.tag.endswith('Cell'):
                    entity1 = None
                    entity2 = None
                    measure = 1.0

                    for child in cell:
                        if child.tag.endswith('entity1'):
                            entity1 = child.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                        elif child.tag.endswith('entity2'):
                            entity2 = child.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                        elif child.tag.endswith('measure'):
                            try:
                                measure = float(child.text)
                            except:
                                measure = 1.0

                    if entity1 and entity2:
                        alignments.append({
                            'source_uri': entity1,
                            'target_uri': entity2,
                            'confidence': measure
                        })

        except Exception as e:
            print(f"    Warning: Failed to parse alignment: {e}")

        return alignments

    def find_candidates(
        self,
        source_concept: Dict[str, Any],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find candidates for source concept from pre-computed AML alignments.

        Args:
            source_concept: Source concept dictionary
            top_k: Number of candidates to return

        Returns:
            List of (target_uri, score) tuples
        """
        source_uri = source_concept['uri']

        # Find all alignments for this source URI
        candidates = []
        for alignment in self.alignments:
            if alignment['source_uri'] == source_uri:
                candidates.append((
                    alignment['target_uri'],
                    alignment['confidence']
                ))

        # Sort by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:top_k]

    def batch_match(
        self,
        source_concepts: List[Dict[str, Any]],
        top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Batch matching using pre-computed alignments.

        Args:
            source_concepts: List of source concepts
            top_k: Number of candidates per source

        Returns:
            Dictionary mapping source_uri to candidates
        """
        results = {}

        for concept in source_concepts:
            source_uri = concept['uri']
            candidates = self.find_candidates(concept, top_k)
            results[source_uri] = candidates

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get AML matcher statistics.

        Returns:
            Statistics dictionary
        """
        stats = super().get_statistics()
        stats['total_alignments'] = len(self.alignments)
        stats['unique_sources'] = len(set(a['source_uri'] for a in self.alignments))
        stats['unique_targets'] = len(set(a['target_uri'] for a in self.alignments))
        stats['avg_confidence'] = (
            sum(a['confidence'] for a in self.alignments) / len(self.alignments)
            if self.alignments else 0.0
        )

        return stats


def main():
    """
    Test AML Matcher.
    """
    from data_loader import load_all_concepts

    print("=" * 70)
    print("AML MATCHER TEST")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading data...")
    df = load_all_concepts('bike')

    s1000d_df = df[df['source'] == 's1000d'].reset_index(drop=True)
    ontology_df = df[df['source'] == 'bike_ontology'].reset_index(drop=True)

    print(f"  S1000D concepts: {len(s1000d_df)}")
    print(f"  Ontology concepts: {len(ontology_df)}")

    # Initialize AML Matcher
    print("\n[2/3] Initializing AML Matcher...")
    aml = AMLMatcher(s1000d_df, ontology_df)

    # Print statistics
    print("\n[3/3] AML Statistics:")
    stats = aml.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test matching
    print("\n" + "=" * 70)
    print("Testing AML matching on sample concepts...")

    for i in range(min(3, len(s1000d_df))):
        concept = s1000d_df.iloc[i].to_dict()

        print(f"\n--- {concept['label']} ---")
        candidates = aml.find_candidates(concept, top_k=5)

        if candidates:
            print(f"Top {len(candidates)} AML matches:")
            for j, (target_uri, score) in enumerate(candidates, 1):
                print(f"  {j}. Score: {score:.3f} - {target_uri}")
        else:
            print("  No matches found")

    print("\n" + "=" * 70)
    print("AML Matcher test completed!")


if __name__ == '__main__':
    main()
