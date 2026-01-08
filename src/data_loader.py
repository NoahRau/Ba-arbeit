"""
Robust Data Loader for S1000D XML and OWL Ontologies.
Creates hierarchical context embeddings (DeepOnto-style).
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Set
from urllib.parse import quote
from collections import defaultdict

import pandas as pd
import requests
from lxml import etree
from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef


# Ontology URLs
ONTOLOGY_URLS = {
    'tbox': 'https://giuliamenna.github.io/BikeOntology/final_data/tbox_bikeo.owl',
    'abox': 'https://giuliamenna.github.io/BikeOntology/final_data/abox_bikeo.ttl',
    'final': 'https://giuliamenna.github.io/BikeOntology/final_data/final_bikeo.owl'
}

# Local cache directory - relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
ONTOLOGY_CACHE_DIR = PROJECT_ROOT / 'data' / 'ontologies'
S1000D_DATA_DIR = PROJECT_ROOT / 'data' / 's1000d'
CACHE_DIR = PROJECT_ROOT / 'cache' / 'embeddings'


def download_ontology(url: str, filename: str) -> Path:
    """
    Download an ontology file if it doesn't exist locally.
    """
    ONTOLOGY_CACHE_DIR.mkdir(exist_ok=True)
    local_path = ONTOLOGY_CACHE_DIR / filename

    if local_path.exists():
        print(f"  Using cached: {filename}")
        return local_path

    print(f"  Downloading: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            f.write(response.content)

        print(f"  Cached: {filename}")
        return local_path

    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        raise


def get_class_hierarchy_path(g: Graph, class_uri: URIRef) -> List[str]:
    """
    Recursively traverse rdfs:subClassOf to build path to root.
    Returns list from root to current class.

    Example: ['Thing', 'BikeComponent', 'Wheel', 'FrontWheel']
    """
    path = []
    visited = set()  # Prevent infinite loops

    def traverse_up(uri: URIRef):
        if uri in visited:
            return
        visited.add(uri)

        # Get label or use local name
        labels = list(g.objects(uri, RDFS.label))
        label = str(labels[0]) if labels else str(uri).split('/')[-1].split('#')[-1]

        # Add to path
        path.insert(0, label)

        # Find parent classes
        parents = list(g.objects(uri, RDFS.subClassOf))

        # Recursively traverse parents
        for parent in parents:
            if isinstance(parent, URIRef):
                # Skip owl:Thing and generic top-level classes
                parent_str = str(parent)
                if 'owl#Thing' not in parent_str and 'XMLSchema' not in parent_str:
                    traverse_up(parent)

    traverse_up(class_uri)

    return path


def load_owl_ontologies(
    tbox_url: str = None,
    abox_url: str = None,
    final_url: str = None
) -> List[Dict[str, Any]]:
    """
    Load OWL ontologies with hierarchical context (DeepOnto-style).

    Returns:
        List of concept dictionaries with hierarchical context_text
    """
    tbox_url = tbox_url or ONTOLOGY_URLS['tbox']
    abox_url = abox_url or ONTOLOGY_URLS['abox']
    final_url = final_url or ONTOLOGY_URLS['final']

    print("=" * 70)
    print("LOADING OWL ONTOLOGIES (DeepOnto-Style)")
    print("=" * 70)

    # Create RDF graph
    g = Graph()

    # Download and parse ontologies
    ontology_files = []

    try:
        if tbox_url and 'example.com' not in tbox_url:
            tbox_path = download_ontology(tbox_url, 'tbox.owl')
            ontology_files.append(('tbox', tbox_path))

        # Skip ABox as it has parsing issues (TTL format)
        # if abox_url and 'example.com' not in abox_url:
        #     abox_path = download_ontology(abox_url, 'abox.owl')
        #     ontology_files.append(('abox', abox_path))

        if final_url and 'example.com' not in final_url:
            final_path = download_ontology(final_url, 'final.owl')
            ontology_files.append(('final', final_path))

    except Exception as e:
        print(f"Warning: Could not download ontologies: {e}")

    # Try local files if download failed
    if not ontology_files:
        print("\nChecking for local ontology files...")
        for name in ['tbox.owl', 'final.owl']:
            local_path = ONTOLOGY_CACHE_DIR / name
            if local_path.exists():
                ontology_files.append((name.replace('.owl', ''), local_path))
                print(f"  Found: {name}")

    if not ontology_files:
        print("No ontology files available.")
        return []

    # Parse ontologies
    print(f"\nParsing {len(ontology_files)} ontology files...")
    for name, path in ontology_files:
        try:
            g.parse(path, format='xml')
            print(f"  Loaded {name}: {len(g)} triples")
        except Exception as e:
            print(f"  Error parsing {name}: {e}")

    # Extract concepts with hierarchical context
    concepts = []

    # Extract OWL Classes
    print("\nExtracting OWL Classes with hierarchical paths...")
    class_count = 0

    for class_uri in g.subjects(RDF.type, OWL.Class):
        if not isinstance(class_uri, URIRef):
            continue

        # Skip imported classes
        class_str = str(class_uri)
        if 'XMLSchema' in class_str or 'owl#Thing' in class_str:
            continue

        # Get hierarchy path
        hierarchy_path = get_class_hierarchy_path(g, class_uri)

        # Get label
        labels = list(g.objects(class_uri, RDFS.label))
        label = str(labels[0]) if labels else hierarchy_path[-1] if hierarchy_path else str(class_uri).split('/')[-1].split('#')[-1]

        # Get comment
        comments = list(g.objects(class_uri, RDFS.comment))
        comment = ' '.join(str(c) for c in comments) if comments else ''

        # Build hierarchical context: Root > Parent > Child > Label | Comment
        if hierarchy_path and len(hierarchy_path) > 1:
            hierarchy_str = ' > '.join(hierarchy_path)
        else:
            hierarchy_str = label

        if comment:
            context_text = f"{hierarchy_str} | {comment}"
        else:
            context_text = hierarchy_str

        # Add additional properties to context
        context_parts = [context_text]
        for pred, obj in g.predicate_objects(class_uri):
            pred_str = str(pred).split('/')[-1].split('#')[-1]
            if pred_str not in ['type', 'label', 'comment', 'subClassOf']:
                obj_str = str(obj).split('/')[-1].split('#')[-1]
                context_parts.append(f"{pred_str}: {obj_str}")

        final_context = ' | '.join(context_parts[:5])  # Limit to avoid too long

        concepts.append({
            'uri': str(class_uri),
            'label': label,
            'context_text': final_context,
            'source': 'bike_ontology'
        })

        class_count += 1

    print(f"  Found {class_count} classes with hierarchical context")

    # Extract Named Individuals
    print("\nExtracting OWL Named Individuals...")
    individual_count = 0

    for individual_uri in g.subjects(RDF.type, OWL.NamedIndividual):
        if not isinstance(individual_uri, URIRef):
            continue

        # Get label
        labels = list(g.objects(individual_uri, RDFS.label))
        label = str(labels[0]) if labels else str(individual_uri).split('/')[-1].split('#')[-1]

        # Get type/class
        types = [t for t in g.objects(individual_uri, RDF.type) if t != OWL.NamedIndividual]
        type_labels = []
        for type_uri in types:
            if isinstance(type_uri, URIRef):
                type_path = get_class_hierarchy_path(g, type_uri)
                if type_path:
                    type_labels.append(' > '.join(type_path))

        # Get comment
        comments = list(g.objects(individual_uri, RDFS.comment))
        comment = ' '.join(str(c) for c in comments) if comments else ''

        # Build context: Type Hierarchy > Instance Label | Comment
        if type_labels:
            context_text = f"{type_labels[0]} > {label}"
        else:
            context_text = label

        if comment:
            context_text += f" | {comment}"

        concepts.append({
            'uri': str(individual_uri),
            'label': label,
            'context_text': context_text,
            'source': 'bike_ontology'
        })

        individual_count += 1

    print(f"  Found {individual_count} individuals")
    print(f"\nTotal ontology concepts: {len(concepts)}")

    return concepts


def extract_dm_code_hierarchy(root: etree._Element) -> str:
    """
    Extract DMC and build hierarchical representation.

    Returns string like: "S1000DBIKE > AAA > D00 > 041"
    """
    try:
        # Try newer S1000D format
        dm_code = root.find('.//dmCode')
        if dm_code is not None:
            parts = []
            if dm_code.get('modelIdentCode'):
                parts.append(dm_code.get('modelIdentCode'))
            if dm_code.get('systemDiffCode'):
                parts.append(dm_code.get('systemDiffCode'))
            if dm_code.get('systemCode'):
                parts.append(dm_code.get('systemCode'))
            if dm_code.get('assyCode'):
                parts.append(dm_code.get('assyCode'))
            if dm_code.get('disassyCode'):
                parts.append(dm_code.get('disassyCode'))
            if dm_code.get('infoCode'):
                parts.append(dm_code.get('infoCode'))

            if parts:
                return ' > '.join(parts)

        # Try older S1000D format (v2.3)
        avee = root.find('.//avee')
        if avee is not None:
            parts = []
            elements = ['modelic', 'sdc', 'chapnum', 'section', 'subsect', 'incode']
            for elem_name in elements:
                elem = avee.find(elem_name)
                if elem is not None and elem.text:
                    parts.append(elem.text.strip())

            if parts:
                return ' > '.join(parts)

    except Exception as e:
        print(f"Error extracting hierarchy: {e}")

    return ""


def extract_tech_name(root: etree._Element) -> str:
    """
    Extract technical name from various S1000D formats.
    """
    try:
        # Try techName (newer)
        tech_name = root.find('.//techName')
        if tech_name is not None and tech_name.text:
            return tech_name.text.strip()

        # Try techname (older)
        tech_name = root.find('.//techname')
        if tech_name is not None and tech_name.text:
            info_name = root.find('.//infoname')
            if info_name is not None and info_name.text:
                return f"{tech_name.text.strip()} - {info_name.text.strip()}"
            return tech_name.text.strip()

        # Try dmTitle
        dm_title = root.find('.//dmTitle')
        if dm_title is not None:
            tech_elem = dm_title.find('techName')
            info_elem = dm_title.find('infoName')
            parts = []
            if tech_elem is not None and tech_elem.text:
                parts.append(tech_elem.text.strip())
            if info_elem is not None and info_elem.text:
                parts.append(info_elem.text.strip())
            if parts:
                return ' - '.join(parts)

        # Try dmtitle (older)
        dm_title = root.find('.//dmtitle')
        if dm_title is not None:
            tech_elem = dm_title.find('techname')
            info_elem = dm_title.find('infoname')
            parts = []
            if tech_elem is not None and tech_elem.text:
                parts.append(tech_elem.text.strip())
            if info_elem is not None and info_elem.text:
                parts.append(info_elem.text.strip())
            if parts:
                return ' - '.join(parts)

    except Exception as e:
        print(f"Error extracting tech name: {e}")

    return ""


def extract_full_description(root: etree._Element) -> str:
    """
    Extract full text from <para> and <descr> tags.
    """
    texts = []

    try:
        # Extract from para tags
        for para in root.findall('.//para'):
            text = ''.join(para.itertext()).strip()
            if text:
                texts.append(text)

        # Extract from descr tags
        for descr in root.findall('.//descr'):
            text = ''.join(descr.itertext()).strip()
            if text:
                texts.append(text)

        # Extract from description tags
        for desc in root.findall('.//description'):
            text = ''.join(desc.itertext()).strip()
            if text:
                texts.append(text)

    except Exception as e:
        print(f"Error extracting description: {e}")

    return ' '.join(texts)


def parse_s1000d_file(file_path: Path) -> Dict[str, Any]:
    """
    Parse single S1000D XML file with hierarchical context.

    Returns dict with uri, label, context_text, source
    """
    try:
        parser = etree.XMLParser(recover=True, remove_blank_text=True)
        tree = etree.parse(str(file_path), parser)
        root = tree.getroot()

        # Extract components
        hierarchy = extract_dm_code_hierarchy(root)
        tech_name = extract_tech_name(root)
        description = extract_full_description(root)

        # Build URI
        if hierarchy:
            encoded_hierarchy = quote(hierarchy.replace(' > ', '-'), safe='')
            uri = f"http://my-company.com/s1000d/{encoded_hierarchy}"
        else:
            uri = f"http://my-company.com/s1000d/{quote(file_path.stem, safe='')}"

        # Build hierarchical context_text: Hierarchy > TechName | Description
        context_parts = []

        if hierarchy:
            context_parts.append(hierarchy)

        if tech_name:
            if context_parts:
                context_parts.append(f"> {tech_name}")
            else:
                context_parts.append(tech_name)

        if description:
            # Limit description length but keep substantial context
            desc_text = description[:1000] if len(description) > 1000 else description
            context_parts.append(f"| {desc_text}")

        context_text = ' '.join(context_parts)

        # Use tech_name as label if available
        label = tech_name if tech_name else (hierarchy.split(' > ')[-1] if hierarchy else file_path.stem)

        return {
            'uri': uri,
            'label': label,
            'context_text': context_text,
            'source': 's1000d'
        }

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def load_s1000d_data(folder_path: str = None) -> List[Dict[str, Any]]:
    """
    Recursively load and parse all S1000D XML files with hierarchical context.
    """
    if folder_path is None:
        folder = S1000D_DATA_DIR
    else:
        folder = Path(folder_path)

    if not folder.exists():
        print(f"Warning: Folder '{folder_path}' does not exist.")
        return []

    # Find all XML files
    xml_files = list(folder.rglob('*.xml')) + list(folder.rglob('*.XML'))

    if not xml_files:
        print(f"Warning: No XML files found in '{folder_path}'.")
        return []

    print(f"Found {len(xml_files)} XML files in '{folder_path}'")

    # Parse files
    concepts = []
    for xml_file in xml_files:
        result = parse_s1000d_file(xml_file)
        if result and result['context_text']:  # Only add if context exists
            concepts.append(result)

    print(f"Successfully parsed {len(concepts)} S1000D concepts")

    return concepts


def load_all_concepts(
    s1000d_folder: str = None,
    include_ontologies: bool = True
) -> pd.DataFrame:
    """
    Load all concepts with hierarchical context embeddings.

    Returns:
        DataFrame with columns: uri, label, context_text, source
    """
    print("=" * 70)
    print("LOADING ALL CONCEPTS WITH HIERARCHICAL CONTEXT")
    print("=" * 70)

    all_concepts = []

    # Load S1000D
    print("\n[1/2] LOADING S1000D DATA")
    print("-" * 70)
    s1000d_concepts = load_s1000d_data(s1000d_folder)
    all_concepts.extend(s1000d_concepts)

    # Load Ontologies
    if include_ontologies:
        print("\n[2/2] LOADING OWL ONTOLOGIES")
        print("-" * 70)
        try:
            ontology_concepts = load_owl_ontologies()
            all_concepts.extend(ontology_concepts)
        except Exception as e:
            print(f"Warning: Could not load ontologies: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(all_concepts)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total concepts: {len(df)}")

    if not df.empty:
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count}")

        # Show context length stats
        if 'context_text' in df.columns:
            avg_len = df['context_text'].str.len().mean()
            max_len = df['context_text'].str.len().max()
            print(f"\nContext Statistics:")
            print(f"  Average length: {avg_len:.0f} chars")
            print(f"  Max length: {max_len:.0f} chars")

    return df


def main():
    """
    Test the data loader.
    """
    df = load_all_concepts(
        s1000d_folder='bike',
        include_ontologies=True
    )

    if df.empty:
        print("\nNo data loaded.")
        return

    print("\n" + "=" * 70)
    print("SAMPLE DATA")
    print("=" * 70)

    pd.set_option('display.max_colwidth', 80)
    pd.set_option('display.width', None)

    # Show samples from each source
    for source in df['source'].unique():
        print(f"\n{source.upper()} Sample:")
        sample = df[df['source'] == source].head(2)
        for _, row in sample.iterrows():
            print(f"\nLabel: {row['label']}")
            print(f"Context: {row['context_text'][:200]}...")


if __name__ == '__main__':
    main()
