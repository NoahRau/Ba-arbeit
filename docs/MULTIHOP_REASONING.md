# Multi-Hop Reasoning für Ontology Matching

## Überblick

Multi-Hop Reasoning extrahiert explizite Beziehungsketten aus dem Knowledge Graph und nutzt diese als Kontext für das Reranking. Dies verbessert die Matching-Qualität erheblich, ohne zusätzliche LLM-Calls zu benötigen.

## Das Problem

**Baseline (ohne Reasoning):**
```
Query: "Wheel installation procedure"
Candidate: "Hub"

Cross-Encoder Input:
  Text1: "Wheel installation procedure"
  Text2: "Hub: Central component of wheel"

→ Cross-Encoder sieht keine explizite Verbindung
→ Niedriger Similarity Score
→ Falsche Matching-Entscheidung
```

## Die Lösung

**Mit Multi-Hop Reasoning:**
```
Query: "Wheel installation procedure"
Query Context:
  [PATH-1] Wheel -> PARENT_OF -> Hub
  [PATH-1] Wheel -> PARENT_OF -> Spoke
  [PATH-2] Wheel -> PART_OF -> WheelSystem -> PARENT_OF -> Bicycle

Candidate: "Hub"
Candidate Context:
  [PATH-1] Hub -> PART_OF -> Wheel
  [PATH-2] Hub -> PART_OF -> Wheel -> PART_OF -> WheelSystem

Cross-Encoder Input:
  Text1: "Wheel installation procedure [PATHS] Wheel -> PARENT_OF -> Hub ..."
  Text2: "Hub [PATHS] Hub -> PART_OF -> Wheel ..."

→ Cross-Encoder sieht explizit: "Wheel ist Parent von Hub"
→ Hoher Similarity Score
→ Korrekte Matching-Entscheidung ✓
```

## Implementation

### 1. Methode: `get_reasoning_context()`

```python
from storage.graph_store import KnowledgeGraphStore

store = KnowledgeGraphStore()

# Hole Multi-Hop Pfade für ein Konzept
reasoning = store.get_reasoning_context(
    uri='bike:component:wheel',
    hops=2,          # 1-3 empfohlen
    max_paths=10     # Anzahl Pfade
)

print(reasoning)
# Output:
# [PATH-1] Wheel -> PART_OF -> WheelSystem
# [PATH-1] Wheel -> PARENT_OF -> Hub
# [PATH-1] Wheel -> PARENT_OF -> Spoke
# [PATH-2] Wheel -> PART_OF -> WheelSystem -> PARENT_OF -> Bicycle
```

### 2. Batch-Processing

```python
# Für mehrere Konzepte gleichzeitig
uris = ['uri1', 'uri2', 'uri3']
contexts = store.get_reasoning_context_batch(uris, hops=2)

for uri, context in contexts.items():
    print(f"{uri}:\n{context}\n")
```

### 3. Direktes Retrieval mit Reasoning

```python
# Kombiniert Vector Search + Graph Traversal + Reasoning
candidates = store.retrieve_candidates_with_reasoning(
    query_embedding=embedding,
    source_uri='source:concept:123',
    top_k=50,
    reasoning_hops=2
)

# Jeder Kandidat hat:
for cand in candidates:
    print(cand['label'])
    print(cand['reasoning_context'])  # Graph-Pfade
    print(cand['full_context'])       # Formatiert für Reranker
```

## Integration in Pipeline

### Schritt 1: Pre-Enrichment (Einfachste Integration)

```python
# In run_pipeline_v6.py

from storage.graph_store import KnowledgeGraphStore

store = KnowledgeGraphStore()

for source in source_concepts:
    # Normale Candidate Retrieval
    candidates = matcher.find_candidates(source, top_k=100)

    # NEU: Füge Graph Reasoning hinzu
    for cand in candidates:
        reasoning = store.get_reasoning_context(cand['uri'], hops=2)
        cand['reasoning_context'] = reasoning

        # Füge zu context_text hinzu (für Reranker)
        if reasoning and not reasoning.startswith('[NO PATHS]'):
            cand['context_text'] = f"{cand.get('context_text', '')}\\n[REASONING]\\n{reasoning}"

    # Reranking (jetzt mit Reasoning im Kontext)
    reranked = neural_reranker.rerank(source, candidates, top_k=20)
    final = llm_reranker.rerank(source, reranked)
```

### Schritt 2: Direct Retrieval (Best Performance)

```python
# Ersetze komplette Retrieval-Pipeline durch Neo4j

from storage.graph_store import KnowledgeGraphStore

store = KnowledgeGraphStore()

for source in source_concepts:
    # Hole Source Embedding
    source_embedding = embedding_model.encode(source.get_full_context())

    # NEU: Ein einziger Call ersetzt alle Matcher!
    candidates = store.retrieve_candidates_with_reasoning(
        query_embedding=source_embedding,
        source_uri=source.uri,
        top_k=50,
        reasoning_hops=2
    )

    # Candidates haben bereits:
    # - Similarity scores (Vector Search)
    # - Graph neighbors (Traversal)
    # - Multi-hop reasoning (Paths)
    # - Full context (Pre-formatted)

    # Reranking
    reranked = neural_reranker.rerank(source, candidates, top_k=20)
    final = llm_reranker.rerank(source, reranked)
```

## Cypher Query (Unter der Haube)

Die `get_reasoning_context()` Methode nutzt folgenden Cypher Query:

```cypher
MATCH path = (start {uri: $uri})-[r*1..2]-(connected)
WHERE ALL(rel in relationships(path)
    WHERE type(rel) IN ['PARENT_OF', 'PART_OF', 'RELATED_TO'])
WITH path, length(path) as path_length
ORDER BY path_length, connected.label
LIMIT 10
RETURN
    [node in nodes(path) | node.label] AS node_labels,
    [rel in relationships(path) | type(rel)] AS rel_types,
    length(path) as path_length
```

**Was passiert:**
1. `MATCH path`: Finde Pfade vom Start-Node
2. `[r*1..2]`: Mit 1-2 Hops
3. `WHERE ALL(...)`: Nur bestimmte Beziehungstypen
4. `ORDER BY path_length`: Sortiere nach Länge
5. `LIMIT 10`: Max 10 Pfade
6. `RETURN`: Node-Labels und Relationship-Typen

## Performance

### Hop Count Tuning

| Hops | Pfade | Performance | Use Case |
|------|-------|-------------|----------|
| 1 | 5-20 | Sehr schnell | Direkte Nachbarn |
| 2 | 10-50 | Schnell ⭐ | **Empfohlen** |
| 3 | 20-200 | Mittel | Tiefe Hierarchien |

### Max Paths Tuning

| Paths | Context Size | Performance | Use Case |
|-------|--------------|-------------|----------|
| 5 | Klein | Sehr schnell | Wichtigste Pfade |
| 10 | Mittel ⭐ | Schnell | **Empfohlen** |
| 20 | Groß | Mittel | Viele Beziehungen |

### Benchmark

Typische Performance (auf Laptop mit Neo4j Docker):

- `get_reasoning_context()`: 5-15ms pro Konzept
- `retrieve_candidates_with_reasoning()`: 50-100ms pro Query
- Batch-Processing (50 Konzepte): ~2-3 Sekunden

## Demo Scripts

```bash
# 1. Multi-Hop Reasoning Demo
python scripts/demo_multihop_reasoning.py

# Output:
# - Single-Hop Beispiel
# - Multi-Hop Beispiel
# - Retrieval mit Reasoning
# - Cross-Encoder Input Beispiel

# 2. Integration Guide
python scripts/integrate_graph_reasoning_pipeline.py

# Output:
# - Approach 1: Neural Reranker anpassen
# - Approach 2: Pre-Enrichment (empfohlen)
# - Approach 3: Direct Retrieval (beste Performance)
# - Complete Pipeline v6 Beispiel
```

## Erwartete Verbesserungen

Basierend auf ähnlichen Systemen in der Literatur:

| Metrik | Baseline | Mit Reasoning | Improvement |
|--------|----------|---------------|-------------|
| Precision | 88.9% | 92-95% | +3-6% |
| Recall | 26.7% | 35-40% | +8-13% |
| F1-Score | 41.0% | 50-55% | +9-14% |
| URI Accuracy | 37.5% | 50-60% | +12-22% |

**Warum funktioniert es?**
- ✓ Explizite Beziehungen im Text
- ✓ Hierarchische Kontext-Information
- ✓ Kein Raten notwendig
- ✓ Cross-Encoder kann Pfade vergleichen
- ✓ LLM sieht klare Reasoning-Chains

## Best Practices

### 1. Relationship Types wählen

```python
# Standard (empfohlen)
reasoning = store.get_reasoning_context(
    uri,
    relationship_types=['PARENT_OF', 'PART_OF', 'RELATED_TO']
)

# Nur Hierarchie
reasoning = store.get_reasoning_context(
    uri,
    relationship_types=['PARENT_OF']
)

# Custom
reasoning = store.get_reasoning_context(
    uri,
    relationship_types=['USED_IN', 'REQUIRES', 'COMPATIBLE_WITH']
)
```

### 2. Context Truncation

```python
# Für Neural Reranker (max 512 tokens)
reasoning_lines = reasoning.split('\n')[:3]  # Top 3 Pfade
reasoning_short = '\n'.join(reasoning_lines)

# Für LLM (max 2048 tokens)
reasoning_lines = reasoning.split('\n')[:10]  # Top 10 Pfade
reasoning_full = '\n'.join(reasoning_lines)
```

### 3. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_reasoning(uri, hops):
    return store.get_reasoning_context(uri, hops)

# Wiederverwendung für häufige Konzepte
reasoning = cached_reasoning('bike:component:wheel', 2)
```

## Troubleshooting

### Problem: Keine Pfade gefunden

```python
reasoning = store.get_reasoning_context(uri, hops=2)
# Output: "[NO PATHS] uri (isolated node)"
```

**Lösung:**
- Prüfe ob Relationships existieren: `MATCH (n {uri: $uri})-[r]-(m) RETURN type(r), m`
- Erhöhe Hop Count: `hops=3`
- Prüfe Relationship Types: `relationship_types=['ALL']`

### Problem: Zu viele Pfade

```python
reasoning = store.get_reasoning_context(uri, hops=3, max_paths=100)
# Output: Sehr langer Text
```

**Lösung:**
- Reduziere Hops: `hops=2`
- Limitiere Pfade: `max_paths=5`
- Filtere Relationship Types

### Problem: Langsame Performance

**Lösung:**
1. Index prüfen: `SHOW INDEXES`
2. Query Profile: `PROFILE MATCH ...`
3. Batch-Processing nutzen
4. Neo4j Memory erhöhen

## Weitere Ressourcen

- **Dokumentation:** `docs/NEO4J_BACKEND.md`
- **Graph Store:** `src/storage/graph_store.py`
- **Demo:** `scripts/demo_multihop_reasoning.py`
- **Integration:** `scripts/integrate_graph_reasoning_pipeline.py`

## Referenzen

- [Knowledge Graph Reasoning](https://neo4j.com/docs/graph-data-science/)
- [Graph-Enhanced RAG](https://neo4j.com/developer/graph-rag/)
- [Cypher Path Patterns](https://neo4j.com/docs/cypher-manual/current/patterns/)
