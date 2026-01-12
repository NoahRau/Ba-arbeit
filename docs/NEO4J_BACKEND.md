# Neo4j Backend für Ontology Matching

Dieses Dokument beschreibt das Neo4j-basierte Graph Backend für das Ontology Matching System.

## Überblick

Das Neo4j Backend ersetzt die In-Memory Pandas DataFrames durch eine persistente Graph-Datenbank mit folgenden Vorteilen:

- **Vector Search**: Schnelle Ähnlichkeitssuche über embeddings
- **Graph Traversal**: Hierarchische Beziehungen zwischen Konzepten
- **Graph-Aware RAG**: Kombination aus Vector Search + Graph Context
- **Skalierbarkeit**: Kann große Ontologien effizient verarbeiten

## Setup

### 1. Neo4j installieren

**Docker (empfohlen):**
```bash
docker run \
    -p 7687:7687 -p 7474:7474 \
    -e NEO4J_AUTH=neo4j/your_password \
    -e NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
    neo4j:5.15
```

**Oder lokal installieren:**
- Download: https://neo4j.com/download/
- Version: Neo4j 5.13+ (für Vector Search Support)

### 2. Python Dependencies

```bash
pip install neo4j pandas numpy
```

### 3. Neo4j Konfiguration

In `neo4j.conf` (für lokale Installation):

```conf
# Enable vector indexes
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*

# Memory settings (anpassen nach Bedarf)
dbms.memory.heap.initial_size=1g
dbms.memory.heap.max_size=2g
```

## Daten Migration

### Schritt 1: Existierende Daten laden

Das Migrations-Script lädt automatisch die Daten aus `data_loader.py`:

```bash
python scripts/migrate_to_neo4j.py
```

**Was passiert:**
1. Lädt S1000D und Ontologie-Konzepte
2. Generiert Embeddings (TODO: durch echte Embeddings ersetzen!)
3. Erstellt Neo4j Schema (Indexes, Constraints)
4. Importiert Nodes mit Properties
5. Erstellt hierarchische Kanten (PARENT_OF)

### Schritt 2: Schema anpassen

Das Schema wird automatisch erstellt:

**Node Labels:**
- `S1000DConcept`: Technische Dokumentations-Konzepte
- `OntologyClass`: Ontologie-Klassen

**Properties:**
- `uri`: Unique identifier
- `label`: Konzept-Name
- `definition`: Beschreibung/Definition
- `context_text`: Hierarchischer Kontext
- `embedding`: Vector embedding (768-dim)

**Relationships:**
- `PARENT_OF`: Hierarchische Beziehung (Parent → Child)
- `RELATED_TO`: Semantische Verwandtschaft

**Indexes:**
- Unique constraint auf `uri`
- Vector index auf `embedding` (für Similarity Search)
- Text index auf `label`

## Verwendung

### Grundlegende Nutzung

```python
from storage.graph_store import KnowledgeGraphStore

# Verbindung herstellen
store = KnowledgeGraphStore(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your_password"
)

# Graph-Aware Retrieval
query_embedding = model.encode("Install wheel on front fork")

candidates = store.retrieve_candidates(
    query_embedding,
    top_k=20,
    target_label='OntologyClass',
    include_neighbors=True
)

for cand in candidates:
    print(f"{cand['label']}: {cand['similarity_score']:.3f}")
    print(f"  Parents: {cand['parents']}")
    print(f"  Children: {cand['children']}")

store.close()
```

### Graph-Aware RAG Pipeline

Das Retrieval funktioniert in zwei Schritten:

**1. Vector Search:**
```cypher
CALL db.index.vector.queryNodes('ontology_embeddings', $top_k, $query_embedding)
YIELD node, score
RETURN node, score
```

**2. Graph Traversal:**
```cypher
MATCH (n:OntologyClass {uri: $uri})
OPTIONAL MATCH (parent)-[:PARENT_OF]->(n)
OPTIONAL MATCH (n)-[:PARENT_OF]->(child)
OPTIONAL MATCH (parent)-[:PARENT_OF]->(sibling)
RETURN parent, child, sibling
```

**Ergebnis:** Kandidaten mit angereichertem Graph-Kontext für besseres Reranking.

### Custom Queries

```python
# Direkter Cypher Query
with store.driver.session() as session:
    result = session.run("""
        MATCH (s:S1000DConcept)-[:MATCHES]->(o:OntologyClass)
        RETURN s.label, o.label, COUNT(*) as freq
        ORDER BY freq DESC
        LIMIT 10
    """)

    for record in result:
        print(record)
```

## Integration mit Matching Pipeline

### Option 1: Retrieval ersetzen

Ersetze die alte Candidate-Retrieval-Logik:

```python
# Alt (Pandas)
candidates = df[df['label'].str.contains(query)].head(50)

# Neu (Neo4j)
candidates = store.retrieve_candidates(
    query_embedding,
    top_k=50,
    include_neighbors=True
)
```

### Option 2: Hybrid Approach

Nutze Neo4j für Retrieval, aber behalte Reranking-Pipeline:

```python
# 1. Retrieval mit Neo4j (Graph-Aware)
candidates = store.retrieve_candidates(query_embedding, top_k=100)

# 2. Reranking mit existierendem Code
reranked = neural_reranker.rerank(source, candidates)
final = llm_reranker.rerank(source, reranked)
```

### Option 3: Matching-Ergebnisse speichern

Schreibe Match-Ergebnisse zurück nach Neo4j:

```python
with store.driver.session() as session:
    session.run("""
        MATCH (s:S1000DConcept {uri: $source_uri})
        MATCH (o:OntologyClass {uri: $target_uri})
        MERGE (s)-[r:MATCHES]->(o)
        SET r.confidence = $confidence,
            r.method = $method,
            r.timestamp = datetime()
    """,
    source_uri=source_uri,
    target_uri=target_uri,
    confidence=0.92,
    method='neural_reranker')
```

## Performance Optimierung

### Vector Index Tuning

```cypher
-- Check index status
SHOW INDEXES;

-- Re-create with different settings
DROP INDEX ontology_embeddings;

CREATE VECTOR INDEX ontology_embeddings
FOR (n:OntologyClass) ON (n.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
  }
};
```

### Batch Processing

```python
# Batch-Retrieval für viele Queries
queries = [emb1, emb2, emb3, ...]

results = []
for query_emb in queries:
    candidates = store.retrieve_candidates(query_emb, top_k=10)
    results.append(candidates)
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_retrieval(query_hash, top_k):
    return store.retrieve_candidates(query_embedding, top_k)
```

## Monitoring & Debugging

### Neo4j Browser

Öffne http://localhost:7474 und exploriere den Graph:

```cypher
-- Alle S1000D Konzepte
MATCH (n:S1000DConcept)
RETURN n LIMIT 25;

-- Hierarchie anzeigen
MATCH path = (parent)-[:PARENT_OF*1..3]->(child)
RETURN path LIMIT 50;

-- Embedding-Statistiken
MATCH (n:OntologyClass)
WHERE n.embedding IS NOT NULL
RETURN count(n) as nodes_with_embeddings;
```

### Statistiken

```python
stats = store.get_statistics()
print(stats)
# {
#   's1000d_concepts': 56,
#   'ontology_classes': 1291,
#   'parent_of_relationships': 150,
#   'total_nodes': 1347
# }
```

## Troubleshooting

### Vector Search funktioniert nicht

**Fehler:** `There is no index named 'ontology_embeddings'`

**Lösung:**
- Neo4j 5.13+ benötigt
- Index manuell erstellen (siehe Schema Init)
- `SHOW INDEXES` prüfen

### Embedding Dimension Mismatch

**Fehler:** `Vector dimension mismatch: expected 768, got 384`

**Lösung:**
```python
# Index mit korrekter Dimension neu erstellen
store.initialize_schema(embedding_dim=384)
```

### Langsame Queries

**Lösung:**
1. Indexes prüfen: `SHOW INDEXES`
2. Query Profile: `PROFILE MATCH ... RETURN ...`
3. Batch size anpassen: `batch_size=1000`

## Migration von Pandas

Alte Struktur:
```python
source_df = pd.DataFrame(...)
target_df = pd.DataFrame(...)
matcher.find_matches(source_df, target_df)
```

Neue Struktur:
```python
store = KnowledgeGraphStore()
store.ingest_dataframe(source_df, 'S1000DConcept')
store.ingest_dataframe(target_df, 'OntologyClass')

candidates = store.retrieve_candidates(query_embedding)
```

## Multi-Hop Reasoning

Eine der mächtigsten Features des Neo4j Backends ist **Multi-Hop Reasoning** - die Fähigkeit, explizite Beziehungsketten aus dem Graph zu extrahieren und als Kontext zu nutzen.

### Grundkonzept

```python
# Get reasoning context for a concept
reasoning = store.get_reasoning_context(
    uri='bike:component:wheel',
    hops=2,
    max_paths=10
)

print(reasoning)
# Output:
# [PATH-1] Wheel -> PART_OF -> WheelSystem
# [PATH-1] Wheel -> PARENT_OF -> Hub
# [PATH-1] Wheel -> PARENT_OF -> Spoke
# [PATH-2] Wheel -> PART_OF -> WheelSystem -> PARENT_OF -> Bicycle
# [PATH-2] Wheel -> PARENT_OF -> Hub -> RELATED_TO -> Bearing
```

### Warum Multi-Hop Reasoning?

**Problem ohne Reasoning:**
```
Query: "Install wheel spoke"
Candidate: "Spoke"
Cross-Encoder: "Are these similar?" → Might miss the connection
```

**Lösung mit Reasoning:**
```
Query: "Install wheel spoke"
Query Context:
  [PATH-1] Spoke -> PART_OF -> Wheel
  [PATH-2] Spoke -> PART_OF -> Wheel -> PART_OF -> WheelSystem

Candidate: "Spoke"
Candidate Context:
  [PATH-1] Spoke -> PART_OF -> Wheel
  [PATH-1] Spoke -> PARENT_OF -> Hub

Cross-Encoder: Sieht explizit: "Beide involvieren Spoke im Wheel-Kontext!" → Match!
```

### Verwendung in der Pipeline

**Option 1: Retrieval mit Reasoning**
```python
candidates = store.retrieve_candidates_with_reasoning(
    query_embedding,
    source_uri='source:concept:123',
    top_k=50,
    reasoning_hops=2
)

# Candidates haben jetzt:
# - reasoning_context: Graph-Pfade
# - full_context: Formatiert für Reranker
# - source_reasoning: Pfade des Source-Konzepts
```

**Option 2: Manuelle Anreicherung**
```python
# Hole Kandidaten
candidates = get_candidates(...)

# Füge Reasoning hinzu
for cand in candidates:
    reasoning = store.get_reasoning_context(cand['uri'], hops=2)
    cand['reasoning_context'] = reasoning
```

### Integration in Neural Reranker

Der Cross-Encoder erhält nun explizite Graph-Pfade im Text:

```python
# Neural Reranker Input (mit Reasoning)
query_text = f"""
[QUERY] {source.label}
[PATHS]
{source_reasoning}
"""

doc_text = f"""
[CANDIDATE] {candidate.label}
[PATHS]
{candidate_reasoning}
"""

pairs = [(query_text, doc_text)]
scores = cross_encoder.predict(pairs)
```

### Performance Tuning

**Hop Count:**
- `hops=1`: Nur direkte Nachbarn (schnell)
- `hops=2`: 2-Hop-Pfade (empfohlen, gute Balance)
- `hops=3`: 3-Hop-Pfade (langsamer, mehr Kontext)

**Pfad-Limit:**
- `max_paths=5`: Wenige, wichtigste Pfade (schnell)
- `max_paths=10`: Standard
- `max_paths=20`: Viele Pfade (mehr Kontext, aber langsamer)

### Beispiel-Scripts

```bash
# Demo: Multi-Hop Reasoning
python scripts/demo_multihop_reasoning.py

# Integration Guide
python scripts/integrate_graph_reasoning_pipeline.py
```

## Nächste Schritte

1. **Echte Embeddings generieren**
   - Ersetze `np.random.rand()` durch echtes Embedding-Modell
   - Z.B. SentenceTransformers, OpenAI, etc.

2. **Hierarchie-Parsing verbessern**
   - Parse `context_text` korrekt für PARENT_OF Kanten
   - Oder: Explizite `parent_uri` in DataFrames hinzufügen

3. **Multi-Hop Reasoning integrieren**
   - Füge `retrieve_candidates_with_reasoning()` zur Pipeline hinzu
   - Update Neural Reranker Input Format
   - Teste mit verschiedenen Hop-Counts

4. **Matching-Pipeline integrieren**
   - Ersetze Retrieval-Schritt durch Neo4j
   - Behalte Reranking-Logik bei

5. **Evaluation**
   - Vergleiche: Mit vs. ohne Reasoning
   - Messe: Recall@k, Precision, F1, URI Accuracy
   - Erwartete Verbesserung: +10-15% F1-Score

6. **Production Deployment**
   - Neo4j Cluster für Hochverfügbarkeit
   - Backup-Strategie
   - Monitoring mit Neo4j Metrics

## Referenzen

- [Neo4j Vector Search Docs](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)
- [Graph RAG Patterns](https://neo4j.com/developer/graph-rag/)
