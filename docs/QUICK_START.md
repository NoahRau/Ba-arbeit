# Quick Start Guide - State-of-the-Art Pipeline

## Sofort starten mit Sprint 1: KROMA Matcher

### Schritt 1: Backup erstellen
```bash
cd "/mnt/d/Software Projekte/Intellj/IdeaProjects/BA-arbeit"
git add -A
git commit -m "Backup vor State-of-the-Art Refactoring"
git tag v1.0-simple-pipeline
```

### Schritt 2: Neue Struktur erstellen
```bash
mkdir -p matchers aggregation reranking validation pipeline evaluation legacy
touch matchers/__init__.py
touch aggregation/__init__.py
touch reranking/__init__.py
touch validation/__init__.py
touch pipeline/__init__.py
touch evaluation/__init__.py
```

### Schritt 3: Legacy Code sichern
```bash
mv bert_matcher.py legacy/
mv llm_reasoner.py legacy/
mv build_knowledge_base.py legacy/
```

### Schritt 4: Dependencies installieren
```bash
source venv/bin/activate
pip install deeponto owlready2 JPype1 --upgrade
```

### Schritt 5: KROMA Matcher implementieren

Ich erstelle jetzt ein funktionsfähiges Template für `matchers/kroma_matcher.py`.

**Was KROMA macht:**
- Extrahiert DMC Chapter Codes aus S1000D URIs (z.B. "D00", "A00", "C00")
- Matched gegen Ontologie-Konzepte via Lookup-Table + String-Similarity
- Nutzt Ihr bereits vorhandenes hierarchisches Parsing aus `data_loader.py`

### Sprint 1 Ziel (Woche 1):
- KROMA läuft standalone
- Findet 80%+ aller DMC-basierten Matches
- Benchmark auf Gold Standard zeigt Verbesserung

---

## Nächste Schritte nach Sprint 1

**Woche 2:** DeepOnto Integration
```bash
git checkout -b feature/deeponto-integration
```

**Woche 3:** AML Wrapper + Aggregation
```bash
git checkout -b feature/aggregation-layer
```

**Woche 4:** LLM Reranking + Pipeline
```bash
git checkout -b feature/hybrid-pipeline
```

---

## Erwartete Timeline

| Sprint | Dauer | Deliverable | F1-Score (geschätzt) |
|--------|-------|-------------|----------------------|
| **Aktuell** | - | Alte Pipeline (BERT+LLM) | 0.69 |
| **Sprint 1** | 1 Woche | KROMA Matcher | 0.74 (+5%) |
| **Sprint 2** | 1 Woche | + DeepOnto | 0.78 (+9%) |
| **Sprint 3** | 1 Woche | + AML + Aggregation | 0.82 (+13%) |
| **Sprint 4** | 1 Woche | + LLM Reranking | **0.85 (+16%)** |
| **Sprint 5** | 1 Woche | Tuning + Validation | **0.87 (+18%)** |

---

## Wichtige Dateien

| Datei | Zweck | Status |
|-------|-------|--------|
| `REFACTORING_PLAN.md` | Detaillierter Plan | ✓ Erstellt |
| `matchers/kroma_matcher.py` | S1000D DMC Heuristik | → Als nächstes |
| `matchers/base_matcher.py` | Abstract Base Class | → Als nächstes |
| `data_loader.py` | Wird beibehalten | ✓ Vorhanden |

---

## Fragen?

1. **Soll ich KROMA jetzt implementieren?**
   - Ja → Ich erstelle `matchers/kroma_matcher.py` + Tests
   - Nein → Ich erkläre erst das Design genauer

2. **DeepOnto Alternative gewünscht?**
   - Falls DeepOnto zu komplex: Ich kann einen "bert_matcher_v2.py" mit Hierarchie-Awareness bauen

3. **Andere Priorität?**
   - Z.B. erst AML testen, dann KROMA?

Sagen Sie mir, womit ich starten soll!
