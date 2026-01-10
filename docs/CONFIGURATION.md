# Pipeline Configuration Guide

Die Hybrid Pipeline kann über die zentrale `config.py` Datei konfiguriert werden.

## 📍 Konfigurationsdatei

**Datei:** `config.py` (im Projekt-Root)

## ⚙️ Wichtige Parameter

### 1. Matcher Gewichtungen

```python
MATCHER_WEIGHTS = {
    'kroma': 0.45,      # DMC-basierte Heuristiken (erhöht von 0.40)
    'deeponto': 0.35,   # BERT semantische Ähnlichkeit
    'string': 0.20      # String-basierte Ähnlichkeit (reduziert von 0.25)
}
```

**Optimierungen:**
- ✅ **KROMA auf 0.45** erhöht - DMC-Heuristiken sind sehr präzise für S1000D
- ✅ **String auf 0.20** reduziert - String-Matching ist weniger zuverlässig allein

**Wie anpassen:**
1. Öffne `config.py`
2. Ändere die Werte in `MATCHER_WEIGHTS`
3. Stelle sicher, dass die Summe ≈ 1.0 ist (wird automatisch normalisiert)

### 2. Top-K Einstellungen

```python
PIPELINE_CONFIG = {
    'matcher_top_k': 10,        # Jeder Matcher gibt Top-10 zurück
    'aggregation_top_k': 5,     # Aggregator kombiniert zu Top-5
    'use_llm': True,            # LLM Reranking aktiviert
    'aggregation_method': 'rank_fusion'  # 'rank_fusion' oder 'weighted_sum'
}
```

**Was bedeutet das:**
- **matcher_top_k=10**: Jeder Matcher (KROMA, DeepOnto, String) findet die Top-10 Kandidaten
- **aggregation_top_k=5**: Der Aggregator kombiniert diese und gibt die besten 5 zurück
- Diese 5 werden dann an das LLM gesendet

**Wie anpassen:**
- **Mehr Kandidaten für LLM:** `aggregation_top_k` auf 7 oder 10 erhöhen
- **Schnellere Verarbeitung:** `matcher_top_k` auf 5 reduzieren
- **Mehr Recall:** `aggregation_top_k` erhöhen

### 3. LLM Confidence Threshold

```python
LLM_CONFIG = {
    'model': 'claude-sonnet-4-5-20250929',
    'confidence_threshold': 0.90,  # Reduziert von 0.95 für mehr Recall
    'temperature': 0.0,             # Deterministisch
    'max_tokens': 1024,
    'enabled': True
}
```

**Optimierungen:**
- ✅ **Threshold auf 0.90** reduziert - von 0.95 auf 0.90
- Dies verbessert **Recall** (findet mehr Matches) mit minimal reduzierter **Precision**

**Trade-offs:**
- **Höherer Threshold (0.95-1.0)**: Sehr konservativ, hohe Precision, niedrige Recall
- **Mittlerer Threshold (0.85-0.95)**: Balance zwischen Precision und Recall ⭐ **Empfohlen**
- **Niedriger Threshold (0.70-0.85)**: Mehr Matches, aber mehr False Positives

**Empfohlene Werte:**
- **Für hohe Precision**: 0.95
- **Für Balance**: 0.90 ⭐ **(aktuell)**
- **Für hohe Recall**: 0.85

### 4. Aggregation Methode

```python
AGGREGATION_CONFIG = {
    'method': 'rank_fusion',  # 'rank_fusion' oder 'weighted_sum'
    'rrf_k': 60,              # Reciprocal Rank Fusion Konstante
    'top_k': 5
}
```

**Methoden:**
- **rank_fusion** ⭐ **(Empfohlen)**: Robuster gegen unterschiedliche Score-Skalen
- **weighted_sum**: Direkter gewichteter Score

**RRF Formel:**
```
score = sum(weight / (k + rank)) für jeden Matcher
```
- Höheres `rrf_k` → konservativere Fusion
- Standard: `k=60`

### 5. KROMA Matcher Konfiguration

```python
KROMA_CONFIG = {
    'dmc_chapter_keywords': {
        'D00': ['bicycle', 'bike', 'general', 'system'],
        'DA0': ['wheel', 'tire', 'hub', 'spoke', 'rim'],
        'DA1': ['steering', 'handlebar', 'stem', 'fork'],
        'DA2': ['braking', 'brake', 'caliper', 'disc', 'pad'],
        # ...
    },

    'score_weights': {
        'chapter_match': 0.40,      # Erhöht von 0.35
        'label_overlap': 0.25,
        'context_keywords': 0.20,
        'hierarchy_bonus': 0.15
    },

    'min_score': 0.3
}
```

**Optimierungen:**
- ✅ **chapter_match auf 0.40** erhöht - DMC Chapter ist sehr wichtig

**Wie eigene Keywords hinzufügen:**
```python
'DA6': ['new_component', 'keyword1', 'keyword2']
```

### 6. DeepOnto Matcher Konfiguration

```python
DEEPONTO_CONFIG = {
    'model_name': 'answerdotai/ModernBERT-base',
    'max_seq_length': 8192,
    'min_similarity': 0.5,

    # Ontologie-Reasoning Penalties
    'subsumption_penalty': 0.5,  # Parent-Child: -50% Score
    'sibling_penalty': 0.7,      # Geschwister: -30% Score

    # Gewichtung
    'label_weight': 0.6,
    'context_weight': 0.4
}
```

**Penalties:**
- **subsumption_penalty=0.5**: Parent-Child Beziehungen bekommen 50% des Scores
- **sibling_penalty=0.7**: Geschwister-Konzepte bekommen 70% des Scores

## 📊 Performance-Tuning

### Für mehr Precision (weniger False Positives):
```python
# config.py
MATCHER_WEIGHTS = {
    'kroma': 0.50,      # KROMA noch höher gewichten
    'deeponto': 0.35,
    'string': 0.15      # String weniger gewichten
}

LLM_CONFIG = {
    'confidence_threshold': 0.95  # Höherer Threshold
}
```

### Für mehr Recall (mehr Matches finden):
```python
# config.py
PIPELINE_CONFIG = {
    'matcher_top_k': 15,        # Mehr Kandidaten pro Matcher
    'aggregation_top_k': 7      # Mehr Kandidaten für LLM
}

LLM_CONFIG = {
    'confidence_threshold': 0.85  # Niedrigerer Threshold
}
```

### Für bessere Balance (aktuell):
```python
# config.py - AKTUELLE EINSTELLUNGEN
MATCHER_WEIGHTS = {
    'kroma': 0.45,
    'deeponto': 0.35,
    'string': 0.20
}

PIPELINE_CONFIG = {
    'matcher_top_k': 10,
    'aggregation_top_k': 5
}

LLM_CONFIG = {
    'confidence_threshold': 0.90
}
```

## 🔧 Änderungen übernehmen

Nachdem du `config.py` geändert hast:

1. **Keine Neuinstallation nötig** - Config wird zur Laufzeit geladen
2. **Script neu starten** - Änderungen werden beim nächsten Durchlauf aktiv
3. **Pipeline neu initialisieren** - Wenn bereits in Python geladen

```python
# In Python
from config import print_config_summary
print_config_summary()  # Aktuelle Config anzeigen
```

## 📈 Empfohlene Anpassungen basierend auf Ergebnissen

### Aktuelles Szenario (Precision 88.9%, Recall 61.5%):

**Problem:** Zu viele False Negatives (5), Pipeline ist zu konservativ

**Lösung:**
```python
# 1. Confidence Threshold senken
LLM_CONFIG['confidence_threshold'] = 0.85  # von 0.90

# 2. Mehr Kandidaten für LLM
PIPELINE_CONFIG['aggregation_top_k'] = 7  # von 5

# 3. KROMA noch höher gewichten (findet komponenten gut)
MATCHER_WEIGHTS['kroma'] = 0.50  # von 0.45
```

**Erwartete Verbesserung:**
- Recall: 61.5% → ~75%
- Precision: 88.9% → ~85%
- F1-Score: 72.7% → ~80%

## 🔍 Debugging

### Config-Werte anzeigen:
```bash
python config.py
```

### In der Pipeline prüfen:
```python
from src.pipeline.hybrid_pipeline import HybridPipeline

pipeline = HybridPipeline(s1000d_df, ontology_df)
print(f"Matcher Top-K: {pipeline.matcher_top_k}")
print(f"Aggregation Top-K: {pipeline.aggregation_top_k}")
print(f"LLM Enabled: {pipeline.use_llm}")
```

## 📝 Hinweise

- **Gewichte werden automatisch normalisiert** - müssen nicht exakt 1.0 sein
- **Änderungen gelten sofort** beim nächsten Pipeline-Durchlauf
- **Alte Embeddings werden gecached** - bei Modell-Änderung Cache löschen:
  ```bash
  rm cache/embeddings/deeponto_embeddings_cache.pkl
  ```

---

**Weitere Informationen:** Siehe `README.md` und `docs/IMPLEMENTATION_SUMMARY.md`
