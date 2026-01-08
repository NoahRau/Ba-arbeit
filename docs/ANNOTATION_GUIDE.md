# Anleitung zur manuellen Annotation

## 📋 Was wurde generiert?

**Datei:** `hybrid_pipeline_matches.csv`

Die Hybrid Pipeline hat **alle 56 S1000D-Konzepte** verarbeitet und folgende Ergebnisse erzielt:

- ✅ **9 Matches gefunden** (16.1%)
- ❌ **47 No Match / NULL** (83.9%)

Die Pipeline ist sehr konservativ (hohe Präzision), daher sind viele Matches als NULL markiert.

---

## 📝 Ihre Aufgabe: Manuelle Annotation

### Schritt 1: CSV-Datei öffnen

Öffnen Sie `hybrid_pipeline_matches.csv` in Excel oder Google Sheets.

### Schritt 2: Spalten verstehen

**Wichtige Spalten:**

| Spalte | Beschreibung |
|--------|--------------|
| `s1000d_label` | S1000D Konzept (z.B. "Bicycle - Description") |
| `s1000d_context` | Hierarchischer Kontext des S1000D-Konzepts |
| `pipeline_selected_uri` | Von der Pipeline gewählter Match (oder "NULL") |
| `selected_label` | Label des gewählten Matches |
| `pipeline_confidence` | Confidence-Score der Pipeline (0-1) |
| `candidate_1_uri` bis `candidate_5_uri` | Top-5 alternative Kandidaten |
| `candidate_1_label` bis `candidate_5_label` | Labels der Kandidaten |
| `candidate_1_score` bis `candidate_5_score` | Aggregierte Scores |
| **`is_match_manual`** | ← **HIER EINTRAGEN!** |
| `correct_match_uri` | ← Optional: Korrekter Match wenn FALSE |
| `notes` | ← Optional: Ihre Notizen |

### Schritt 3: Annotation durchführen

Für **jede Zeile** (56 insgesamt):

#### Fall 1: Pipeline hat Match gefunden (`pipeline_selected_uri` ≠ "NULL")

**Frage:** Ist `selected_label` korrekt für `s1000d_label`?

- ✅ **Ja, korrekt** → Eintragen: `TRUE`
- ❌ **Nein, falsch** → Eintragen: `FALSE`
  - Optional: In `correct_match_uri` den richtigen URI aus `candidate_1_uri` bis `candidate_5_uri` eintragen
  - Optional: In `notes` Begründung schreiben

**Beispiel:**
```
s1000d_label: "Bicycle - Description of how it is made"
selected_label: "Bike"
→ Passt "Bike" zu "Bicycle"? → JA → is_match_manual = TRUE
```

#### Fall 2: Pipeline hat KEINEN Match gefunden (`pipeline_selected_uri` = "NULL")

**Frage:** Sollte es einen Match geben?

- ✅ **Ja, es sollte einen Match geben** → Eintragen: `FALSE` (Pipeline hat Fehler gemacht)
  - Optional: In `correct_match_uri` den richtigen URI aus `candidate_1_uri` bis `candidate_5_uri` eintragen
  - Optional: In `notes` schreiben "Should match candidate X"

- ❌ **Nein, NULL ist korrekt** → Eintragen: `TRUE` (Pipeline hatte Recht)

**Beispiel:**
```
s1000d_label: "Mountain bicycle - Business rules"
selected_label: (leer, weil NULL)
→ Gibt es ein passendes Konzept in den Kandidaten? → NEIN → is_match_manual = TRUE (NULL ist korrekt)
```

**Beispiel 2:**
```
s1000d_label: "Wheel - Description"
selected_label: (leer, weil NULL)
candidate_1_label: "Wheel"
→ "Wheel" passt perfekt! → is_match_manual = FALSE (Pipeline hätte matchen sollen)
→ correct_match_uri = http://purl.org/ontology/bikeo#Wheel
```

---

## ✏️ Annotation-Regeln

### Was ist ein MATCH?

✅ **Match = TRUE**, wenn:
- Beide Konzepte **exakt dasselbe** repräsentieren
- Funktionale Äquivalenz gegeben ist
- Beispiele:
  - "Bicycle" ↔ "Bike" ✓
  - "Wheel" ↔ "Wheel" ✓
  - "Brake System - Description" ↔ "Brake" ✓ (wenn Kontext passt)

❌ **Match = FALSE**, wenn:
- Nur verwandt, aber nicht identisch
- Parent-Child Beziehung (z.B. "Wheel" ≠ "Hub")
- Geschwister (z.B. "Front Brake" ≠ "Rear Brake")
- Verschiedene Aspekte (z.B. "Maintenance Procedure" ≠ "Description")

### Bei Unsicherheit

- Schauen Sie sich **alle 5 Kandidaten** an
- Lesen Sie den **Kontext** (`s1000d_context`)
- Im Zweifel: Konservativ sein (lieber FALSE)

---

## 💾 Speichern

Nach der Annotation:
1. Speichern Sie die Datei als `hybrid_pipeline_matches_ANNOTATED.csv`
2. Stellen Sie sicher, dass die Spalte `is_match_manual` für **alle 56 Zeilen** ausgefüllt ist

---

## 🔬 Evaluation

Nach der Annotation wird das Evaluation-Script ausgeführt:

```bash
python evaluate_annotated_matches.py hybrid_pipeline_matches_ANNOTATED.csv --plot
```

Das generiert:
- **Evaluation Report** (Markdown mit MCC, F1, Precision, Recall)
- **Confusion Matrix** (Visualisierung)
- **Error Analysis** (Welche Fehler hat die Pipeline gemacht?)

---

## 📊 Beispiel-Zeilen

### Beispiel 1: Pipeline korrekt (Match gefunden)
```csv
s1000d_label: "Bicycle - Description of function"
selected_label: "Bike"
pipeline_confidence: 0.95
→ is_match_manual: TRUE  ← Korrekt!
```

### Beispiel 2: Pipeline korrekt (NULL)
```csv
s1000d_label: "Mountain bicycle - Business rules"
selected_label: (NULL)
→ is_match_manual: TRUE  ← Korrekt, es gibt kein passendes Konzept
```

### Beispiel 3: False Positive
```csv
s1000d_label: "Lighting - Maintenance"
selected_label: "Bike"
→ is_match_manual: FALSE  ← Falsch! "Bike" passt nicht zu "Lighting"
→ notes: "Should be NULL or Light-related concept"
```

### Beispiel 4: False Negative
```csv
s1000d_label: "Wheel - Description"
selected_label: (NULL)
candidate_1_label: "Wheel"
→ is_match_manual: FALSE  ← Pipeline hat Fehler gemacht
→ correct_match_uri: http://purl.org/ontology/bikeo#Wheel
→ notes: "Candidate 1 is perfect match"
```

---

## ⏱️ Zeitaufwand

- **Geschätzte Dauer:** 20-30 Minuten für 56 Konzepte
- **Pro Zeile:** ~30 Sekunden

---

## ❓ Fragen?

Bei Unklarheiten:
1. Schauen Sie sich die Top-5 Kandidaten an
2. Lesen Sie den Kontext
3. Nutzen Sie die `notes` Spalte für Unsicherheiten

**Viel Erfolg bei der Annotation!** 📝
