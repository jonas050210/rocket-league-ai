# Implementierungsbericht – Nexto-Gen2-Ausbau

Stand: 19.10.2026 · Branch `arena/a9c58dba-rocket-league-ai` · Commit `38b7710`

Umgebung der Messungen: Debian 12, 1 physischer Kern / 2 vCPU, 3,8 GiB RAM,
**keine GPU**, kein Rocket League, kein Windows. Torch 2.9.1+cu128 (CPU-only),
RocketSim 2.2.1, RLGym-Rocket-League 2.0.1, RLBot 2.0.0b56 (installiert,
aber nicht startbar).

---

## 1. Umgesetzte Merkmale

### 1.1 Trainingszeit (verlässlich gemessen)

| Baustein | Datei | Inhalt |
| --- | --- | --- |
| Zeitmessung | `src/rlai/timing.py` (neu) | `SessionTimer` (monotone Uhr + UTC-Start/-Ende), `TrainingClock` (akkumulierte Summe über Sitzungen, `restore()`/`adopt()`), `format_duration`, `merge_durations`, `duration_summary`, Status-Konstanten |
| Persistenz | `src/rlai/store.py` | Tabelle `sessions` (Status `completed`/`stopped`/`interrupted`/`inherited`/`running`), `start_session`/`finish_session`/`list_sessions`/`session_totals`/`recover_interrupted_sessions`; Run-Spalten `duration_s`, `session_duration_s`, `last_checkpoint_at`, `last_checkpoint_id` |
| Checkpoint | `src/rlai/checkpoints.py` | `extra.timing = {accumulated_s, session_s, session_started_at, measured_at, source}`, Manifest `duration_s`/`session_duration_s`, `CheckpointValidation.has_timing/duration_s/session_duration_s` |
| Laufzeit | `src/rlai/trainer.py` | Sitzung vor dem ersten Schritt öffnen, in `finally` schließen, Absturz-Sitzungen beim nächsten Start als *interrupted* markieren (Dauer **unbekannt**), geerbte Zeit als eigene Sitzung mit Status `inherited` sichtbar machen |
| Bericht | `src/rlai/report.py` | eigener Timing-Abschnitt: gemessene Zeit, Sitzungsliste, Wanduhr getrennt ausgewiesen, aktuelle Sitzung, Hinweis „unbekannt“ bei Altläufen |

Regeln, die eingehalten werden: monotone Uhr für jede Dauer, UTC nur für
Zeitpunkte, **kein** Zählen der Standby-Zeit zwischen zwei Sitzungen, keine
Doppelzählung (ein Checkpoint trägt höchstens die bis dahin gemessene Zeit),
Alt-Checkpoints bleiben ladbar und melden die historische Dauer als
*unbekannt* statt als 0.

### 1.2 Statistik (Stats-Tab) und Performance

* `src/rlai/stats.py` (neu): **eine** Quelle für alle Zahlen des Stats-Tabs –
  `timing_block`, `progress_block`, `learning_block`, `curves_block`,
  `performance_block`, `checkpoint_block`, `evaluation_block`,
  `checkpoint_comparison`, `build_stats`. GUI, Report und Headless-Bericht
  können nicht auseinanderlaufen.
* GUI: neue Ansichten **Stats**, **Lab**, **Teams**, **Memory**
  (`src/rlai/web/stats.js`, `src/rlai/web/features.js`).
* Nicht gemessene Werte erscheinen als *unavailable* mit Begründung, nie
  als 0. Alte Checkpoints zeigen „unknown“.
* Kurven werden auf 400 Punkte heruntergesampelt, Metrik-Schreibvorgänge
  gebündelt (`metric_flush_interval`), Metrik-/Log-/Episoden-Tabellen sind
  begrenzt (`metric_retention`, `log_retention`, `episode_retention`).
* Essenzielle Metriken, Fehlerlogs, Checkpoints und die Zeitmessung hängen
  **nicht** an der GUI-Überwachung.

### 1.3 Ausführungsmodi

| | GUI | Headless |
| --- | --- | --- |
| Start | `POST /api/runs` mit `confirm=true` (Dialog fragt vorher) | `rlai headless --config-file …` |
| Webdienst | ja | **nein** |
| Abschluss | Report + Ansichten | `rlai-headless-summary-v1` (Zeit, Schritte, Durchsatz, Lernmetriken, Episoden, Evaluation, Checkpoint, Abbruchstatus) |
| Logik | identisch (Trainer, Checkpoints, Zeitmessung) | identisch |

`POST /api/lab/run` verlangt `execute=true`, `POST /api/lab/plan` startet nie
eine Simulation.

### 1.4 Die fehlenden Merkmale (funktional, nicht dekorativ)

| Merkmal | Umsetzung | Grenze (dokumentiert) |
| --- | --- | --- |
| Bot Brain | `src/rlai/botbrain.py` – vier Profile, deterministisch, simulatorfrei | beschreibt Verhalten, nicht Stärke |
| Schwierigkeitsprofile | `src/rlai/difficulty.py` – Rock/Tiny/Medium/Pro mit Zielwert **und** Mechanismus; `difficulty:<level>` als Gegner | Zielwerte 0/50/75/100 % sind **nicht gemessen** (`target_is_measured = false`) |
| Team Builder | `src/rlai/team_builder.py` – Bot-Slots, Validierung mit klarer Meldung, Speicherung in eigener Tabelle | beschreibt Teams; ausgeführt wird der Gegner-Code |
| Spielerspeicher | `src/rlai/player_memory.py` – eigene Tabelle, Grenzen (500 Einträge / 90 Tage), Notizen, Export/Import | reine Daten, nie Teil der Gewichte |
| Rival-System | `src/rlai/rivals.py` – benannte Gegner, Historie, Rangfolge, Engpass, Trend | nur gemessene Evaluationen |
| Adaptive Match Dynamics | `src/rlai/adaptive.py` – Grenzen, Hysterese, Mindest-Spielzahl/-Zeit, Zustandswiederherstellung; die neuen Parameter werden an die Worker übergeben und **wirken sich aus** | verändert nie eine trainierte Policy; ohne Schwierigkeitsstufe wird nicht angepasst |
| Match-/Replay-Analyse | `src/rlai/replay.py` – wertet die protokollierten Spiel- und Episodenzeilen aus | echte `.replay`-Dateien sind hier **nicht lesbar**; das UI sagt das |
| Bot Evaluation Lab | Plan → Bestätigung → Lauf → Bericht, Checkpoint-Vergleich | ein Lauf startet nur mit `execute=true` |
| Ressourcenwächter | `src/rlai/resources.py` – CPU-/Speicherbudget, Warnungen, Klemmen der Workerzahl | Empfehlung, keine Garantie |

### 1.5 Korrektheit und Zuverlässigkeit

* Benchmark-Standardmodus ist gültig (`soccer-1v1` statt `soccar-1v1`).
* Gegnernamen werden **vor** dem Start der Worker geprüft; „self“ fällt auf
  einen skriptgesteuerten Gegner zurück, statt einen unauflösbaren Namen zu
  reichen.
* Der Torch-Thread-Budget wird explizit gesetzt (vor der ersten Rollout) und
  auf den Worker begrenzt; das Modul-Level-`set_num_threads(1)` ist weg.
* Die Evaluation baut die RocketSim-Arena **pro Seite einmal** und schließt
  sie.
* Der globale RNG-Zustand wird um `env.reset()` gerettet und wiederhergestellt.
* Curriculum-Stufen validieren ihre Schwellen (Siegquote 0–1, Fenster,
  Mindest-Episoden, bekannter Gegner).
* Der Gegnerpool lehnt Near-Duplicate-Snapshots ab (Schritt- und
  Distanztoleranz).
* CPU-/GPU-Monitoring bezieht sich auf den **Trainerprozess** (`psutil`,
  `nvidia-smi` pro Prozess).
* Checkpoints werden vor dem Laden auf Architektur, Modellzustand,
  Optimizerzustand und Resume-Kompatibilität geprüft.
* Transfer Learning: passende Tensoren werden wiederverwendet, der Rest neu
  initialisiert und **gemeldet**.
* Beim Resume bleiben RNG, Curriculum, Gegnerpool, Optimizer und
  Iterationszähler erhalten.

### 1.6 RLBot

* **Gefundener Fehler:** RLBot sortiert die Boost-Pads nach (y, x), RocketSim
  nicht. Die Permutation `[0..12, 14, 13, 15..18, 20, 19, 21..33]` ist
  gemessen und in `src/rlai/rlbot/pad_order.py` dokumentiert.
* Der Adapter rechnet die Pads um und **verweigert** den Betrieb bei einer
  anderen Pad-Anzahl; der Bot warnt bei abweichendem Stadion.
* `tests/test_boost_pad_order.py` (10 Tests) prüft die Permutation, ihre
  Umkehrung, die Übereinstimmung mit der Live-Arena und den Adapter.

---

## 2. Geänderte Bereiche

| Bereich | Dateien | Art |
| --- | --- | --- |
| Zeitmessung | `timing.py` (neu), `store.py`, `checkpoints.py`, `trainer.py`, `report.py` | neu/erweitert |
| Statistik | `stats.py` (neu), `api/app.py`, `web/stats.js` (neu), `web/features.js` (neu), `web/app.js`, `web/index.html`, `web/web.css` | neu/erweitert |
| Merkmale | `botbrain.py`, `difficulty.py`, `team_builder.py`, `player_memory.py`, `rivals.py`, `adaptive.py`, `replay.py` (alle neu) | neu |
| Betrieb | `cli.py` (`headless`, `benchmarks`), `config.py`, `opponents.py`, `workers.py`, `evaluation.py`, `envs.py`, `curriculum.py`, `league.py`, `hardware.py`, `resources.py` (neu), `benchmarks.py` (neu) | erweitert |
| RLBot | `rlbot/pad_order.py` (neu), `rlbot/obs_adapter.py`, `rlbot/bot.py` | korrigiert |
| Dokumentation | `docs/ARCHITECTURE.md`, `docs/TRAINING.md`, `docs/TESTING.md`, `docs/IMPLEMENTATION_REPORT.md` (neu) | aktualisiert |
| Tests | 10 neue Testdateien, `tests/gui/smoke.mjs`, `tests/packet_utils.py`, `tests/test_api.py` | erweitert |

---

## 3. Tests und Messungen

### 3.1 Automatisierte Tests

```
python -m pytest -q          →  464 passed, 0 failed, 0 skipped  (2 min, 2 vCPU)
ruff check src tests rlbot   →  All checks passed
cd tests/gui && node smoke.mjs →  all GUI smoke checks passed
```

Vorher: 289 Tests. Neu hinzugekommen (10 Dateien, 175 Tests):
`test_timing.py` (18), `test_trainer_timing.py` (8), `test_stats.py` (26),
`test_features.py` (54), `test_analysis.py` (16), `test_resources.py` (13),
`test_benchmarks.py` (6), `test_api_features.py` (25), `test_cli_modes.py` (7),
`test_boost_pad_order.py` (10).

### 3.2 Gemessene Trainingszeit

| Lauf | Art | Schritte | Sitzungen | Akkumuliert |
| --- | --- | --- | --- | --- |
| `a69a85a6b4dc` | fresh | 0 → 2 048 | 1 × `completed` (4,41 s) | 4,41 s |
| `run-b` | continue | 2 048 → 4 096 | `inherited` 4,11 s + `completed` 4,81 s | 8,91 s |

### 3.3 Benchmarks (`rlai benchmarks`)

| Benchmark | Ergebnis |
| --- | --- |
| Schwierigkeit (Ballberührungen) | rock 28 · tiny 27 · medium 36 · pro 42 (Spreizung 15) |
| Arena-Wiederverwendung (4 Spiele) | 0,518 s mit Wiederverwendung (inkl. Simulation) · 0,144 s allein der Arena-Aufbau |
| Torch-Threads (Forward-Pass) | 1 Thread 2 693/s · 2 Threads 2 900/s · 4 Threads 956/s |

Der Thread-Benchmark zeigt, warum der Thread-Budget nicht erhöht wird: bei
diesem kleinen MLP kostet der Wechsel auf vier Threads mehr, als er bringt.

### 3.4 Adaptive Match Dynamics (60 000-Schritte-Smoke)

Gemessene Siegquote 0 % gegen Ziel 20 % → Gegner wurde leichter gestellt:
`mistake_rate` 0,18 → 0,38, `reaction_delay` 0,25 s → 0,35 s,
`boost_efficiency` 0,75 → 0,55. Die Werte stehen als Metriken
(`adaptive/*`) in der Datenbank und im Checkpoint-Zustand.

### 3.5 Nicht ausgeführt (und warum)

| Punkt | Grund |
| --- | --- |
| Echter RLBot-Match | kein Windows, kein Rocket League; nur Adapter-Parität mit synthetischen Paketen und die Live-Arena-Prüfung |
| GPU-Pfad | keine GPU vorhanden (`torch.cuda.is_available() == False`) |
| Langzeittraining / Spielstärke | bewusst nicht gestartet (Ressourcen, keine belastbare Basis) |
| Validierung der Schwierigkeits-Zielwerte | braucht je Stufe eine vollständige Evalutionsserie |
| Replay-Parsing | kein Parser verfügbar |

---

## 4. Bekannte Grenzen

1. Die vier Schwierigkeits-Prozentzahlen sind **Zielwerte**, keine Messung.
   Gemessen wurde nur, dass der Mechanismus die Spielweise verändert.
2. Echte Rocket-League-Replays können hier nicht gelesen werden; die
   Analyse nutzt die protokollierten Spiel- und Episodenzeilen.
3. Kein echter RLBot-Match wurde getestet – nur Mocks, synthetische Pakete
   und ein Vergleich mit der Live-Arena.
4. Der GPU-Pfad ist ungetestet (keine CUDA-Geräte).
5. Die Box ist zu klein für aussagekräftiges Training; alle Durchsatzzahlen
   sind Smoke-Maßstab, kein Langlaufnachweis.
6. Aus der Trainingsoptimierung folgt **keine** Aussage über Spielstärke.
7. Beim Ende eines Laufs wird der Report geschrieben; die Auswertung der
   Kurven über einen Abbruch hinweg ist nicht implementiert.

---

## 5. Empfohlene nächste Schritte

1. **Schwierigkeitsziele validieren:** je Stufe eine Serie von ≥ 50 Spielen
   gegen eine feste Referenz und das Ergebnis im GUI als „gemessen“ markieren
   (`target_is_measured` dann `true`).
2. **Replay-Unterstützung:** entweder einen Parser für `.replay` ergänzen
   oder die Aussage im UI als dauerhafte Grenze formulieren.
3. **RLBot-Abnahme:** auf einem Windows-Rechner einen echten Match gegen den
   Bot fahren und die Pad-Reihenfolge im Live-Paket verifizieren.
4. **Längere Läufe:** auf einer Maschine mit ≥ 8 Kernen und 16 GB RAM einen
   Lauf mit festem Seed und mehreren Konfigurationen (Rollout, Minibatch,
   Worker) vergleichen; der Baseline-Vergleich (`rlai benchmark`) liefert
   die Vergleichsgrößen.
5. **Speicherbedarf messen:** die Schätzung von 700 MB je Worker in
   `resources.py` ist eine grobe Annahme; ein Messlauf über mehrere
   Worker-Zahlen würde sie ersetzen.
6. **Auswertung über Abbruchgrenzen:** Kurven und Kennzahlen eines
   abgebrochenen Laufs in der GUI zusammenführen.
