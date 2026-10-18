# Rocket League AI

Lokale, reproduzierbare Trainings- und Deployment-Pipeline für Rocket-League-Bots:
RocketSim-Simulation, diskretes PPO, vier Spielmodi, Web-Dashboard und ein
RLBot-v5-Adapter. Alles läuft auf einem Rechner; GPU ist optional.

**Ehrlicher Status:** Training, Evaluation, Modellverwaltung und Dashboard sind
implementiert und in dieser Umgebung **mit echten Simulationen getestet**.
Ein Live-Match in Rocket League konnte hier **nicht** ausgeführt werden
(Linux, kein Spiel installiert). Der RLBot-Adapter ist über Paritätstests und
einen synthetischen `GamePacket` abgesichert, nicht über ein echtes Spiel.
Details: `docs/TESTING.md`.

## Was drin ist

| Bereich | Zustand | Nachweis |
| --- | --- | --- |
| 1v1 / 2v2 / 3v3 / Heatseeker 2v2 | implementiert | `tests/test_env_modes.py`, `tests/test_heatseeker.py` |
| PPO-Training, Checkpoints, Resume | implementiert | `tests/test_ppo.py`, Smoke-Runs über die API |
| Parallele Worker (Multiprocessing) | implementiert | 2-Worker-Runs für 3v3 und Heatseeker |
| Evaluation mit festen Seeds | implementiert | `tests/test_evaluation.py` |
| Modell-Registry, Promotion, Schutzregeln | implementiert | `tests/test_checkpoints.py` |
| Dashboard + Steuerung + Logs | implementiert | 7 Routen headless gerendert, API-Tests |
| RLBot-v5-Adapter | implementiert, **nicht im Spiel verifiziert** | `tests/test_rlbot_adapter.py` |
| Windows-/Epic-Anleitung | dokumentiert | `docs/SETUP_WINDOWS.md` |

## Schnellstart (Linux/macOS)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"      # torch, RLGym, RocketSim, FastAPI, pytest
python -m rlai.cli serve --port 8000
```

Danach `http://localhost:8000` öffnen. Training startet **nie** automatisch –
es muss im Tab *Training* explizit gestartet werden.

Erster sinnvoller Lauf: Preset **1v1 vs chaser** oder eine kleine eigene
Konfiguration. Ein Smoke-Test dauert wenige Sekunden:

```bash
python -m rlai.cli train --config-file configs/smoke-1v1.json
python -m rlai.cli evaluate --model data/models/<id>/model.pt --games 2 --match-seconds 20
python -m rlai.cli benchmark --mode heatseeker-2v2 --workers 2 --seconds 2
```

Unter Windows: `powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1`
(siehe `docs/SETUP_WINDOWS.md`).

## Spielmodi

| ID | Autos | Regeln |
| --- | --- | --- |
| `soccer-1v1` | 2 | Standard-Soccar, ein Auto pro Team |
| `soccer-2v2` | 4 | Standard-Soccar, geteilte Policy für beide blauen Autos |
| `soccer-3v3` | 6 | Standard-Soccar, Beobachtung auf drei Slots pro Team gepaddet |
| `heatseeker-2v2` | 4 | Homing-Ball, Tempoplus pro Touch, unendlich Boost |

Heatseeker wird von RocketSim nativ unterstützt (`GameMode.HEATSEEKER`).
Verifiziert wurde: Berührung setzt `heatseeker_target_dir` auf `+1` (blau → oranges
Tor), Zieltempo steigt von 2900 auf 2985 uu/s, der Ball fliegt ins Tor und wird
als Tor für Blau gewertet (`tests/test_heatseeker.py`). Die Beobachtung ist in
allen Modi **identisch 190-dimensional**; ein Checkpoint kann daher als
Initialisierung für größere Teams dienen.

## Beobachtung, Aktionen, Belohnung

* Beobachtung: 190 Werte – eigener Zustand, Ball (relativ im Auto-Frame),
  2 Teamkollegen- und 3 Gegner-Slots (gepadet), 34 Boost-Pads, Restzeit,
  Score-Differenz und die letzten 90 Aktionen als One-Hot.
* Aktionen: 90 diskrete Lookup-Table-Aktionen (Gas/Bremse, Lenken, Boost,
  Handbremse, Sprung/Flip, Luftsteuerung), identisch in Training, Evaluation
  und RLBot-Adapter.
* Belohnung: Tore (+10) und Gegentore (−10) dominieren; Potenzial-Shaping für
  Ballfortschritt und Ausrichtung, kleine Boni für Touch, Boost-Effizienz und
  Verteidigung. Alle Gewichte sind pro Lauf konfigurierbar und werden geloggt.
  Beim Tor wird kein Distanz-Shaping angewendet, damit die Torprämie nicht
  teilweise aufgehoben wird.

## Training

* PPO mit GAE, Advantage-Normalisierung, Gradient-Clipping, LR-Annealing,
  orthogonaler Initialisierung; Actor und Critic getrennt.
* Jeder Worker ist ein eigener Prozess mit einer RocketSim-Arena; der Trainer
  sammelt Rollouts, trainiert zentral und schreibt Metriken in SQLite.
* Checkpoints sind atomar und unveränderlich; der stabile Zeiger wird nur
  explizit gesetzt. Der letzte Checkpoint ist gegen Löschen geschützt.
* Resume nach Unterbrechung lädt Modell, Optimizer und Schrittstand.
* Selbstspiel nutzt eingefrorene Snapshots als Gegnerpool (optional).
* Alle Kennzahlen im Dashboard kommen aus der Datenbank des laufenden Prozesses –
  keine erfundenen Werte.

Details: `docs/TRAINING.md`.

## Web-Dashboard

Sieben Ansichten: Dashboard, Training, Monitoring, Models, Evaluation, Live match,
Settings. Es gibt keine Dummy-Buttons: nicht verfügbare Aktionen sind deaktiviert
und erklären sich, fehlende Abhängigkeiten werden als Diagnose angezeigt.
Live-Matches sind außerhalb von Windows deaktiviert (Serverprüfung, kein Fake).

## Tests

```bash
python -m pytest -q
```

Ergebnis in dieser Umgebung: **76 passed** in ca. 4,6 s (2 vCPU, kein GPU, kein
Spiel). Abgedeckt: Modi, Aktionen, Beobachtung/Parität, Belohnungen, PPO,
Checkpoint-Schutz, Store, API, Evaluation, Heatseeker-Regeln. Nicht abgedeckt:
echtes Rocket League, RLBot-Start, GPU-Pfad.

## Grenzen

* Kein Nachweis von Spielstärke gegenüber Nexto oder einem bestimmten Rang.
  Die gezeigten Zahlen sind Smoke-Tests, keine trainierten Fähigkeiten.
* Live-Deployment ist ungetestet, weil hier kein Windows/Spiel/RLBot-Lauf
  möglich ist. Die Anleitung nennt exakte Schritte und bekannte Risiken.
* 2 vCPU bedeutet: 3v3/Heatseeker mit 8 Workern sind auf dieser Maschine
  langsamer als die Workerzahl suggeriert; das Dashboard zeigt die echte Rate.
* `data/` ist Laufzeitverzeichnis und nicht versioniert.

## Dokumentation

* `docs/MODES.md` – Regeln, Beobachtung, Aktionen, Belohnung pro Modus
* `docs/TRAINING.md` – PPO, Curriculum, Metriken, Resume, Selbstspiel
* `docs/RLBOT.md` – v5-Adapter, Konfiguration, Grenzen
* `docs/SETUP_WINDOWS.md` – Windows + Epic Schritt für Schritt
* `docs/TESTING.md` – Testmatrix, Messwerte, was nicht lief
* `docs/ARCHITECTURE.md` – Module und Datenfluss
