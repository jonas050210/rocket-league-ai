# Architektur

```
Browser (HTML/CSS/JS, ui/theme)
        │  fetch / SSE
        ▼
FastAPI (src/rlai/api/app.py)
        │
        ├── SQLite-Store (Runs, Metriken, Logs, Modelle, Evaluationen)
        ├── JobManager (Subprozesse: train, evaluate, live)
        └── Hardware/Diagnose (psutil, torch, Import-Checks)
                 │
                 ▼
        Trainer (src/rlai/trainer.py)
        ├── PPO-Update (src/rlai/ppo.py)
        ├── Worker-Pool (src/rlai/workers.py, multiprocessing)
        │      └── MatchEnv (src/rlai/envs.py)
        │             ├── RLGym + RocketSim
        │             ├── SoccerObsBuilder (src/rlai/obs.py)
        │             └── SoccerReward (src/rlai/rewards.py)
        └── Checkpoints (src/rlai/checkpoints.py)
                 │
                 ├── Evaluation (src/rlai/evaluation.py)
                 └── RLBot-Bot (src/rlai/rlbot/*, rlbot/bot.py)
```

## Datenfluss

1. Der Browser startet einen Lauf über `POST /api/runs`. Der Server validiert
   die Konfiguration (Pydantic) und legt einen Run in SQLite an.
2. `JobManager` startet `python -m rlai.cli train` als Subprozess mit
   `PYTHONPATH=<repo>/src` und `RLAI_DATA_DIR`.
3. Der Trainer erzeugt Worker, verteilt Gewichte, sammelt Rollouts und
   schreibt Metriken, Episoden und Logs.
4. Das Dashboard liest Metriken per HTTP und abonniert `/api/events` für
   Live-Updates. Es gibt keinen zweiten Kanal und keine erfundenen Werte.
5. Checkpoints werden atomar gespeichert. Promotion ist ein expliziter
   Schreibvorgang auf `stable.json`.
6. Evaluation lädt einen Checkpoint und spielt deterministische Serien gegen
   `idle`, `random`, `chase` oder ein anderes Modell.

## Verzeichnisse

| Pfad | Inhalt |
| --- | --- |
| `src/rlai/` | Python-Paket (API, Training, Simulation, RLBot) |
| `src/rlai/web/` | Dashboard (statisch, kein Build-Schritt) |
| `ui/theme/` | Designsystem (unverändert aus Phase 1) |
| `configs/` | vollständige Trainingskonfigurationen |
| `data/` | Laufzeitdaten: SQLite, Runs, Modelle, Live-Matches (nicht versioniert) |
| `tests/` | pytest-Suite |
| `docs/` | Betriebs- und Architektur-Dokumentation |
| `scripts/` | Setup- und Startskripte |

## Threading und Blockierung

* FastAPI-Endpunkte sind synchron, außer dem SSE-Generator; ein Training
  blockiert die API nicht, weil es in einem Subprozess läuft.
* Der Trainer selbst ist single-threaded; die Simulation läuft in Workern.
* Das Frontend führt Rendering und Netzwerkzugriffe asynchron aus; es gibt
  keine blockierenden Schleifen im UI-Thread.

## Sicherheit und Schutzregeln

* Optionale Capability-Token (`RLAI_AUTH_TOKEN`) für alle schreibenden
  Endpunkte; Lesen bleibt lokal offen.
* Checkpoints sind unveränderlich; der letzte Checkpoint und das stabile
  Modell sind gegen Löschen geschützt.
* Kein automatisches Starten von Training, kein automatisches Promoten.
