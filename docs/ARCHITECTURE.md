# Architektur

```
Browser (HTML/CSS/JS, ui/theme)
        │  fetch / SSE
        ▼
FastAPI (src/rlai/api/app.py)
        │
        ├── SQLite-Store (Runs, Metriken, Episoden, Logs, Modelle,
        │                 Evaluationen, Reports, Settings)
        ├── JobManager (Subprozesse: train, evaluate, live)
        └── Hardware/Diagnose (psutil, torch, Import-Checks)
                 │
                 ▼
        Trainer (src/rlai/trainer.py)
        ├── Resume-Plan (src/rlai/resume.py: Kandidaten, Validierung, Fallback)
        ├── Report (src/rlai/report.py: Statistik nach jedem Lauf)
        ├── PPO-Update (src/rlai/ppo.py)
        ├── Gegnerauswahl (src/rlai/league.py, src/rlai/curriculum.py)
        ├── Periodische Evaluation (src/rlai/evaluation.py)
        ├── Worker-Pool (src/rlai/workers.py, multiprocessing)
        │      └── MatchEnv (src/rlai/envs.py)
        │             ├── RLGym + RocketSim
        │             ├── SoccerObsBuilder (src/rlai/obs.py, v1/v2)
        │             └── SoccerReward (src/rlai/rewards.py)
        └── Checkpoints + best-Zeiger (src/rlai/checkpoints.py)
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
   Schreibvorgang auf `stable.json`; `best.json` hält nur die beste gemessene
   Evaluation fest und deployt nichts.
6. Evaluation lädt einen Checkpoint und spielt deterministische Serien gegen
   `idle`, `random`, `chase`, ein anderes Modell oder eine Pool-Kopie. Sie
   meldet Siege, Tore, Ballberührungen und ein 95-%-Wilson-Intervall.
7. Bei aktivem Curriculum wählt der Trainer die Gegnerstufe anhand
   abgeschlossener Episoden; bei Selbstspiel sampelt der Pool nach
   PFSP-Gewichten. Beides wird im Checkpoint gespeichert und beim Resume
   fortgesetzt (inklusive RNG-Zuständen und, mit `league_states`, den
   Gewichten der Pool-Snapshots).
8. Vor einem Resume baut `resume.plan_resume` einen Plan: Kandidaten der
   Lauf-Linie, Kompatibilitätsprüfung, Fallback bei defekten Checkpoints,
   Ablehnung bei explizit angeforderten, unbrauchbaren Checkpoints. CLI
   (`resume-plan`, `train --plan-only`) und API (`POST /api/resume/plan`)
   zeigen denselben Plan, bevor etwas startet.
9. Nach jedem Lauf – auch nach Stopp oder Fehler – schreibt der Trainer den
   Report nach `data/runs/<id>/report.json` und in die Tabelle `reports`;
   die GUI-Ansichten `#/reports` und `#/report?run=<id>` lesen ihn.

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
  Endpunkte; Lesen bleibt lokal offen. Der Vergleich läuft über
  `hmac.compare_digest`. `EventSource` kann keine Header senden, deshalb
  akzeptiert **nur** `/api/events` das Token als `?token=`; das Frontend
  schickt es sonst als `x-rlai-token` und legt es im localStorage ab.
* `serve --host` bindet standardmäßig an `127.0.0.1`; wer `0.0.0.0` ohne
  Token nutzt, bekommt eine Warnung auf stderr. CORS ist standardmäßig aus
  und wird über `RLAI_CORS_ORIGINS` (kommagetrennte Origin-Liste) explizit
  freigeschaltet – das Dashboard wird vom selben Prozess ausgeliefert.
* Checkpoints werden mit `torch.load(weights_only=True)` geladen
  (`checkpoints.load_policy`, gemeinsam genutzt von Evaluation,
  `ModelOpponent` und RLBot-Bot); `weights_only=False` ist ein ausdrücklicher
  Opt-in für vertrauenswürdige Altdateien. Modellreferenzen aus HTTP-Anfragen
  bleiben über `resolve_model_path` auf `data/models` beschränkt, Modell-IDs
  werden gegen `^[A-Za-z0-9._-]+$` geprüft (auch beim Live-Start).
* `JobManager` startet keinen zweiten Trainingslauf, solange einer aktiv ist
  (API: 409), außer `allow_concurrent` wird bewusst gesetzt. Evaluationen
  haben eine Obergrenze für simulierte Match-Sekunden pro Anfrage.
* `WorkerPool` wartet nicht unbegrenzt: Start und Sammlung haben Timeouts,
  ein toter Kindprozess wird mit Exit-Code gemeldet statt als Hängen, und
  `close()` ist idempotent (terminate → kill).
* Checkpoints sind unveränderlich; der letzte Checkpoint und das stabile
  Modell sind gegen Löschen geschützt. `best.json` ist ein Zeiger ohne
  Lösch- oder Promotionswirkung.
* Kein automatisches Starten von Training, kein automatisches Promoten.
