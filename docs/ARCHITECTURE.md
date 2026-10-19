# Architektur

```
Browser (HTML/CSS/JS, ui/theme)
        │  fetch / SSE
        ▼
FastAPI (src/rlai/api/app.py)
        │
        ├── SQLite-Store (Runs, Metriken, Episoden, Logs, Modelle,
        │                 Evaluationen, Reports, Settings, Sessions,
        │                 Teams, Rivalen, Spielerspeicher)
        ├── JobManager (Subprozesse: train, evaluate, live)
        ├── Statistik (src/rlai/stats.py: eine Quelle für GUI, Report, CLI)
        └── Hardware/Diagnose (psutil, torch, Import-Checks)
                 │
                 ▼
        Trainer (src/rlai/trainer.py)
        ├── Trainingszeit (src/rlai/timing.py: monotoner Timer, Akkumulation)
        ├── Ressourcenplan (src/rlai/resources.py: CPU-/Speicherbudget)
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

Ohne Browser, gleiche Trainings- und Checkpoint-Logik:
CLI (src/rlai/cli.py) → Trainer → Report/Summary
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

## Trainingszeit

`src/rlai/timing.py` trennt drei Uhren:

* **UTC-Wanduhr** – *wann* etwas passiert (Start/Ende, Checkpoint-Zeit).
  Sie darf springen und wird nie für eine Dauer verwendet.
* **Monotone Uhr** – *wie lange* der Prozess wirklich gearbeitet hat. Nur
  sie erzeugt Dauern.
* **Akkumulierte Summe** – die Summe aller beendeten Sitzungen eines Laufs.
  Sie liegt in der Datenbank und im Checkpoint, damit eine Fortsetzung den
  Wert fortschreibt statt neu zu beginnen.

Regeln: Ein Checkpoint speichert höchstens die bis dahin gemessene Zeit; die
Zeit zwischen zwei Sitzungen (Prozess gestoppt) zählt nicht; eine Sitzung, die
ein abgestürzter Prozess offen gelassen hat, wird beim nächsten Start als
*interrupted* markiert und behält ihre **unbekannte** Dauer (ein Absturz kann
die eigene Laufzeit nicht messen). Alte Checkpoints ohne Zeit-Metadaten
bleiben ladbar und melden die historische Dauer als unbekannt.

## Neue Module

| Modul | Aufgabe |
| --- | --- |
| `src/rlai/timing.py` | Sitzungs-Timer, akkumulierte Trainingszeit, Formatierung |
| `src/rlai/stats.py` | alle Zahlen des Stats-Tabs (GUI, Report und CLI rechnen damit gleich) |
| `src/rlai/resources.py` | CPU-/Speicherbudget und Warnungen vor dem Start |
| `src/rlai/botbrain.py` | Verhaltensprofil (aggressiv/defensiv/ausgewogen/adaptiv) |
| `src/rlai/difficulty.py` | vier Schwierigkeitsstufen mit Zielwert + Mechanismus |
| `src/rlai/team_builder.py` | Teams aus Bot-Slots, Validierung, Beschreibung |
| `src/rlai/player_memory.py` | begrenztes Gedächtnis über Begegnungen (eigene Tabelle) |
| `src/rlai/rivals.py` | benannte Rivalen mit Historie, Rangfolge, Engpass |
| `src/rlai/adaptive.py` | begrenzte Anpassung des skriptgesteuerten Gegners |
| `src/rlai/replay.py` | Match-/Episodenanalyse; meldet ehrlich, dass Replays nicht lesbar sind |
| `src/rlai/benchmarks.py` | drei reproduzierbare Mikro-Benchmarks |
| `src/rlai/rlbot/pad_order.py` | kanonische Boost-Pad-Reihenfolge und Remap |

## Zwei Ausführungsmodi

| | GUI-Modus | Headless-Modus |
| --- | --- | --- |
| Start | `POST /api/runs` mit `confirm=true` (der GUI-Dialog holt die Bestätigung vorher ein) | `rlai headless --config-file …` – der Aufruf *ist* die Bestätigung |
| Webdienst | ja (`rlai serve`) | nein, kein Browser, keine GUI-Überwachung |
| Überwachung | HTTP/SSE, konfigurierbare Intervalle | nur die essenziellen Metriken, Fehlerlogs, Checkpoints, Dauer |
| Abschluss | Report in `data/runs/<id>/report.json` + Ansichten | kompakter Abschlussbericht `rlai-headless-summary-v1` |
| Trainingslogik | identisch | identisch |

Beide Modi benutzen denselben Trainer, denselben Checkpoint-Code und dieselbe
Zeitmessung.

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
* `POST /api/runs` startet **nur** mit `confirm=true`. Die GUI holt die
  Bestätigung vorher im Dialog ein; der Server weist eine unbestätigte
  Anfrage ab. Dasselbe gilt für `POST /api/lab/run` (`execute=true`).
* `POST /api/lab/plan` plant nur: es startet nie eine Simulation.
* `WorkerPool` wartet nicht unbegrenzt: Start und Sammlung haben Timeouts,
  ein toter Kindprozess wird mit Exit-Code gemeldet statt als Hängen, und
  `close()` ist idempotent (terminate → kill).
* Checkpoints sind unveränderlich; der letzte Checkpoint und das stabile
  Modell sind gegen Löschen geschützt. `best.json` ist ein Zeiger ohne
  Lösch- oder Promotionswirkung.
* Kein automatisches Starten von Training, kein automatisches Promoten.
