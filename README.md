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
| PPO-Training, Checkpoints, Resume | implementiert | `tests/test_ppo.py`, `tests/test_trainer.py` |
| GAE über Zeitlimit-Truncations | korrigiert | `tests/test_gae_truncation.py` |
| Automatische Fortsetzung (fresh/continue/transfer) | implementiert | `tests/test_resume.py`, `tests/test_continuation.py` |
| Unveränderliche Selbstspiel-Snapshots | korrigiert | `tests/test_continuation.py`, `tests/test_league.py` |
| Lauf-Report (Statistik nach jedem Lauf) | implementiert | `tests/test_report.py` |
| Parallele Worker (Multiprocessing) | implementiert | 2-Worker-Runs für 3v3 und Heatseeker |
| Durchsatz-Benchmark (Simulation/Inferenz getrennt) | korrigiert | `tests/test_benchmark.py` |
| Seitengeglichene Evaluation + Gegner-Suiten | implementiert | `tests/test_evaluation_v2.py`, `tests/test_side_balance.py` |
| Beobachtungs-Spiegelung (180° um z) | implementiert | `tests/test_obs_mirror.py`, `tests/test_side_balance.py` |
| Modell-Registry, Promotion, Schutzregeln | implementiert | `tests/test_checkpoints.py` |
| Dashboard + Steuerung + Logs + Reports | implementiert | alle Ansichten headless gerendert (jsdom), API-Tests |
| RLBot-v5-Adapter inkl. Seiten-Spiegelung | implementiert, **nicht im Spiel verifiziert** | `tests/test_rlbot_adapter.py`, `tests/test_rlbot_bot.py` |
| Windows-/Epic-Anleitung | dokumentiert | `docs/SETUP_WINDOWS.md` |

## Schnellstart (Linux/macOS)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"      # torch, RLGym, RocketSim, FastAPI, pytest
python -m rlai.cli serve --port 8000
```

Danach `http://localhost:8000` öffnen. Der Server bindet standardmäßig nur an
`127.0.0.1`; `--host 0.0.0.0` exponiert die API im Netz und sollte nur zusammen
mit `RLAI_AUTH_TOKEN` (oder `--token`) verwendet werden. Cross-Origin-Zugriff ist
standardmäßig aus und wird über `RLAI_CORS_ORIGINS=https://host` explizit
freigeschaltet. Training startet **nie** automatisch – weder beim Start der App,
noch nach einem Absturz, noch wenn ein Checkpoint auftaucht. Jeder Lauf muss im
Tab *Training* explizit gestartet und bestätigt werden.

Erster sinnvoller Lauf: im Web-Dashboard unter *Training* die Ziel-Schritte
wählen – optional mit **Add these steps to the restored checkpoint**, damit
die Zahl wie `--steps` auf den Checkpoint aufaddiert wird statt ein absolutes
Ziel zu sein – und starten (1v1 ist der einzige Modus des Workflows; alle
übrigen Parameter kommen aus den Standardwerten), oder per CLI:

```bash
python3 main.py --steps 500000      # 500.000 Schritte, Fortsetzung inklusive
python3 main.py --plan-only         # was würde passieren? (trainiert nicht)
# Warmstart: der Actor imitiert zuerst den gescripteten Chaser
python -m rlai.cli pretrain --config-file configs/1v1-mastery-1v1.json --episodes 100
python -m rlai.cli train --config-file configs/smoke-1v1.json
python -m rlai.cli evaluate --model data/models/<id>/model.pt --games 2 --match-seconds 20
python -m rlai.cli benchmark --mode heatseeker-2v2 --workers 2 --seconds 2
```

`main.py` ist nur eine Abkürzung: es ergänzt `train` und
`configs/1v1-mastery-1v1.json`, wenn beides fehlt, und funktioniert auch ohne
`pip install -e .` (Torch/RocketSim müssen trotzdem installiert sein). Die
vollständige CLI bleibt `python -m rlai.cli`.

### Warmstart: `pretrain` (Behaviour Cloning)

Eine frisch initialisierte Policy ist nahezu gleichverteilt über 90 Aktionen;
in kurzen Läufen berührt sie den Ball fast nie, und PPO hat dann kaum ein
Signal, aus dem „fahr zum Ball“ entstehen könnte. `pretrain` nimmt dem
gescripteten Chaser (dem stärksten eingebauten Gegner, ~8 Ballkontakte pro
Episode) `(Beobachtung, Aktion)`-Paare ab und trainiert den Actor per
Kreuzentropie darauf:

```bash
python -m rlai.cli pretrain --config-file configs/1v1-mastery-1v1.json \
    --episodes 100 --epochs 3          # schreibt einen Checkpoint mit Label "pretrain"
```

Im Dashboard liegt derselbe Schritt unter **Models → Warm start** (Experte,
Episoden, Epochen, Lernrate), und `POST /api/pretrain` macht ihn
skriptbar. Der Checkpoint hat `step = 0` und ist ein ganz normaler Checkpoint: `train`
bzw. `python3 main.py --steps N` setzt ihn fort, echte Trainings-Checkpoints
haben wegen ihres höheren Schritts weiter Vorrang. Für längere Läufe gibt es
`configs/1v1-selfplay.json` (10M Schritte, Selbstspiel-Pool, Batch 8192,
fortsetzbar mit `--steps`). Gemessen (gleiche Config,
gleicher Seed, 30.000 PPO-Schritte): **3,4 statt 0,8 Ballkontakte pro Episode
und 45 statt 7 Tore**, mittlerer Return +4,9 statt 0,0; die reine
BC-Policy gewinnt gegen `idle` 3 von 10 Spielen (4:0 Tore) und spielt gegen
`chase` 2:7 bei 6:9 Toren. Details und Einschränkungen: `docs/TRAINING.md`.

### Schrittbudget: `--steps` oder `--max-steps`

* `--steps N` (Aliase `--maxsteps`, `--Maxsteps`) heißt **N Schritte
  zusätzlich** zu dem Schritt, den der fortgesetzte Checkpoint hat: ein
  Checkpoint bei 2.000.000 läuft mit `--steps 500000` bis 2.500.000 (das Ziel
  wird auf volle Rollout-Batches aufgerundet, weil der Lauf nur dort enden
  kann – hier 2.500.096). Der
  Lernraten-Anneal beginnt dabei neu beim wiederhergestellten Schritt
  (`lr_anneal_start`), damit die zusätzlichen Schritte nicht auf dem
  10-%-Boden einer langen Historie starten. Ohne Checkpoint ist `N` die
  Gesamtzahl.
* `--max-steps N` bleibt das historische **absolute** Ziel. Beides zusammen ist
  ein Fehler; pro Lauf gilt genau eine Angabe.

Fortsetzung, Report und seitengeglichene Evaluation:

```bash
# Was würde ein Resume wiederherstellen? (startet kein Training)
python -m rlai.cli resume-plan --config-file configs/smoke-1v1.json --run-id <run-id>

# Standard: neuester gültiger, kompatibler Checkpoint wird automatisch fortgesetzt
python -m rlai.cli train --config-file configs/smoke-1v1.json --steps 40000
# Explizit von einem Checkpoint / nur Gewichte übernehmen / bewusst neu anfangen
python -m rlai.cli train --config-file configs/smoke-1v1.json --resume-from <model-id> --resume-mode transfer
python -m rlai.cli train --config-file configs/smoke-1v1.json --resume-mode fresh
python -m rlai.cli train --config-file configs/smoke-1v1.json --plan-only

# Statistik-Report eines Laufs (bzw. Vergleich mehrerer Läufe)
python -m rlai.cli report --run-id <run-id> --summary
python -m rlai.cli report --compare <run-a>,<run-b>

# Suite gegen Diagnose- und Stärkegegner, auf beiden Seiten
python -m rlai.cli evaluate --model <model-id> --opponents idle,chase \
    --games 10 --sides blue,orange --phase manual
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
allen Modi **identisch aufgebaut** (Version 2: 198 Werte, Version 1: 190);
ein Checkpoint kann daher als Initialisierung für größere Teams dienen.

## Beobachtung, Aktionen, Belohnung

* Beobachtung: 198 Werte (v2) – eigener Zustand, Ball (relativ im Auto-Frame),
  2 Teamkollegen- und 3 Gegner-Slots (gepadet), 34 Boost-Pads, Restzeit,
  Score-Differenz, **8 Werte Feldrahmen** (eigene Position, Blickrichtung,
  Ballposition, Torrichtung und -distanz) und die letzten 90 Aktionen als
  One-Hot. Die alte 190er-Version bleibt für bestehende Checkpoints ladbar.
* Aktionen: 90 diskrete Lookup-Table-Aktionen (Gas/Bremse, Lenken, Boost,
  Handbremse, Sprung/Flip, Luftsteuerung), identisch in Training, Evaluation
  und RLBot-Adapter.
* Belohnung: Tore (+10) und Gegentore (−10) dominieren; Potenzial-Shaping für
  Ballfortschritt und Ausrichtung, kleine Boni für Touch, Boost-Effizienz,
  Verteidigung und Karosseriekontakt (Bump). Alle Gewichte sind pro Lauf konfigurierbar und werden geloggt.
  Beim Tor wird kein Distanz-Shaping angewendet, damit die Torprämie nicht
  teilweise aufgehoben wird.

## Training

* PPO mit GAE, Advantage-Normalisierung, Gradient-Clipping, LR-Annealing,
  orthogonaler Initialisierung; Actor und Critic getrennt.
* Jeder Worker ist ein eigener Prozess mit einer RocketSim-Arena; der Trainer
  sammelt Rollouts, trainiert zentral und schreibt Metriken in SQLite.
* Checkpoints werden atomar geschrieben; die Modelldateien sind unveränderlich,
  nur Labels werden im Manifest aktualisiert. Der stabile Zeiger wird nur
  explizit gesetzt. Der letzte Checkpoint ist gegen automatisches Löschen
  geschützt; `latest.json` hält ihn fest. Laden
  erfolgt standardmäßig mit `torch.load(weights_only=True)`, Modellreferenzen
  aus HTTP-Anfragen bleiben auf `data/models` beschränkt.
* **Fortsetzung ist automatisch**: Ohne Resume-Option wird der neueste
  gültige, kompatible Checkpoint gesucht und fortgesetzt; `fresh` (nichts wird
  geladen), `continue` (Gewichte, Optimizer, Schrittstand, RNG, Curriculum,
  Gegnerpool, geerbte Config-Felder) und `transfer` (nur Gewichte,
  Schrittzähler beginnt bei 0) bleiben explizit wählbar. Sortiert wird nach
  Trainingsschritt und zuletzt erfolgreichem Speichern, nicht nach Dateiname.
  Ist der neueste Checkpoint defekt oder inkompatibel, wird auf einen älteren
  zurückgefallen – mit Begründung im Log und im Report. Existieren nur
  unbrauchbare Checkpoints, startet der Lauf **nicht** und nennt die Gründe
  (CLI-Exit 3, API 400); ein bewusstes `--resume-mode fresh` ist der einzige
  Weg, vorhandenen Fortschritt zu ignorieren. Ein **namentlich** angeforderter
  Checkpoint wird niemals stillschweigend durch einen anderen ersetzt:
  dann startet der Lauf nicht. `rlai resume-plan` bzw. `POST /api/resume/plan`
  zeigen vorher, was wiederhergestellt würde und was fehlt.
* Gegnerpool-Snapshots sind echte Kopien (`freeze_state`), die sich beim
  Weiterlernen nicht verändern; ihre Gewichte werden mit
  `league_states=true` (Standard) im Checkpoint gespeichert, damit Selbstspiel
  nach einem Resume gegen dieselben Gegner weiterläuft.
* Selbstspiel nutzt einen begrenzten Pool eingefrorener Snapshots mit
  Elo-/PFSP-Auswahl (optional); ein Curriculum kann Gegnerstufen anhand
  gemessener Siegquoten weiterschalten.
* Evaluation läuft auf Wunsch auf beiden Seiten (`eval_sides`), wobei die
  Beobachtung für Orange um 180° um die Hochachse gedreht wird
  (`eval_mirror`); die Aktion bleibt unverändert, weil sie autorelativ ist.
  Gegner-Suiten (`eval_opponents`) trennen Diagnosebaselines (idle, random)
  von Stärkegegnern (chase, mixed, self, model:*). Jede Serie schreibt
  Siegquote, Tore, Touches, 95-%-Wilson-Intervall **und das Protokoll**
  (Modus, Gegner, Spiele, Seed, Matchlänge, Seiten, Spiegelung, Sampling,
  Obs-Version); `best.json` wird nur nach einem Protokoll und nur mit
  mindestens einem Sieg gesetzt.
* Nach jedem Lauf – abgeschlossen, gestoppt oder fehlgeschlagen – wird ein
  Report geschrieben (`data/runs/<id>/report.json` und Datenbank): Schritte
  von/bis, Dauer, zwei Durchsatzzahlen (Wandzeit und pro Iteration), Gerät und
  Worker, Konfiguration, Resume-Herkunft, PPO-Kennzahlen, Zeitanteile
  (Simulation, Inferenz, Update, Evaluation, Checkpoint), Belohnungskomponenten,
  Episoden, Before/After-Evaluation **nur bei gleichem Gegner und gleichem
  Protokoll**, Checkpoint-Gültigkeit, Kurven und Vergleichsläufe.
  Nicht Gemessenes steht als `available: false` mit Begründung darin.
* Alle Kennzahlen im Dashboard kommen aus der Datenbank des laufenden Prozesses –
  keine erfundenen Werte. Trainings-Reward belegt keine Spielstärke; dafür ist
  ausschließlich der Evaluations-Teil des Reports da.

Details: `docs/TRAINING.md`.

## Web-Dashboard

Vier Hauptansichten – **Dashboard, Training, Stats, Bot Lab** – plus die
Systemgruppe **Models, Live match, Settings** in der Sidebar. **Monitoring**
(Telemetrie eines Laufs) und der **Report** sind als Detailansichten über
Links erreichbar (tiefe Links wie `#/monitoring?run=<id>` oder `#/report?run=<id>`).
Alte Adressen leiten weiter: `#/reports` → Stats, `#/evaluation` → Bot Lab/Serien,
`#/teams` → Bot Lab/Teams, `#/memory` → Bot Lab/Memory. Es gibt keine
Dummy-Buttons: nicht verfügbare Aktionen sind deaktiviert und erklären sich,
fehlende Abhängigkeiten werden als Diagnose angezeigt.

* **Dashboard** zeigt Zustand (läuft/stopped/unterbrochen), aktuellen und
  kumulierten Training-Zeitwert, Schritt gegen Ziel mit Fortschrittsbalken,
  gemessenen Durchsatz, den neuesten Checkpoint mit Fortsetzungsstatus
  („Nächster Lauf setzt fort von …" – automatisch, keine Auswahl) und die
  letzten Läufe, Evaluationen und Modelle.
* **Training** ist auf den 1v1-Workflow reduziert: Ziel-Schritte (primärer
  Regler, mit Vorschlägen), optionaler Laufname und ein Start-Button mit
  Bestätigungsdialog. Alle technischen Parameter (LR, Batch/Minibatch,
  PPO-Epochen/Clip, Gamma/GAE, Architektur, Worker, Sim-Einstellungen) sind
  keine Benutzerfelder mehr; die Werte kommen aus den stabilen Standardwerten
  plus der Hardware-/Konfigurationslogik und bleiben in der Backend-Config.
  Das Fortsetzungs-Panel zeigt vor dem Start, was der Server finden würde
  (welcher Checkpoint, welche Schritte, welche Zustände werden wiederhergestellt)
  – ein passender Checkpoint wird nie stillschweigend überschrieben. Ohne
  Checkpoint startet der Lauf neu; existieren nur unbrauchbare Checkpoints,
  verweigert der Server den Start und nennt die Gründe. Beides wird im Dialog
  benannt. Die
  erweiterten Optionen (paralleler Lauf, Evaluations-Regler) sind für die
  Entwicklung im Bot Lab (Serien) verfügbar.
* **Monitoring** zeigt zusätzlich die Iterationskosten (Collect, Simulation,
  Inferenz, Update, Evaluation, Checkpoint), Curriculum-Stufe, Pool-Größe und
  bei Fortsetzungen, ab welchem Schritt gezählt wird.
* **Stats** konsolidiert Läufe, Lernkurven, Durchsatz über die Zeit,
  Session-Historie, Checkpoints und Evaluationen; nicht gemessene Kennzahlen
  werden als nicht verfügbar ausgewiesen, nicht mit Nullen aufgefüllt. Der
  Abschnitt *Lauf-Reports* listet alle Läufe mit Status, Schritten, Dauer,
  Durchsatz, finaler Evaluation und Checkpoint-Gültigkeit, erlaubt den
  Vergleich ausgewählter Läufe (inkl. „nicht vergleichbar"-Warnung) und kann
  Reports älterer Läufe aus der Datenbank neu berechnen.
* **Bot Lab** bündelt die Experimentierfläche in Unterabschnitten:
  *Experimente* (Modell-Auswahl, Gegner, Spiele – erst nach Bestätigung
  ausgeführt), *Serien* (Einzelserien und Suiten mit Seiten, Spiegelung,
  Phasen-Label, Protokoll-Chips, Konfidenzintervallen und getrennten
  Stärke-/Diagnose-Baselines), *Schwierigkeit* (Gegnerprofile und
  Verhaltensprofile; Prozentwerte sind Design-Ziele, keine Messwerte),
  *Teams* (Bot-Editor, gespeicherte Teams) und *Memory* (Bekanntheitsliste,
  Rivalen, Export/Reset – Spiel-Zustand, nie Modellgewichte).
* **Models** prüft einen Checkpoint gegen eine Konfiguration
  (`POST /api/models/{id}/validate`) und zeigt, ob `continue` bzw. `transfer`
  möglich ist und welcher Zustand fehlt.
* **Settings** misst den Durchsatz wahlweise parallel/sequenziell und mit oder
  ohne Inferenzanteil und verwaltet das API-Token (localStorage, Header
  `x-rlai-token`; der SSE-Strom nutzt `?token=`, weil `EventSource` keine
  Header senden kann).

Live-Matches sind außerhalb von Windows deaktiviert (Serverprüfung, kein Fake);
Schaltflächen für Setup-Prüfung und reines Vorbereiten der Match-Dateien
funktionieren auch dort und zeigen echte Ergebnisse.

## Tests

```bash
python -m pytest -q                    # 616 passed, 1 skipped, ca. 3 min (2 vCPU, keine GPU)
```

Ergebnis in dieser Umgebung: **616 passed, 1 skipped** in ca. 3 min (skip:
jsdom fehlt). Abgedeckt: Modi,
Aktionen, Beobachtung v1/v2 inkl. Parität und **Spiegelung**, Belohnungs-Events,
PPO inkl. **GAE über Truncations** und Early Stop, Checkpoint-Schutz,
Validierung und best-Zeiger, Gegnerpool (inkl. **Unveränderlichkeit** und
 serialisierter Gewichte), Curriculum, **Resume-Pläne/Fallback/Ablehnungen**,
**echte unterbrochene und fortgesetzte Läufe**, **Report-Inhalte und
Before/After-Paarung**, Store, API (inkl. 409 bei parallelem Lauf,
Resume-Vorschau, Validierung, Suite, Token, CORS), CLI (Run-Registrierung,
`resume-plan`, `report`, Suite-Evaluation, Live-Preflight/Dry-Run),
Evaluation inkl. Konfidenzintervall und **Seitengleichgewicht**,
**Benchmark-Messung** (parallel/sequenziell/Inferenz), RLBot-Parser für alle
generierten Match-Configs sowie **Bot-Laden, Spiegelung und Fehlerverhalten**.

Das Web-Frontend wird headless mit jsdom gerendert (`tests/gui/smoke.mjs`,
automatisch als `tests/test_gui_smoke.py` ausgeführt; einmalig
`cd tests/gui && npm install`). Fehlt node oder jsdom, wird der Test
übersprungen statt rot zu werden.

Nicht abgedeckt: echtes Rocket League, RLBot-Start, GPU-Pfad, Spielstärke.

## Grenzen

* Kein Nachweis von Spielstärke gegenüber Nexto oder einem bestimmten Rang.
  Die gezeigten Zahlen sind Smoke-Tests, keine trainierten Fähigkeiten.
* Live-Deployment ist ungetestet, weil hier kein Windows/Spiel/RLBot-Lauf
  möglich ist. Die Anleitung nennt exakte Schritte und bekannte Risiken.
* 2 vCPU bedeutet: 3v3/Heatseeker mit 8 Workern sind auf dieser Maschine
  langsamer als die Workerzahl suggeriert; das Dashboard zeigt die echte Rate
  (Benchmark mit parallelen Prozessen, ohne Env-Aufbau, plus Inferenzanteil).
* Alte Checkpoints (vor `CHECKPOINT_VERSION = 2`) enthalten weder RNG- noch
  Gegnerpool-Gewichte; ein Resume daraus läuft, meldet aber laut, was fehlt.
* `data/` ist Laufzeitverzeichnis und nicht versioniert.

## Dokumentation

* `docs/MODES.md` – Regeln, Beobachtung, Aktionen, Belohnung pro Modus
* `docs/TRAINING.md` – PPO, Curriculum, Metriken, Resume, Selbstspiel
* `docs/RLBOT.md` – v5-Adapter, Konfiguration, Grenzen
* `docs/SETUP_WINDOWS.md` – Windows + Epic Schritt für Schritt
* `docs/TESTING.md` – Testmatrix, Messwerte, was nicht lief
* `docs/ARCHITECTURE.md` – Module und Datenfluss
