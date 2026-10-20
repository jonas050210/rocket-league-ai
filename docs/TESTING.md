# Testmatrix und Messergebnisse

Umgebung: Linux (Sandbox), 2 vCPU, 3,85 GB RAM, **keine GPU**, kein
Rocket-League-Installation, kein RLBot-Match. Python 3.11.2, Torch 2.9.1,
RocketSim 2.2.1, RLGym-Rocket-League 2.0.1, FastAPI 0.115.6.

## Automatisierte Tests

```bash
python -m pytest -q
```

| Bereich | Datei | Ergebnis |
| --- | --- | --- |
| Trainingszeit: monotoner Timer, Akkumulation über Sitzungen, unbekannte Historie, Sitzungs-CRUD | `test_timing.py` | bestanden |
| Trainingszeit end-to-end: gemessene Sitzung, Checkpoint-Metadaten, Fortsetzung, Safe-Stop, Absturz-Rettung, kein Zählen der Standby-Zeit | `test_trainer_timing.py` | bestanden (slow) |
| Statistik (Stats-Tab): Zeit, Fortschritt, Lernkurven, Checkpoints, Evaluation, Retention, unbekannte Werte | `test_stats.py` | bestanden |
| Merkmale: Bot-Brain, Schwierigkeiten, Team-Builder, Spielerspeicher, Rivalen, adaptive Dynamik | `test_features.py` | bestanden |
| Match-/Replay-Analyse: nur echte Messwerte, Replay-Parsing explizit nicht verfügbar | `test_analysis.py` | bestanden |
| Ressourcenwächter: CPU-/Speicherbudget, Warnungen, abschaltbare Klemme | `test_resources.py` | bestanden |
| Mikro-Benchmarks: Schwierigkeitsmechanik, Arena-Wiederverwendung, Thread-Budget | `test_benchmarks.py` | bestanden |
| API der neuen Merkmale: Stats, Teams, Schwierigkeiten, Verhalten, Speicher, Rivalen, Analyse, Lab | `test_api_features.py` | bestanden |
| Ausführungsmodi: Headless ohne Webdienst, Resume, Benchmarks-Kommando | `test_cli_modes.py` | bestanden |
| Boost-Pad-Reihenfolge: RLBot (y,x) vs. RocketSim, Permutation, Live-Arena-Abgleich | `test_boost_pad_order.py` | bestanden |
| Moduskatalog, Konfigurationsvalidierung | `test_modes.py` | bestanden |
| Aktionstabelle (90 Aktionen, Grenzen, Mapping aller Indizes) | `test_actions.py` | bestanden |
| Alle vier Modi: Reset/Step, Beobachtungsform, Padding, Seed | `test_env_modes.py` | bestanden |
| Heatseeker: Homing, Tempoplus, Tor, unendlich Boost | `test_heatseeker.py` | bestanden |
| Belohnungen: Tor/Gegentor, Shaping ohne Tor, Touch als Event | `test_rewards.py` | bestanden |
| PPO: GAE, Dones, Truncation-Bootstrap, Early Stop, ein Update | `test_ppo.py` | bestanden |
| GAE über Zeitlimits: Maske für Bootstrap **und** Rekursion getrennt, Oracle | `test_gae_truncation.py` | bestanden |
| Beobachtungs-Spiegelung: 180° um z, Invarianzen, Boost-Pad-Reihenfolge | `test_obs_mirror.py` | bestanden |
| Seitengleichgewicht: gespiegelte Trajektorien, 0 Aktionsabweichungen, Klassifikation | `test_side_balance.py` | bestanden |
| Benchmark: parallele/sequenzielle Messung, Speed-up, Inferenz, Klemmen | `test_benchmark.py` | bestanden |
| Resume-Pläne: automatische Auswahl nach Trainingsschritt (nicht Dateiname), Fallback, Ablehnungen, Config-Merge, Modi, explizite Referenz | `test_resume.py` | bestanden |
| Echte Fortsetzung: unterbrochener Lauf, Restore, Report, Snapshot-Immutabilität | `test_continuation.py` | bestanden (slow) |
| Report: nicht Gemessenes, Schritt-Zählung, Before/After-Paarung, Persistenz | `test_report.py` | bestanden |
| GUI: alle Ansichten headless gerendert (jsdom), Plan-Vorschau, Token | `test_gui_smoke.py`, `tests/gui/smoke.mjs` | bestanden |
| Beobachtung: v1/v2-Größen, Feldrahmen, Padding, Determinismus | `test_obs.py` | bestanden |
| Beobachtung v2: Größen, Torrichtung, Padding, Determinismus | `test_obs_v2.py` | bestanden |
| Checkpoints: atomar (auch bei Save-Fehler), latest-Zeiger, Promotion, Löschschutz, best-Zeiger, Retention mit Ankern und Ladbarkeitsprüfung, Auswahl nach Schritt | `test_checkpoints.py` | bestanden |
| Gegnerpool: Kapazität, PFSP-Gewichte, Elo, Serialisierung,unveränderliche und serialisierte Gewichte | `test_league.py` | bestanden |
| Curriculum: Aufstiegskriterien, Fenster, Resume-Zähler | `test_curriculum.py` | bestanden |
| SQLite-Store: Runs, Metriken, Logs, Episoden, Evaluationsdetails (CI) | `test_store.py` | bestanden |
| HTTP-API inkl. Auth (Header/Bearer/SSE-Token), CORS opt-in, 409 bei parallelem Lauf, Resume-Vorschau, `additional_steps`, Checkpoint-Validierung, Reports, Suite, Budget-Grenzen, Pfad-Ablehnung, Statik, Preflight, Dry-Run, Presets | `test_api.py` | bestanden |
| Evaluation: Ergebnis, Seed-Reproduzierbarkeit, Wilson-Intervall, v1-Checkpoints, best-Regel | `test_evaluation.py` | bestanden |
| Evaluation v2: CI-Grenzen, Layout-Erkennung, v1-Checkpoint-Evaluation, Suiten, Seiten | `test_evaluation_v2.py` | bestanden |
| Trainer: fehlende Run-Zeile, automatischer Resume, Abbruch bei unbrauchbaren Checkpoints, Final-Dedupe, Ctrl+C-Save, fehlgeschlagener Shutdown-Save, best-Regel, Pool-Gewichte im Checkpoint | `test_trainer.py` | bestanden |
| CLI: automatischer Resume, Exit 3 bei unbrauchbaren Checkpoints, `--resume-mode fresh`, `--resume-from`, `--plan-only`, `resume-plan`, `report`, Suite-Evaluation, `live --preflight/--dry-run/--list` | `test_cli.py` | bestanden |
| RLBot-Parität: Packet → Beobachtung (1v1, 3v3, Heatseeker) | `test_rlbot_adapter.py` | bestanden |
| Bot: sicheres Laden (Pickle-Ablehnung), Inferenz, deterministische Aktionen, Spiegelung je Team, Fehlerrobustheit, `initialize`, fehlendes/externes Modell | `test_rlbot_bot.py` | bestanden |
| JobManager: Steuerdateien, Subprozess-Umgebung | `test_jobs.py` | bestanden |
| Alle ausgelieferten Konfigurationen laden und validieren | `test_modes.py` | bestanden |
| Live-Match-TOML für alle vier Modi, Team-/Fill-Optionen, Preflight, Dry-Run | `test_live.py` | bestanden |
| Generierte Match-Configs durch RLBot-Parser (alle Modi, beide Fill-Arten) | `test_live.py` | bestanden |
| Launcher `--dry-run` mit RLBot-Parser, Rückgabecodes | `test_live.py`, `test_cli.py` | bestanden |
| Live-Start außerhalb Windows wird mit 501 abgelehnt | `test_api.py` | bestanden |
| Exposure-Settings: Loopback-Standard, Host-/Port-/CORS-Validierung, Token-Warnung, `serve` | `test_app_settings.py` | bestanden |
| `--steps N` (zusätzliche Schritte ab Checkpoint, Aufrunden auf Rollout-Batches), Lernraten-Fenster, Plan-Felder, CLI-Aliase, `main.py`-Wrapper, Episode-Retention mit geschütztem Anfang, Recovery-Checkpoint, No-Touch-Truncation | `test_additional_steps.py` | bestanden |
| Behaviour Cloning und DAgger: Expertendaten sammeln, Kreuzentropie, Learner-Rollouts mit Experten-Labels, aggregiertes Dataset, Checkpoint mit `step=0`, Resume-Priorität, CLI/`main.py` | `test_imitation.py` | bestanden |
**Gesamt: 603 passed, 0 failed, 0 skipped** (ca. 3 min, 2 vCPU, mit
installiertem jsdom für den GUI-Smoke-Test), `ruff check src tests rlbot`
sauber. Ohne jsdom wird der GUI-Smoke-Test übersprungen (602 passed,
1 skipped). Die einzige Warnung stammt aus `starlette`/`anyio`
(Drittanbieter), nicht aus diesem Projekt.

Der GUI-Test braucht node und jsdom; fehlen sie, wird er übersprungen statt zu
fehlschlagen:

```bash
cd tests/gui && npm install && npm test     # einmalig bzw. direkt
python -m pytest tests/test_gui_smoke.py -q # läuft automatisch mit
```

## Projekt-Mikro-Benchmarks (`python -m rlai.cli benchmarks`)

Drei Messungen, die jeweils eine konkrete Frage beantworten. Sie laufen ohne
Training und ohne GUI:

| Benchmark | Was er misst | Ergebnis in dieser Umgebung |
| --- | --- | --- |
| `difficulty` | Ballberührungen des skriptgesteuerten Gegners je Stufe (3×500 Schritte) | rock 28 · tiny 27 · medium 36 · pro 42 (Spreizung 15) |
| `evaluation_arena_reuse` | Kosten einer Eval-Serie (4 Spiele) mit Arena-Wiederverwendung gegenüber dem reinen Arena-Aufbau | 0,518 s Wiederverwendung (inkl. Simulation) · 0,144 s allein der Arena-Aufbau von 4 frischen Arenas |
| `threads` | Kosten einer Forward-Passage je Torch-Thread-Zahl | 1 Thread 2 693/s · 2 Threads 2 900/s · 4 Threads 956/s |

Die Schwierigkeitsmessung belegt, dass der *Mechanismus* (Fehlerrate,
Reaktionszeit, Boost-Effizienz) die Spielweise messbar verändert. Sie belegt
**nicht**, dass die Zielwerte 0/50/75/100 % erreicht sind – das braucht eine
vollständige Evalutionsserie je Stufe. Die Prozentzahlen bleiben im Code und
in der GUI als *Zielwerte* markiert (`target_is_measured = false`).

Die Thread-Messung zeigt, warum der Trainer den Thread-Budget nicht
standardmäßig erhöht: bei diesem kleinen MLP kostet der Wechsel auf vier
Threads mehr, als er bringt (2 vCPU in dieser Umgebung).

## Gemessene Trainingszeit (diese Umgebung)

Zwei reale Läufe über die CLI (`rlai headless`), 1 Worker, `hidden=[16]`:

| Lauf | Art | Schritte | Sitzungen | Akkumulierte Trainingszeit |
| --- | --- | --- | --- | --- |
| `a69a85a6b4dc` | fresh | 0 → 2 048 | 1 (`completed`, 4,41 s) | 4,41 s |
| `run-b` | continue aus `a69a85a6b4dc` | 2 048 → 4 096 | 2 (`inherited` 4,11 s + `completed` 4,81 s) | 8,91 s |

Die Fortsetzung liest `restored accumulated training time: 4.1s from the
checkpoint`. Der geerbte Wert (4,11 s) ist die Zeit, die bis zum Schreiben
des Checkpoints gemessen war; die eigene Sitzung misst 4,81 s. Die
Sitzungstabelle trennt beides, und die Summe ist exakt die akkumulierte
Trainingszeit. Die Zeit zwischen den Läufen (Prozess gestoppt) wird nicht
mitgezählt. Im Checkpoint steht
`timing: {accumulated_s, session_s, session_started_at, measured_at,
source: "monotonic clock"}`; ältere Checkpoints ohne diese Angaben bleiben
ladbar und melden die historische Dauer als *unbekannt*.

## Headless-Modus (ohne GUI-Dienste)

`python -m rlai.cli headless --config-file <cfg>` startet **keinen**
Webdienst und keine Browser-Überwachung. Der Abschlussbericht
(`rlai-headless-summary-v1`) enthält: gemessene Gesamtzeit, aktuelle Sitzung,
Schritte, Durchsatz, Lernmetriken, Episoden, Evaluation, Checkpoint,
Abbruchstatus. `POST /api/runs` verlangt zusätzlich `confirm=true` – dieselbe
Bestätigung, die der GUI-Dialog einholt; eine unbestätigte Anfrage startet
keinen Lauf.

## Gemessene Fortsetzung über die CLI (diese Umgebung)

Zwei reale Läufe mit RocketSim-Workern (`RLAI_DATA_DIR=/tmp/rlai-t4`,
`hidden=[16]`, 1 Worker, 5-s-Episoden, Suite `idle,chase` auf beiden Seiten):

| Lauf | Modus | Schritte | Dauer | Schritte/s (Wandzeit) | Schritte/s (Iteration) |
| --- | --- | --- | --- | --- | --- |
| `run-a` | `fresh` | 0 → 2 048 | 5,73 s | 358 | ~1 100 |
| `run-b` | `continue` aus `run-a` | 2 048 → 4 096 | 5,78 s | 354 | ~1 177 |

`run-b` meldet `resume_mode=continue`, `parent_run=run-a`, `start_step=2048`,
führt die Iterationszählung bei 9 fort und stellt Optimizer, RNG und
Gegnerpool wieder her. `rlai resume-plan` nennt vorab Kandidat, Elternlauf,
`restored_step`, was wiederhergestellt wird und was fehlt. Der Report von
`run-b` enthält 2 Before/After-Paare (`chase` als Stärkegegner in der
Kopfzeile, `idle` als Diagnose), jeweils mit `same_seed: false`, weil der
Trainer pro Phase einen neuen Kickoff-Seed ableitet; die Protokolle sind
identisch (Modus, 2 Spiele, 5 s, Seiten `blue+orange`, gespiegelt, obs v2).
Zeitanteile pro Iteration: Collect 0,214 s (davon Simulation 0,111 s,
Inferenz 0,097 s), Update 0,004 s, Checkpoint 0,013 s, Evaluation 0,653 s
pro Serie – die Evaluation dominiert bei kurzen Serien, nicht das Lernen.

## Was zusätzlich manuell geprüft wurde

* `GET /api/health`, `/api/system`, `/api/resources`, `/api/diagnostics`,
  `/api/modes`, `/api/config/defaults`, `/api/runs`, `/api/models`,
  `/api/evaluations`, `/api/live/status` → HTTP 200 mit echten Daten.
* `POST /api/runs` mit *soccer-1v1* → Prozess startet, trainiert 2 560
  Schritte, endet `completed`, Checkpoints und Metriken geschrieben.
* Derselbe Pfad mit *soccer-2v2*, *heatseeker-2v2* und *soccer-3v3*, jeweils
  2 Worker → `completed` nach 3 072 Schritten. Damit ist jeder der vier Modi
  einmal durch den vollständigen Prozess gelaufen.
* Pause → Status `paused`; Resume → `running`; Stop → `stopped` mit
  Nachricht „stopped by user“.
* Evaluation über die API für 1v1-, 3v3- und Heatseeker-Checkpoints →
  Ergebnisse in der Datenbank, z. B. 1v1: 0W 2L (0:5), Heatseeker: 1W 0L 1D.
* SSE-Stream `/api/events?run_id=...` liefert Log- und Metrik-Events.
* Alle sieben Dashboard-Routen wurden in Phase 2 in einem headless Chromium
  gerendert: kein JavaScript-Fehler, kein horizontaler Überlauf, kein
  hängender Ladezustand. Nach den Änderungen an der Live-Ansicht in Phase 3
  wurde nur die JavaScript-Syntax geprüft und die zugehörigen API-Pfade
  getestet; ein erneutes Browser-Rendering war in dieser Umgebung nicht
  möglich.

## Durchsatz (gemessen, nicht geschätzt)

Benchmark mit Zufallsaktionen, zwei Umgebungen, je 2 Sekunden
(`python -m rlai.cli benchmark`), Messung nach der Beobachtungs- und
Reward-Überarbeitung:

| Modus | Umgebungsschritte/s | Simulator-Ticks/s |
| --- | --- | --- |
| soccer-1v1 | 7 102 | 56 817 |
| soccer-2v2 | 3 709 | 29 669 |
| soccer-3v3 | 2 108 | 16 861 |
| heatseeker-2v2 | 3 539 | 28 310 |

Wichtig: Der Benchmark startet die Umgebungen **nacheinander im selben
Prozess** und summiert die Schritte. Er misst den Simulationsdurchsatz, nicht
die Skalierung über Kerne. Ein paralleler Worker-Pool ist langsamer pro Kern
und wird hier nicht als „linear skalierend“ behauptet.

Direkter Vorher/Nachher-Vergleich derselben 1v1-Umgebung (1 Worker,
1 500 Schritte, gesäte Zufallsaktionen, `idle`-Gegner, je 3 Läufe). Der alte
Stand ist `git archive` von Commit `21d7339` (Phase-2-Code):

| Build | Beobachtung | Schritte/s (Median) | Einzelmessungen |
| --- | --- | --- | --- |
| alt (Phase 2) | v1, 190 | 2 850 | 3 006 / 2 850 / 2 442 |
| neu | v1, 190 | 3 622 | 3 622 / 3 242 / 3 730 |
| neu | v2, 198 | 3 576 | 3 580 / 3 576 / 3 170 |

Das sind **+27 %** gegenüber dem alten Stand bei gleicher Beobachtung und
**+25 %** mit der größeren v2-Beobachtung. Der Unterschied zwischen v1 und v2
liegt mit −1,3 % im Rauschen der drei Läufe. Ursache der Beschleunigung sind
skalare Clamps statt `numpy.clip` und der Wegfall der Dict-Allokation pro
Reward, nicht die Beobachtungserweiterung.

Mit PPO-Update im selben Prozess (echte Trainingsläufe):

| Lauf | Worker | Schritte/s |
| --- | --- | --- |
| learning-smoke (1v1 vs idle, 150k Schritte, 2 Worker) | 2 | 1 733–1 973 |
| Phase-3-Smoke (1v1, Curriculum + Evaluation) | 1 | 921–1 105 |
| Phase-3-Smoke (1v1, Selbstspiel, Snapshot je Iteration) | 1 | 693–823 |
| smoke-1v1 (Phase 2) | 1 | 810 |
| smoke-heatseeker-2v2 (Phase 2) | 2 | 1 138 |
| smoke-soccer-2v2 (Phase 2) | 2 | 1 123 |
| smoke-soccer-3v3 (Phase 2) | 2 | 782 |

Einordnung: Das sind Funktionsnachweise auf 2 vCPUs. Sie sagen nichts über
Lernqualität aus. Für 3v3 ist die Rate erwartbar niedriger (6 Autos,
aufwendigere Kollisionen).

## Phase-3/4-Nachweise (Funktion, nicht Spielstärke)

* **Neuer Trainer end-to-end:** 2 560 Schritte, 1 Worker, `curriculum = true`,
  Evaluation alle 3 Iterationen, 2 Spiele à 5 s. Ergebnis: `completed`,
  Metriken, Episoden, Logs, Checkpoints und Evaluationszeilen geschrieben.
  Evaluation: 0W 0L 2D (0:0) gegen `chase` – eine untrainierte Policy, wie
  erwartet.
* **Resume:** derselbe Lauf wurde aus dem finalen Checkpoint bei Schritt
  2 560 fortgesetzt und lief bis 3 072. Modell, Optimizer, Schrittstand,
  Liga-Bilanz und Curriculum-Stufe wurden geladen; die Logzeile
  „restored opponent-pool ratings“ bestätigt den Pool-Restore.
* **Selbstspiel:** eigener Lauf mit `self_play = true` und
  `snapshot_interval = 1`, 2 048 Schritte, `completed`. Die Worker laden die
  eingefrorenen Gewichte als `ModelOpponent`, der Trainer aktualisiert Elo
  und Pool. Nach dem Resume aus diesem Checkpoint greift der dokumentierte
  Fallback auf den festen Gegner, bis der nächste Snapshot Gewichte hat;
  genau diese Logzeile wurde beobachtet.
* **Alte Beobachtung:** ein vollständiger Lauf mit `obs_version = 1` über
  2 048 Schritte inklusive Evaluation und ein 2v2-Lauf mit
  `obs_version = 2` über 3 072 Schritte liefen `completed`; die
  Evaluations-API lädt den 190er-Checkpoint ohne manuelle Layout-Angabe.
* **Live-Vorstufen:** `GET /api/live/preflight` und
  `POST /api/live/launch` mit `dry_run = true` sind getestet; der Dry-Run
  schreibt `match.toml`, `bot-N.toml` und `launch.json`, parst die Datei mit
  `rlbot.config.load_match_config` und startet nichts.
* **RLBot-Parser statt Vermutung:** Beim Roundtrip aller generierten
  Match-Configs durch den echten RLBot-Parser schlug jede Nicht-Heatseeker-
  Datei mit `ConfigParsingException: Invalid value 'Soccer'` fehl. RLBot v5
  erwartet `Soccar` (Psyonix-Schreibweise). Der Fehler ist behoben, alle vier
  Modi und beide Fill-Arten werden akzeptiert; `test_live.py` prüft den
  Roundtrip und einen gezielt manipulierten `"Soccer"`-Fall.
* **CLI-Live-Pfad:** `rlai live --preflight` meldet alle Prüfungen mit
  Rückgabecode 1 außerhalb Windows, `--dry-run` schreibt und validiert die
  Dateien, `--list` zeigt laufende/beendete Matches mit letzter Logzeile,
  `--timeout` beendet den Launcher nach N Sekunden. `human_team`, `fill` und
  `no_human` sind in CLI, API und Tests abgedeckt.
* **Bot-Entrypoint:** `rlbot/bot.py` akzeptiert `--model`, `--mode`,
  `--data-dir`, `--log-level`, schreibt `data/logs/bot.log` und gibt bei
  fehlendem Checkpoint Code 2 zurück.
* **Resume nach dem Trainer-Rewrite (manuell):** Lauf `smoke-1v1` bis
  Schritt 2 048 (`completed`), dann `--resume-from` des finalen Checkpoints
  bis 3 072. Log bestätigt `restored opponent-pool ratings from checkpoint`
  und den dokumentierten Fallback auf den festen Gegner; Schrittstand,
  Modell und Optimizer wurden geladen. Dabei fiel auf, dass ein per
  `--run-id` ohne vorherigen API-Lauf gestarteter Prozess keine
  `runs`-Zeile hatte; der Trainer legt sie jetzt selbst an
  (`test_trainer.py`, `test_cli.py`).
* **`best.json`:** Der Zeiger wird nur bei einem Sieg gesetzt; die 0:0-Läufe
  oben haben ihn daher nicht verändert. Die Regel ist als `is_best_candidate`
  unit-getestet.
* **Alte Checkpoints:** Ein 190-dimensionaler v1-Checkpoint wurde nach der
  Obs-Änderung geladen und evaluiert (`test_legacy_v1_policy_is_still_evaluable`).
* **Begrenzter Lernnachweis:** `configs/learning-smoke-1v1.json` lief
  150 528 Schritte (2 Worker, ~99 s, ~1 900 Schritte/s). Mittlere
  Episodenbelohnung 1,56 → 5,36, explained variance 0,22 → 0,79,
  Value-Loss 0,242 → 0,125, Entropie 4,50 → 3,80. 5 von 294
  Trainings-Episoden hatten ein Tor. Die deterministische Evaluation blieb
  bei 0W/4D (0:0), stochastisch 2 Ballberührungen in 4 Spielen. Das ist ein
  Optimierungs-, kein Spielstärkenachweis.
* **Keine Spielstärke-Aussage:** Es gab keinen Langlauf und keinen
  mehrfach gesäten Vergleich v1 gegen v2. Kurzläufe belegen nur, dass die
  Pipeline funktioniert.
* **CI:** `.github/workflows/verify.yml` installiert die CPU-Variante von
  Torch, führt `ruff check src tests rlbot` und `pytest -q` unter Python 3.11
  aus. Der Workflow wurde hier nicht auf GitHub ausgeführt; die Prüfung
  erfolgt beim Pull Request.

## Nicht getestet

| Punkt | Grund |
| --- | --- |
| Bot in einem echten RLBot-Match | kein Windows, kein Rocket League, kein RLBot-Start |
| Akzeptanz der generierten `match.toml` durch RLBot v5 | **erledigt**: Parser-Roundtrip mit `rlbot.config.load_match_config`; ein echter Matchstart fehlt weiterhin |
| GPU-Training | keine GPU vorhanden |
| Langzeittraining / Spielstärke | bewusst nicht gestartet (Ressourcen, keine belastbare Basis) |
| Parität mit echten GamePackets | nur synthetische Pakete aus Simulatorzuständen |
| Maus-/Tastatur-Interaktion im Dashboard | Klickpfade über API getestet; die neuen Live-Schaltflächen wurden hier nicht in einem Browser gerendert |
| Zielwerte der Schwierigkeitsstufen (0/50/75/100 %) | **keine Messung**: gemessen wurde nur der Mechanismus (Ballberührungen). Eine Validierung braucht je Stufe eine vollständige Evalutionsserie |
| Parsen echter Rocket-League-Replays | kein Replay-Parser verfügbar; die Analyse nutzt die protokollierten Spiel- und Episodenzeilen und sagt das im UI auch |
| Speicherpfad des Wettbewerbs- und Spieler-Gedächtnisses über einen echten Langlauf | nur Unit- und API-Tests mit künstlichen Zeilen |
| Adaptive Match Dynamics über mehrere Stufen | Unit-Tests decken Grenzen, Hysterese und Zustandswiederherstellung ab; ein langer Lauf hat es hier nicht gegeben |
| Wirkung der adaptiven Dynamik über mehrere Anpassungen | in einem 60 000-Schritte-Smoke einmal gemessen (0 % Siegquote → Gegner leichter: `mistake_rate` 0,18 → 0,38); eine Serie über mehrere Anpassungen gab es hier nicht |

## Bekannte Warnungen

* FastAPI `on_event` ist veraltet (Lifespan-Empfehlung), funktioniert aber.
* `numpy` 1.26 mit Torch 2.9 erzeugt keine Fehler; die Deprecation-Warnung
  beim Konvertieren von Ein-Element-Arrays wurde behoben.
* Der RLBot-Import im Repo löst trotz des Ordners `rlbot/` auf das
  installierte Paket auf (verifiziert).
