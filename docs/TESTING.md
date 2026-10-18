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
| Moduskatalog, Konfigurationsvalidierung | `test_modes.py` | bestanden |
| Aktionstabelle (90 Aktionen, Grenzen, Mapping aller Indizes) | `test_actions.py` | bestanden |
| Alle vier Modi: Reset/Step, Beobachtungsform, Padding, Seed | `test_env_modes.py` | bestanden |
| Heatseeker: Homing, Tempoplus, Tor, unendlich Boost | `test_heatseeker.py` | bestanden |
| Belohnungen: Tor/Gegentor, Shaping ohne Tor, Touch als Event | `test_rewards.py` | bestanden |
| PPO: GAE, Dones, Truncation-Bootstrap, Early Stop, ein Update | `test_ppo.py` | bestanden |
| Beobachtung: v1/v2-Größen, Feldrahmen, Padding, Determinismus | `test_obs.py` | bestanden |
| Beobachtung v2: Größen, Torrichtung, Padding, Determinismus | `test_obs_v2.py` | bestanden |
| Checkpoints: atomar, Promotion, Löschschutz, best-Zeiger, Prune-Schutz | `test_checkpoints.py` | bestanden |
| Gegnerpool: Kapazität, PFSP-Gewichte, Elo, Serialisierung | `test_league.py` | bestanden |
| Curriculum: Aufstiegskriterien, Fenster, Resume-Zähler | `test_curriculum.py` | bestanden |
| SQLite-Store: Runs, Metriken, Logs, Episoden, Evaluationsdetails (CI) | `test_store.py` | bestanden |
| HTTP-API inkl. Auth, Statik, Preflight, Dry-Run, Presets | `test_api.py` | bestanden |
| Evaluation: Ergebnis, Seed-Reproduzierbarkeit, Wilson-Intervall, v1-Checkpoints, best-Regel | `test_evaluation.py` | bestanden |
| Evaluation v2: CI-Grenzen, Layout-Erkennung, v1-Checkpoint-Evaluation | `test_evaluation_v2.py` | bestanden |
| Trainer: fehlende Run-Zeile, Resume-Restore, Architektur-Mismatch, best-Regel | `test_trainer.py` | bestanden |
| CLI: Train-Run-Registrierung, `live --preflight/--dry-run/--list` | `test_cli.py` | bestanden |
| RLBot-Parität: Packet → Beobachtung (1v1, 3v3, Heatseeker) | `test_rlbot_adapter.py` | bestanden |
| Bot: Checkpoint laden, Inferenz, deterministische Aktionen, fehlendes Modell | `test_rlbot_bot.py` | bestanden |
| JobManager: Steuerdateien, Subprozess-Umgebung | `test_jobs.py` | bestanden |
| Alle ausgelieferten Konfigurationen laden und validieren | `test_modes.py` | bestanden |
| Live-Match-TOML für alle vier Modi, Team-/Fill-Optionen, Preflight, Dry-Run | `test_live.py` | bestanden |
| Generierte Match-Configs durch RLBot-Parser (alle Modi, beide Fill-Arten) | `test_live.py` | bestanden |
| Launcher `--dry-run` mit RLBot-Parser, Rückgabecodes | `test_live.py`, `test_cli.py` | bestanden |
| Live-Start außerhalb Windows wird mit 501 abgelehnt | `test_api.py` | bestanden |

**Gesamt: 152 passed, 0 failed, 0 skipped** (ca. 8,4 s), `ruff check src tests`
sauber.

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
* **Keine Spielstärke-Aussage:** Es gab keinen Langlauf und keinen
  mehrfach gesäten Vergleich v1 gegen v2. Kurzläufe belegen nur, dass die
  Pipeline funktioniert.

## Nicht getestet

| Punkt | Grund |
| --- | --- |
| Bot in einem echten RLBot-Match | kein Windows, kein Rocket League, kein RLBot-Start |
| Akzeptanz der generierten `match.toml` durch RLBot v5 | **erledigt**: Parser-Roundtrip mit `rlbot.config.load_match_config`; ein echter Matchstart fehlt weiterhin |
| GPU-Training | keine GPU vorhanden |
| Langzeittraining / Spielstärke | bewusst nicht gestartet (Ressourcen, keine belastbare Basis) |
| Parität mit echten GamePackets | nur synthetische Pakete aus Simulatorzuständen |
| Maus-/Tastatur-Interaktion im Dashboard | Klickpfade über API getestet; die neuen Live-Schaltflächen wurden hier nicht in einem Browser gerendert |

## Bekannte Warnungen

* FastAPI `on_event` ist veraltet (Lifespan-Empfehlung), funktioniert aber.
* `numpy` 1.26 mit Torch 2.9 erzeugt keine Fehler; die Deprecation-Warnung
  beim Konvertieren von Ein-Element-Arrays wurde behoben.
* Der RLBot-Import im Repo löst trotz des Ordners `rlbot/` auf das
  installierte Paket auf (verifiziert).
