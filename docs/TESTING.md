# Testmatrix und Messergebnisse

Umgebung: Linux (Sandbox), 2 vCPU, 3,85 GB RAM, **keine GPU**, kein
Rocket-League-Installat, kein RLBot-Match. Python 3.11.2, Torch 2.9.1,
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
| Belohnungen: Tor/Gegentor, Shaping ohne Tor | `test_rewards.py` | bestanden |
| PPO: GAE, Dones, ein Update, Batch-Form | `test_ppo.py` | bestanden |
| Checkpoints: atomar, Promotion, Löschschutz | `test_checkpoints.py` | bestanden |
| SQLite-Store: Runs, Metriken, Logs, Episoden | `test_store.py` | bestanden |
| HTTP-API inkl. Auth und Statik | `test_api.py` | bestanden |
| Evaluation: vollständiges Ergebnis, Seed-Reproduzierbarkeit | `test_evaluation.py` | bestanden |
| RLBot-Parität: Packet → Beobachtung (1v1, 3v3, Heatseeker) | `test_rlbot_adapter.py` | bestanden |
| Bot: Checkpoint laden, Inferenz, deterministische Aktionen, fehlendes Modell | `test_rlbot_bot.py` | bestanden |
| JobManager: Steuerdateien, Subprozess-Umgebung | `test_jobs.py` | bestanden |
| Alle ausgelieferten Konfigurationen laden und validieren | `test_modes.py` | bestanden |
| Live-Match-TOML für alle vier Modi | `test_live.py` | bestanden |
| Live-Start außerhalb Windows wird mit 501 abgelehnt | `test_api.py` | bestanden |

**Gesamt: 76 passed, 0 failed, 0 skipped** (ca. 4,6 s).

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
* Alle sieben Dashboard-Routen in einem headless Chromium gerendert:
  kein JavaScript-Fehler, kein horizontaler Überlauf, kein hängender
  Ladezustand.

## Durchsatz (gemessen, nicht geschätzt)

Benchmark mit Zufallsaktionen, 2 Worker, 2 Sekunden je Modus (`/api/benchmark`):

| Modus | Umgebungsschritte/s | Simulator-Ticks/s |
| --- | --- | --- |
| soccer-1v1 | 6 261 | 50 090 |
| soccer-2v2 | 2 815 | 22 518 |
| soccer-3v3 | 1 899 | 15 192 |
| heatseeker-2v2 | 2 936 | 23 485 |

Mit PPO-Update im selben Prozess (echte Trainingsläufe):

| Lauf | Worker | Schritte/s |
| --- | --- | --- |
| smoke-1v1 | 1 | 810 |
| smoke-controls (1v1) | 1 | 817 |
| smoke-heatseeker-2v2 | 2 | 1 138 |
| smoke-soccer-2v2 | 2 | 1 123 |
| smoke-soccer-3v3 | 2 | 782 |

Einordnung: Das sind Funktionsnachweise auf 2 vCPUs. Sie sagen nichts über
Lernqualität aus. Für 3v3 ist die Rate erwartbar niedriger (6 Autos,
aufwendigere Kollisionen).

## Nicht getestet

| Punkt | Grund |
| --- | --- |
| Bot in einem echten RLBot-Match | kein Windows, kein Rocket League, kein RLBot-Start |
| Akzeptanz der generierten `match.toml` durch RLBot v5 | nur Schema-/Doku-Abgleich |
| GPU-Training | keine GPU vorhanden |
| Langzeittraining / Spielstärke | bewusst nicht gestartet (Ressourcen, keine belastbare Basis) |
| Parität mit echten GamePackets | nur synthetische Pakete aus Simulatorzuständen |
| Maus-/Tastatur-Interaktion im Dashboard | headless geprüft, Klickpfade über API getestet |

## Bekannte Warnungen

* FastAPI `on_event` ist veraltet (Lifespan-Empfehlung), funktioniert aber.
* `numpy` 1.26 mit Torch 2.9 erzeugt keine Fehler; die Deprecation-Warnung
  beim Konvertieren von Ein-Element-Arrays wurde behoben.
* Der RLBot-Import im Repo löst trotz des Ordners `rlbot/` auf das
  installierte Paket auf (verifiziert).
