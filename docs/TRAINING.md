# Training

## PPO

Implementiert in `src/rlai/ppo.py`, ohne Stable-Baselines-Abhängigkeit:

* getrennte Actor- und Critic-MLPs (Standard: 256×256, Tanh),
  orthogonale Initialisierung, kleine Ausgangsverstärkung für den Actor;
* GAE mit konfigurierbarem `gamma`/`gae_lambda`, Advantage-Normalisierung;
* Clipping (`clip_coef`), Value-Loss, Entropie-Bonus, globales
  Gradient-Clipping;
* lineares LR-Annealing über den Lauf;
* mehrere Epochen und Minibatches pro Rollout.

Standardwerte orientieren sich an verbreiteten PPO-Rezepten (LR 3e-4,
γ 0.995, λ 0.95, Clip 0.2, 4 Epochen) und sind in `configs/*.json`
dokumentiert. Sie sind **Startwerte**, keine empirisch optimierten
Hyperparameter für gute Spielstärke.

## Parallele Umgebungen

`src/rlai/workers.py` startet pro Umgebung einen Prozess (`spawn`, damit
Windows funktioniert). Jeder Worker besitzt eine eigene RocketSim-Arena und
führt `rollout_steps` Entscheidungen aus. Der Trainer schickt die aktuellen
Gewichte, sammelt die Rollouts ein und trainiert zentral.

* Entscheidungsintervall: `tick_skip = 8` Ticks = 66,7 ms bei 120 Hz.
* Aktionen werden pro Entscheidung wiederholt (RLBot-kompatibel).
* Der Beobachtungsvektor ist 190-dimensional, die Aktion diskret (90).
* `n_workers` sollte die Anzahl physischer Kerne nicht überschreiten.
  Auf einer 2-vCPU-Maschine sind 2–3 Worker sinnvoll, nicht 8.

## Metriken

Der Trainer schreibt pro Iteration:

| Metrik | Bedeutung |
| --- | --- |
| `charts/episodic_return` | mittlere Episodenbelohnung |
| `charts/win_rate` | Siegquote der letzten Episoden |
| `losses/policy`, `losses/value`, `losses/entropy` | PPO-Verluste |
| `time/sps` | Umgebungsschritte pro Sekunde |
| `episodes/*` | Länge, Tore, Ergebnis pro Episode (SQLite-Tabelle) |

Das Dashboard liest diese Werte über SQLite und SSE. Es gibt keine
synthetischen Kurven; wenn keine Daten existieren, zeigt die Oberfläche
„No data recorded yet“.

## Checkpoints und Resume

* Checkpoints liegen unter `data/models/<model-id>/` und enthalten Modell,
  Optimizer, Policy-Konfiguration, Schrittstand und Trainingskonfiguration.
* Schreiben ist atomar (temporäre Datei + `os.replace`).
* Vorhandene IDs werden nie überschrieben; bei Kollision wird ein Suffix
  angehängt.
* Der stabile Zeiger (`stable.json`) wird nur durch eine explizite Promotion
  gesetzt; die vorherige stabile Version wird archiviert, nicht gelöscht.
* `keep_checkpoints` begrenzt Kandidaten; stabile Modelle und der letzte
  Checkpoint bleiben immer erhalten.
* Resume: `python -m rlai.cli train --config-file ... --run-id ... --resume-from data/models/<id>/model.pt`.
  Das Dashboard bietet denselben Pfad über die Run-Ansicht an.

## Curriculum (empfohlen)

1. **Ballkontrolle** – kurze Episoden, Gegner `idle`, dichte Touch- und
   Annäherungsbelohnung (`ball-control-warmup`).
2. **1v1 gegen Chaser** – erster echter Match-Baseline (`1v1-vs-chaser`).
3. **1v1 gegen feste Baselines** – Evaluation gegen `idle`, `random`, `chase`
   mit festen Seeds; erst danach Selbstspiel.
4. **1v1 Selbstspiel** – Gegnerpool aus eingefrorenen Snapshots
   (`1v1-selfplay`).
5. **2v2 / 3v3** – vom 1v1-Checkpoint starten, Teamgröße erhöhen
   (`2v2-teamplay`, `3v3-teamplay`).
6. **Heatseeker 2v2** – separat trainieren (`heatseeker-2v2`).

Diese Stufen sind ein Rezept, kein Nachweis von Spielstärke. Ob eine Stufe
besser ist, entscheidet die Evaluation mit festen Seeds, nicht die
Trainingskurve allein.

## Reproduzierbarkeit

* `seed` steuert Kickoffs, Gegner und Sampling.
* Evaluation nutzt feste Seeds pro Spiel und deterministische Aktionen.
* Gleiche Seeds + gleiche Konfiguration ergeben in dieser Umgebung identische
  Episodenlängen und Ergebnisse (`tests/test_evaluation.py`).
* Worker-Prozesse erhalten `seed + worker_index`, damit sie nicht dieselben
  Zufallszahlen ziehen.

## Grenzen

* Es gibt keine Garantie, dass ein Smoke-Lauf eine sinnvolle Policy erzeugt.
  Die mitgelieferten Läufe sind ausdrücklich Funktionsnachweise.
* Auf CPU ist 3v3 mit 8 Workern langsam; das Dashboard zeigt die gemessene
  Rate. GPU beschleunigt nur den PPO-Update-Schritt, nicht die Simulation.
* Langfristiges Training wurde hier nicht durchgeführt (Ressourcenlimit).

## Designentscheidungen (evidenzbasiert)

* **MLP statt Attention.** Die Beobachtung ist ein fester 190-Vektor, die
  Entscheidungsrate liegt bei 15 Hz. Ein Aufmerksamkeitsmodell über sechs
  Auto-Slots würde Parameter und Latenz erhöhen, ohne dass ein Engpass
  nachgewiesen ist. Wenn 3v3-Evaluationen zeigen, dass die feste Slot-Reihenfolge
  limitiert, ist der Obs-Builder die Stelle für eine Aufmerksamkeitsvariante.
* **Zentrales PPO statt verteiltem Trainer.** Der gemessene Durchsatz liegt bei
  6 261 Umgebungsschritten/s (1v1, 2 Worker) bzw. 1 899 (3v3, 2 Worker) auf
  2 vCPU. Das Netz hat ~250k Parameter; ein verteilter Parameter-Server würde
  mehr Ausfallmodi als Geschwindigkeit bringen.
* **Frame-Skip 8.** Entspricht 66,7 ms Entscheidungen und damit dem
  RLBot-Tick-Muster; der Wechsel auf 4 oder 16 ist konfigurierbar.
* **Selbstspiel optional.** Der Chaser ist ein reproduzierbarer, billiger
  Baseline-Gegner. Selbstspiel startet erst danach, weil ein eingefrorener
  Anfänger-Pool sonst nur Anfängerfehler verstärkt. Ein Liga-/PFSP-System ist
  bewusst nicht eingebaut: Für seinen Nutzen gibt es in diesem Projekt noch
  keine Messung.
* **Checkpoints statt "bestes Modell automatisch".** Automatische Promotion
  würde bei verrauschten Kurzläufen schlechte Modelle stabil setzen; die
  Entscheidung bleibt explizit und wird durch eine Evaluation gestützt.
