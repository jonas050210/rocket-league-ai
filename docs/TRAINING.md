# Training

## PPO

Implementiert in `src/rlai/ppo.py`, ohne Stable-Baselines-Abhängigkeit:

* getrennte Actor- und Critic-MLPs (Standard: 256×256, Tanh),
  orthogonale Initialisierung, kleine Ausgangsverstärkung für den Actor;
* GAE mit konfigurierbarem `gamma`/`gae_lambda`, Advantage-Normalisierung;
  Zeitlimit-Enden werden als Truncation behandelt und bootstrappen aus dem
  Wert des Endzustands (`truncated`/`bootstrap` im Rollout). Ohne diese
  Korrektur wirkt jede 60-s-Episode wie eine Niederlage;
* Clipping (`clip_coef`), Value-Loss mit eigenem Clipping (`clip_value`),
  Entropie-Bonus, globales Gradient-Clipping;
* optionales Early-Stop pro Update, wenn die approximierte KL-Divergenz
  `target_kl` überschreitet (verhindert zu große Schritte bei kleinen
  Batches); `target_kl = 0` deaktiviert die Prüfung;
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
* Der Beobachtungsvektor ist 198-dimensional (Version 2), die Aktion diskret
  (90). Checkpoints mit der alten 190er-Beobachtung bleiben über
  `obs_version = 1` nutzbar.
* `n_workers` sollte die Anzahl physischer Kerne nicht überschreiten.
  Auf einer 2-vCPU-Maschine sind 2–3 Worker sinnvoll, nicht 8.

## Beobachtungsversionen

| Version | Größe | Inhalt | Verwendung |
| --- | --- | --- | --- |
| 1 | 190 | nur relative Merkmale | alte Checkpoints, Vergleich |
| 2 (Standard) | 198 | zusätzlich 8 Werte Feldrahmen (Position, Blickrichtung, Ballposition, Torausrichtung, Tordistanz) | neue Läufe |

Die Feldrahmen-Werte sind keine Lerntheorie, sondern notwendige Information:
Die Tore stehen an festen Weltkoordinaten, die relativen Blöcke enthalten
keinen Anker dorthin. Ein Netz kann die Torposition aus relativen Vektoren
nicht rekonstruieren. Ein Kurzlauf belegt die Funktion der Pipeline, nicht
einen Spielstärkegewinn; ein belastbarer Vergleich von Version 1 und 2 würde
deutlich längere, mehrfach gesäte Läufe erfordern und wurde hier nicht
durchgeführt.

## Liga, Gegnerpool und Curriculum

* `self_play = true` füllt einen begrenzten Pool (`src/rlai/league.py`) mit
  eingefrorenen Snapshots (Standard: 8). Jeder Snapshot hat eine Elo-Zahl.
* Die Auswahl nutzt PFSP-Gewichte: Begegnungen nahe 50 % werden bevorzugt,
  damit der Lernende nicht überwiegend gegen deutlich schwächere Kopien
  spielt. Ein Mindestgewicht hält jede Kopie samplerbar.
* Ergebnisse aus Rollouts aktualisieren die Elo-Zahlen (k = 16); die
  Pool-Bilanz wird im Checkpoint gespeichert und beim Resume
  wiederhergestellt. Die Modellgewichte der Pool-Einträge werden bewusst
  nicht serialisiert (sie sind groß und aus Checkpoints reproduzierbar).
* `curriculum = true` aktiviert gestufte Gegner mit messbaren
  Aufstiegskriterien (`src/rlai/curriculum.py`):

| Modus | Stufen |
| --- | --- |
| 1v1 | Ballkontrolle (`idle`) → `chase` → gemischt (`chase`/`random`) → Selbstspiel |
| 2v2, 3v3 | `chase` → gemischt → Selbstspiel |
| Heatseeker 2v2 | Ballkontrolle → `chase` → Selbstspiel |

Aufgestiegen wird nur, wenn über ein Fenster abgeschlossener Episoden die
Ziel-Siegquote erreicht ist und Mindestiterationen sowie Mindestepisoden
erfüllt sind. Die Kriterien sind bewusst kurz; sie sind ein Startpunkt und
keine empirisch optimierten Schwellen. Die Stufe, die Iterationszähler und
die Pool-Bilanz stehen im Checkpoint und werden beim Resume fortgesetzt.

Der Nutzen von PFSP ist in diesem Projekt **nicht gemessen**. Der Pool ist
implementiert, weil er die Datenverteilung im Selbstspiel kontrolliert; ob er
auf 2 vCPUs in kurzen Läufen hilft, ist offen.

## Metriken

Der Trainer schreibt pro Iteration:

| Metrik | Bedeutung |
| --- | --- |
| `charts/episodic_return` | mittlere Episodenbelohnung |
| `charts/win_rate` | Siegquote der letzten Episoden |
| `losses/policy`, `losses/value`, `losses/entropy` | PPO-Verluste |
| `losses/approx_kl`, `losses/clipfrac` | Update-Diagnose |
| `losses/epochs_run`, `losses/early_stopped` | Early-Stop-Diagnose |
| `charts/goals_for`, `charts/goals_against` | Tore pro Episode |
| `eval/win_rate`, `eval/win_rate_low`, `eval/win_rate_high`, `eval/goal_diff` | periodische Evaluation mit 95-%-Wilson-Intervall |
| `eval/touches`, `eval/draw_rate` | Ballberührungen und Remis-Anteil der Evaluation |
| `curriculum/stage`, `curriculum/window_win_rate` | aktuelle Stufe und Fenster-Siegquote |
| `league/agent_elo`, `league/pool_size` | Elo des Lernenden, Poolgröße |
| `time/sps` | Umgebungsschritte pro Sekunde |
| `episodes/*` | Länge, Tore, Ergebnis pro Episode (SQLite-Tabelle) |

Das Dashboard liest diese Werte über SQLite und SSE. Es gibt keine
synthetischen Kurven; wenn keine Daten existieren, zeigt die Oberfläche
„No data recorded yet“. Evaluationszeilen speichern zusätzlich das
Wilson-Intervall im JSON-Feld und zeigen es in der Verlaufsansicht.

### Deterministisch vs. stochastisch evaluieren

Die Standard-Evaluation wählt `argmax` (deterministisch), weil der RLBot-Bot
genau so spielt. Bei einem kaum trainierten Netz kann `argmax` dauerhaft
dieselbe Aktion wählen – die Evaluation misst dann eine degenerierte
Konstante, nicht das Verhalten der Trainings-Policy. Für Frühdiagnose gibt es
deshalb zusätzlich:

```bash
python -m rlai.cli evaluate --model <pfad> --opponent idle --games 4 --match-seconds 20 --stochastic
```

`--stochastic` sampelt Aktionen gemäß der Policy. Beide Werte gehören getrennt
berichtet; der deterministische Wert ist die Deployment-Metrik.

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
* `best.json` ist ein reiner Messzeiger: Er zeigt auf den Checkpoint mit der
  besten dokumentierten Evaluation (Siegquote, Wilson-Intervall in der
  Evaluationstabelle). Er wird nie schlechter und ersetzt **nicht** die
  stabile Promotion. Ein Checkpoint ohne Sieg wird nicht als „best“
  markiert.
* Resume: `python -m rlai.cli train --config-file ... --run-id ... --resume-from data/models/<id>/model.pt`.
  Das Dashboard bietet denselben Pfad über die Run-Ansicht an. Modell,
  Optimizer, Schrittstand, Liga-Bilanz und Curriculum-Stufe werden
  wiederhergestellt; die Policy-Architektur muss exakt passen.

## Begrenzter Lernnachweis (kein Langlauf)

Ein bewusst kurzer Lauf (`configs/learning-smoke-1v1.json`, 2 Worker,
1v1 gegen `idle`, 150 528 Schritte, 99 s, ~1 900 Schritte/s) zeigt, dass die
Optimierung tatsächlich läuft:

| Metrik | erste 10 Iterationen | letzte 10 Iterationen |
| --- | --- | --- |
| `charts/episodic_return` | 1,56 | 5,36 |
| `losses/explained_variance` | 0,22 | 0,79 |
| `losses/value` | 0,242 | 0,125 |
| `losses/entropy` | 4,50 | 3,80 |

Zusätzlich wurden 5 von 294 Trainings-Episoden mit mindestens einem Tor
beendet (maximal 1 Tor). Die periodische Evaluation (4 Spiele à 20 s,
deterministisch) blieb bei 0W 0D 4D, 0:0; stochastisch wurden 2
Ballberührungen in 4 Spielen gemessen. Das belegt **Optimierung, keine
Spielstärke**: Das Netz hängt nach 150k Schritten noch fast konstant an einer
Aktion. Für verwertbares Spiel sind Millionen Schritte nötig; ein solcher
Lauf wurde ohne explizite Freigabe nicht gestartet.

## Empfohlener Ablauf

1. **Ballkontrolle** – kurze Episoden, Gegner `idle`, dichte Touch- und
   Annäherungsbelohnung (`ball-control-warmup`).
2. **1v1 gegen Chaser** – erster echter Match-Baseline (`1v1-vs-chaser`).
3. **1v1 gegen feste Baselines** – Evaluation gegen `idle`, `random`, `chase`
   mit festen Seeds; erst danach Selbstspiel.
4. **1v1 Selbstspiel** – Gegnerpool aus eingefrorenen Snapshots
   (`1v1-selfplay`). Wer die Gegnerstufen automatisch weiterschalten will,
   nimmt `1v1-curriculum` (auch als Preset im Dashboard).
5. **2v2 / 3v3** – vom 1v1-Checkpoint starten, Teamgröße erhöhen
   (`2v2-teamplay`, `3v3-teamplay`).
6. **Heatseeker 2v2** – separat trainieren (`heatseeker-2v2`).

Diese Stufen sind ein Rezept, kein Nachweis von Spielstärke. Ob eine Stufe
besser ist, entscheidet die Evaluation mit festen Seeds, nicht die
Trainingskurve allein. Die automatische Curriculum-Variante ist ein
Gegner-Scheduler, kein Qualitätsbeweis.

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

* **Feldrahmen statt nur relativer Merkmale.** Tore haben feste
  Weltkoordinaten; ein relatives Beobachtungsbild enthält diese Information
  nicht. Die 8 zusätzlichen Werte sind deshalb die minimale Ergänzung, die
  Torrichtung und -distanz überhaupt beobachtbar macht. Ein Spielstärkenachweis
  steht aus (kein Langzeittraining).
* **MLP statt Attention.** Die Beobachtung ist ein fester 198-Vektor, die
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
* **Selbstspiel optional, Pool statt ungewichteter Zufall.** Der Chaser ist
  ein reproduzierbarer, billiger Baseline-Gegner. Selbstspiel startet erst
  danach, weil ein eingefrorener Anfänger-Pool sonst nur Anfängerfehler
  verstärkt. Im Pool verhindert PFSP-Gewichtung, dass überwiegend gegen
  deutlich schwächere Kopien gespielt wird. Der Nutzen ist hier nicht
  gemessen; die Poolgröße bleibt klein, weil Pickling und Speicher auf
  2 vCPUs teuer sind.
* **Checkpoints statt "bestes Modell automatisch".** Automatische Promotion
  würde bei verrauschten Kurzläufen schlechte Modelle stabil setzen; die
  Entscheidung bleibt explizit und wird durch eine Evaluation gestützt.
  `best.json` dokumentiert nur die bisher beste Messung, es deployt nichts.
