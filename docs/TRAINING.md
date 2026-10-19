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
* Snapshots sind echte Kopien: `freeze_state` legt für jeden Pool-Eintrag
  eigene Tensoren an (`detach().to("cpu", copy=True)`), weil
  `state_dict()`-Tensoren eines CPU-Modells sonst Speicher der lebenden
  Parameter aliasieren und der Pool still zu „immer die aktuelle Policy"
  degeneriert. `tests/test_continuation.py` prüft, dass sich ein Snapshot
  nach drei Adam-Schritten mit lr = 0.5 nicht verändert.
* Ergebnisse aus Rollouts aktualisieren die Elo-Zahlen (k = 16). Bilanz
  **und Gewichte** der Pool-Einträge werden im Checkpoint gespeichert
  (`league_states = true`, Standard) und beim Resume wiederhergestellt, damit
  Selbstspiel gegen dieselben Gegner weiterläuft. Ein Snapshot kostet etwa
  eine Policy (~1 MB bei 256×256), maximal `capacity` Stück; mit
  `league_states = false` werden nur die Bilanzen gespeichert – dann meldet
  der Trainer laut, dass der Pool bis zum nächsten Snapshot leer bleibt.
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
### Fortsetzung: drei explizite Modi

| Modus | Bedeutung | Was wiederhergestellt wird |
| --- | --- | --- |
| `fresh` | Standard, wenn nichts angegeben wird | nichts – neue Policy, Schritt 0 |
| `continue` | Lauf fortsetzen | Gewichte, Optimizer, Schrittstand, Iteration, RNG (numpy/torch/python/CUDA), Curriculum-Stufe, Gegnerpool inkl. Gewichten, geerbte Config-Felder (`gamma`, `learning_rate`, …), Elternlauf |
| `transfer` | Fine-Tuning auf neuer Aufgabe | nur die Gewichte; Schrittzähler, Optimizer, RNG und Curriculum beginnen neu |

```bash
# Vorschau ohne Training – zeigt Kandidaten, Ablehnungen und was fehlt
python -m rlai.cli resume-plan --config-file configs/1v1-vs-chaser.json --run-id <run-id>
python -m rlai.cli train --config-file ... --plan-only

# neuester gültiger, kompatibler Checkpoint der Lauf-Linie
python -m rlai.cli train --config-file ... --auto-resume --run-id <kind> --max-steps 40000

# explizit, oder bewusst nur Gewichte
python -m rlai.cli train --config-file ... --resume-from <model-id> --resume-mode continue
python -m rlai.cli train --config-file ... --resume-from <model-id> --resume-mode transfer
```

Regeln, die `src/rlai/resume.py` durchsetzt:

* Kompatibilität wird **vor** dem Laden geprüft (`validate_checkpoint`):
  Dateiformat, endliche Gewichte, exakt passende Policy-Architektur,
  Beobachtungsversion und Modus. Jede Ablehnung nennt den Grund.
* Automatische Fortsetzung fällt auf einen älteren Kandidaten zurück, wenn der
  neueste defekt oder inkompatibel ist – mit Warnung und Eintrag im Report
  (`fallback_used`). Ein **namentlich** angeforderter Checkpoint wird dagegen
  niemals ersetzt: ist er unbrauchbar, startet der Lauf nicht
  (`explicit_unavailable`).
* Nichts wird stillschweigend neu begonnen. Fehlt der Optimizer- oder
  RNG-Zustand (alte Checkpoints), läuft das Resume trotzdem, sagt aber im Log,
  im Plan und im Report, was fehlt und was das bedeutet.
* Die API verweigert den Start, bevor eine Run-Zeile angelegt wird
  (`POST /api/runs` → 400), und meldet 409, wenn bereits ein Lauf aktiv ist
  (`allow_concurrent` setzt das bewusst außer Kraft).
* `parent_run` und `resumed_step` werden in der Datenbank gespeichert; der
  Report zählt deshalb nur die Schritte, die dieser Prozess wirklich gemacht hat.

### Report nach jedem Lauf

`src/rlai/report.py` schreibt nach jedem Lauf – abgeschlossen, gestoppt oder
fehlgeschlagen – `data/runs/<id>/report.json` und eine Zeile in der Tabelle
`reports`. Inhalt: Schritte von/bis/gesamt/Ziel, Dauer, **zwei**
Durchsatzzahlen (Wandzeit inkl. Worker-Start/Evaluation/Checkpointing und
Iterationen ohne diese Anteile, jeweils mit Angabe der Basis), Gerät, Worker,
Plattform, torch-Threads, vollständige Konfiguration, Resume-Herkunft,
PPO-Kennzahlen (first/last/mean/min/max), Zeitanteile pro Phase,
Belohnungskomponenten, Episoden (dieser Lauf, gesamt, letzte 100),
Evaluation vor/nach/periodisch inkl. Protokoll und Wilson-Intervall,
Gültigkeit des finalen Checkpoints, Kurven, Vergleichsläufe, Warnungen.

Before/After wird **nur** als Änderung berichtet, wenn Gegner und Protokoll
identisch sind (Modus, Spiele, Matchlänge, Episodenlänge, Seiten, Spiegelung,
Sampling, Beobachtungsversion). Unterschiedliche Kickoff-Seeds sind erlaubt und
werden als `same_seed: false` markiert, weil der Trainer pro Phase einen neuen
Seed ableitet – die Serien sind dann unabhängige Stichproben desselben
Protokolls. Alles Nicht-Gemessene steht als `{"available": false, "reason": …}`
im Report; eine fehlende Evaluation wird nie als 0 % interpretiert.

```bash
python -m rlai.cli report --run-id <id> [--summary] [--rebuild]
python -m rlai.cli report --compare <run-a>,<run-b>
```

## Trainingszeit

Die Trainingszeit wird **gemessen**, nicht geschätzt: `src/rlai/timing.py`
benutzt eine monotone Uhr für die Dauer und eine UTC-Uhr nur für die
Zeitpunkte. Der Trainer öffnet beim Start eine Sitzung, schließt sie beim
Ende (auch bei Stopp und Fehler) und schreibt die gemessene Dauer in die
Tabelle `sessions`, in die Run-Zeile und in jeden Checkpoint
(`extra.timing`).

* **Aktuelle Sitzung** – die Zeit des laufenden Prozesses.
* **Akkumuliert** – die Summe aller beendeten Sitzungen des Laufs, inklusive
  der Zeit, die eine Fortsetzung aus dem Checkpoint übernimmt.
* **Checkpoint** – trägt die akkumulierte Zeit, die **bis zum Schreiben**
  gemessen war, plus die Dauer der laufenden Sitzung. Ein Checkpoint kann
  nie behaupten, dass weitertrainiert wurde, während der Prozess gestoppt
  war.
* **Absturz** – eine offen gelassene Sitzung wird beim nächsten Start als
  *interrupted* markiert; ihre Dauer bleibt unbekannt, weil ein Absturz die
  eigene Laufzeit nicht messen kann. Die akkumulierte Zeit kommt aus dem
  Checkpoint.
* **Alte Checkpoints** ohne Zeit-Metadaten bleiben ladbar; die historische
  Dauer wird als *unbekannt* gemeldet, nicht als 0.

```bash
# der Headless-Abschlussbericht enthält die gemessene Zeit
python -m rlai.cli headless --config-file configs/smoke-1v1.json
```

## Statistik im GUI (Stats-Tab)

Der Stats-Tab liest `/api/stats`. Alle Zahlen kommen aus
`src/rlai/stats.py` – dieselbe Quelle, die Report und Headless-Bericht
benutzen, damit GUI und Bericht nie auseinanderlaufen. Der Tab zeigt:

* **Trainingszeit** – aktuelle Sitzung, akkumulierte Zeit, Start-/Endzeit,
  letzte Checkpoint-Zeit, Sitzungsverlauf (Status `completed` / `stopped` /
  `interrupted` / `inherited`), Historie der früheren Läufe;
* **Fortschritt** – Umweltschritte, Iteration, beendete Episoden, Status;
* **Lernmetriken** – Belohnungsverlauf, Episodenergebnisse, PPO-Verluste
  (Policy/Value/KL/Entropy/Clip-Fraktion/erklärte Varianz), Kurven und eine
  Kompakttabelle;
* **Performance** – Schritte/s (letzte Iteration und über den Lauf),
  CPU/RAM/GPU, wenn verfügbar;
* **Checkpoints** – Erstellungszeit, akkumulierte Trainingszeit bei
  Erstellung, Iteration, Schrittzahl, gemessener Eval-Score („unbekannt“,
  wenn nie gemessen);
* **Evaluation** – Siegquote je Gegner, einzelne Serien, Vergleich der
  Checkpoints nach gemessenem Score.

Kurven werden auf 400 Punkte heruntergesampelt; die GUI fragt nur die
geänderten Werte ab. Metrik-, Log- und Episoden-Zeilen sind begrenzt
(`metric_retention`, `log_retention`, `episode_retention`), Schreibvorgänge
werden gebündelt (`metric_flush_interval`).

## Weitere Merkmale

| Merkmal | Nutzfläche | Grenze |
| --- | --- | --- |
| Team Builder | Teams aus Bot-Slots (Name, Schwierigkeit, Verhalten, Modell, Rolle) speichern, prüfen, als Gegner verwenden | beschreibt Teams; die Ausführung nutzt die Gegner-Implementierung |
| Schwierigkeitsprofile | vier Stufen (Rock/Tiny/Medium/Pro) mit Zielwert **und** Mechanismus (Fehlerrate, Reaktionszeit, Boost-Effizienz) | die Prozentzahlen sind **Zielwerte**, keine Messung (`target_is_measured = false`) |
| Bot Brain | vier Verhaltensprofile, deterministisch und ohne Simulator | beschreibt Verhalten, nicht Stärke |
| Adaptive Match Dynamics | passt den skriptgesteuerten Gegner (`opponent: "difficulty:<level>"`) innerhalb konfigurierter Grenzen an; die neuen Parameter werden an die Worker übergeben und wirken sich wirklich aus | verändert nie eine trainierte Policy; braucht Messungen, Mindest-Spielzahl und Hysterese; ein Gegner ohne Schwierigkeitsstufe wird nicht angepasst |
| Rival-System | benannte Gegner mit Historie, Rangfolge, Engpass, Trend | nur gemessene Evalutionsergebnisse |
| Spielerspeicher | Begegnungen mit Notizen, begrenzt (500 Einträge / 90 Tage) | reine Daten, nie Teil der Gewichte oder des Checkpoints |
| Match-/Replay-Analyse | Auswertung der protokollierten Spiele und Episoden | echte `.replay`-Dateien können hier nicht gelesen werden; die Oberfläche sagt das |
| Bot Evaluation Lab | wiederholbare Experimente, Plan **vor** der Ausführung, Vergleich, Bericht | ein Lauf startet nur mit `execute=true` |
| Ressourcenwächter | CPU-/Speicherbudget, Warnungen vor dem Start, automatisches Klemmen der Workerzahl | Empfehlung, keine Garantie |

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
