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
* Der Beobachtungsvektor ist 198-dimensional (Version 2, Standard), die Aktion
  diskret (90). Version 3 (204 Werte, `obs_version = 3`) ergänzt die eigene
  Linear- und Winkelgeschwindigkeit – ohne sie ist die eigene Geschwindigkeit
  für die gedächtnislose Policy nicht beobachtbar. Checkpoints mit der alten
  190er-Beobachtung bleiben über `obs_version = 1` nutzbar; ein Resume mischt
  keine Versionen (die Validierung lehnt das mit Begründung ab).
* `n_workers` sollte die Anzahl physischer Kerne nicht überschreiten.
  Auf einer 2-vCPU-Maschine sind 2–3 Worker sinnvoll, nicht 8.
* **Episodenende (Truncation).** Neben dem Zeitlimit `episode_seconds` gibt es
  `no_touch_timeout_seconds` (Standard 0 = aus; empfohlene 1v1-Configs: 10 s):
  Berührt die Policy den Ball so lange nicht, endet die Episode vorzeitig.
  Eine frische Policy fährt sonst die volle Horizontzeit auf einen Ball zu,
  den sie nie erreicht – ein langer, fast gradientenfreier Versuch statt
  mehrerer kurzer. Truncations sind **keine** Terminals: `rlai.ppo.compute_gae`
  bootstrappt mit `V(s_end)` (siehe `tests/test_gae_truncation.py`), der
  Wertschätzer lernt also kein „Zeitlimit = verlorenes Spiel“.

## Beobachtungsversionen

| Version | Größe | Inhalt | Verwendung |
| --- | --- | --- | --- |
| 1 | 190 | nur relative Merkmale | alte Checkpoints, Vergleich |
| 2 (Standard) | 198 | zusätzlich 8 Werte Feldrahmen (Position, Blickrichtung, Ballposition, Torausrichtung, Tordistanz) | bestehende Checkpoints |
| 3 | 204 | zusätzlich eigene Linear-/Winkelgeschwindigkeit (Auto-Rahmen) | Übergang |
| 4 | 222 | zusätzlich eigener Up-Vektor (Welt) + Blickrichtung aller anderen Autos (Auto-Rahmen) | **neue Läufe** (`configs/1v1-curriculum.json`) |

Die Feldrahmen-Werte sind keine Lerntheorie, sondern notwendige Information:
Die Tore stehen an festen Weltkoordinaten, die relativen Blöcke enthalten
keinen Anker dorthin. Ein Netz kann die Torposition aus relativen Vektoren
nicht rekonstruieren. Ein Kurzlauf belegt die Funktion der Pipeline, nicht
einen Spielstärkegewinn; ein belastbarer Vergleich von Version 1 und 2 würde
deutlich längere, mehrfach gesäte Läufe erfordern und wurde hier nicht
durchgeführt.

## Belohnung

`src/rlai/rewards.py`, neu gefasst nach dem Stillstand über 2 Mio. Schritte
(Analyse: `docs/LEARNING_OVERHAUL.md`):

* **Aufgabe zuerst.** Tore/Gegentore (±10) dominieren; das stärkste dichte
  Signal ist `ball_progress` – die Potentialdifferenz der Distanz
  Ball→Gegnertor. `progress` (Auto→Ball) ist nur noch ein schwacher
  Frühgradient.
* **Potentiale statt Zustände.** Alle dichten Terme sind Differenzen; sie
  teleskopieren über die Episode, sodass Stillstehen, Kreisen und Pendeln
  netto ~0 ergeben – und ein frühes Tor keine Bonus-Einnahmen mehr kostet.
  Die alten Zustandsboni `alignment`/`approach` (Ball ansehen / darauf
  zufahren wurde pro Schritt bezahlt; Optimum: parken und schauen, 94 %
  des Alignment-Maximums bei ~0,06 Berührungen/Episode) stehen auf 0 und
  existieren nur noch, damit alte Konfigurationen ihr Verhalten exakt
  reproduzieren.
* **Ereignisse mit Richtung und Treffergenauigkeit.** Eine Berührung zahlt
  nach der Ballgeschwindigkeit Richtung Gegnertor; `shot` gibt es nur, wenn
  die Flugbahn-Projektion (mit Gravitation) auch zwischen Pfosten/Querlatte
  landet — Fehlschüsse zählen als `shots_off_target`, bleiben aber unbezahlt.
  Eine Cooldown von 4 Entscheidungen pro Auto verhindert Tipp-Farming an
  langsamen Bällen (gezählt werden weiterhin alle Kontakte). Komponenten und Ereigniszähler werden pro Agent geführt; der
  Report zeigt nur das Lernteam (früher wurden beide Teams summiert).

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
* `curriculum = true` aktiviert die **Mastery-Stufen** mit unabhängig
  gemessenen Aufstiegskriterien (`src/rlai/curriculum.py`,
  `src/rlai/skills.py`, ausführlich in `docs/MASTERY_CURRICULUM.md`):

| Modus | Stufen |
| --- | --- |
| 1v1 (`mastery`, Standard) | `approach` → `control` → `shooting` → `defense` → `pressure` → `integrated` → Selbstspiel |
| 1v1 (`legacy`) | Erster Kontakt → Schießen → `chase` → gemischt → Selbstspiel |
| 2v2, 3v3 | Erster Kontakt → `chase` → gemischt → Selbstspiel |
| Heatseeker 2v2 | Erster Kontakt → `chase` → Selbstspiel |

Aufstieg entscheidet **nie** der Trainings-Reward, sondern eine unabhängige
Messung: `rlai.skills` spielt geskriptete, geseedete Startsituationen auf
**beiden Feldseiten** und liest den Simulatorzustand (Ballposition,
Berührungen, Torereignisse, Auf-Vektor) – nicht die Belohnung. Probes:
`approach`, `progress`, `shooting`, `defense`, `reengage`; die letzten beiden
Stufen verlangen zusätzlich echte Spielserien (`match_chase`, `match_mixed`).

Eine Stufe steigt erst auf, wenn

1. alle Gates im Punktwert **und** in der unteren 95-%-Konfidenzgrenze
   bestehen (ein Glückstreffer mit 4 Trials steigt nicht auf),
2. dieselbe Messung `confirmations = 2` Mal **in Folge** bestanden wurde,
3. `min_iterations` und `min_episodes` erreicht sind und
4. keine früher gemeisterte Fertigkeit um mehr als 0.15 gefallen ist
   (Retention – ein Gate misst deshalb immer auch die früheren Stufen mit).

Gate-Messungen laufen alle `curriculum_gate_interval` Iterationen
(Standard 25) mit `curriculum_gate_trials` Trials pro Probe (Standard 24,
also 12 pro Seite). `curriculum_gate_interval = 0` wird zusammen mit einem
gated Preset schon bei der Konfigurationsprüfung abgelehnt, weil sonst keine
Stufe je verlassen werden könnte. Blockierte Aufstiege nennen den exakten
Grund inklusive Messwert, Sollwert und Konfidenzintervall
(`Curriculum.blocked_reason()`, `Curriculum.diagnostics()`, im Log, im Report
und als Metrik `curriculum/blocked`); Regressionen werden als WARNING
gemeldet und setzen den Bestätigungszähler zurück. Eine einstürzende Stufe
kann absteigen, eine stagnierende meldet sich laut.

Jede Stufe setzt außerdem Gegner, Startverteilung und Episodenlänge. `reach`
(neu) startet den Lernenden 250–900 uu hinter einem ruhenden Ball – gemessen
der entscheidende Hebel für frühe Ballkontakte
(`docs/MASTERY_CURRICULUM.md`, Experiment 7: 1.34 statt 0.46 Berührungen pro
Episode, 3 Seeds). `drill`-Starts sind randomisierte Spielsituationen in drei
Szenarien: Angriff (55 %), Verteidigung gegen einen aufs eigene Tor rollenden
Ball (25 %) und Mittelfeld-Rennen (20 %); `mixed` mischt Kickoff und Drills.
Jeder Worker zieht außerdem seinen eigenen Gegner (`mixed` = chase/random pro
Worker, Selbstspiel = eigene PFSP-Losung), damit ein Batch nicht gegen genau
einen Gegner korreliert ist.

Stufe, Zähler, das Episoden-Fenster (Beweislage) **und der Gate-Zustand**
(gemessene Werte, Bestätigungszähler, Retention-Baselines) stehen im
Checkpoint und werden beim Resume fortgesetzt; die Stufe wird dabei über
ihren **Namen** zugeordnet, nicht über ihren Index.

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
| `charts/touches`, `charts/shots` | Ballberührungen und Schüsse pro Episode (Lernteam) |
| `curriculum/stage`, `curriculum/window_win_rate` | aktuelle Stufe und Fenster-Siegquote |
| `curriculum/window_touches`, `curriculum/window_goals_for`, `curriculum/blocked` | Fenster-Kriterien der Stufe und ob der Aufstieg blockiert ist |
| `league/agent_elo`, `league/pool_size` | Elo des Lernenden, Poolgröße |
| `time/sps` | Umgebungsschritte pro Sekunde |
| `episodes/*` | Länge (**Entscheidungsschritte**, nicht Ticks), Tore, Berührungen, Schüsse, Ergebnis pro Episode (SQLite-Tabelle) |

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
* Schreiben ist atomar: `model.pt` und `manifest.json` werden in eine
  temporäre Datei geschrieben, mit `fsync` gesichert und per `os.replace`
  veröffentlicht. Ein abgebrochener Schreibvorgang hinterlässt kein
  halbes, aber als gültig gelistetes Checkpoint-Verzeichnis; ein Fehler kann
  den vorherigen Checkpoint nicht beschädigen.
* Vorhandene IDs werden nie überschrieben; bei Kollision wird ein Suffix
  angehängt.
* `latest.json` hält den zuletzt **erfolgreich** abgeschlossenen Speichervorgang
  (ID, Schritt, Lauf, Label, Zeitpunkt) fest. Die Auswahl des neuesten
  Checkpoints nutzt dieses Metadaten-Feld und den im Manifest gespeicherten
  Trainingsschritt – nicht den Zeitstempel im Verzeichnisnamen.
* Der stabile Zeiger (`stable.json`) wird nur durch eine explizite Promotion
  gesetzt; die vorherige stabile Version wird archiviert, nicht gelöscht.
* `keep_checkpoints` begrenzt die Zahl der regulären, aufeinanderfolgenden
  Checkpoints (die neuesten nach Trainingsschritt bleiben). Zusätzlich geschützt
  und nie von der Aufräumroutine entfernt werden: das stabile Modell, **jeder**
  Best-Checkpoint (`best*.json`), das Ziel von `latest.json` sowie der neueste
  `final`-/`interrupt`-Checkpoint. Vor jedem Löschen prüft die Routine, dass
  mindestens ein ladbarer Checkpoint übrig bleibt; sind alle Dateien defekt,
  wird nichts gelöscht. Das Aufräumen ist auf den eigenen Lauf begrenzt, damit
  ein neuer Lauf nicht die Historie eines anderen löscht.
* `best.json` (und je Messprotokoll ein eigenes `best-*.json`) ist ein reiner
  Messzeiger: Er zeigt auf den Checkpoint mit der besten dokumentierten
  Evaluation (Siegquote, Wilson-Intervall in der Evaluationstabelle). Er wird
  nie schlechter und ersetzt **nicht** die stabile Promotion. Ein Checkpoint
  ohne Sieg wird nicht als „best“ markiert.
* **Während des Trainings wird kein Routine-Checkpoint mehr geschrieben.**
  Ausnahmen sind nur der reguläre Abschluss und ein Abbruch (siehe unten)
  sowie der **Recovery-Checkpoint**: `recovery_checkpoint_minutes` (Standard
  30, `0` = aus) schreibt höchstens einen zusätzlichen `recovery`-Checkpoint
  pro Zeitfenster, begrenzt durch dieselbe `keep_checkpoints`-Retention. Das
  ist die Versicherung gegen harte Abstürze (Stromausfall, SIGKILL) und
  ausdrücklich kein „Checkpoint pro Intervall“ (das erzeugte in einem
  50M-Lauf ~10.000 Verzeichnisse).
* **`total_steps` gegen den wiederhergestellten Schritt.** `total_steps` ist
  ein *Wunsch*-Feld: beim Resume gewinnt der konfigurierte Wert über den im
  Checkpoint gespeicherten. Liegt das Ziel also auf oder unter dem
  wiederhergestellten Schritt, wird **kein** einziger Schritt trainiert. Das
  ist nicht still: `rlai.resume.plan_resume` gibt einen WARNING aus, der
  Trainer meldet denselben Grund als Lauf-WARNING, `steps_completed` ist 0 im
  Ergebnis, und es wird kein Checkpoint geschrieben
  (`tests/test_trainer_gates.py::test_restored_step_beyond_total_steps_is_reported_not_silent`).
  Wer einen abgebrochenen Lauf fortsetzt, nimmt `--steps N` (Ziel = aktueller
  Schritt + N, siehe „Wie viel weiter?“) oder hebt `total_steps` über den
  erreichten Schritt an.

  Es gibt genau zwei Auslöser:
  * Erreicht der Lauf regulär `total_steps`, wird **ein** `final`-Checkpoint
    geschrieben (vor der abschließenden Evaluation, damit die Gewichte auch
    dann auf der Platte liegen, wenn die Messung unterbrochen wird).
  * Wird der Lauf abgebrochen – Stop-Anforderung über `control.json`,
    Ctrl+C/SIGINT oder SIGTERM (z. B. durch den Job-Manager) –, wird **ein**
    `interrupt`-Checkpoint des aktuellen Zustands geschrieben.
  Existiert auf demselben Schritt bereits ein Checkpoint (etwa weil ein
  fortgesetzter Lauf keine weiteren Schritte mehr macht), wird dessen Manifest
  atomar mit dem neuen Label aktualisiert, statt ein zweites identisches
  Modell zu schreiben. `checkpoint_interval` bleibt aus
  Kompatibilitätsgründen in der Konfiguration, hat aber keine Wirkung mehr.
* Periodische Evaluationen schreiben keinen Checkpoint und sind deshalb nicht
  mit einer Modell-ID verknüpft; nur die abschließende Evaluation eines
  regulär beendeten Laufs hängt am `final`-Checkpoint. `best.json` wird
  entsprechend nur noch dort aktualisiert.

### Fortsetzung: automatisch, aber explizit unterscheidbar

| Modus | Bedeutung | Was wiederhergestellt wird |
| --- | --- | --- |
| `continue` | **Standard**, auch ohne Angabe | Gewichte, Optimizer, Schrittstand, Iteration, RNG (numpy/torch/python/CUDA), Curriculum-Stufe, Gegnerpool inkl. Gewichten, geerbte Config-Felder (`gamma`, `learning_rate`, …), Elternlauf |
| `fresh` | nur auf ausdrücklichen Wunsch | nichts – neue Policy, Schritt 0; vorhandene Checkpoints werden bewusst ignoriert |
| `transfer` | Fine-Tuning auf neuer Aufgabe | nur die Gewichte; Schrittzähler, Optimizer, RNG und Curriculum beginnen neu |

`rlai train` und `rlai headless` suchen beim Start automatisch den neuesten
gültigen, kompatiblen Checkpoint und setzen den Lauf fort. Die Auswahl prüft
Architektur, Beobachtungsversion und Modus vor dem Laden; der höchste
Trainingsschritt gewinnt, bei gleichem Schritt der zuletzt erfolgreich
gespeicherte Checkpoint. `--auto-resume` bleibt als veralteter Kompatibilitäts-
Schalter akzeptiert, ist aber ohne Wirkung, weil das Verhalten jetzt der
Standard ist.

Unterscheidung beim Start:

* **Kein Checkpoint vorhanden:** der Lauf startet wie bisher mit einer neuen
  Policy; Log und CLI sagen ausdrücklich „no checkpoint found … fresh
  training“.
* **Checkpoints vorhanden, aber keiner ist kompatibel oder ladbar:** der Lauf
  wird mit den Ablehnungsgründen abgelehnt (CLI-Exitcode 3, Trainer-
  `ValueError`, API 400). Es wird **nicht** stillschweigend ein neues Modell
  initialisiert. Bewusst neu starten: `--resume-mode fresh`; nur Gewichte
  übernehmen: `--resume-mode transfer`.
* **Neuester Checkpoint defekt, älterer gültig:** die automatische Fortsetzung
  fällt mit Warnung auf den älteren gültigen Kandidaten zurück
  (`fallback_used`).
* **Namentlich angeforderter Checkpoint unbrauchbar:** niemals Ersatz durch
  einen anderen (CLI-Exitcode 2, API 400).

```bash
# Vorschau ohne Training – zeigt Kandidaten, Ablehnungen und was fehlt
python -m rlai.cli resume-plan --config-file configs/1v1-vs-chaser.json --run-id <run-id>
python -m rlai.cli train --config-file ... --plan-only

# Standard: neuester gültiger, kompatibler Checkpoint wird automatisch fortgesetzt
python -m rlai.cli train --config-file ... --run-id <kind> --steps 40000

# explizit, oder bewusst nur Gewichte
python -m rlai.cli train --config-file ... --resume-from <model-id> --resume-mode continue
python -m rlai.cli train --config-file ... --resume-from <model-id> --resume-mode transfer

# bewusst neu anfangen (ignoriert vorhandene Checkpoints)
python -m rlai.cli train --config-file ... --resume-mode fresh

# neue Phase auf denselben Gewichten: geerbte Felder bewusst ändern
python -m rlai.cli train --config-file ... --override-config reward
```

### Wie viel weiter? `--steps` statt `--max-steps`

`total_steps` in der Konfiguration ist ein **absolutes** Ziel: die Schleife
läuft, solange `step < total_steps`. Ein Checkpoint bei 2.000.000 Schritten
meldete deshalb mit `--max-steps 500000` sofort „completed“ und trainierte null
Schritte. Dafür gibt es `--steps` (Aliase `--maxsteps`, `--Maxsteps`):

```bash
python3 main.py --steps 500000                       # 2.000.000 -> 2.500.000
python -m rlai.cli train --config-file ... --steps 500000 --plan-only
python -m rlai.cli resume-plan --config-file ... --steps 500000
```

* Ziel = `restored_step + N`; ohne Checkpoint ist N die Gesamtzahl. Der Lauf
  endet nur an Iterationsgrenzen, deshalb wird das Ziel auf volle
  Rollout-Batches (`rollout_steps × n_workers`) **aufgerundet** – höchstens um
  einen Batch; die Note im Plan nennt den exakten Ausgangswert. Der Plan zeigt
  `additional_steps`, `target_total_steps`, `notes` und in `effective_config`
  das tatsächlich benutzte `total_steps`.
* `lr_anneal_start` wird auf den wiederhergestellten Schritt gesetzt: die
  Lernrate fällt über die **neuen** N Schritte von 100 % auf 10 %, statt beim
  Fortsetzen sofort auf dem Boden der alten Rampe zu stehen. Der Report
  protokolliert das Feld unter `steps.lr_anneal_start`.
* `--max-steps N` bleibt das absolute Ziel (mutually exclusive zu `--steps`).
  Wer `--max-steps` unter dem Checkpoint-Schritt angibt, bekommt eine Warnung
  im Plan – genau der Fall, der früher stillschweigend nichts tat.
* Weniger als eine Rollout-Batch (`rollout_steps × n_workers`) wird als Fehler
  abgelehnt; so ein Lauf hätte keine einzige Aktualisierung.

Die API verhält sich genauso: Fehlt `resume_mode` in `POST /api/runs`, wird
automatisch fortgesetzt, wenn ein nutzbarer Checkpoint existiert; gibt es
keinen, startet der Lauf frisch. Existieren nur unbrauchbare Checkpoints,
antwortet die API mit 400. Ein explizites `resume_mode: "continue"` verlangt
weiterhin einen nutzbaren Checkpoint, `"fresh"` ignoriert ihn bewusst.

Betriebsparameter (`episode_seconds`, `rollout_steps`, `total_steps`, Worker,
Evaluation, `reset_mode`) folgen immer der Anfrage; lernkritische Felder
(`reward`, PPO-Hyperparameter, …) kommen standardmäßig aus dem Checkpoint,
damit die Fortsetzung vergleichbar bleibt. Der Plan zeigt **vor** dem Start
jede Abweichung mit dem tatsächlich verwendeten Wert (`overrides`,
`effective_config`) – nichts wird still ersetzt. Wer ein geerbtes Feld
bewusst ändern will (z. B. eine korrigierte Belohnung auf einen bestehenden
Checkpoint anwenden), nennt es in `--override-config`; das wird als
Phasenwechsel protokolliert (Log, Plan, Report), inklusive des Hinweises,
dass Wertfunktion und Optimizer-Momente noch auf den alten Werten beruhen.

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

### Ctrl+C und sicheres Beenden

Der Trainer fängt `SIGINT` (Ctrl+C) ab, schließt zuerst die Worker-Prozesse und
schreibt danach den letzten Zustand als Checkpoint mit dem Label `interrupt`.
Die Sitzung wird als `stopped` beendet, der Run-Status und der Report werden
normal geschrieben; der nächste Start findet diesen Checkpoint automatisch.
Während des Schreibens wird ein erstes Signal zurückgestellt, damit der
Checkpoint, der `latest.json`-Zeiger und die Buchhaltung konsistent bleiben.
Ein zweites Ctrl+C während des Aufräumens wird nur gezählt und kann das
Schreiben nicht mehr abbrechen.

Schlägt das Schreiben beim Beenden fehl, bleibt der vorherige Checkpoint
unangetastet. Ergebnis und Report melden dann ausdrücklich `resumable: false`
und „the latest state could NOT be saved“; der Lauf wird **nicht** als
resumierbar ausgegeben. `result["checkpoint"]` nennt in diesem Fall keinen
neuen Stand.

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
