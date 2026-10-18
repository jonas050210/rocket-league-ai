# Spielmodi

Alle Modi teilen denselben Aufbau des Beobachtungsraums (198 Werte in der
aktuellen Version 2; die alte Version 1 mit 190 Werten bleibt ladbar) und
denselben Aktionsraum (90 diskrete Aktionen). Unterschiede sind reine Konfiguration:
Teamgröße, Arena-Modus, Kickoff-Zustand, Boost-Regeln und Gegner.

## Übersicht

| ID | Teamgröße | Autos | Arena | Besonderheit |
| --- | --- | --- | --- | --- |
| `soccer-1v1` | 1 | 2 | `GameMode.SOCCAR` | Referenzmodus für die Pipeline |
| `soccer-2v2` | 2 | 4 | `GameMode.SOCCAR` | geteilte Policy für beide blauen Autos |
| `soccer-3v3` | 3 | 6 | `GameMode.SOCCAR` | Beobachtung nutzt drei Slots pro Team |
| `heatseeker-2v2` | 2 | 4 | `GameMode.HEATSEEKER` | Homing-Ball, Tempoplus, unendlich Boost |

## Beobachtung (Version 2, 198 Dimensionen)

| Block | Werte | Inhalt |
| --- | --- | --- |
| Selbst | 10 | Boost, Bodenkontakt, Supersonic, Flip-Fähigkeit, Demo, Sprung/Flip-Status, Luftzeit, Handbremse |
| Ball | 7 | relative Position und Geschwindigkeit im Auto-Frame, Balltempo |
| Teamkollegen | 2 × 9 | relative Position/Geschwindigkeit, Boost, Bodenkontakt; fehlende Slots = 0 |
| Gegner | 3 × 9 | wie Teamkollegen |
| Boost-Pads | 34 | Cooldown-Zeiten der großen und kleinen Pads |
| Global | 4 | Restzeit, Score-Differenz, Heatseeker-Flag, Team-Flag |
| Feldrahmen | 8 | eigene x/y-Position, Vorwärtsvektor x/y, Ball-x/y, Ausrichtung Vorwärts zu Ball→Gegnertor, Distanz zum Gegnertor |
| letzte Aktion | 90 | One-Hot der zuletzt ausgeführten Aktion |

Die relative Transformation in das Auto-Frame macht die übrigen Blöcke
unabhängig von Teamseite und Spawnrotation. Sie kann aber die feste Position
der Tore nicht ausdrücken: Aus rein relativen Vektoren ist nicht ableitbar,
wo das gegnerische Tor liegt. Genau diese Information ergänzt der
Feldrahmen-Block. Deshalb ist Version 2 Standard; Version 1 (190 Werte, ohne
Feldrahmen) wird weiter erzeugt, damit ältere Checkpoints ladbar und
evaluierbar bleiben. Welche Version gilt, entscheidet die `obs_size` des
Checkpoints, nicht eine globale Einstellung.

Der RLBot-Adapter erzeugt denselben Vektor aus einem echten `GamePacket`;
ein Paritätstest vergleicht beide Werte für Version 1 und 2 in allen Modi
(`tests/test_rlbot_adapter.py`, maximale Abweichung < 1e-3).

## Aktionen (90)

Die Tabelle entspricht der verbreiteten RLGym-Lookup-Table:

* Bodenaktionen: Gas vorwärts/rückwärts, Lenken, Boost, Handbremse.
* Luftaktionen: Pitch/Yaw/Roll, Sprung, Flip, Boost.

`controller_dict(index)` übersetzt eine Aktion in RLBot-Felder
(`throttle`, `steer`, `pitch`, `yaw`, `roll`, `jump`, `boost`, `handbrake`).
Tabelle und Mapping sind durch `tests/test_actions.py` abgesichert.

## Belohnung

Standardgewichte (pro Lauf änderbar):

| Komponente | Gewicht | Bedeutung |
| --- | --- | --- |
| Tor | +10 | Ball im gegnerischen Tor |
| Gegentor | −10 | Ball im eigenen Tor |
| Touch | 0.35 | Ballberührung, skaliert mit Balltempo |
| Fortschritt | 0.9 | Potenzialdifferenz Ball→Gegnertor, pro 1000 uu |
| Ausrichtung | 0.02 | Auto zeigt zum Ball |
| Annäherung | 0.03 | Geschwindigkeit in Ballrichtung |
| Boost | 0.02 | Strafe für verschwendeten Boost |
| Verteidigung | 0.15 | Auto zwischen Ball und eigenem Tor, Ball bewegt sich zum eigenen Tor |
| Bump | 0.3 | Karosseriekontakt mit einem Gegner (RLGym meldet `bump_victim_id`); eine Demo-Erkennung ist nicht implementiert |

Beim Tor wird das Potenzial-Shaping übersprungen, damit die 10 Punkte nicht
durch die große Ballbewegung teilweise kompensiert werden. Jede Komponente
wird in `shared_info["reward_stats"]` summiert und kann geloggt werden.

## Heatseeker

RocketSim implementiert den Modus nativ; wir setzen nur die Arena auf
`GameMode.HEATSEEKER` und reproduzieren den Kickoff-Zustand:

* Ballstart bei `(-1000, -2220, 92.75)` mit Geschwindigkeit `(0, -65, 650)`.
* Vier Spawnpunkte bei `y = -4620`, zufällig auf beide Teams verteilt,
  Blickrichtung Richtung Spielfeldmitte.
* `boost_consumption = 0` (unendlich Boost).

Verifizierte Regeln (Tests, echte Simulation):

1. Vor der ersten Berührung ist `heatseeker_target_dir == 0`, Zieltempo 2900 uu/s.
2. Eine Berührung durch Blau setzt `target_dir = +1` (oranges Tor), das
   Zieltempo steigt auf 2985 uu/s und der Ball beschleunigt auf dieses Tempo.
3. Der Ball krümmt sich ins Tor; das Tor wird als Blau-Treffer gewertet
   (`scoring_team == 0`) und der Spielstand erhöht sich.
4. Boost wird bei laufendem Boost nicht verbraucht.

Die dritte Berührung erhöht das Zieltempo weiter (bis 4600 uu/s); das ist
RocketSim-Verhalten und muss nicht nachgebaut werden. Wichtig: Die
Torerkennung prüft zusätzlich zur `y`-Position die Tormündung
(`|x| ≤ 892.755`, `z ≤ 642.775`), weil RLGyms Standardprüfung nur `|y|`
betrachtet und sonst Bälle außerhalb des Tors zählen würde.

## Modi wechseln

Ein Checkpoint aus 1v1 kann in 2v2/3v3 weitertrainiert werden, weil die
Eingabedimension gleich bleibt. Für Heatseeker ist ein separater Lauf
empfohlen: Ballphysik und Belohnungsdynamik unterscheiden sich deutlich.
