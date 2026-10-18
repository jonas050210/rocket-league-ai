# RLBot v5 Deployment

## Status

| Prüfung | Ergebnis |
| --- | --- |
| RLBot-Pakete installieren und importieren | **ja**, `rlbot 2.0.0b56`, `rlbot_flatbuffers 0.19.0` |
| Bot-Entrypoint importiert | **ja** (`rlai.rlbot.bot`, `rlai.rlbot.run`) |
| Beobachtung aus `GamePacket` = Trainingsbeobachtung | **ja**, Paritätstest mit synthetischem Paket, v1 und v2 in allen Modi (max. Abweichung ≤ 2,4e-7) |
| Checkpoint laden und Inferenz ausführen | **ja** (Unit-Test) |
| Aktion → `ControllerState` | **ja** (Mapping-Test aller 90 Aktionen) |
| Bot-Prozess in RLBot / echtes Match | **nein** – Windows, Rocket League und RLBot fehlen in der Build-Umgebung |
| Match-Konfiguration (`match.toml`) von RLBot akzeptiert | **ja** – alle vier Modi, beide Fill-Varianten, geparst mit `rlbot.config.load_match_config` (nicht gestartet) |

Die Aussage „RLBot v5 kompatibel“ ist deshalb bewusst eingeschränkt: Die
Schnittstelle ist gegen die installierte Bibliothek geprüft, ein echter
Match-Lauf steht aus.

## Architektur

```
rlbot/bot.py                 Entrypoint, den RLBot startet
  └─ rlai.rlbot.bot         Bot-Klasse, Modell laden, Entscheidungen
       ├─ rlai.rlbot.obs_adapter   GamePacket → GameState → 198-dim (v2) bzw. 190-dim (v1) Beobachtung
       ├─ rlai.policy             TorchPolicy, deterministische Aktion
       └─ rlai.actions            Aktion → ControllerState
```

Der Beobachtungsbau ist **derselbe Code** wie im Training. Es gibt keine
zweite Feature-Implementierung, die auseinanderlaufen könnte. Die
Beobachtungsversion wird aus `policy.model.obs_size` abgeleitet und beim
Start geloggt; ein 190er-Checkpoint bekommt automatisch das v1-Layout.

## Voraussetzungen

* Windows 10/11, Rocket League über Epic Games.
* RLBot v5 (Installer von rlbot.org) und Python 3.11/3.12.
* Dieses Repository mit installierten Abhängigkeiten (siehe
  `docs/SETUP_WINDOWS.md`).
* Ein trainierter Checkpoint unter `data/models/`, entweder als stabiler
  Zeiger oder explizit per `--model`/`RLAI_MODEL`.

## Manueller Start

```powershell
# 1) Prüfen, ob die Umgebung vollständig ist
python -m rlai.cli live --preflight

# 2) Dateien schreiben und von RLBot parsen lassen (startet nichts)
python -m rlai.cli live --mode soccer-1v1 --model-id <model-id> --dry-run

# 3) Starten; --timeout beendet den Launcher notfalls nach 600 s
python -m rlai.cli live --mode soccer-1v1 --model-id <model-id> --timeout 600

# Alternativ direkt, wenn match.toml schon existiert
python -m rlai.rlbot.run --config data\live\<mode>-<timestamp>\match.toml --timeout 600
```

`python -m rlai.cli live --list` zeigt die letzten Matches mit `alive`,
Status und letzter Logzeile. `--human-team orange`, `--fill policy` und
`--no-human` steuern die Teamzusammensetzung (siehe unten).

Alternativ über das Dashboard: *Live match* → Modell und Modus wählen →
*Launch match*. Der Server erzeugt den Match-Ordner, schreibt `match.toml`,
`bot-N.toml` und `launch.json` und startet den RLBot-Prozess.

Zwei Vorstufen ohne Risiko:

* **Run setup checks** (`GET /api/live/preflight`, CLI `rlai live --preflight`)
  prüft Plattform, RLBot-Import, RLBotServer, Bot- und Launcher-Entrypoint,
  vorhandenes Modell, Schreibrechte, den Match-Parser und – falls `psutil`
  verfügbar ist – ob `RocketLeague.exe` läuft. Jeder Punkt hat `ok`,
  `required` und eine Begründung; `ok` bezieht sich nur auf Pflichtpunkte.
* **Prepare files only** (`POST /api/live/launch` mit `dry_run = true`)
  schreibt alle Match-Dateien, startet aber nichts. Das funktioniert auch
  unter Linux und ist die beste Prüfung, bevor RLBot ins Spiel kommt.

Erzeugte Dateien:

```
data/live/<mode>-<timestamp>/
  match.toml        # Arena, Mutatoren, Mensch + Policy-Team + Filler
  bot-0.toml ...    # je Bot: agent_id, root_dir, run_command
  launch.json       # Modell, Python, Mensch/Fill, Status, PID, Kommando
  launch.log        # stdout/stderr des Launchers
```

Zusätzlich schreibt der Bot-Prozess `data/logs/bot.log` (Start, Modellpfad,
Modus, Traceback). Der Launcher beendet sich mit Code 2 bei fehlender oder
ungültiger `match.toml`, 3 bei Timeout und 4 bei einem Startfehler des
Managers.

## Teamzusammensetzung

| Option | Standard | Bedeutung |
| --- | --- | --- |
| `--human-team blue\|orange` | `blue` | Seite des Menschen; die Policy übernimmt automatisch die andere Seite |
| `--fill psyonix\|policy` | `psyonix` | füllt das menschliche Team in 2v2/3v3 mit Psyonix-Bots oder weiteren Policy-Autos |
| `--no-human` | aus | reines Schau-Match: beide Teams werden von der Policy gesteuert |

Ein 2v2 besteht damit immer aus vier Autos, ein 3v3 aus sechs – unabhängig
davon, ob ein Mensch mitspielt. Ohne Filler wäre ein „2v2“ faktisch ein
1-gegen-2.

## Gegen Menschen spielen

**Unterstützt und regelkonform:**

* Ein Mensch auf demselben PC über `type = "Human"` in der generierten
  `match.toml` (Standard). Das ist ein lokales Offline-Match, kein
  Matchmaking.
* RLBot unterstützt LAN-/VLAN-Spiele, bei denen weitere Menschen auf einem
  zweiten PC im selben Netzwerk teilnehmen. Dafür ist ein LAN-Match im
  lokalen Netz nötig; Console-Spieler sind nicht möglich. Diese Variante ist
  hier **nicht getestet** und erfordert zusätzliche RLBot-Konfiguration.

**Nicht unterstützt und ausdrücklich nicht implementiert:**

* Online-Matchmaking oder das Beitreten zu öffentlichen/privaten
  Online-Lobbys mit dem Bot. RLBot blockiert Online-Funktionen; Psyonix
  verfolgt Bots im Online-Spiel mit Null-Toleranz (Cheating und
  Matchmaking-Missbrauch).
* Das Umgehen von Easy Anti-Cheat, der Online-Sperre oder anderer
  Schutzmechanismen.
* Automatisiertes Spielen auf fremden Accounts.

Der Bot ist damit ein Trainings- und Forschungsgegenstand für lokale
Offline-Spiele, kein Tool für Online-Ranked.

## Umgebungsvariablen des Bots

| Variable | Pflicht | Bedeutung |
| --- | --- | --- |
| `RLAI_MODEL` | nein | Pfad oder Modell-ID; ohne Wert wird das stabile Modell geladen |
| `RLAI_MODE` | nein | Modus-ID, Standard `soccer-1v1` |
| `RLAI_TICK_SKIP` | nein | Entscheidungsintervall, Standard 8 |
| `RLAI_DATA_DIR` | nein | Datenverzeichnis, Standard `<repo>/data` |

## Match-Konfiguration

`src/rlai/live.py` erzeugt:

* `[rlbot] launcher = "Epic"`, `auto_start_agents = true`,
  `wait_for_agents = true`
* `[match] game_mode = "Soccar" | "Heatseeker"`, `skip_replays = true`,
  `instant_start = true`, `existing_match_behavior = "Restart"`.
  **Wichtig:** RLBot schreibt den Modus `Soccar` (Psyonix-Schreibweise);
  `"Soccer"` wird mit `ConfigParsingException` abgelehnt. Dieser Fehler trat
  real auf und ist durch `tests/test_live.py` (Parser-Roundtrip) abgesichert.
* `[mutators] match_length = "TenMinutes"`, unbegrenztes Score/Overtime,
  bei Heatseeker zusätzlich `boost_amount = "UnlimitedBoost"`
* `[[cars]]`: optional ein Mensch, `team_size` Policy-Bots auf der
  Gegenseite und in 2v2/3v3 `team_size - 1` Filler (Psyonix oder Policy),
  jeweils mit eigener `config_file`, `auto_start = true`
* `[rlbot]` setzt `launcher = "Epic"`, `auto_start_agents` und
  `wait_for_agents`.

## Bekannte Risiken / was zuerst zu prüfen ist

1. **Echter Matchstart.** Die Dateien werden von `rlbot.config.load_match_config`
   akzeptiert, aber der Start von RLBotServer, Epic und Rocket League ist
   hier nicht ausführbar. Bei Fehlern zuerst `launch.log` und
   `data/logs/bot.log` prüfen.
2. **Python-Interpreter.** `run_command` verwendet den Interpreter, der den
   Server startet. Wenn RLBot einen anderen Interpreter erwartet, in
   `bot-N.toml` anpassen.
3. **Paket-Shadowing.** Das Verzeichnis `rlbot/` im Repo enthält keinen
   `__init__.py`; der Import von `rlbot` löst weiterhin auf das installierte
   Paket auf (geprüft). Beim Start als Skript wird `sys.path` korrekt gesetzt.
4. **Modell-/Modus-Mismatch.** Ein 1v1-Checkpoint kann in 3v3 geladen werden,
   ist aber nicht dafür trainiert. Das Dashboard verhindert das nicht – die
   Verantwortung liegt beim Nutzer.
5. **Heatseeker.** Arena und Mutatoren werden gesetzt; ob RLBot den Modus
   im Match tatsächlich aktiviert, ist nicht verifiziert.

## Fehlersuche

| Symptom | Prüfung |
| --- | --- |
| Bot startet nicht | `python rlbot/bot.py --model <pfad> --mode <id>` manuell ausführen; Traceback in `data/logs/bot.log` |
| „No model configured“ | `RLAI_MODEL` setzen oder ein Modell promoten |
| Beobachtungsdimension falsch | `tests/test_rlbot_adapter.py` ausführen |
| Aktionen wirken falsch | `controller_dict` gegen `tests/test_actions.py` prüfen |
| Match startet nicht | RLBot-Installation, Epic-Launcher, `launch.log` |

## Was hier bewusst nicht behauptet wird

* Keine Aussage über Siegquote gegen Menschen oder Nexto.
* Keine Aussage, dass ein trainierter Smoke-Checkpoint sinnvoll spielt.
* Keine Aussage, dass der Live-Start ohne Anpassung funktioniert.
