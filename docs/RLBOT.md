# RLBot v5 Deployment

## Status

| Prüfung | Ergebnis |
| --- | --- |
| RLBot-Pakete installieren und importieren | **ja**, `rlbot 2.0.0b56`, `rlbot_flatbuffers 0.19.0` |
| Bot-Entrypoint importiert | **ja** (`rlai.rlbot.bot`, `rlai.rlbot.run`) |
| Beobachtung aus `GamePacket` = Trainingsbeobachtung | **ja**, Paritätstest mit synthetischem Paket (max. Abweichung < 1e-3) |
| Checkpoint laden und Inferenz ausführen | **ja** (Unit-Test) |
| Aktion → `ControllerState` | **ja** (Mapping-Test aller 90 Aktionen) |
| Bot-Prozess in RLBot / echtes Match | **nein** – Windows, Rocket League und RLBot fehlen in der Build-Umgebung |
| Match-Konfiguration (`match.toml`) von RLBot akzeptiert | **nein** – nicht ausführbar hier |

Die Aussage „RLBot v5 kompatibel“ ist deshalb bewusst eingeschränkt: Die
Schnittstelle ist gegen die installierte Bibliothek geprüft, ein echter
Match-Lauf steht aus.

## Architektur

```
rlbot/bot.py                 Entrypoint, den RLBot startet
  └─ rlai.rlbot.bot         Bot-Klasse, Modell laden, Entscheidungen
       ├─ rlai.rlbot.obs_adapter   GamePacket → GameState → 190-dim Beobachtung
       ├─ rlai.policy             TorchPolicy, deterministische Aktion
       └─ rlai.actions            Aktion → ControllerState
```

Der Beobachtungsbau ist **derselbe Code** wie im Training. Es gibt keine
zweite Feature-Implementierung, die auseinanderlaufen könnte.

## Voraussetzungen

* Windows 10/11, Rocket League über Epic Games.
* RLBot v5 (Installer von rlbot.org) und Python 3.11/3.12.
* Dieses Repository mit installierten Abhängigkeiten (siehe
  `docs/SETUP_WINDOWS.md`).
* Ein trainierter Checkpoint unter `data/models/`, entweder als stabiler
  Zeiger oder explizit per `RLAI_MODEL`.

## Manueller Start

```powershell
$env:RLAI_MODEL = "data\models\<model-id>\model.pt"
$env:RLAI_MODE  = "soccer-1v1"
$env:PYTHONPATH = "<repo>\src"
python -m rlai.rlbot.run --config data\live\<mode>-<timestamp>\match.toml
```

Alternativ über das Dashboard: *Live match* → Modell und Modus wählen →
*Launch match*. Der Server erzeugt den Match-Ordner, schreibt `match.toml`,
`bot-N.toml` und `launch.json` und startet den RLBot-Prozess.

Erzeugte Dateien:

```
data/live/<mode>-<timestamp>/
  match.toml        # Arena, Mutatoren, ein Mensch (Blau), N Bots (Orange)
  bot-0.toml ...    # je Bot: agent_id, root_dir, run_command
  launch.json       # Modell, Python, Zeitstempel
  launch.log        # stdout/stderr des Launchers
```

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
* `[match] game_mode = "Soccer" | "Heatseeker"`, `skip_replays = true`,
  `start_without_countdown = true`, `existing_match_behavior = "Restart"`
* `[mutators] match_length = "TenMinutes"`, unbegrenztes Score/Overtime,
  bei Heatseeker zusätzlich `boost_amount = "UnlimitedBoost"`
* `[[cars]]`: ein Mensch im blauen Team, `team_size` Bots im orangen Team,
  jeweils mit eigener `config_file`, `auto_start = true`

## Bekannte Risiken / was zuerst zu prüfen ist

1. **Schema-Akzeptanz.** Die Felder sind aus der v5-Dokumentation und dem
   python-example abgeleitet, aber nicht gegen einen echten MatchManager
   validiert. Bei Fehlern zuerst `launch.log` prüfen.
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
| Bot startet nicht | `python rlbot/bot.py` manuell im Repo ausführen, Traceback lesen |
| „No model configured“ | `RLAI_MODEL` setzen oder ein Modell promoten |
| Beobachtungsdimension falsch | `tests/test_rlbot_adapter.py` ausführen |
| Aktionen wirken falsch | `controller_dict` gegen `tests/test_actions.py` prüfen |
| Match startet nicht | RLBot-Installation, Epic-Launcher, `launch.log` |

## Was hier bewusst nicht behauptet wird

* Keine Aussage über Siegquote gegen Menschen oder Nexto.
* Keine Aussage, dass ein trainierter Smoke-Checkpoint sinnvoll spielt.
* Keine Aussage, dass der Live-Start ohne Anpassung funktioniert.
