# Windows + Epic: Installation und erster Lauf

Getestet wurde die Pipeline in einer Linux-Umgebung. Die folgenden Schritte
sind die exakte Prozedur für Windows; sie sind **nicht** in dieser
Build-Umgebung ausgeführt worden. Abweichungen bitte im ersten Lauf notieren.

## 1. Voraussetzungen

* Windows 10/11 (64-bit)
* Python 3.11 oder 3.12 von python.org – **„Add python.exe to PATH“** ankreuzen
* Git für Windows
* Rocket League über Epic Games installiert und einmal gestartet
* RLBot v5 von <https://rlbot.org> installiert
* Optional: NVIDIA-Treiber + CUDA-fähige GPU (nicht erforderlich)

Prüfen:

```powershell
python --version
git --version
```

## 2. Repository und virtuelle Umgebung

```powershell
cd $HOME
git clone https://github.com/jonas050210/rocket-league-ai.git
cd rocket-league-ai
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

Das Skript legt `.venv` an, installiert die gepinnten Abhängigkeiten und führt
die Tests aus. Manuell entspricht das:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m pytest -q
```

Falls PowerShell das Aktivieren blockiert:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### GPU (optional)

Die Standardinstallation nutzt das PyPI-Torch-Paket. Für CUDA:

```powershell
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
```

Das Dashboard zeigt unter *Settings*, ob CUDA erkannt wurde. Ohne GPU läuft
alles auf CPU – langsamer, aber vollständig.

## 3. Anwendung starten

```powershell
.\.venv\Scripts\Activate.ps1
python -m rlai.cli serve --port 8000
```

Browser: <http://localhost:8000>

* *Settings* prüft Abhängigkeiten und misst den Simulator-Durchsatz.
* *Training* startet **nichts** automatisch. Erst Konfiguration prüfen,
  dann *Start training*.
* Ein kleiner Test: `configs\smoke-1v1.json` über die CLI:

```powershell
python -m rlai.cli train --config-file configs\smoke-1v1.json
```

## 4. Ersten Checkpoint evaluieren

```powershell
python -m rlai.cli evaluate --model data\models\<id>\model.pt `
  --mode soccer-1v1 --opponent idle --games 4 --match-seconds 30
```

Die Ausgabe enthält W/L/D, Tore, Ballberührungen und ein 95-%-Wilson-Intervall
sowie Spieldetails. Dieselbe Auswertung ist im Dashboard unter *Models →
Evaluate* möglich.

## 5. Live-Match (Epic + RLBot)

1. Rocket League starten und im Hauptmenü bleiben.
2. Ein Modell promoten (*Models → Promote*) oder die Modell-ID notieren.
3. Zuerst die Umgebung prüfen – in der CLI oder im Dashboard
   (*Live match → Run setup checks*):

```powershell
python -m rlai.cli live --preflight
```

   Geprüft werden Windows, RLBot-Import, laufender/installierter
   RLBotServer, Bot- und Launcher-Entrypoint, Modell, Schreibrechte,
   Match-Parser und optional `RocketLeague.exe`. `--preflight` endet mit
   Code 1, solange ein Pflichtpunkt fehlt.
4. Dateien schreiben und von RLBot parsen lassen, ohne zu starten:

```powershell
python -m rlai.cli live --mode soccer-1v1 --model-id <id> --dry-run
```

   Das erzeugt `data\live\<mode>-<timestamp>\match.toml`, `bot-N.toml`
   und `launch.json`; der Launcher wird mit `--dry-run` aufgerufen und
   meldet `invalid match config`, falls RLBot die Datei ablehnt.
5. Starten:

```powershell
python -m rlai.cli live --mode soccer-1v1 --model-id <id> --timeout 600
```

   Ohne `--timeout` wartet der Launcher auf das Matchende; mit Timeout
   beendet er sich nach N Sekunden. Varianten:
   `--human-team orange` (Mensch spielt Orange), `--fill policy` (Policy
   füllt das Mensch-Team statt Psyonix-Bots), `--no-human` (beide Teams
   Policy). Im Dashboard heißen dieselben Schritte *Prepare files only*
   und *Launch match*.
6. Status prüfen:

```powershell
python -m rlai.cli live --list
```

   Zeigt pro Match `alive`, `status` und die letzte Zeile aus
   `data\live\...\launch.log`. Der Bot schreibt zusätzlich
   `data\logs\bot.log`.

Wenn der Start fehlschlägt:

```powershell
# Bot isoliert testen
$env:PYTHONPATH = "$PWD\src"
python rlbot\bot.py --model "$PWD\data\models\<id>\model.pt" --mode soccer-1v1
```

Der Bot sollte ohne Traceback starten und auf ein Paket warten (er blockiert,
bis ein RLBotServer ihn verbindet). Fehler hier sind von
RLBot-Startfehlern zu unterscheiden. Meldet RLBot `Invalid value 'Soccer'`,
ist eine alte `match.toml` im Spiel: neu erzeugen (`--dry-run`) – gültig ist
`game_mode = "Soccar"`.

## 6. Tests und Diagnose

```powershell
python -m pytest -q
python -m rlai.cli benchmark --mode heatseeker-2v2 --workers 2 --seconds 2
```

Erwartung: alle Tests grün. Der Benchmark misst echte Schritte/s; die Zahl
hängt von CPU, Workerzahl und Hintergrundlast ab.

## Häufige Probleme

| Problem | Ursache / Lösung |
| --- | --- |
| `RocketSim` importiert nicht | Microsoft Visual C++ Redistributable 2015–2022 installieren |
| `torch` installiert riesige CUDA-Pakete | CPU-Variante mit `--index-url .../cpu` installieren |
| Port 8000 belegt | `--port 8010` verwenden |
| Dashboard zeigt „API offline“ | Server läuft? `http://localhost:8000/api/health` prüfen |
| Bot lädt kein Modell | `RLAI_MODEL` setzen oder stabiles Modell promoten |
| RLBot findet den Bot nicht | Pfad in `bot-N.toml` prüfen; Leerzeichen in Pfaden sind erlaubt, aber in Logs schwer lesbar |

## Deinstallation

`.venv` löschen, `data` nach Bedarf sichern. Es werden keine systemweiten
Änderungen vorgenommen außer den genannten Paketen und RLBot.
