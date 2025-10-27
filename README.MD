# Rust GPU Miner für das Hashing Game

Dies ist ein leistungsstarker Rust-Client für das [hash.h10a.de](http://hash.h10a.de/) Hashing-Spiel.

Diese Anleitung setzt voraus, dass Sie dieses Repository bereits geklont haben und sich im Projektverzeichnis befinden.

## 1. Voraussetzungen

Bevor Sie kompilieren können, müssen die folgenden Abhängigkeiten installiert sein:

1.  **Rust Toolchain:**

    - Falls noch nicht geschehen, installieren Sie Rust: [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)

2.  **OpenCL-Treiber/SDK:**
    - **NVIDIA:** Installieren Sie das [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). Es enthält die notwendigen OpenCL-Treiber.
    - **AMD:** Installieren Sie die AMD Radeon ROCm-Treiber für Ihre GPU.
    - **Intel:** Installieren Sie das Intel SDK für OpenCL-Anwendungen.

## 2. Kompilieren

Da alle Quelldateien (`src/*`) und die `Cargo.toml` bereits vorhanden sind, können Sie das Projekt direkt mit dem Release-Flag kompilieren. Dieser Befehl lädt auch automatisch alle Abhängigkeiten (wie `ocl`, `rayon` etc.) herunter.

```bash
cargo build --release
```

Die fertige, optimierte Programmdatei finden Sie anschließend unter: `./target/release/rust_miner`.

## 3\. Starten & Benutzen

Der Client benötigt einen Nickname, um Hashes einzureichen. Dieser wird über die Umgebungsvariable `HASH_NAME` festgelegt.

### GPU Miner starten (Empfohlen)

Um die bestmögliche Leistung zu erzielen, starten Sie den Client mit GPU-Beschleunigung. Setzen Sie dazu die `USE_GPU` Umgebungsvariable auf `true`.

```bash
# Unter Linux/macOS
HASH_NAME="DeinNickname" USE_GPU=true ./target/release/rust_miner

# Unter Windows (PowerShell)
$env:HASH_NAME="DeinNickname"; $env:USE_GPU="true"; ./target/release/rust_miner
```

Der Miner initialisiert OpenCL, sucht Ihre GPU und beginnt mit dem Hashing.

### CPU Miner starten (Fallback)

Wenn Sie `USE_GPU=true` nicht setzen oder der GPU-Miner (z.B. wegen fehlender Treiber) nicht initialisiert werden kann, fällt das Programm automatisch auf den CPU-Miner zurück. Dieser nutzt alle verfügbaren CPU-Kerne.

```bash
# Unter Linux/macOS
HASH_NAME="DeinNickname" ./target/release/rust_miner

# Unter Windows (PowerShell)
$env:HASH_NAME="DeinNickname"; ./target/release/rust_miner
```
