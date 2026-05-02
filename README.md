# Ollama Model Manager

An interactive Python helper for finding, downloading, and updating Ollama models.

The script checks whether Ollama is installed, detects models you already have, refreshes an online model catalog from the Ollama library, and lets you download all models, presets, selected catalog entries, or custom model names.

## Features

- Checks for the Ollama CLI before doing anything destructive.
- Reads installed models from Ollama and skips duplicates by default.
- Refreshes model options from `https://ollama.com/library`.
- Falls back to a built-in catalog if the network or library page is unavailable.
- Supports interactive picking, presets, custom names, and full-catalog downloads.
- Can re-pull installed models to update them.
- Includes `--dry-run` so you can preview downloads first.
- Uses only the Python standard library.

## Requirements

- Python 3.10 or newer
- Ollama installed and available from your terminal

Install Ollama from:

```powershell
irm https://ollama.com/install.ps1 | iex
```

Or download it from [ollama.com/download](https://ollama.com/download).

## Usage

Run the interactive picker:

```powershell
python .\ollama_model_manager.py
```

Preview the recommended starter set:

```powershell
python .\ollama_model_manager.py --preset recommended --dry-run
```

Download the recommended starter set:

```powershell
python .\ollama_model_manager.py --preset recommended
```

Download specific models:

```powershell
python .\ollama_model_manager.py --models llama3.2:3b qwen3:8b
```

Download every model in the refreshed catalog:

```powershell
python .\ollama_model_manager.py --all
```

Update every model you already have installed:

```powershell
python .\ollama_model_manager.py --update-installed
```

Use the built-in fallback catalog without going online:

```powershell
python .\ollama_model_manager.py --offline
```

## Useful Options

| Option | What it does |
| --- | --- |
| `--all` | Queue every model in the current catalog. |
| `--preset recommended` | Queue a practical starter set. |
| `--preset small` | Queue lightweight models. |
| `--preset coding` | Queue coding-focused models. |
| `--preset reasoning` | Queue reasoning-focused models. |
| `--preset embedding` | Queue embedding models. |
| `--preset vision` | Queue vision/multimodal models. |
| `--models ...` | Queue explicit model names. |
| `--update-installed` | Re-pull installed models to update them. |
| `--dry-run` | Show what would happen without downloading. |
| `--offline` | Skip the online catalog refresh. |
| `--online-limit N` | Control how many Ollama library families to inspect. |
| `--force` | Pull selected models even if already installed. |
| `--yes` | Skip confirmation prompts. |

## Notes

Ollama does not currently expose an official public API for every remotely pullable model. This tool refreshes from the public Ollama library pages and safely falls back to its built-in catalog if that refresh fails.

Local installed models are read from Ollama itself. Pulling an installed model again is how Ollama checks for updates.

## Project Name

Recommended repository name:

```text
ollama-model-manager
```
