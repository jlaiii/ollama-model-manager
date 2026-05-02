#!/usr/bin/env python3
"""
Interactive Ollama model downloader.

Run after installing Ollama:
    python ollama_model_manager.py

Optional shortcuts:
    python ollama_model_manager.py --preset recommended
    python ollama_model_manager.py --models llama3.2 qwen2.5-coder:7b
    python ollama_model_manager.py --all --yes
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import os
import platform
import re
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from typing import Iterable


OLLAMA_API = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_LIBRARY_URL = "https://ollama.com/library"


@dataclass(frozen=True)
class Model:
    name: str
    label: str
    category: str
    size: str
    notes: str


CATALOG: list[Model] = [
    Model("qwen3:0.6b", "Qwen 3 0.6B", "general", "~523 MB", "Tiny current Qwen model."),
    Model("qwen3:1.7b", "Qwen 3 1.7B", "general", "~1.4 GB", "Small current Qwen model."),
    Model("qwen3:4b", "Qwen 3 4B", "general", "~2.5 GB", "Modern small model with long context."),
    Model("qwen3:8b", "Qwen 3 8B", "general", "~5.2 GB", "Current balanced Qwen model."),
    Model("qwen3:14b", "Qwen 3 14B", "general", "~9.3 GB", "Larger current Qwen model."),
    Model("gemma4:e2b", "Gemma 4 E2B", "general", "~7.2 GB", "Modern Gemma model for edge devices."),
    Model("gemma4:e4b", "Gemma 4 E4B", "general", "~9.6 GB", "Modern Gemma model, default Gemma 4 tag."),
    Model("gemma4:26b", "Gemma 4 26B", "general", "~18 GB", "Workstation-class Gemma 4 MoE model."),
    Model("gpt-oss:20b", "GPT OSS 20B", "reasoning", "~14 GB", "Open-weight reasoning and agentic model."),
    Model("llama3.2:1b", "Llama 3.2 1B", "general", "~1.3 GB", "Tiny, fast general chat model."),
    Model("llama3.2:3b", "Llama 3.2 3B", "general", "~2.0 GB", "Good small general model."),
    Model("llama3.1:8b", "Llama 3.1 8B", "general", "~4.9 GB", "Balanced general-purpose model."),
    Model("mistral:7b", "Mistral 7B", "general", "~4.1 GB", "Strong compact general model."),
    Model("gemma3:1b", "Gemma 3 1B", "general", "~815 MB", "Very small Google Gemma model."),
    Model("gemma3:4b", "Gemma 3 4B", "general", "~3.3 GB", "Compact multimodal-capable family member."),
    Model("qwen2.5:7b", "Qwen 2.5 7B", "general", "~4.7 GB", "Strong multilingual general model."),
    Model("phi4", "Phi 4", "general", "~9.1 GB", "Reasoning-focused compact model."),
    Model("deepseek-r1:1.5b", "DeepSeek R1 1.5B", "reasoning", "~1.1 GB", "Small reasoning model."),
    Model("deepseek-r1:7b", "DeepSeek R1 7B", "reasoning", "~4.7 GB", "Reasoning model with manageable size."),
    Model("deepseek-r1:14b", "DeepSeek R1 14B", "reasoning", "~9.0 GB", "Larger reasoning model."),
    Model("qwen2.5-coder:1.5b", "Qwen 2.5 Coder 1.5B", "coding", "~986 MB", "Tiny coding helper."),
    Model("qwen2.5-coder:7b", "Qwen 2.5 Coder 7B", "coding", "~4.7 GB", "Solid coding model."),
    Model("qwen2.5-coder:14b", "Qwen 2.5 Coder 14B", "coding", "~9.0 GB", "Bigger coding model."),
    Model("codellama:7b", "Code Llama 7B", "coding", "~3.8 GB", "Classic code model."),
    Model("starcoder2:3b", "StarCoder2 3B", "coding", "~1.7 GB", "Small code completion model."),
    Model("nomic-embed-text", "Nomic Embed Text", "embedding", "~274 MB", "Text embeddings for search/RAG."),
    Model("all-minilm", "All MiniLM", "embedding", "~46 MB", "Very small embedding model."),
    Model("llava:7b", "LLaVA 7B", "vision", "~4.7 GB", "Vision-language model."),
    Model("bakllava", "BakLLaVA", "vision", "~4.7 GB", "Vision-language model."),
]

PRESETS: dict[str, list[str]] = {
    "recommended": ["qwen3:4b", "gemma4:e2b", "qwen2.5-coder:7b", "nomic-embed-text"],
    "small": ["qwen3:0.6b", "qwen3:1.7b", "llama3.2:1b", "deepseek-r1:1.5b", "all-minilm"],
    "coding": ["qwen2.5-coder:1.5b", "qwen2.5-coder:7b", "codellama:7b", "starcoder2:3b"],
    "reasoning": ["deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:14b", "gpt-oss:20b"],
    "embedding": ["nomic-embed-text", "all-minilm"],
    "vision": ["gemma4:e2b", "gemma4:e4b", "llava:7b", "bakllava"],
}


def main() -> int:
    args = parse_args()

    print_header()
    catalog = build_catalog(args)
    ollama = find_ollama()
    if not ollama:
        print_missing_ollama()
        return 1

    print(f"Found Ollama CLI: {ollama}")
    print_ollama_version(ollama)
    warn_if_server_unreachable()

    installed = get_installed_models(ollama)
    if installed:
        print(f"Installed models found: {', '.join(sorted(installed))}")
    else:
        print("No installed models detected yet.")

    selected = sorted(installed) if args.update_installed else select_models(args, catalog)
    selected = unique_preserving_order(selected)

    if not selected:
        print("No models selected. Nothing to download.")
        return 0

    installed_keys = installed_model_keys(installed)
    missing = [model for model in selected if normalize_model_name(model) not in installed_keys]
    already_have = [model for model in selected if normalize_model_name(model) in installed_keys]

    if already_have and not args.force and not args.update_installed:
        print("\nSkipping already installed:")
        for model in already_have:
            print(f"  - {model}")

    to_pull = selected if args.force or args.update_installed else missing
    if not to_pull:
        print("\nEverything selected is already installed.")
        return 0

    print("\nModels queued for download:")
    for model in to_pull:
        detail = catalog_lookup(model, catalog)
        suffix = format_model_detail(detail)
        print(f"  - {model}{suffix}")

    if args.dry_run:
        print("\nDry run only. No downloads started.")
        return 0

    if not args.yes and not confirm("\nStart downloading these models?"):
        print("Cancelled.")
        return 0

    failures = pull_models(ollama, to_pull, keep_going=args.keep_going)
    print_summary(ollama, failures)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick and download Ollama models.")
    parser.add_argument("--all", action="store_true", help="Download every model in this script's catalog.")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Download a preset group.")
    parser.add_argument("--models", nargs="+", help="Download explicit model names, e.g. llama3.2 qwen2.5-coder:7b.")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without pulling models.")
    parser.add_argument("--update-installed", action="store_true", help="Re-pull every installed model to update it.")
    parser.add_argument("--offline", action="store_true", help="Use only the built-in fallback catalog.")
    parser.add_argument(
        "--online-limit",
        type=int,
        default=40,
        help="Maximum online Ollama library families to inspect. Default: 40.",
    )
    parser.add_argument("--force", action="store_true", help="Pull selected models even if already installed.")
    parser.add_argument(
        "--keep-going",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue after a failed download. Default: true.",
    )
    return parser.parse_args()


def build_catalog(args: argparse.Namespace) -> list[Model]:
    if args.offline:
        print("Using built-in fallback catalog only.")
        return CATALOG

    print("Refreshing model catalog from Ollama library...")
    online = fetch_online_catalog(limit=max(args.online_limit, 0))
    if not online:
        print("Could not refresh online catalog. Using built-in fallback catalog.")
        return CATALOG

    merged = merge_catalogs(online, CATALOG)
    print(f"Loaded {len(online)} online model tags; {len(merged)} total choices after fallback merge.")
    return merged


def fetch_online_catalog(limit: int) -> list[Model]:
    try:
        index_html = fetch_text(OLLAMA_LIBRARY_URL, timeout=15)
    except Exception as exc:
        print(f"Online catalog fetch failed: {exc}")
        return []

    families = parse_library_families(index_html)
    if limit:
        families = families[:limit]

    models: list[Model] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for family_models in executor.map(fetch_family_models, families):
            models.extend(family_models)

    return unique_models(models)


def parse_library_families(page: str) -> list[str]:
    families: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r'href="/library/([^"#?]+)"', page):
        name = html.unescape(match.group(1)).strip("/")
        if name and name not in seen:
            families.append(name)
            seen.add(name)
    return families


def fetch_family_models(family: str) -> list[Model]:
    try:
        page = fetch_text(f"{OLLAMA_LIBRARY_URL}/{family}", timeout=12)
    except Exception:
        return [Model(family, title_from_name(family), "online", "unknown", "Online Ollama library model.")]

    description = extract_description(page) or "Online Ollama library model."
    category = infer_category(family, description)
    names = parse_family_tags(page, family)
    if not names:
        names = [family]

    return [Model(name, title_from_name(name), category, "unknown", description) for name in names]


def parse_family_tags(page: str, family: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    pattern = rf"\b{re.escape(family)}:[A-Za-z0-9][A-Za-z0-9._-]*"

    for name in re.findall(pattern, page):
        normalized = normalize_model_name(html.unescape(name))
        if "cloud" in normalized:
            continue
        if normalized not in seen:
            names.append(html.unescape(name))
            seen.add(normalized)

    if not names and re.search(rf"\bollama\s+(run|pull)\s+{re.escape(family)}\b", page):
        names.append(family)

    return names


def infer_category(name: str, description: str) -> str:
    text = f"{name} {description}".lower()
    if "embedding" in text:
        return "embedding"
    if "code" in text or "coding" in text or "coder" in text:
        return "coding"
    if "reasoning" in text or "thinking" in text:
        return "reasoning"
    if "vision" in text or "image" in text or "multimodal" in text:
        return "vision"
    return "online"


def extract_description(page: str) -> str | None:
    match = re.search(r'<meta\s+name="description"\s+content="([^"]*)"', page)
    if not match:
        match = re.search(r'<meta\s+property="og:description"\s+content="([^"]*)"', page)
    if not match:
        return None
    description = html.unescape(match.group(1)).strip()
    return re.sub(r"\s+", " ", description)[:140]


def fetch_text(url: str, timeout: int) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "ollama-model-manager/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def merge_catalogs(primary: list[Model], fallback: list[Model]) -> list[Model]:
    merged = unique_models(primary)
    seen = {normalize_model_name(model.name) for model in merged}
    for model in fallback:
        key = normalize_model_name(model.name)
        if key not in seen:
            merged.append(model)
            seen.add(key)
    return merged


def unique_models(models: Iterable[Model]) -> list[Model]:
    unique: list[Model] = []
    seen: set[str] = set()
    for model in models:
        key = normalize_model_name(model.name)
        if key and key not in seen:
            unique.append(model)
            seen.add(key)
    return unique


def title_from_name(name: str) -> str:
    return name.replace("-", " ").replace("_", " ").replace(":", " ").title()


def print_header() -> None:
    print("=" * 72)
    print("Ollama Model Manager")
    print("=" * 72)
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Ollama API target: {OLLAMA_API}")
    print()


def find_ollama() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found

    if platform.system().lower() == "windows":
        candidates = [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
            os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

    return None


def print_missing_ollama() -> None:
    print("Ollama was not found on PATH.")
    if platform.system().lower() == "windows":
        print("\nInstall it from https://ollama.com/download/windows")
        print("PowerShell installer command:")
        print("  irm https://ollama.com/install.ps1 | iex")
        print("\nAfter installing, open a new terminal and run this script again.")
    else:
        print("\nInstall it from https://ollama.com/download")
        print("Linux/macOS installer command:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")


def print_ollama_version(ollama: str) -> None:
    result = run([ollama, "--version"], capture=True)
    if result.returncode == 0 and result.stdout.strip():
        print(result.stdout.strip())
    elif result.stderr.strip():
        print(result.stderr.strip())


def warn_if_server_unreachable() -> None:
    try:
        request_json("/api/version", timeout=2)
    except Exception:
        print("Note: Ollama's local API did not respond yet.")
        print("      If downloads fail, open the Ollama app or run: ollama serve")


def get_installed_models(ollama: str) -> set[str]:
    from_api = get_installed_models_from_api()
    if from_api:
        return from_api

    result = run([ollama, "list"], capture=True)
    if result.returncode != 0:
        return set()

    models: set[str] = set()
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.add(parts[0])
    return models


def get_installed_models_from_api() -> set[str]:
    try:
        payload = request_json("/api/tags", timeout=2)
    except Exception:
        return set()

    models = set()
    for item in payload.get("models", []):
        name = item.get("name")
        if name:
            models.add(name)
    return models


def installed_model_keys(installed: Iterable[str]) -> set[str]:
    keys: set[str] = set()
    for model in installed:
        normalized = normalize_model_name(model)
        keys.add(normalized)
        if normalized.endswith(":latest"):
            keys.add(normalized.removesuffix(":latest"))
        elif ":" not in normalized:
            keys.add(f"{normalized}:latest")
    return keys


def request_json(path: str, timeout: int) -> dict:
    with urllib.request.urlopen(f"{OLLAMA_API}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def select_models(args: argparse.Namespace, catalog: list[Model]) -> list[str]:
    if args.all:
        return [model.name for model in catalog]
    if args.preset:
        return PRESETS[args.preset]
    if args.models:
        return args.models

    while True:
        print_menu()
        choice = input("\nChoose an option: ").strip().lower()

        if choice in {"q", "quit", "exit"}:
            return []
        if choice == "1":
            return PRESETS["recommended"]
        if choice == "2":
            return choose_preset()
        if choice == "3":
            return choose_from_catalog(catalog)
        if choice == "4":
            return [model.name for model in catalog]
        if choice == "5":
            return enter_custom_models()

        print("Please enter 1, 2, 3, 4, 5, or q.")


def print_menu() -> None:
    print("\nWhat would you like to download?")
    print("  1. Recommended starter set")
    print("  2. Preset category")
    print("  3. Pick specific catalog models")
    print("  4. Every catalog model")
    print("  5. Type custom model names")
    print("  q. Quit")


def choose_preset() -> list[str]:
    names = sorted(PRESETS)
    print("\nPresets:")
    for index, name in enumerate(names, 1):
        print(f"  {index}. {name} ({', '.join(PRESETS[name])})")

    choice = input("\nPreset number/name: ").strip().lower()
    if choice.isdigit() and 1 <= int(choice) <= len(names):
        return PRESETS[names[int(choice) - 1]]
    if choice in PRESETS:
        return PRESETS[choice]

    print("Unknown preset.")
    return []


def choose_from_catalog(catalog: list[Model]) -> list[str]:
    print_catalog(catalog)
    raw = input("\nEnter numbers/ranges (example: 1,3,7-10) or model names: ").strip()
    return parse_selection(raw, catalog)


def enter_custom_models() -> list[str]:
    raw = input("\nEnter model names separated by commas or spaces: ").strip()
    return [item.strip() for item in raw.replace(",", " ").split() if item.strip()]


def print_catalog(catalog: list[Model]) -> None:
    print("\nCatalog:")
    for index, model in enumerate(catalog, 1):
        print(f"  {index:>2}. {model.name:<24} {model.size:<8} {model.category:<9} {model.notes}")


def parse_selection(raw: str, catalog: list[Model]) -> list[str]:
    selected: list[str] = []
    tokens = [token.strip() for token in raw.replace(",", " ").split() if token.strip()]

    for token in tokens:
        if "-" in token and all(part.isdigit() for part in token.split("-", 1)):
            start, end = [int(part) for part in token.split("-", 1)]
            if start > end:
                start, end = end, start
            for number in range(start, end + 1):
                add_catalog_number(selected, number, catalog)
        elif token.isdigit():
            add_catalog_number(selected, int(token), catalog)
        else:
            selected.append(token)

    return selected


def add_catalog_number(selected: list[str], number: int, catalog: list[Model]) -> None:
    if 1 <= number <= len(catalog):
        selected.append(catalog[number - 1].name)
    else:
        print(f"Ignoring catalog number outside range: {number}")


def catalog_lookup(name: str, catalog: list[Model]) -> Model | None:
    normalized = normalize_model_name(name)
    for model in catalog:
        if normalize_model_name(model.name) == normalized:
            return model
    return None


def format_model_detail(model: Model | None) -> str:
    if not model:
        return ""
    if model.size == "unknown" and model.category == "online":
        return " (online catalog)"
    if model.size == "unknown":
        return f" ({model.category})"
    return f" ({model.size}, {model.category})"


def normalize_model_name(name: str) -> str:
    return name.strip().lower()


def unique_preserving_order(models: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for model in models:
        model = model.strip()
        key = normalize_model_name(model)
        if model and key not in seen:
            unique.append(model)
            seen.add(key)
    return unique


def confirm(prompt: str) -> bool:
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def pull_models(ollama: str, models: list[str], keep_going: bool) -> list[str]:
    failures: list[str] = []
    total = len(models)

    for index, model in enumerate(models, 1):
        print("\n" + "-" * 72)
        print(f"[{index}/{total}] Pulling {model}")
        print("-" * 72)
        started = time.time()
        result = run([ollama, "pull", model], capture=False)
        elapsed = format_seconds(time.time() - started)

        if result.returncode == 0:
            print(f"Finished {model} in {elapsed}.")
        else:
            print(f"Failed to pull {model} after {elapsed}.")
            failures.append(model)
            if not keep_going:
                break

    return failures


def print_summary(ollama: str, failures: list[str]) -> None:
    print("\n" + "=" * 72)
    if failures:
        print("Done, but these models failed:")
        for model in failures:
            print(f"  - {model}")
    else:
        print("All requested downloads completed.")

    print("\nCurrent Ollama models:")
    result = run([ollama, "list"], capture=True)
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print("Could not read model list.")


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def run(command: list[str], capture: bool) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(command, returncode=127, stdout="", stderr="Command not found")
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)


if __name__ == "__main__":
    raise SystemExit(main())
