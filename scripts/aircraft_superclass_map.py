"""Shared aircraft-to-superclass mapping for the Kaggle military aircraft dataset."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

AIRCRAFT_TO_SUPERCLASS: Dict[str, str] = {
    # Fighter
    "F16": "Fighter",
    "F18": "Fighter",
    "F35": "Fighter",
    "F15": "Fighter",
    "F14": "Fighter",
    "J20": "Fighter",
    "EF2000": "Fighter",
    "J10": "Fighter",
    "Rafale": "Fighter",
    "F4": "Fighter",
    "F22": "Fighter",
    "JAS39": "Fighter",
    "Su57": "Fighter",
    "Mirage2000": "Fighter",
    "Mig29": "Fighter",
    "JF17": "Fighter",
    "F2": "Fighter",
    "Tejas": "Fighter",
    "J35": "Fighter",
    "FCK1": "Fighter",
    "KF21": "Fighter",
    "KAAN": "Fighter",

    # Cargo
    "C130": "Cargo",
    "A400M": "Cargo",
    "C2": "Cargo",
    "C17": "Cargo",
    "C5": "Cargo",
    "Y20": "Cargo",
    "C1": "Cargo",
    "C390": "Cargo",
    "An72": "Cargo",
    "Il76": "Cargo",
    "An124": "Cargo",
    "An22": "Cargo",
    "An225": "Cargo",

    # Attack Aircraft
    "A10": "Attack Aircraft",
    "Su34": "Attack Aircraft",
    "Su24": "Attack Aircraft",
    "AV8B": "Attack Aircraft",
    "Su25": "Attack Aircraft",
    "EMB314": "Attack Aircraft",
    "JH7": "Attack Aircraft",

    # Tiltrotor
    "V22": "Tiltrotor",
    "V280": "Tiltrotor",

    # Bomber
    "B1": "Bomber",
    "B52": "Bomber",
    "B2": "Bomber",
    "Tu22M": "Bomber",
    "H6": "Bomber",
    "Tu95": "Bomber",
    "Tu160": "Bomber",
    "Vulcan": "Bomber",
    "B21": "Bomber",

    # Helicopter
    "AH64": "Helicopter",
    "CH47": "Helicopter",
    "Mi24": "Helicopter",
    "UH60": "Helicopter",
    "Mi8": "Helicopter",
    "Ka52": "Helicopter",
    "Mi28": "Helicopter",
    "Ka27": "Helicopter",
    "Z10": "Helicopter",
    "Mi26": "Helicopter",
    "Z19": "Helicopter",
    "CH53": "Helicopter",
    "NH90": "Helicopter",
    "WZ10": "Helicopter",

    # Recon / Patrol / Reconnaissance
    "P3": "Recon / Patrol",
    "U2": "Reconnaissance",
    "SR71": "Reconnaissance",
    "WZ9": "Reconnaissance",

    # Interceptor
    "Mig31": "Interceptor",

    # Multirole Combat
    "Tornado": "Multirole Combat",

    # AEW&C
    "E2": "AEW&C",
    "E7": "AEW&C",
    "KJ600": "AEW&C",

    # Tanker
    "KC135": "Tanker",

    # Amphibious Aircraft
    "US2": "Amphibious Aircraft",
    "CL415": "Amphibious Aircraft",
    "Be200": "Amphibious Aircraft",
    "AG600": "Amphibious Aircraft",

    # Drone
    "TB2": "Drone",
    "MQ9": "Drone",
    "RQ4": "Drone",
    "WZ7": "Drone",
    "AKINCI": "Drone",
    "TB001": "Drone",
    "XQ58": "Drone",
    "KIZILELMA": "Drone",
    "MQ25": "Drone",
    "MQ28": "Drone",
    "MQ20": "Drone",

    # Prototype Aircraft
    "F117": "Prototype Aircraft",
    "T50": "Prototype Aircraft",
    "XB70": "Prototype Aircraft",
    "YF23": "Prototype Aircraft",
    "J36": "Prototype Aircraft",
    "X32": "Prototype Aircraft",
    "X29": "Prototype Aircraft",
    "Su47": "Prototype Aircraft",
    "J50": "Prototype Aircraft",
}


def build_superclass_lists() -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for aircraft, superclass in AIRCRAFT_TO_SUPERCLASS.items():
        grouped[superclass].append(aircraft)
    return {k: sorted(v) for k, v in grouped.items()}


SUPERCLASS_ORDER = list(build_superclass_lists().keys())


def resolve_device(preferred: str = "auto") -> str:
    """Choose a GPU-capable device if available; fail if none exists and auto is requested."""
    import torch

    pref = str(preferred).strip().lower()
    if pref not in {"auto", "gpu", "mps", "cuda", "cpu", "0", "1"}:
        return preferred

    if pref == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "0"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    if pref in {"auto", "gpu", "cuda", "mps"}:
        raise RuntimeError("No GPU/MPS device available. This pipeline is configured to run on GPU only.")

    return "cpu"
