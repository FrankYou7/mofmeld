import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm


TEMPLATES = {
    "void_fraction": {
        "questions": [
            "What is the void fraction of this mof?",
            "How porous is this mof?",
            "Can you tell me the porosity of this mof?",
            "Please tell me the void fraction for this mof.",
            "Could you tell me the void fraction of this mof?"
        ],
        "answers": [
            "The void fraction of this mof is {value}.",
            "This mof has a porosity of {value}.",
            "The porosity of this mof is {value}.",
            "The void fraction for this mof is {value}.",
            "The void fraction of this mof is {value}."
        ]
    },
    "surface_area_m2g": {
        "questions": [
            "What is the surface area of this mof?",
            "Can you give me the surface area of this mof?",
            "How much surface area does this mof have?",
            "Please tell me the surface area for this mof.",
            "Could you tell me the surface area for this mof?"
        ],
        "answers": [
            "The surface area of this mof is {value} m²/g.",
            "This mof has a surface area of {value} m²/g.",
            "This mof has a surface area of {value} m²/g.",
            "The surface area for this mof is {value} m²/g.",
            "The surface area for this mof is {value} m²/g."
        ]
    },
    "pld": {
        "questions": [
            "What is the pld of this mof?",
            "Can you tell me the pld of this mof?",
            "What is the smallest pore channel of this mof?",
            "Please provide the pld for the mof.",
            "Could you tell me the pld of this mof?"
        ],
        "answers": [
            "The pld of this mof is {value}.",
            "The pld of this mof is {value}.",
            "The smallest pore channel of this mof is {value}.",
            "The pld for the mof is {value}.",
            "The pld of this mof is {value}."
        ]
    },
    "lcd": {
        "questions": [
            "What is the lcd of this mof?",
            "Can you tell me the lcd of this mof?",
            "Please provide the lcd for the mof.",
            "Could you tell me the lcd of this mof?",
            "What is the maximum pore size of this mof?"
        ],
        "answers": [
            "The lcd of this mof is {value}.",
            "The lcd of this mof is {value}.",
            "The lcd for the mof is {value}.",
            "The lcd of this mof is {value}.",
            "The maximum pore size of this mof is {value}."
        ]
    },
    "co2": {
        "questions": [
            "What is the CO2 uptake at {temp} K and {pressure} {punit} of this mof?",
            "How much CO2 does this mof adsorb at {temp} K and {pressure} {punit}?",
            "Give me the CO2 adsorption capacity under {temp} K and {pressure} {punit} of this mof.",
            "Can you tell me the CO2 adsorption capacity of this mof under {temp} K and {pressure} {punit}?",
            "What is the CO2 adsorption capacity of this mof at {temp} K and {pressure} {punit}?"
        ],
        "answers": [
            "The CO2 uptake is {uptake} {uunit} at {temp} K and {pressure} {punit} of this mof.",
            "This mof adsorbs {uptake} {uunit} of CO2 at {temp} K and {pressure} {punit}.",
            "This mof can hold {uptake} {uunit} CO2 at {temp} K and {pressure} {punit}.",
            "The CO2 adsorption capacity of this mof under {temp} K and {pressure} {punit} is {uptake} {uunit}.",
            "The CO2 adsorption capacity of this mof at {temp} K and {pressure} {punit} is {uptake} {uunit}."
        ]
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build stage-II fine-tuning QA-style task data from embedding .pt files."
    )
    parser.add_argument(
        "--pt_dir",
        type=str,
        required=True,
        help="Directory containing embedding .pt files.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the generated fine-tuning JSONL file.",
    )
    parser.add_argument(
        "--max_co2_points",
        type=int,
        default=5,
        help="Maximum number of CO2 isotherm points sampled per structure.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for template and CO2-point sampling.",
    )
    return parser.parse_args()


def build_static_property_samples(name: str, path_str: str, properties: dict) -> list[dict]:
    records = []

    for key in ["void_fraction", "surface_area_m2g", "pld", "lcd"]:
        if key in properties and key in TEMPLATES:
            value = properties[key]
            idx = random.randrange(len(TEMPLATES[key]["questions"]))
            question = TEMPLATES[key]["questions"][idx]
            answer = TEMPLATES[key]["answers"][idx].format(value=round(value, 6))
            qa_id = f"{name}-{key}-{idx}"

            records.append(
                {
                    "id": qa_id,
                    "embedding_path": path_str,
                    "input": question,
                    "output": answer,
                }
            )

    return records


def collect_co2_points(properties: dict) -> list[tuple]:
    co2_points = []

    for iso_idx, iso in enumerate(properties.get("co2_isotherms", [])):
        temp = iso.get("temperature", 298)
        punit = iso.get("unit_pressure", "atm")
        uunit = iso.get("unit_loading", "mmol/g")

        for pt_idx, pt in enumerate(iso.get("points", [])):
            co2_points.append(
                (
                    iso_idx,
                    pt_idx,
                    temp,
                    punit,
                    uunit,
                    pt["pressure"],
                    pt["uptake"],
                )
            )

    return co2_points


def build_co2_samples(
    name: str,
    path_str: str,
    properties: dict,
    max_co2_points: int,
) -> list[dict]:
    records = []
    co2_points = collect_co2_points(properties)

    if not co2_points:
        return records

    selected_points = random.sample(co2_points, min(max_co2_points, len(co2_points)))

    for iso_idx, pt_idx, temp, punit, uunit, pressure, uptake in selected_points:
        idx = random.randrange(len(TEMPLATES["co2"]["questions"]))

        question = TEMPLATES["co2"]["questions"][idx].format(
            temp=temp,
            pressure=round(pressure, 6),
            punit=punit,
        )
        answer = TEMPLATES["co2"]["answers"][idx].format(
            uptake=round(uptake, 6),
            uunit=uunit,
            temp=temp,
            pressure=round(pressure, 6),
            punit=punit,
        )

        qa_id = f"{name}-co2-{iso_idx}-{pt_idx}-{idx}"
        records.append(
            {
                "id": qa_id,
                "embedding_path": path_str,
                "input": question,
                "output": answer,
            }
        )

    return records


def main():
    args = parse_args()
    random.seed(args.seed)

    pt_dir = Path(args.pt_dir)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    qa_list = []
    pt_files = sorted(pt_dir.glob("*.pt"))

    for pt_file in tqdm(pt_files, desc="Building stage-II fine-tuning tasks"):
        data = torch.load(pt_file, map_location="cpu")

        properties = data["properties"]
        name = data["name"]
        path_str = str(pt_file)

        qa_list.extend(build_static_property_samples(name, path_str, properties))
        qa_list.extend(build_co2_samples(name, path_str, properties, args.max_co2_points))

    with open(save_path, "w", encoding="utf-8") as f:
        for qa in qa_list:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"Generated {len(qa_list)} QA pairs.")
    print(f"Saved fine-tuning data to: {save_path}")


if __name__ == "__main__":
    main()