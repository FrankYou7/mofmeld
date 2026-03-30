import argparse
import json
from pathlib import Path

import torch
from pymatgen.core import Structure
from chgnet.model import CHGNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract 64-dimensional CHGNet structure embeddings from CIF files."
    )
    parser.add_argument(
        "--cif_dir",
        type=str,
        required=True,
        help="Directory containing input CIF files.",
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        required=True,
        help="Directory containing JSON property files with matching names.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory for saving extracted embedding .pt files.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Path to save the list of failed files.",
    )
    return parser.parse_args()


def extract_embedding_from_cif(cif_path: Path, chgnet_model: CHGNet):
    """
    Extract a mean-pooled 64-dimensional CHGNet embedding from a CIF file.

    Args:
        cif_path: Path to the CIF file.
        chgnet_model: Loaded CHGNet model.

    Returns:
        graph_vec: Mean-pooled embedding tensor of shape [64]
        num_atoms: Number of atoms in the structure
    """
    structure = Structure.from_file(cif_path)
    graph = chgnet_model.graph_converter(structure)
    output = chgnet_model.predict_graph(graph, return_atom_feas=True)

    atom_fea = torch.tensor(output["atom_fea"], dtype=torch.float32)  # [n_atoms, 64]
    graph_vec = atom_fea.mean(dim=0)                                  # [64]
    num_atoms = atom_fea.shape[0]

    return graph_vec, num_atoms


def load_properties(json_path: Path):
    """
    Load per-structure metadata or properties from a JSON file.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_mof_embedding(
    save_dir: Path,
    name: str,
    embedding: torch.Tensor,
    properties: dict,
    num_atoms: int,
):
    """
    Save a structure embedding file in .pt format.

    Saved format:
        {
            "name": <str>,
            "embedding": <tensor of shape [64]>,
            "properties": <dict>,
            "num_atoms": <int>
        }
    """
    save_path = save_dir / f"{name}.pt"
    torch.save(
        {
            "name": name,
            "embedding": embedding,
            "properties": properties,
            "num_atoms": num_atoms,
        },
        save_path,
    )
    print(f"Saved: {save_path.name} | embedding_shape={tuple(embedding.shape)} | num_atoms={num_atoms}")


def main():
    args = parse_args()

    cif_dir = Path(args.cif_dir)
    json_dir = Path(args.json_dir)
    save_dir = Path(args.save_dir)
    log_path = Path(args.log_path)

    save_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading CHGNet model...")
    chgnet_model = CHGNet.load()

    all_cif_files = sorted(cif_dir.glob("*.cif"))
    total = len(all_cif_files)
    failed = []

    print(f"Found {total} CIF files in: {cif_dir}")

    for i, cif_file in enumerate(all_cif_files, start=1):
        name = cif_file.stem
        json_file = json_dir / f"{name}.json"

        print(f"[{i}/{total}] Processing: {name}")

        if not json_file.exists():
            print(f"Skipped: missing JSON file for {name}")
            failed.append(name)
            continue

        try:
            emb, num_atoms = extract_embedding_from_cif(cif_file, chgnet_model)
            props = load_properties(json_file)
            save_mof_embedding(save_dir, name, emb, props, num_atoms)
        except Exception as e:
            print(f"Error processing {name}: {e}")
            failed.append(name)

    if failed:
        with open(log_path, "w", encoding="utf-8") as f:
            for name in failed:
                f.write(name + "\n")
        print(f"Finished with {len(failed)} failures. Failed file list saved to: {log_path}")
    else:
        print("All files processed successfully.")


if __name__ == "__main__":
    main()