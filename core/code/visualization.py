from __future__ import annotations

import csv
import ast
from pathlib import Path

from chemistry import smiles_to_half_graph
from graph import HalfGraph
from symmetry import mirror_complete


def draw_symmetric_grid_from_csv(
    csv_path: str | Path,
    output_path: str | Path,
    smiles_column: str = "canonical_smiles",
    id_column: str = "usable_id",
    limit: int | None = 80,
    mols_per_row: int = 4,
) -> int:
    """Draw generated molecules in a fixed mirror-axis layout."""

    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]

    from PIL import Image, ImageDraw

    tile_w, tile_h = 260, 220
    images = []
    for index, row in enumerate(rows, start=1):
        smiles = row.get(smiles_column, "").strip()
        label = row.get(id_column, "").strip() or f"candidate_{index:06d}"
        images.append(_draw_symmetric_tile(smiles, label, row, tile_w, tile_h))

    if not images:
        output = Image.new("RGB", (tile_w, tile_h), "white")
    else:
        rows_count = (len(images) + mols_per_row - 1) // mols_per_row
        output = Image.new("RGB", (tile_w * mols_per_row, tile_h * rows_count), "white")
        for index, image in enumerate(images):
            x = (index % mols_per_row) * tile_w
            y = (index // mols_per_row) * tile_h
            output.paste(image, (x, y))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)
    return len(images)


def _draw_symmetric_tile(smiles: str, label: str, row: dict[str, str], width: int, height: int):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.rectangle((0, 0, width - 1, height - 1), outline=(225, 225, 225))
    try:
        half_graph = _half_graph_from_row(row, smiles)
        edges_full = mirror_complete(half_graph)
        positions = _symmetric_positions(half_graph, width, height)

        axis_x = width // 2
        draw.line((axis_x, 22, axis_x, height - 46), fill=(180, 180, 180), width=1)
        draw.text((axis_x + 4, 24), "axis", fill=(120, 120, 120), font=font)

        for u, v in edges_full:
            fill = (35, 35, 35)
            width_px = 2
            if _is_n2_mirror_bond(half_graph, u, v):
                fill = (190, 60, 55)
                width_px = 3
            draw.line((*positions[u], *positions[v]), fill=fill, width=width_px)

        for node, (x, y) in positions.items():
            fill = _node_color(half_graph, node)
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=fill, outline=(20, 20, 20))
            draw.text((x + 7, y - 6), str(node), fill=(50, 50, 50), font=font)

        meta = (
            f"{label} C{row.get('carbon_count', '?')} "
            f"n1/n2/n3={half_graph.n1}/{half_graph.n2}/{half_graph.n3}"
        )
        five_member_rings = row.get(
            "five_member_rings", row.get("projected_five_member_rings", "?")
        )
        six_member_rings = row.get(
            "six_member_rings", row.get("projected_six_member_rings", "?")
        )
        rings = (
            f"5/6={five_member_rings}/{six_member_rings} "
            f"score={row.get('screen_score', '?')}"
        )
        draw.text((8, height - 36), meta, fill=(0, 0, 0), font=font)
        draw.text((8, height - 18), rings, fill=(0, 0, 0), font=font)
    except Exception as exc:
        draw.text((8, 20), label, fill=(0, 0, 0), font=font)
        draw.text((8, 42), "symmetry layout failed", fill=(160, 0, 0), font=font)
        draw.text((8, 62), str(exc)[:40], fill=(160, 0, 0), font=font)
    return image


def _half_graph_from_row(row: dict[str, str], smiles: str) -> HalfGraph:
    edges_text = row.get("edges_half_predicted", "").strip()
    if edges_text:
        edges = tuple(tuple(edge) for edge in ast.literal_eval(edges_text))
        return HalfGraph(
            n1=int(row["n1"]),
            n2=int(row["n2"]),
            n3=int(row["n3"]),
            edges=edges,
        )
    return smiles_to_half_graph(smiles)


def _symmetric_positions(graph: HalfGraph, width: int, height: int) -> dict[int, tuple[int, int]]:
    top, bottom = 42, height - 62
    center_x = width // 2
    positions: dict[int, tuple[int, int]] = {}

    for node, y in zip(graph.n1_range, _spread(graph.n1, top, bottom)):
        positions[node] = (center_x, y)

    n2_originals = tuple(graph.n2_original_range)
    for node, y in zip(n2_originals, _spread(len(n2_originals), top, bottom)):
        positions[node] = (center_x - int(width * 0.23), y)
        positions[graph.mirror_node(node)] = (center_x + int(width * 0.23), y)

    n3_originals = tuple(graph.n3_original_range)
    for node, y in zip(n3_originals, _spread(len(n3_originals), top, bottom)):
        positions[node] = (center_x - int(width * 0.39), y)
        positions[graph.mirror_node(node)] = (center_x + int(width * 0.39), y)

    return positions


def _spread(count: int, top: int, bottom: int) -> tuple[int, ...]:
    if count <= 0:
        return ()
    if count == 1:
        return ((top + bottom) // 2,)
    return tuple(round(top + (bottom - top) * index / (count - 1)) for index in range(count))


def _node_color(graph: HalfGraph, node: int) -> tuple[int, int, int]:
    if node in graph.n1_range:
        return (35, 35, 35)
    if node in graph.n2_original_range or node in graph.n3_original_range:
        return (69, 117, 180)
    return (231, 138, 72)


def _is_n2_mirror_bond(graph: HalfGraph, u: int, v: int) -> bool:
    for node in graph.n2_original_range:
        if {u, v} == {node, graph.mirror_node(node)}:
            return True
    return False
