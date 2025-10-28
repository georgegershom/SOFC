#!/usr/bin/env python3
"""
SOFC Abaqus .inp generator (coupled CPS4T, default HR4 schedule)

This script generates an Abaqus/Standard input deck for a 2D SOFC cross-section
using coupled temperature–displacement elements (CPS4T). It encodes geometry,
mesh, layer partitions, materials (temperature-dependent elastic + thermal),
boundary conditions, amplitude-based bottom temperature schedule (HR1/HR4/HR10),
and a film convection boundary at the top edge.

Notes and assumptions
- Units: SI (m, kg, s, K, N, Pa). The original geometry given in mm is
  converted to meters here.
- Geometry: 10.0 mm × 1.0 mm rectangle partitioned at y = 0.40, 0.50, 0.90 mm.
- Mesh: 80 elements along x. Elements across y per layer are proportional to
  thickness with at least 12 in the electrolyte. This yields a total of 120
  elements across the thickness: [48, 12, 48, 12] for anode, electrolyte,
  cathode, interconnect.
- Coupling: Single coupled temperature–displacement step, with bottom edge
  temperature ramp/hold/cool and film convection on the top edge.
- Mechanical BCs: Roller in x on left edge, roller in y on bottom edge.
- Thermal BCs: Prescribed temperature on bottom edge via amplitude; top edge
  film convection h=25 W/m^2-K to 25 °C (298 K) ambient.
- Materials: Temperature-dependent elastic, expansion, conductivity, specific
  heat are included. Densities are provided as reasonable defaults to enable
  transient heat capacity (rho*cp). Plasticity and creep are omitted by default
  in this generator to maximize robustness of first runs; they can be added
  later.

Usage
  python generate_sofc_inp.py --schedule HR4 --output /path/to/sofc_cps4t_HR4.inp

Schedules
  HR1  : 1 °C/min to 900 °C, hold 10 min, cool 1 °C/min (total 1760 min)
  HR4  : 4 °C/min to 900 °C, hold 10 min, cool 4 °C/min (total 447.5 min)
  HR10 : 10 °C/min to 900 °C, hold 10 min, cool 10 °C/min (total 185 min)

"""

import argparse
from typing import List, Tuple


def generate_x_coordinates(length_x_m: float, num_elements_x: int) -> List[float]:
    node_count_x = num_elements_x + 1
    return [i * (length_x_m / num_elements_x) for i in range(node_count_x)]


def generate_y_coordinates(layer_boundaries_m: List[float], elems_per_layer: List[int]) -> List[float]:
    assert len(layer_boundaries_m) == len(elems_per_layer) + 1
    y_coords: List[float] = []
    for layer_index, num_elems in enumerate(elems_per_layer):
        y0 = layer_boundaries_m[layer_index]
        y1 = layer_boundaries_m[layer_index + 1]
        dy = (y1 - y0) / num_elems
        # for each layer, add nodes from bottom to top (excluding the top boundary to avoid duplicates)
        if layer_index == 0:
            # include the very first y=0 on first layer
            for j in range(num_elems):
                y_coords.append(y0 + j * dy)
        else:
            # for subsequent layers, skip the first node to avoid duplicating partition boundary
            for j in range(num_elems):
                if j == 0:
                    continue
                y_coords.append(y0 + j * dy)
    # finally, include the very top boundary once
    y_coords.append(layer_boundaries_m[-1])
    return y_coords


def build_schedule(schedule_name: str) -> Tuple[str, List[Tuple[float, float]]]:
    schedule_name_upper = schedule_name.upper()
    if schedule_name_upper not in {"HR1", "HR4", "HR10"}:
        raise ValueError("schedule must be one of: HR1, HR4, HR10")

    # All schedules: start 25 C (298 K), target 900 C (1173 K), hold 10 min
    t_start_K = 298.0
    t_target_K = 1173.0

    if schedule_name_upper == "HR1":
        ramp_min = 875.0
        hold_min = 10.0
        cool_min = 875.0
    elif schedule_name_upper == "HR4":
        ramp_min = 218.75
        hold_min = 10.0
        cool_min = 218.75
    else:  # HR10
        ramp_min = 87.5
        hold_min = 10.0
        cool_min = 87.5

    ramp_s = ramp_min * 60.0
    hold_s = hold_min * 60.0
    cool_s = cool_min * 60.0

    t0 = 0.0
    t1 = t0 + ramp_s
    t2 = t1 + hold_s
    t3 = t2 + cool_s

    amplitude_points = [
        (t0, t_start_K),
        (t1, t_target_K),
        (t2, t_target_K),
        (t3, t_start_K),
    ]
    return schedule_name_upper, amplitude_points


def format_list(values: List[int], per_line: int = 16) -> str:
    lines = []
    for i in range(0, len(values), per_line):
        chunk = values[i : i + per_line]
        lines.append(", ".join(str(v) for v in chunk))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SOFC Abaqus .inp (CPS4T)")
    parser.add_argument("--schedule", default="HR4", choices=["HR1", "HR4", "HR10"], help="Thermal schedule amplitude")
    parser.add_argument("--output", default="./sofc_cps4t.inp", help="Output .inp file path")
    parser.add_argument("--nelx", type=int, default=80, help="Number of elements along x")
    # y-element counts per layer: [anode, electrolyte, cathode, interconnect]
    parser.add_argument("--ny", type=str, default="48,12,48,12", help="Comma-separated element counts per layer across thickness")
    args = parser.parse_args()

    # Geometry in meters
    width_m = 10.0e-3
    thickness_m = 1.0e-3
    y_partitions_mm = [0.0, 0.40, 0.50, 0.90, 1.00]
    y_partitions_m = [v * 1.0e-3 for v in y_partitions_mm]

    nelx = int(args.nelx)
    nely_per_layer = [int(v) for v in args.ny.split(",")]
    if len(nely_per_layer) != 4:
        raise ValueError("--ny must have 4 comma-separated integers for the four layers")

    # Build mesh coordinates
    x_coords = generate_x_coordinates(width_m, nelx)
    y_coords = generate_y_coordinates(y_partitions_m, nely_per_layer)

    num_nodes_x = len(x_coords)
    num_nodes_y = len(y_coords)
    total_nodes = num_nodes_x * num_nodes_y

    # Node id mapping: (i, j) -> nid = j*num_nodes_x + i + 1
    def nid(i: int, j: int) -> int:
        return j * num_nodes_x + i + 1

    # Build elements (CPS4T): connectivity n1, n2, n3, n4 (bl, br, tr, tl)
    el_connectivity: List[Tuple[int, int, int, int, int]] = []
    elem_id = 1
    for j in range(num_nodes_y - 1):
        for i in range(num_nodes_x - 1):
            n1 = nid(i, j)
            n2 = nid(i + 1, j)
            n3 = nid(i + 1, j + 1)
            n4 = nid(i, j + 1)
            el_connectivity.append((elem_id, n1, n2, n3, n4))
            elem_id += 1

    total_elements = len(el_connectivity)

    # Identify element j-bands per layer (in element index along y)
    nely_layers = nely_per_layer
    cum_elems_y = [0]
    for c in nely_layers:
        cum_elems_y.append(cum_elems_y[-1] + c)
    # cum_elems_y = [0, 48, 60, 108, 120] for default

    # Helper: element index band along y for a given global element row j (0-based)
    # Element row j corresponds to nodes between y_coords[j] and y_coords[j+1]
    layer_bands = [
        ("ANODE", cum_elems_y[0], cum_elems_y[1] - 1),
        ("ELYTE", cum_elems_y[1], cum_elems_y[2] - 1),
        ("CATH", cum_elems_y[2], cum_elems_y[3] - 1),
        ("INTCONN", cum_elems_y[3], cum_elems_y[4] - 1),
    ]

    # Build element id lists for each layer
    layer_to_element_ids = {"ANODE": [], "ELYTE": [], "CATH": [], "INTCONN": []}
    # Also top-row and interface-adjacent rows for surfaces/history
    top_row_element_ids: List[int] = []
    anode_top_row_ids: List[int] = []  # j = cum_elems_y[1]-1
    elyte_top_row_ids: List[int] = []  # j = cum_elems_y[2]-1
    cath_top_row_ids: List[int] = []   # j = cum_elems_y[3]-1

    # Map (i, j_el) -> element id for convenience
    def eid(i: int, j_el: int) -> int:
        return j_el * (num_nodes_x - 1) + i + 1

    for j_el in range(num_nodes_y - 1):
        # Determine which layer this j_el belongs to
        if layer_bands[0][1] <= j_el <= layer_bands[0][2]:
            layer = "ANODE"
        elif layer_bands[1][1] <= j_el <= layer_bands[1][2]:
            layer = "ELYTE"
        elif layer_bands[2][1] <= j_el <= layer_bands[2][2]:
            layer = "CATH"
        else:
            layer = "INTCONN"

        for i_el in range(num_nodes_x - 1):
            layer_to_element_ids[layer].append(eid(i_el, j_el))

        if j_el == (num_nodes_y - 2):
            # very top row elements
            for i_el in range(num_nodes_x - 1):
                top_row_element_ids.append(eid(i_el, j_el))

        if j_el == (cum_elems_y[1] - 1):
            for i_el in range(num_nodes_x - 1):
                anode_top_row_ids.append(eid(i_el, j_el))

        if j_el == (cum_elems_y[2] - 1):
            for i_el in range(num_nodes_x - 1):
                elyte_top_row_ids.append(eid(i_el, j_el))

        if j_el == (cum_elems_y[3] - 1):
            for i_el in range(num_nodes_x - 1):
                cath_top_row_ids.append(eid(i_el, j_el))

    # Node sets for boundaries
    x0_node_ids: List[int] = []
    y0_node_ids: List[int] = []
    ytop_node_ids: List[int] = []
    x_tol = 1e-12
    y_tol = 1e-12
    for j in range(num_nodes_y):
        for i in range(num_nodes_x):
            node_id = nid(i, j)
            x = x_coords[i]
            y = y_coords[j]
            if abs(x - 0.0) < x_tol:
                x0_node_ids.append(node_id)
            if abs(y - 0.0) < y_tol:
                y0_node_ids.append(node_id)
            if abs(y - thickness_m) < y_tol:
                ytop_node_ids.append(node_id)

    schedule_name, amplitude_points = build_schedule(args.schedule)

    # Material data: T-dependent elastic, expansion, conductivity, specific heat
    # Densities are assumed to enable transient capacity; adjust as needed.
    materials = {
        "ANODE": {
            "density": 6500.0,
            "elastic": [(140e9, 0.30, 298.0), (91e9, 0.30, 1273.0)],
            "alpha": [(12.5e-6, 298.0), (13.5e-6, 1273.0)],
            "k": [(6.0, 298.0), (4.0, 1273.0)],
            "cp": [(450.0, 298.0), (570.0, 1273.0)],
        },
        "ELYTE": {
            "density": 6000.0,
            "elastic": [(210e9, 0.28, 298.0), (170e9, 0.28, 1273.0)],
            "alpha": [(10.5e-6, 298.0), (11.2e-6, 1273.0)],
            "k": [(2.6, 298.0), (2.0, 1273.0)],
            "cp": [(400.0, 298.0), (600.0, 1273.0)],
        },
        "CATH": {
            "density": 6600.0,
            "elastic": [(120e9, 0.30, 298.0), (84e9, 0.30, 1273.0)],
            "alpha": [(11.5e-6, 298.0), (12.4e-6, 1273.0)],
            "k": [(2.0, 298.0), (1.8, 1273.0)],
            "cp": [(480.0, 298.0), (610.0, 1273.0)],
        },
        "INTCONN": {
            "density": 7700.0,
            "elastic": [(205e9, 0.30, 298.0), (150e9, 0.30, 1273.0)],
            "alpha": [(12.5e-6, 298.0), (13.2e-6, 1273.0)],
            "k": [(20.0, 298.0), (15.0, 1273.0)],
            "cp": [(500.0, 298.0), (700.0, 1273.0)],
        },
    }

    # Build .inp text
    out_lines: List[str] = []
    add = out_lines.append

    add("*Heading")
    add("** SOFC CPS4T coupled model - generated by generate_sofc_inp.py")
    add("** Domain: 10 mm x 1 mm; units SI; schedule {}".format(schedule_name))
    add("*Preprint, echo=NO, model=NO, history=NO, contact=NO")

    # Part definition
    add("*Part, name=SOFC2D")
    add("*Node")
    for j, y in enumerate(y_coords):
        for i, x in enumerate(x_coords):
            add("{nid}, {x:.9e}, {y:.9e}".format(nid=nid(i, j), x=x, y=y))

    add("*Element, type=CPS4T")
    for (eid_val, n1, n2, n3, n4) in el_connectivity:
        add("{eid}, {n1}, {n2}, {n3}, {n4}".format(eid=eid_val, n1=n1, n2=n2, n3=n3, n4=n4))

    # Element sets per layer (in part space first; we will redefine in assembly for BCs/surfaces)
    for lname in ("ANODE", "ELYTE", "CATH", "INTCONN"):
        add("*Elset, elset={}".format(lname))
        add(format_list(layer_to_element_ids[lname]))

    # Elements for top row and interface-adjacent rows
    add("*Elset, elset=ELTOP")
    add(format_list(top_row_element_ids))
    add("*Elset, elset=EL_ANODE_TOPROW")
    add(format_list(anode_top_row_ids))
    add("*Elset, elset=EL_ELYTE_TOPROW")
    add(format_list(elyte_top_row_ids))
    add("*Elset, elset=EL_CATH_TOPROW")
    add(format_list(cath_top_row_ids))

    # Define surfaces from element faces for convection and interface outputs
    add("*Surface, name=YTOP_SURF, type=ELEMENT")
    add("ELTOP, S3")
    add("*Surface, name=INT_AE, type=ELEMENT")
    add("EL_ANODE_TOPROW, S3")
    add("*Surface, name=INT_EC, type=ELEMENT")
    add("EL_ELYTE_TOPROW, S3")
    add("*Surface, name=INT_CI, type=ELEMENT")
    add("EL_CATH_TOPROW, S3")

    add("*End Part")

    # Assembly and instance
    add("*Assembly, name=Assembly")
    add("*Instance, name=SOFC2D-1, part=SOFC2D")
    add("*End Instance")

    # Assembly-level node sets for boundaries
    add("*Nset, nset=X0, instance=SOFC2D-1")
    add(format_list(x0_node_ids))
    add("*Nset, nset=Y0, instance=SOFC2D-1")
    add(format_list(y0_node_ids))
    add("*Nset, nset=YTOP, instance=SOFC2D-1")
    add(format_list(ytop_node_ids))

    # Assembly-level element sets (reuse names)
    for lname in ("ANODE", "ELYTE", "CATH", "INTCONN", "ELTOP", "EL_ANODE_TOPROW", "EL_ELYTE_TOPROW", "EL_CATH_TOPROW"):
        # Re-declare with instance scoping for convenience in steps/outputs
        if lname == "ANODE":
            ids = layer_to_element_ids["ANODE"]
        elif lname == "ELYTE":
            ids = layer_to_element_ids["ELYTE"]
        elif lname == "CATH":
            ids = layer_to_element_ids["CATH"]
        elif lname == "INTCONN":
            ids = layer_to_element_ids["INTCONN"]
        elif lname == "ELTOP":
            ids = top_row_element_ids
        elif lname == "EL_ANODE_TOPROW":
            ids = anode_top_row_ids
        elif lname == "EL_ELYTE_TOPROW":
            ids = elyte_top_row_ids
        else:
            ids = cath_top_row_ids
        add("*Elset, elset={lname}, instance=SOFC2D-1".format(lname=lname))
        add(format_list(ids))

    # Recreate surfaces at assembly level for loads/outputs
    add("*Surface, name=YTOP_SURF_ASM, type=ELEMENT, internal")
    add("ELTOP, S3")
    add("*Surface, name=INT_AE_ASM, type=ELEMENT, internal")
    add("EL_ANODE_TOPROW, S3")
    add("*Surface, name=INT_EC_ASM, type=ELEMENT, internal")
    add("EL_ELYTE_TOPROW, S3")
    add("*Surface, name=INT_CI_ASM, type=ELEMENT, internal")
    add("EL_CATH_TOPROW, S3")

    add("*End Assembly")

    # Materials and sections
    for lname, props in materials.items():
        add("*Material, name={}".format(lname))
        add("*Density")
        add("{:.6g}".format(props["density"]))
        add("*Elastic")
        for E, nu, T in props["elastic"]:
            add("{:.6g}, {:.6g}, {:.6g}".format(E, nu, T))
        add("*Expansion")
        for alpha, T in props["alpha"]:
            add("{:.6g}, {:.6g}".format(alpha, T))
        add("*Conductivity")
        for k, T in props["k"]:
            add("{:.6g}, {:.6g}".format(k, T))
        add("*Specific Heat")
        for cp, T in props["cp"]:
            add("{:.6g}, {:.6g}".format(cp, T))

    # Section assignments (homogeneous solid sections per layer)
    for lname in ("ANODE", "ELYTE", "CATH", "INTCONN"):
        add("*Solid Section, elset={lname}, material={lname}".format(lname=lname))
        add(",")

    # Amplitudes
    add("*Amplitude, name={}".format(schedule_name))
    # pairs t, T
    amp_vals = []
    for t, T in amplitude_points:
        amp_vals.append("{:.6g}, {:.6g}".format(t, T))
    add(",\n".join(amp_vals))

    # Step: Coupled Temperature-Displacement
    total_time_s = amplitude_points[-1][0]
    add("*Step, name=COUPLED, nlgeom=YES")
    add("*Coupled Temperature-Displacement")
    add("{:.6g}, 1., 1e-06, {:.6g}".format(total_time_s, total_time_s))
    # Mechanical BCs
    add("*Boundary")
    add("X0, 1, 1, 0.")  # U1=0 on left edge
    add("Y0, 2, 2, 0.")  # U2=0 on bottom edge
    # Thermal BC: bottom temperature via amplitude
    add("*Temperature, amplitude={}".format(schedule_name))
    add("Y0, 298.")
    # Film convection on top edge surface to ambient 298 K
    add("*Film")
    add("YTOP_SURF_ASM, 25., 298.")
    # Outputs
    add("*Output, field")
    add("*Element Output, directions=YES")
    add("S, E, LE, HFL, NT")
    add("*Node Output")
    add("U, NT")
    # History outputs along interface-adjacent element sets (stress components)
    add("*Output, history")
    add("*Element Output, elset=EL_ANODE_TOPROW")
    add("S11, S22, S12")
    add("*Element Output, elset=EL_ELYTE_TOPROW")
    add("S11, S22, S12")
    add("*Element Output, elset=EL_CATH_TOPROW")
    add("S11, S22, S12")
    add("*End Step")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    print("Wrote {}".format(args.output))


if __name__ == "__main__":
    main()

