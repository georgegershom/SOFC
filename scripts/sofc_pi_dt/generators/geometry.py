from pathlib import Path
from ..utils import write_json


def generate_geometry(output_dir: Path) -> None:
    geom_dir = output_dir / 'geometry'
    geom_dir.mkdir(parents=True, exist_ok=True)

    # Provide simple parametric CAD-like JSON and a placeholder STEP text
    cad = {
        'cell': {'length_mm': 100.0, 'width_mm': 100.0, 'thickness_mm': 0.2},
        'interconnect': {'channel_width_mm': 1.0, 'rib_width_mm': 1.0, 'height_mm': 2.0},
        'seal': {'thickness_mm': 0.3},
        'stack': {'cells': 5, 'manifold_diameter_mm': 10.0}
    }
    write_json(geom_dir / 'cad_parameters.json', cad)

    (geom_dir / 'stack_placeholder.step').write_text('''
ISO-10303-21;
HEADER; FILE_DESCRIPTION(('STEP AP203'),'1'); ENDSEC;
DATA;  /* Placeholder STEP content for demo */ ENDSEC; END-ISO-10303-21;
''')

    (geom_dir / 'assembly_drawing.txt').write_text('Parametric assembly drawing placeholder with dimensions in cad_parameters.json')
