from __future__ import annotations
import typer
from pathlib import Path
from typing import Optional

from .config import GeneratorConfig, GridConfig
from .orchestrator import generate_dataset

app = typer.Typer(add_completion=False, help="SOFC multi-fidelity dataset generator")


@app.command()
def generate(
    out_dir: Path = typer.Option(Path("/workspace/data/sofc_mf_dataset"), help="Output directory"),
    num_lf: int = typer.Option(200, help="Number of LF samples"),
    num_mf: int = typer.Option(40, help="Number of MF samples (subset of LF)"),
    num_hf: int = typer.Option(8, help="Number of HF samples (subset of MF)"),
    lf_t_points: int = typer.Option(128, help="LF 1D T points"),
    mf_nx: int = typer.Option(48, help="MF grid X"),
    mf_ny: int = typer.Option(48, help="MF grid Y"),
    hf_nx: int = typer.Option(48, help="HF grid X"),
    hf_ny: int = typer.Option(48, help="HF grid Y"),
    hf_nz: int = typer.Option(16, help="HF grid Z"),
    seed: Optional[int] = typer.Option(1234, help="Random seed"),
):
    grid = GridConfig(
        lf_t_points=lf_t_points,
        mf_nx=mf_nx,
        mf_ny=mf_ny,
        hf_nx=hf_nx,
        hf_ny=hf_ny,
        hf_nz=hf_nz,
    )
    cfg = GeneratorConfig(
        num_lf=num_lf,
        num_mf=num_mf,
        num_hf=num_hf,
        grid=grid,
        seed=seed,
        out_dir=out_dir,
    )
    out = generate_dataset(cfg)
    typer.echo(f"Dataset generated at: {out}")


if __name__ == "__main__":
    app()
