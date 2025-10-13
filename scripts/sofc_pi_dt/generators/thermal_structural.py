import numpy as np
from pathlib import Path
from ..utils import write_json


def generate_thermal_structural(output_dir: Path, seed: int, num_tc: int, ir_frames: int, ir_size: int, num_sg: int, dic_frames: int) -> None:
    rng = np.random.default_rng(seed + 3)
    ts_dir = output_dir / 'thermo_structural'
    ts_dir.mkdir(parents=True, exist_ok=True)

    # Thermocouples: multiple points
    t = np.linspace(0, 3600, 720)
    tc = []
    for k in range(num_tc):
        base = 750 + 20*np.sin(2*np.pi*t/1500 + k*0.3)
        noise = rng.normal(scale=0.5, size=t.shape)
        tc.append(base + noise)
    tc_data = np.column_stack([t] + tc).astype(np.float32)
    np.save(ts_dir / 'thermocouples.npy', tc_data)

    # IR camera: 2D frames with a hotspot drifting
    ir = np.zeros((ir_frames, ir_size, ir_size), dtype=np.float32)
    yy, xx = np.mgrid[0:ir_size, 0:ir_size]
    for f in range(ir_frames):
        cx = ir_size/2 + 15*np.sin(2*np.pi*f/20)
        cy = ir_size/2 + 10*np.cos(2*np.pi*f/25)
        sigma = 8.0
        hotspot = np.exp(-((xx-cx)**2 + (yy-cy)**2)/(2*sigma**2))
        background = 720 + 10*np.sin(2*np.pi*f/30)
        frame = background + 25*hotspot + rng.normal(scale=0.3, size=(ir_size, ir_size))
        ir[f] = frame
    np.save(ts_dir / 'ir_frames.npy', ir)

    # Strain gauges time-series
    sg = []
    for k in range(num_sg):
        base = 0.001 + 0.0002*np.sin(2*np.pi*t/1800 + k*0.5)
        noise = rng.normal(scale=1e-5, size=t.shape)
        sg.append(base + noise)
    sg_data = np.column_stack([t] + sg).astype(np.float32)
    np.save(ts_dir / 'strain_gauges.npy', sg_data)

    # DIC frames: 2D strain fields (ex-situ)
    dic = np.zeros((dic_frames, ir_size, ir_size), dtype=np.float32)
    for f in range(dic_frames):
        gradient = (np.linspace(0, 1e-3, ir_size)[None, :] + np.linspace(0, 1e-3, ir_size)[:, None])
        localized = 8e-4*np.exp(-((xx-0.6*ir_size)**2 + (yy-0.4*ir_size)**2)/(2*9**2))
        dic[f] = gradient + localized + rng.normal(scale=2e-5, size=(ir_size, ir_size))
    np.save(ts_dir / 'dic_strain_frames.npy', dic)

    write_json(ts_dir / 'metadata.json', {
        'thermocouples': {'columns': ['t'] + [f'TC{k}' for k in range(num_tc)]},
        'ir': {'frames': ir_frames, 'size': ir_size},
        'strain_gauges': {'columns': ['t'] + [f'SG{k}' for k in range(num_sg)]},
        'dic': {'frames': dic_frames, 'size': ir_size}
    })
