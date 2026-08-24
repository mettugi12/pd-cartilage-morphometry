"""
viewer_app.py  –  PD vs DESS Interactive Slice Viewer

Standalone Tkinter viewer.  Reads from ./nifti/<patient>/ (5 NIfTI files per patient).
No hardcoded paths — works on any machine.

Panel layout:
  [0,0] DESS MR (raw)            [0,1] DESS MR + DESS seg       [0,2] DESS MR + PD reg seg
  [1,0] PD MR native (closest)   [1,1] PD MR registered + seg   [1,2] Overlap diff

Controls:
  • Patient dropdown (sidebar)
  • Slice slider at the bottom
  • ← / →  ±1 slice,  ↑ / ↓  ±5 slices
  • Label checkboxes and alpha slider (sidebar)

Required files in ./nifti/<patient>/:
  DESS_MR.nii.gz              (0.7 mm iso, 160×200×200)
  DESS_labels.nii.gz          (label map 0-11)
  PD_MR_registered.nii.gz    (same grid)
  PD_labels_registered.nii.gz
  PD_MR_native.nii.gz        (in-plane resampled, native z-spacing)

Launch:
  python viewer_app.py
  — or double-click launch_viewer.bat —
"""

import gzip, struct, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation
import tkinter as tk
from tkinter import ttk, messagebox

# ─── DATA DIRECTORY ──────────────────────────────────────────────────────────
# Works both as a plain .py script and as a PyInstaller --onedir bundle.
# When frozen, sys.executable is the .exe path; its parent holds the nifti/ folder.
if getattr(sys, 'frozen', False):
    _BASE = Path(sys.executable).parent   # PyInstaller bundle dir
else:
    _BASE = Path(__file__).parent         # normal Python execution
DATA_DIR = _BASE / "nifti"

# ─── SEGMENTATION APPEARANCE ─────────────────────────────────────────────────
CHANNEL_INFO = {
    0:  ("Patella",         "#FF2D2D"),
    1:  ("Femur",           "#00B4D8"),
    2:  ("Tibia",           "#0096C7"),
    3:  ("Patella Cart.",   "#FF6D00"),
    4:  ("Femur Cart.",     "#00C896"),
    5:  ("Tibia Cart.",     "#FFD000"),
    6:  ("Med. Meniscus",   "#00BCD4"),
    7:  ("Lat. Meniscus",   "#9C27B0"),
    8:  ("ACL",             "#FF4081"),
    9:  ("PCL",             "#1976D2"),
    10: ("Patellar Tend.",  "#FFD700"),
}
DEFAULT_ACTIVE = {4, 5}   # Femur cart + Tibia cart on by default
N_CH = len(CHANNEL_INFO)


# ─── IMAGE HELPERS ───────────────────────────────────────────────────────────
def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def load_nifti(path):
    """Load .nii.gz → (D, H, W) array + (x_sp, y_sp, z_sp) spacing in mm.
    Round-trip compatible with SimpleITK-written files.
    """
    with gzip.open(str(path), 'rb') as f:
        header = f.read(352)
        raw    = f.read()
    dim    = struct.unpack('<8h', header[40:56])
    pixdim = struct.unpack('<8f', header[76:108])
    dtype_map = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 512: np.uint16}
    dtype = dtype_map.get(struct.unpack('<h', header[70:72])[0], np.int16)
    arr = np.frombuffer(raw, dtype=dtype).copy().reshape((dim[3], dim[2], dim[1]))
    return arr, (float(pixdim[1]), float(pixdim[2]), float(pixdim[3]))


def label_to_masks(label_arr, n_ch=N_CH):
    """(D, H, W) uint8 label map → (n_ch, D, H, W) uint8 binary channels."""
    masks = np.zeros((n_ch, *label_arr.shape), dtype=np.uint8)
    for ch in range(n_ch):
        masks[ch] = (label_arr == ch + 1)
    return masks


def normalize(img):
    img = img.astype(np.float32)
    m = img > 0
    if not m.any():
        return np.zeros_like(img)
    lo, hi = np.percentile(img[m], [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def overlay(mr_2d, masks_2d, active_chs, alpha):
    """Return (H, W, 3) RGB: grayscale MR with colored seg overlay + edge highlight."""
    rgb = np.stack([normalize(mr_2d)] * 3, axis=-1)
    for ch in active_chs:
        if ch >= masks_2d.shape[0]:
            continue
        fg = masks_2d[ch] > 0
        if not fg.any():
            continue
        c = np.array(hex_to_rgb(CHANNEL_INFO[ch][1]))
        rgb[fg] = rgb[fg] * (1 - alpha) + c * alpha
        edge = binary_dilation(fg, iterations=1) & ~fg
        rgb[edge] = np.minimum(1.0, c * 1.3)
    return np.clip(rgb, 0, 1)


# ─── DATA LOADING (dict cache — loads each patient once) ─────────────────────
_cache = {}

def load_pair(patient_dir: Path) -> dict:
    key = str(patient_dir)
    if key in _cache:
        return _cache[key]

    required = ["DESS_MR.nii.gz", "DESS_labels.nii.gz",
                "PD_MR_registered.nii.gz", "PD_labels_registered.nii.gz",
                "PD_MR_native.nii.gz"]
    missing = [f for f in required if not (patient_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing in {patient_dir.name}/:\n  " + "\n  ".join(missing))

    dess_mr,    _       = load_nifti(patient_dir / "DESS_MR.nii.gz")
    dess_lbl,   _       = load_nifti(patient_dir / "DESS_labels.nii.gz")
    pd_mr_reg,  _       = load_nifti(patient_dir / "PD_MR_registered.nii.gz")
    pd_lbl,     _       = load_nifti(patient_dir / "PD_labels_registered.nii.gz")
    pd_native,  pd_sp   = load_nifti(patient_dir / "PD_MR_native.nii.gz")

    D, H, W = dess_mr.shape
    result = dict(
        name=patient_dir.name,
        dess_mr=dess_mr.astype(np.float32),
        dess_mask=label_to_masks(dess_lbl),
        pd_mr_reg=pd_mr_reg.astype(np.float32),
        pd_mask=label_to_masks(pd_lbl),
        pd_mr_native=pd_native.astype(np.float32),
        dess_z_sp=0.7,                  # DESS is always 0.7 mm isotropic
        pd_z_sp=float(pd_sp[2]),        # native PD z-spacing from header
        D=D, H=H, W=W,
    )
    _cache[key] = result
    return result


# ─── MAIN APPLICATION ────────────────────────────────────────────────────────
class ViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PD vs DESS Slice Viewer")
        self.state('zoomed')

        patients = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
        if not patients:
            messagebox.showerror("No data",
                f"No patient folders found in:\n{DATA_DIR}\n\n"
                "Run export_nifti_pairs.py first, then place the 'nifti' folder\n"
                "next to viewer_app.py.")
            self.destroy()
            return

        self._patients = patients
        self._data     = None
        self._build_ui()
        self._load_pair(patients[0])

        self.bind('<Left>',  lambda e: self._step(-1))
        self.bind('<Right>', lambda e: self._step(+1))
        self.bind('<Up>',    lambda e: self._step(+5))
        self.bind('<Down>',  lambda e: self._step(-5))

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        sidebar = ttk.Frame(self, width=215)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
        sidebar.pack_propagate(False)

        ttk.Label(sidebar, text="Patient",
                  font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        names = [p.name for p in self._patients]
        self._pair_var = tk.StringVar(value=names[0])
        cb = ttk.Combobox(sidebar, textvariable=self._pair_var,
                          values=names, state='readonly', width=27)
        cb.pack(fill=tk.X, pady=(0, 6))
        cb.bind('<<ComboboxSelected>>',
                lambda e: self._load_pair(
                    next(p for p in self._patients if p.name == self._pair_var.get())))

        ttk.Separator(sidebar, orient='horizontal').pack(fill=tk.X, pady=4)
        ttk.Label(sidebar, text="Segmentation labels",
                  font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self._label_vars = {}
        for ch, (name, _) in CHANNEL_INFO.items():
            v = tk.BooleanVar(value=(ch in DEFAULT_ACTIVE))
            ttk.Checkbutton(sidebar, text=name, variable=v,
                            command=self._draw).pack(anchor=tk.W)
            self._label_vars[ch] = v

        ttk.Separator(sidebar, orient='horizontal').pack(fill=tk.X, pady=4)
        ttk.Label(sidebar, text="Overlay alpha").pack(anchor=tk.W)
        self._alpha_var = tk.DoubleVar(value=0.65)
        ttk.Scale(sidebar, variable=self._alpha_var, from_=0.0, to=1.0,
                  orient=tk.HORIZONTAL,
                  command=lambda _: self._draw()).pack(fill=tk.X)

        ttk.Separator(sidebar, orient='horizontal').pack(fill=tk.X, pady=4)
        self._status_var = tk.StringVar(value="")
        ttk.Label(sidebar, textvariable=self._status_var,
                  wraplength=205, foreground='gray',
                  justify=tk.LEFT).pack(anchor=tk.W)
        ttk.Label(sidebar,
                  text="← → : ±1 slice\n↑ ↓  : ±5 slices",
                  foreground='gray',
                  font=("Segoe UI", 8)).pack(anchor=tk.W, pady=(8, 0))

        # Canvas + slice slider
        main = ttk.Frame(self)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._fig, self._axes = plt.subplots(2, 3, figsize=(14, 8))
        self._fig.patch.set_facecolor('#111111')
        for ax in self._axes.flat:
            ax.axis('off')
            ax.set_facecolor('#111111')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        self._canvas = FigureCanvasTkAgg(self._fig, master=main)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        bot = ttk.Frame(main)
        bot.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(bot, text="DESS slice").pack(side=tk.LEFT)
        self._slice_var = tk.IntVar(value=80)
        self._slider = ttk.Scale(bot, variable=self._slice_var,
                                 from_=0, to=159, orient=tk.HORIZONTAL,
                                 command=lambda _: self._draw())
        self._slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self._slice_lbl = ttk.Label(bot, text="80 / 159", width=9)
        self._slice_lbl.pack(side=tk.LEFT)

    # ── Loading ───────────────────────────────────────────────────────────────
    def _load_pair(self, patient_dir: Path):
        self._status_var.set(f"Loading {patient_dir.name}…")
        self.update_idletasks()
        try:
            data = load_pair(patient_dir)
            self._data = data
            D = data['D']
            self._slider.configure(to=D - 1)
            self._slice_var.set(D // 2)
            self._status_var.set(
                f"{data['name']}\n"
                f"DESS z = {data['dess_z_sp']:.2f} mm\n"
                f"PD   z = {data['pd_z_sp']:.2f} mm"
            )
            self._draw()
        except Exception as e:
            self._status_var.set(f"Error:\n{e}")
            messagebox.showerror("Load error", str(e))

    # ── Keyboard ──────────────────────────────────────────────────────────────
    def _step(self, delta):
        if self._data is None:
            return
        val = int(np.clip(self._slice_var.get() + delta, 0, self._data['D'] - 1))
        self._slice_var.set(val)
        self._draw()

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _draw(self):
        if self._data is None:
            return
        d      = self._data
        sl     = int(self._slice_var.get())
        alpha  = float(self._alpha_var.get())
        active = [ch for ch, v in self._label_vars.items() if v.get()]

        D, H, W = d['D'], d['H'], d['W']
        D_pd   = d['pd_mr_native'].shape[0]
        pd_idx = int(np.clip(round(sl * d['dess_z_sp'] / d['pd_z_sp']), 0, D_pd - 1))

        ds = d['dess_mr'][sl, :H, :W]
        dm = d['dess_mask'][:, sl, :H, :W]
        pm = d['pd_mask'][:, sl, :H, :W]
        pr = d['pd_mr_reg'][sl, :H, :W]
        pn = d['pd_mr_native'][pd_idx]

        # Overlap diff panel
        bg   = normalize(ds)
        diff = np.stack([bg * 0.4] * 3, axis=-1)
        for ch in active:
            if ch >= dm.shape[0]:
                continue
            df = dm[ch] > 0
            pf = pm[ch] > 0
            diff[df & pf]  = [0.1, 0.8, 0.2]
            diff[df & ~pf] = [0.9, 0.15, 0.1]
            diff[~df & pf] = [0.1, 0.3, 0.9]

        panels = [
            (normalize(ds),                  "DESS MR"),
            (overlay(ds, dm, active, alpha),  "DESS + DESS seg"),
            (overlay(ds, pm, active, alpha),  "DESS + PD reg seg"),
            (normalize(pn),                  f"PD native  (sl {pd_idx})"),
            (overlay(pr, pm, active, alpha),  "PD MR reg + PD seg"),
            (np.clip(diff, 0, 1),            "Overlap   green=agree  red=DESS  blue=PD"),
        ]

        for ax, (img, title) in zip(self._axes.flat, panels):
            ax.clear()
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None,
                      interpolation='bilinear', aspect='auto')
            ax.set_title(title, color='#dddddd', fontsize=8, pad=2)
            ax.axis('off')
            ax.set_facecolor('#111111')

        handles = [Patch(facecolor=hex_to_rgb(CHANNEL_INFO[ch][1]),
                         label=CHANNEL_INFO[ch][0]) for ch in active]
        self._fig.suptitle(
            f"{d['name']}   slice {sl}/{D-1}  (z = {sl * d['dess_z_sp']:.1f} mm)",
            color='white', fontsize=10)
        self._fig.patch.set_facecolor('#111111')
        if handles:
            self._fig.legend(handles=handles, loc='lower center',
                             ncol=min(len(active), 8), fontsize=7,
                             frameon=False, labelcolor='white',
                             bbox_to_anchor=(0.5, 0.0))

        self._slice_lbl.config(text=f"{sl} / {D - 1}")
        self._canvas.draw()


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ViewerApp()
    app.mainloop()
