"""
Add a per-vertex `compartment` label to the unified tibia template.

  0 = non-articular (subch_prob < threshold)
  1 = medial plateau   (subchondral, KMeans cluster on ML axis)
  2 = lateral plateau  (subchondral, other KMeans cluster)

Usage:
  python add_tibia_compartment.py [--vtk path] [--subch_thresh 0.5]

Modifies the input VTK in place by default; use --out to save elsewhere.

For RIGHT knees: medial side has lower X (axis-2 = ML). The script auto-detects
which cluster is medial by checking which one has the smaller mean-X centroid.
"""
import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from sklearn.cluster import KMeans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vtk",
                   default="MOAKS_seg/standardization/tibia_template_full_with_subch.vtk")
    p.add_argument("--out", default=None,
                   help="Save to this path (default: overwrite input)")
    p.add_argument("--subch_thresh", type=float, default=0.5)
    p.add_argument("--ml_axis", type=int, default=2,
                   help="Index of the ML axis (default 2 = X in mm coords)")
    p.add_argument("--side", choices=["RIGHT", "LEFT"], default="RIGHT",
                   help="For RIGHT knees, medial = smaller-X cluster.")
    args = p.parse_args()

    mesh = pv.read(str(Path(args.vtk)))
    print(f"loaded {args.vtk}: {mesh.n_points} verts")
    print(f"  fields: {list(mesh.point_data.keys())}")

    if "subch_prob" not in mesh.point_data:
        print("ERROR: subch_prob field missing"); return

    subch = mesh.point_data["subch_prob"] >= args.subch_thresh
    n_subch = int(subch.sum())
    print(f"subch verts (>= {args.subch_thresh}): {n_subch}")
    if n_subch < 10:
        print("Too few subch verts for clustering"); return

    # KMeans on the ML coordinate of subch verts
    subch_idx = np.where(subch)[0]
    ml_coords = mesh.points[subch_idx, args.ml_axis].reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(ml_coords)
    cluster_centers = km.cluster_centers_.flatten()
    cluster_labels = km.labels_  # 0 or 1 per subch vert

    # Determine which cluster is medial:
    # RIGHT knee: medial side = smaller-X cluster
    # LEFT knee:  medial side = larger-X cluster
    if args.side == "RIGHT":
        medial_cluster = int(np.argmin(cluster_centers))
    else:
        medial_cluster = int(np.argmax(cluster_centers))
    lateral_cluster = 1 - medial_cluster
    print(f"cluster centers (ML mm): {cluster_centers}")
    print(f"medial cluster idx={medial_cluster} (center {cluster_centers[medial_cluster]:.1f})")
    print(f"lateral cluster idx={lateral_cluster} (center {cluster_centers[lateral_cluster]:.1f})")

    # Build per-vertex compartment label
    compartment = np.zeros(mesh.n_points, dtype=np.uint8)  # 0 = non-articular
    medial_mask = cluster_labels == medial_cluster
    lateral_mask = cluster_labels == lateral_cluster
    compartment[subch_idx[medial_mask]] = 1
    compartment[subch_idx[lateral_mask]] = 2

    n_med = int((compartment == 1).sum())
    n_lat = int((compartment == 2).sum())
    print(f"compartment counts: medial={n_med}, lateral={n_lat}, "
          f"non-articular={int((compartment == 0).sum())}")

    mesh["compartment"] = compartment
    out_path = Path(args.out) if args.out else Path(args.vtk)
    mesh.save(str(out_path))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
