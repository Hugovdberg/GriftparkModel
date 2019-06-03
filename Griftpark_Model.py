# Importing packages
from pathlib import Path  # Filepath maniputlation

import flopy
import fiona
import gstools  # GeoStat-tools
import matplotlib.path as mpth
import matplotlib.pyplot as plt
import numpy as np  # Numeric array manipulation
import pandas as pd
import scipy.ndimage as si  # Numpy array manipulation

# Local utility functions and definitions
from plot_tools import plot_model, xs_lines
from run_tools import run_flow_model, run_transport
from utils import StressPeriod

# Name and location
modelname = "Griftpark"
model_workspace = Path(modelname)
model_output_dir = Path("output")
if not model_output_dir.exists():
    model_output_dir.mkdir()

layer_boundaries = [2.5, -2.5, -7.5, -32.5, -50, -60, -100]
n_sublayers = [1, 2, 10, 7, 4, 5]
wvp = np.repeat([1, 1, 1, 1, 2, 2], n_sublayers)
kh_park = np.repeat([10, 20, 80, 40, 1, 50], n_sublayers)
kh_buiten = np.repeat([1, 20, 80, 40, 1, 50], n_sublayers)
kv_park = np.repeat([1, 2, 8, 4, 10 / 100, 5], n_sublayers)
kv_buiten = np.repeat([0.1, 2, 8, 4, 10 / 2000, 5], n_sublayers)

x_zones = [136600, 137000, 137090, 137350, 137450, 138000]
ncol_zones = [8, 9, 52, 10, 11]
y_zones = [455600, 456550, 456670, 457170, 457250, 458200]
nrow_zones = [19, 12, 100, 8, 19]

model_extent = np.array(
    [(min(x_zones), min(y_zones)), (max(x_zones), max(y_zones))],
    dtype=[("X", np.float), ("Y", np.float)],
)

x_vertices = np.unique(
    np.hstack(
        [
            np.linspace(x0, x1, nc + 1)
            for x0, x1, nc in zip(x_zones[:-1], x_zones[1:], ncol_zones)
        ]
    )
)
col_width = np.diff(x_vertices)
x = x_vertices[:-1] + col_width / 2
y_vertices = np.unique(
    np.hstack(
        [
            np.linspace(y0, y1, nr + 1)
            for y0, y1, nr in zip(y_zones[:-1], y_zones[1:], nrow_zones)
        ]
    )
)[::-1]
row_height = -np.diff(y_vertices)
y = y_vertices[1:] + row_height / 2
x, y = np.meshgrid(x, y)

nov = pd.read_csv("data/neerslagoverschot.csv", index_col=0)
startjaar = 2019
nov.loc[startjaar, "NOV"] = nov.loc[nov.index <= startjaar, "NOV"].mean()
nov = nov.loc[startjaar:]
# stress_periods = [
#     StressPeriod(
#         period_length=(pd.Timestamp(f"{j+1}0401") - pd.Timestamp(f"{j}0401")).days,
#         n_steps=(j == startjaar) + 12 * (j > startjaar),
#         step_multiplier=1,
#         steady_state=j == nov.index[0],
#     )
#     for j in nov.index
# ]
stress_periods = [
    StressPeriod(
        period_length=30 * 365, n_steps=1, step_multiplier=1, steady_state=True
    )
]

with fiona.open("data/aquifers.shp") as src:
    aquifers = list(src)

ibounds = np.ones((len(aquifers), *x.shape), dtype=np.bool)
const_heads = np.ones((len(aquifers), *x.shape), dtype=np.bool)
for ia, aq in enumerate(aquifers):
    aq_path = mpth.Path(aq["geometry"]["coordinates"][0])
    inside_aq = aq_path.contains_points(np.vstack((x.ravel(), y.ravel())).T)
    inside_aq = np.reshape(inside_aq, x.shape)
    ibounds[ia] = inside_aq
    const_heads[ia] = ~inside_aq & ~si.binary_erosion(~inside_aq, border_value=False)

seed = gstools.random.MasterRNG(20190517)

with fiona.open("data/location_wells.shp", "r") as wellshp:
    wells = [f["geometry"]["coordinates"] for f in wellshp]

with fiona.open("data/resistive_wall.shp", "r") as shapefile:
    wall = [f["geometry"] for f in shapefile]

wall_path = mpth.Path(wall[0]["coordinates"][0])
inside_wall = wall_path.contains_points(np.vstack((x.ravel(), y.ravel())).T)
inside_wall = np.reshape(inside_wall, x.shape)

wall_mask = inside_wall & ~si.binary_erosion(inside_wall, border_value=False)
outside_wall_mask = ~inside_wall & ~si.binary_erosion(~inside_wall, border_value=True)
wall_mask_row, wall_mask_col = np.where(wall_mask)

hfb_conductance = 1 / 1000
hfb_data = []
for r, c in zip(wall_mask_row, wall_mask_col):
    if outside_wall_mask[r, c - 1]:
        # West boundary
        for l in range(sum(wvp == 1)):
            hfb_data.append([l, r, c - 1, r, c, hfb_conductance])
    if outside_wall_mask[r, c + 1]:
        # East boundary
        for l in range(sum(wvp == 1)):
            hfb_data.append([l, r, c, r, c + 1, hfb_conductance])
    if outside_wall_mask[r - 1, c]:
        # North boundary
        for l in range(sum(wvp == 1)):
            hfb_data.append([l, r - 1, c, r, c, hfb_conductance])
    if outside_wall_mask[r + 1, c]:
        # South boundary
        for l in range(sum(wvp == 1)):
            hfb_data.append([l, r, c, r + 1, c, hfb_conductance])

model_config = {
    "layers": {
        "boundaries": layer_boundaries,
        "n_sublayers": n_sublayers,
        "aquifer": wvp,
    },
    "grid": {
        "extent": model_extent,
        "col_width": col_width,
        "row_height": row_height,
        "x": x,
        "y": y,
    },
    "time": {"startjaar": startjaar, "stress_periods": stress_periods},
    "flow": {
        "ibound": ibounds,
        "const_heads": const_heads,
        "generator": {"active": False, "seed": seed},
        "k_h": {"regional": kh_buiten, "park": kh_park},
        "k_v": {"regional": kv_buiten, "park": kv_park},
    },
    "wall": {"inside_wall": inside_wall, "hfb_data": hfb_data},
    "wells": {"locations": wells, "discharge": -10 * 24 / len(wells)},
    "recharge": nov,
}

mf = run_flow_model(modelname, model_workspace, model_config)

init_conc_PAH = np.zeros((mf.dis.nlay, mf.dis.nrow, mf.dis.ncol), dtype=np.float)
init_conc_cyanide = np.zeros((mf.dis.nlay, mf.dis.nrow, mf.dis.ncol), dtype=np.float)

with fiona.open("data/cyanide/cyanide_0_5.shp") as src:
    contours = [mpth.Path(f["geometry"]["coordinates"][0]) for f in src]
init_conc_cyanide[0, :, :] = np.reshape(
    np.sum(
        np.vstack(
            [
                contour.contains_points(np.vstack((x.ravel(), y.ravel())).T)
                for contour in contours
            ]
        ),
        axis=0,
    ),
    x.shape,
)

with fiona.open("data/PAK/PAK_0_5.shp") as src:
    contours = [mpth.Path(f["geometry"]["coordinates"][0]) for f in src]
init_conc_PAH[0, :, :] = np.reshape(
    np.sum(
        np.vstack(
            [
                contour.contains_points(np.vstack((x.ravel(), y.ravel())).T)
                for contour in contours
            ]
        ),
        axis=0,
    ),
    x.shape,
)
with fiona.open("data/PAK/PAK_5_15.shp") as src:
    for contour in src:
        # print(contour["properties"]["diepte"])
        contour_path = mpth.Path(contour["geometry"]["coordinates"][0])
        in_contour = contour_path.contains_points(np.vstack((x.ravel(), y.ravel())).T)
        in_contour = np.reshape(in_contour, x.shape)
        icr, icc = np.where(in_contour)
        # print(layer_boundaries[0] - contour["properties"]["diepte"])
        for r, c in zip(icr, icc):
            l_top = mf.dis.get_layer(r, c, -2.51)
            l_bottom = mf.dis.get_layer(
                r, c, layer_boundaries[0] - contour["properties"]["diepte"] + 0.01
            )
            for l in range(l_top, l_bottom + 1):
                init_conc_PAH[l, r, c] = 1
with fiona.open("data/PAK/PAK_15_23.shp") as src:
    for contour in src:
        contour_path = mpth.Path(contour["geometry"]["coordinates"][0])
        in_contour = contour_path.contains_points(np.vstack((x.ravel(), y.ravel())).T)
        in_contour = np.reshape(in_contour, x.shape)
        icr, icc = np.where(in_contour)
        for r, c in zip(icr, icc):
            l_top = mf.dis.get_layer(r, c, -12.51)
            l_bottom = mf.dis.get_layer(r, c, -20.49)
            for l in range(l_top, l_bottom + 1):
                init_conc_PAH[l, r, c] = 1

model_config["transport"] = {
    "initial_conc": {"cyanide": init_conc_cyanide, "PAH": init_conc_PAH}
}

mt = run_transport(mf, model_config)

plot_model(mf, mt, model_output_dir)

hds_file = flopy.utils.HeadFile(model_workspace / f"{modelname}.hds")
heads = hds_file.get_data()

cbc_file = flopy.utils.CellBudgetFile(model_workspace / f"{modelname}.cbc")
frf = cbc_file.get_data(text="FLOW RIGHT FACE")[0]
fff = cbc_file.get_data(text="FLOW FRONT FACE")[0]
flf = cbc_file.get_data(text="FLOW LOWER FACE")[0]
xs_lines_full = {
    "A-A'": {
        "line": {
            "column": mf.dis.get_rc_from_node_coordinates(
                *xs_lines["A-A'"][0], local=False
            )[1]
        },
        "extent": (1000, 1600, -100, 2.5),
    },
    "B-B'": {
        "line": {
            "row": mf.dis.get_rc_from_node_coordinates(
                *xs_lines["B-B'"][0], local=False
            )[0]
        },
        "extent": (400, 780, -100, 2.5),
    },
    "C-C'": {
        "line": {
            "row": mf.dis.get_rc_from_node_coordinates(
                *xs_lines["C-C'"][0], local=False
            )[0]
        },
        "extent": (450, 780, -100, 2.5),
    },
}
for line_title, xs_line in xs_lines_full.items():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_title(f"Head and flow vectors on transect {line_title}")
    pxs = flopy.plot.PlotCrossSection(model=mf, ax=ax, **xs_line)
    pxs.plot_grid(linewidths=0.5, alpha=0.5)
    c = pxs.plot_array(heads, head=heads, masked_values=[-999.99])
    plt.colorbar(c, ax=ax)
    pxs.plot_bc("WEL")
    pxs.plot_discharge(
        frf=frf,
        fff=fff,
        flf=flf,
        head=heads,
        color="#ffffff",
        normalize=True,
        hstep=1,
        kstep=1,
    )
    fig.tight_layout()
    fig.savefig(model_output_dir / f"crosssection_flows_{line_title[0]}.png")
    plt.close(fig)
