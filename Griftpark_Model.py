# Importing packages
from pathlib import Path  # Filepath maniputlation

import fiona
import flopy  # MODFLOW interface for Python
import gstools  # GeoStat-tools
import matplotlib.path as mpth
import matplotlib.pyplot as plt  # Plotting
import numpy as np  # Numeric array manipulation
import scipy.ndimage as si  # Numpy array manipulation
import pandas as pd

import config
from utils import (
    StressPeriod,  # Local utility functions and definitions
    stress_period_dtype,
)


def run_flow_model():
    # Extract tops and bottoms from the layer boundaries (defined outside
    # function), and interpolate layers to sublayers for random field generation.
    top = layer_boundaries[0]
    bottoms = np.hstack(
        [
            np.linspace(u, l, n + 1)[1:]
            for u, l, n in zip(layer_boundaries[:-1], layer_boundaries[1:], n_sublayers)
        ]
    )

    # Convert stress period list (defined outside function) to recarray for easy
    # data extraction.
    stress_periods_ = np.array(stress_periods, dtype=stress_period_dtype)

    # Extract number of layers, columns, rows and stress periods
    n_layers = bottoms.size
    n_cols = col_width.size
    n_rows = row_height.size
    n_stress_periods = stress_periods_.size

    # Setup basic model, including reference location for spatial projections
    mf = flopy.modflow.Modflow(
        modelname=modelname,
        version="mf2005",
        exe_name=config.mfexe,
        model_ws=model_workspace,
        xul=model_extent["X"][0],
        yul=model_extent["Y"][1],
        proj4_str="EPSG:28992",
    )
    # Setup numeric discretisation (DIS package) using data collected above
    dis = flopy.modflow.ModflowDis(
        model=mf,
        nlay=n_layers,
        nrow=n_rows,
        ncol=n_cols,
        nper=n_stress_periods,
        itmuni=4,
        lenuni=2,
        delr=col_width,
        delc=row_height,
        top=top,
        botm=bottoms,
        perlen=stress_periods_["period_length"],
        nstp=stress_periods_["n_steps"],
        tsmult=stress_periods_["step_multiplier"],
        steady=stress_periods_["steady_state"],
        xul=model_extent["X"][0],
        yul=model_extent["Y"][1],
        proj4_str="+init=epsg:28992",
        start_datetime=f"4/1/{startjaar}",
    )

    # # Get node coordinates and volumes for random field generation.
    _, _, node_z = dis.get_node_coordinates()
    node_vol = dis.get_cell_volumes()
    node_x = np.tile(x, (len(node_z), 1, 1))
    node_y = np.tile(y, (len(node_z), 1, 1))

    # Setup the IBOUND and STRT arrays
    ibound = np.ones((dis.nlay, dis.nrow, dis.ncol), dtype=np.int)
    init_head = np.zeros_like(ibound, dtype=np.float)

    # Broadcast ibounds of first and second aquifer to corresponding layers
    ibound = ibounds[(wvp - 1)].astype(np.int)
    # Set the constant head at the top of the first and bottom of the second
    # aquifer.
    for r, c in zip(*np.where(const_heads[0])):
        ibound[0, r, c] = -1
        if y[r, c] < 456_750:
            init_head[0, r, c] = 0.25
    for r, c in zip(*np.where(const_heads[1])):
        ibound[-1, r, c] = -1
        if x[r, c] < 137_600:
            init_head[-1, r, c] = -0.25

    # Setup the basic flow (BAS package)
    bas = flopy.modflow.ModflowBas(model=mf, ibound=ibound, strt=init_head)

    # Generate random fields for hydraulic conductivity (data now based purely
    # on random mean and variance)
    # model_layer_1 = gstools.Gaussian(dim=3, var=1, len_scale=[50, 50, 5])
    # model_layer_2 = gstools.Gaussian(dim=3, var=2, len_scale=[40, 40, 3])

    # srf_layer_1 = gstools.SRF(model=model_layer_1, mean=25, upscaling="coarse_graining")
    # srf_layer_2 = gstools.SRF(model=model_layer_2, mean=35, upscaling="coarse_graining")

    # field_layer_1 = srf_layer_1(
    #     pos=(node_x[:5].ravel(), node_y[:5].ravel(), node_z[:5].ravel()),
    #     seed=seed(),
    #     point_volumes=node_vol[:5].ravel(),
    # )
    # field_layer_1 = np.reshape(field_layer_1, node_x[:5].shape)
    # field_layer_2 = srf_layer_2(
    #     pos=(node_x[5:].ravel(), node_y[5:].ravel(), node_z[5:].ravel()),
    #     seed=seed(),
    #     point_volumes=node_vol[5:].ravel(),
    # )
    # field_layer_2 = np.reshape(field_layer_1, node_x[5:].shape)
    # horizontal_conductivity[:5] = field_layer_1
    # horizontal_conductivity[5:] = field_layer_2
    # Set vertical conductivity to one tenth of the horizontal conductivity in
    # every node.
    # vertical_conductivity = horizontal_conductivity / 10
    horizontal_conductivity = (
        kh_buiten[:, np.newaxis, np.newaxis] * ~inside_wall[np.newaxis, :]
    )
    horizontal_conductivity += (
        kh_park[:, np.newaxis, np.newaxis] * inside_wall[np.newaxis, :]
    )
    vertical_conductivity = (
        kv_buiten[:, np.newaxis, np.newaxis] * ~inside_wall[np.newaxis, :]
    )
    vertical_conductivity += (
        kv_park[:, np.newaxis, np.newaxis] * inside_wall[np.newaxis, :]
    )

    # Setup the Layer Property Flow properties (LPF package)
    lpf = flopy.modflow.ModflowLpf(
        model=mf, hk=horizontal_conductivity, vka=vertical_conductivity
    )

    hfb = flopy.modflow.ModflowHfb(model=mf, nphfb=0, nacthfb=0, hfb_data=hfb_data)

    welldata = flopy.modflow.ModflowWel.get_empty(ncells=len(wells) * dis.nlay)
    wellflux = -10 * 24 / len(wells)
    w = 0
    z_top = -21
    z_bot = -43
    for well in wells:
        r, c = dis.get_rc_from_node_coordinates(*well, local=False)
        l_top = dis.get_layer(r, c, z_top)
        l_bot = dis.get_layer(r, c, z_bot)
        kD = np.array(
            [
                horizontal_conductivity[l, r, c]
                * (min(dis.botm[l - 1, r, c], z_top) - max(dis.botm[l, r, c], z_bot))
                for l in range(l_top, l_bot + 1)
            ]
        )
        for il, l in enumerate(range(l_top, l_bot + 1)):
            welldata[w]["k"] = l
            welldata[w]["i"] = r
            welldata[w]["j"] = c
            welldata[w]["flux"] = wellflux * kD[il] / kD.sum()
            w += 1
    welldata = welldata[:w]
    first_year = np.where(nov.index == 1990)[0][0]
    wel = flopy.modflow.ModflowWel(model=mf, stress_period_data={first_year: welldata})

    # Setup recharge (RCH package) using a time dependent recharge rate.
    # Add recharge only to active cells of the top aquifer
    recharge = {j: ibounds[0] * n / 1000 / 365 for j, n in enumerate(nov["NOV"])}
    rch = flopy.modflow.ModflowRch(model=mf, rech=recharge)

    # Setup the output control (OC package), saving heads and budgets for all
    # timesteps in all stress periods
    oc_spd = {
        (p, s): ["SAVE HEAD", "SAVE BUDGET"]
        for p, sp in enumerate(stress_periods_)
        for s in range(sp["n_steps"])
    }
    oc = flopy.modflow.ModflowOc(model=mf, stress_period_data=oc_spd)
    oc.reset_budgetunit()

    # Setup MODFLOW solver (Preconditioned Conjugate-Gradient, PCG package)
    solver = flopy.modflow.ModflowPcg(model=mf)

    # Write MODFLOW input files to disk and run MODFLOW executable
    mf.write_input()
    mf.run_model()

    return mf


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
ncol_zones = [8, 9, 26, 10, 11]
y_zones = [455600, 456550, 456670, 457170, 457250, 458200]
nrow_zones = [19, 12, 50, 8, 19]

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
startjaar = 1989
nov.loc[startjaar, "NOV"] = nov.loc[nov.index <= startjaar, "NOV"].mean()
nov = nov.loc[startjaar:]
stress_periods = [
    StressPeriod(
        period_length=(pd.Timestamp(f"{j+1}0401") - pd.Timestamp(f"{j}0401")).days,
        n_steps=(j == startjaar) + 12 * (j > startjaar),
        step_multiplier=1,
        steady_state=j == nov.index[0],
    )
    for j in nov.index
]

with fiona.open("data/h1.shp") as isohead:
    contours_h1 = [f for f in isohead.filter(bbox=tuple(model_extent.view(np.float)))]
with fiona.open("data/h2.shp") as isohead:
    contours_h2 = [f for f in isohead.filter(bbox=tuple(model_extent.view(np.float)))]

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
wall_mask_row, wall_mask_col = np.where(wall_mask)

hfb_conductance = 1 / 1000
hfb_data = []
for r, c in zip(wall_mask_row, wall_mask_col):
    if c == min(wall_mask_col[wall_mask_row == r]):
        # West boundary
        for l in range(5):
            hfb_data.append([l, r, c - 1, r, c, hfb_conductance])
    if c == max(wall_mask_col[wall_mask_row == r]):
        # East boundary
        for l in range(5):
            hfb_data.append([l, r, c, r, c + 1, hfb_conductance])
    if r == min(wall_mask_row[wall_mask_col == c]):
        # North boundary
        for l in range(5):
            hfb_data.append([l, r - 1, c, r, c, hfb_conductance])
    if r == max(wall_mask_row[wall_mask_col == c]):
        # South boundary
        for l in range(5):
            hfb_data.append([l, r - 1, c, r, c, hfb_conductance])


mf = run_flow_model()

hds_file = flopy.utils.HeadFile(model_workspace / f"{modelname}.hds")
heads = hds_file.get_data()
cbc_file = flopy.utils.CellBudgetFile(model_workspace / f"{modelname}.cbc")
frf = cbc_file.get_data(text="FLOW RIGHT FACE")[0]
fff = cbc_file.get_data(text="FLOW FRONT FACE")[0]
flf = cbc_file.get_data(text="FLOW LOWER FACE")[0]


ts = hds_file.get_ts((0, 72, 51))
plt.plot(pd.to_datetime(ts[:, 0], unit="D", origin="1989-04-01").normalize(), ts[:, 1])
plt.close(fig)

fig, ax = plt.subplots(figsize=(16, 16))
ilay = 0
ax.set_title(f"Layer {ilay+1}")
ax.set_aspect(1)
extent = None
# extent = [137000, 137400, 456600, 457400]  # Wall
# extent = [137150, 137300, 456900, 457150]  # Wells
pmv = flopy.plot.PlotMapView(model=mf, ax=ax, layer=ilay, extent=extent)
pmv.plot_grid(linewidths=1, alpha=0.5)
# c = pmv.plot_array(heads, alpha=1, masked_values=[-999.99])
c = pmv.plot_array(ibounds[0].astype(int))  # + const_heads[1].astype(int))
plt.colorbar(c, ax=ax)
quiver = pmv.plot_discharge(
    frf=frf, fff=fff, flf=flf, head=heads, color="#ffffff", istep=3, jstep=3
)
# for contour in contours_h1:
#     points = np.array(contour["geometry"]["coordinates"])
#     ax.plot(points[:, 0], points[:, 1], "--", color="k")
# for contour in contours_h2:
#     points = np.array(contour["geometry"]["coordinates"])
#     ax.plot(points[:, 0], points[:, 1], "-.", color="k")
pmv.plot_bc("WEL", plotAll=True)
# pmv.plot_ibound()
# pmv.plot_bc("HFB6")
# pmv.plot_shapefile("data/aquifers.shp")
# pmv.plot_shapefile("data/resistive_wall.shp", alpha=1, facecolor="None")
pmv.plot_shapefile("data/location_wells.shp", radius=1, edgecolor="blue", facecolor="b")

fig, ax = plt.subplots(figsize=(16, 8))
# ax.set_aspect(15)
# row=72, column=51
pxs = flopy.plot.PlotCrossSection(
    model=mf, ax=ax, line={"column": 51}, extent=(1100, 1400, -80, -30)
)
pxs.plot_grid(linewidths=0.5, alpha=0.5)
# pxs.plot_bc("WEL")
c = pxs.plot_array(heads, head=heads, vmin=-0.125, vmax=0.025)
plt.colorbar(c, ax=ax)
# pxs.plot_ibound(color_noflow="#aaaaaa", head=heads)
quiver = pxs.plot_discharge(
    frf=frf, fff=fff, flf=flf, head=heads, color="#ffffff", normalize=False
)
fig.tight_layout()
fig.savefig(model_output_dir / "crosssection.png")
