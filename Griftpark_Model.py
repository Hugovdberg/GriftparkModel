# Importing packages
from pathlib import Path  # Filepath maniputlation

import fiona
import flopy  # MODFLOW interface for Python
import gstools  # GeoStat-tools
import matplotlib.path as mpth
import matplotlib.pyplot as plt  # Plotting
import numpy as np  # Numeric array manipulation
import rasterio.features as rf
import rasterio.transform as rtf
import scipy.ndimage as si  # Numpy array manipulation
import shapefile as shp  # ESRI Shapefile manipulation - pip install pyshp

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
            np.linspace(u, l, n_sublayers + 1)[1:]
            for u, l in zip(layer_boundaries[:-1], layer_boundaries[1:])
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
        proj4_str="EPSG:28992",
    )

    # # Get node coordinates and volumes for random field generation.
    # node_y, node_x, node_z = dis.get_node_coordinates()
    # node_vol = dis.get_cell_volumes()
    # node_x, node_y = np.meshgrid(node_x, node_y)
    # node_x = np.tile(node_x, (len(node_z), 1, 1))
    # node_y = np.tile(node_y, (len(node_z), 1, 1))
    # node_x.shape, node_y.shape, node_z.shape, node_vol.shape
    # node_x_world = model_extent["X"][0] + node_x
    # node_y_world = model_extent["Y"][0] + node_y

    # # Get locations of isohead contours on the model grid to fix constant heads
    # east_l2 = node_y_world[0] < (
    #     458170 + (node_x_world[0] - 136613) * (455529 - 458170) / (137238 - 136613)
    # )
    # idx_east_l2 = (
    #     east_l2 & (~si.binary_erosion(east_l2, structure=np.ones((1, 3)))).astype(int)
    # )[:, 1:].argmax(axis=1) + 1

    # west_l2 = node_y_world[0] < (
    #     455522 + (node_x_world[0] - 137947) * (458167 - 455522) / (137960 - 137947)
    # )
    # idx_west_l2 = (
    #     west_l2 & (~si.binary_erosion(west_l2, structure=np.ones((1, 3)))).astype(int)
    # )[:, 1:].argmax(axis=1) + 1

    # Setup the IBOUND and STRT arrays
    ibound = np.ones(dis.botm.shape, dtype=np.int)
    init_head = np.zeros(dis.botm.shape, dtype=np.float)

    ibound[:5] = ibounds[0]
    # ibound[:5][const_heads[0]] = -1
    ibound[5:] = ibounds[1]
    # ibound[5:][const_heads[1]] = -1
    # # Disable everything outside the isohead contours
    # ibound[5:, east_l2] = 0
    # ibound[5:, west_l2] = 0
    # # Set the isohead contours themselves as constant head lines
    for r, c in zip(*np.where(const_heads[0])):
        ibound[:5, r, c] = -1
        if y[r, c] < 456900:
            init_head[:5, r, c] = 0.25
    for r, c in zip(*np.where(const_heads[1])):
        ibound[5:, r, c] = -1
        if x[r, c] < 137500:
            init_head[5:, r, c] = -50.25
    # for r, c in enumerate(idx_west_l2):
    #     ibound[5:, r, c] = -1
    #     init_head[5:, r, c] = 0

    # Setup the basic flow (BAS package)
    bas = flopy.modflow.ModflowBas(model=mf, ibound=ibound, strt=init_head)

    # Generate random fields for hydraulic conductivity (data now based purely
    # on random mean and variance)
    # model_layer_1 = gstools.Gaussian(dim=3, var=1, len_scale=[50, 50, 5])
    # model_layer_2 = gstools.Gaussian(dim=3, var=2, len_scale=[40, 40, 3])

    # srf_layer_1 = gstools.SRF(model=model_layer_1, mean=25, upscaling="coarse_graining")
    # srf_layer_2 = gstools.SRF(model=model_layer_2, mean=35, upscaling="coarse_graining")

    # field_layer_1 = srf_layer_1(
    #     pos=(node_x[:5].flatten(), node_y[:5].flatten(), node_z[:5].flatten()),
    #     seed=seed(),
    #     point_volumes=node_vol[:5].flatten(),
    # )
    # field_layer_1 = np.reshape(field_layer_1, node_x[:5].shape)
    # field_layer_2 = srf_layer_2(
    #     pos=(node_x[5:].flatten(), node_y[5:].flatten(), node_z[5:].flatten()),
    #     seed=seed(),
    #     point_volumes=node_vol[5:].flatten(),
    # )
    # field_layer_2 = np.reshape(field_layer_1, node_x[5:].shape)
    horizontal_conductivity = 40
    # horizontal_conductivity[:5] = field_layer_1
    # horizontal_conductivity[5:] = field_layer_2
    # Set vertical conductivity to one tenth of the horizontal conductivity in
    # every node.
    vertical_conductivity = horizontal_conductivity / 10

    # Setup the Layer Property Flow properties (LPF package)
    lpf = flopy.modflow.ModflowLpf(
        model=mf, hk=horizontal_conductivity, vka=vertical_conductivity
    )

    hfb = flopy.modflow.ModflowHfb(model=mf, nphfb=0, nacthfb=0, hfb_data=hfb_data)

    wel = flopy.modflow.ModflowWel(model=mf, stress_period_data={0: welldata})

    # Setup recharge (RCH package) using a constant net recharge of 200mm/yr
    recharge = np.ones(dis.botm.shape[1:], dtype=np.float) * 0.200 / 365
    rch = flopy.modflow.ModflowRch(model=mf, rech=recharge)

    # Setup the output control (OC package), saving heads and budgets for all
    # timesteps in all stress periods
    oc_spd = {
        (p, s): ["SAVE HEAD", "SAVE BUDGET"]
        for p, sp in enumerate(stress_periods_)
        for s in range(sp["n_steps"])
    }
    oc = flopy.modflow.ModflowOc(model=mf, stress_period_data=oc_spd)

    # Setup MODFLOW solver (Preconditioned Conjugate-Gradient, PCG package)
    solver = flopy.modflow.ModflowPcg(model=mf)

    # Write MODFLOW input files to disk and run MODFLOW executable
    mf.write_input()
    mf.run_model()

    return mf


# Name and location
modelname = "Griftpark"
model_workspace = Path(modelname)

layer_boundaries = [0, -20, -60]
n_sublayers = 5

x_zones = [136600, 137000, 137090, 137350, 137450, 138000]
ncol_zones = [8, 9, 52, 10, 11]
y_zones = [455600, 456550, 456670, 457170, 457250, 458200]
nrow_zones = [19, 12, 100, 8, 19]

model_extent = np.array(
    [(136600, 455600), (138000, 458200)], dtype=[("X", np.float), ("Y", np.float)]
)

# nrow, ncol = 100, 100
# x_vertices = np.linspace(*model_extent["X"], ncol + 1)
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
# y_vertices = np.linspace(*model_extent["Y"][::-1], nrow + 1)
row_height = -np.diff(y_vertices)
y = y_vertices[1:] + row_height / 2
x, y = np.meshgrid(x, y)

stress_periods = [
    StressPeriod(period_length=1, n_steps=1, step_multiplier=1, steady_state=True)
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
    inside_aq = aq_path.contains_points(np.vstack((x.flatten(), y.flatten())).T)
    inside_aq = np.reshape(inside_aq, x.shape)
    ibounds[ia] = inside_aq
    const_heads[ia] = ~inside_aq & ~si.binary_erosion(~inside_aq, border_value=False)
# ibounds


seed = gstools.random.MasterRNG(20190517)


with fiona.open("data/location_wells.shp", "r") as wellshp:
    wells = [f["geometry"]["coordinates"] for f in wellshp]

welldata = flopy.modflow.ModflowWel.get_empty(ncells=len(wells))
for w, well in enumerate(wells):
    welldata[w]["i"] = np.abs(y[:, 0] - well[1]).argmin()
    welldata[w]["j"] = np.abs(x[0, :] - well[0]).argmin()
    welldata[w]["k"] = 0
    welldata[w]["flux"] = -1500

with fiona.open("data/resistive_wall.shp", "r") as shapefile:
    wall = [f["geometry"] for f in shapefile]

wall_path = mpth.Path(wall[0]["coordinates"][0])
inside_wall = wall_path.contains_points(np.vstack((x.flatten(), y.flatten())).T)
mask = np.reshape(inside_wall, x.shape)

wall_mask = mask & ~si.binary_erosion(mask, border_value=False)
wall_mask_row, wall_mask_col = np.where(wall_mask)

hfb_conductance = 1e-7
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

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_aspect(1)
extent = None
# extent = [137000, 137400, 456600, 457400]  # Wall
# extent = [137150, 137300, 456900, 457150]  # Wells
pmv = flopy.plot.PlotMapView(model=mf, ax=ax, layer=5, extent=extent)
pmv.plot_grid(linewidths=1, alpha=0.5)
c = pmv.plot_array(heads, masked_values=[-999.99])
# c = pmv.plot_array(ibounds[1].astype(int) + const_heads[1].astype(int))
plt.colorbar(c)
for contour in contours_h1:
    points = np.array(contour["geometry"]["coordinates"])
    ax.plot(points[:, 0], points[:, 1], "--", color="k")
for contour in contours_h2:
    points = np.array(contour["geometry"]["coordinates"])
    ax.plot(points[:, 0], points[:, 1], "-.", color="k")
pmv.plot_bc("WEL")
# pmv.plot_ibound()
# pmv.plot_bc("HFB6")
# pmv.plot_shapefile("data/aquifers.shp")
pmv.plot_shapefile("data/resistive_wall.shp", alpha=1, facecolor="None")
pmv.plot_shapefile("data/location_wells.shp", radius=1, edgecolor="blue", facecolor="b")

# dis = mf.get_package("DIS")
# x, y, z = dis.get_node_coordinates()
# dis = mf.dis
# dis.get

# flopy.discretization.structuredgrid.StructuredGrid

cells = []
for ix, x0 in enumerate(x_vertices[:-1]):
    x1 = x_vertices[ix + 1]
    for iy, y0 in enumerate(y_vertices[:-1]):
        y1 = y_vertices[iy + 1]
        cells.append(mpth.Path(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])))


with fiona.open("data/location_wells.shp", "r") as wellshp:
    wells = [
        (
            # f,
            mf.dis.get_rc_from_node_coordinates(
                *mf.modelgrid.get_local_coords(*f["geometry"]["coordinates"])
            ),
        )
        for f in wellshp
    ]
    wellc = [f["geometry"]["coordinates"] for f in wellshp]
