import flopy
import gstools
import numpy as np

import config
from utils import stress_period_dtype


def run_flow_model(modelname, model_workspace, model_config):
    model_extent = model_config["grid"]["extent"]
    col_width = model_config["grid"]["col_width"]
    row_height = model_config["grid"]["row_height"]
    x = model_config["grid"]["x"]
    y = model_config["grid"]["y"]
    layer_boundaries = model_config["layers"]["boundaries"]
    n_sublayers = model_config["layers"]["n_sublayers"]
    wvp = model_config["layers"]["aquifer"]
    startjaar = model_config["time"]["startjaar"]
    stress_periods = model_config["time"]["stress_periods"]
    ibounds = model_config["flow"]["ibound"]
    const_heads = model_config["flow"]["const_heads"]
    seed = model_config["flow"]["generator"]["seed"]
    kh_buiten = model_config["flow"]["k_h"]["regional"]
    kh_park = model_config["flow"]["k_h"]["park"]
    kv_buiten = model_config["flow"]["k_v"]["regional"]
    kv_park = model_config["flow"]["k_v"]["park"]
    inside_wall = model_config["wall"]["inside_wall"]
    hfb_data = model_config["wall"]["hfb_data"]
    wells = model_config["wells"]["locations"]
    nov = model_config["recharge"]
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
    flopy.modflow.ModflowBas(model=mf, ibound=ibound, strt=init_head)

    if model_config["flow"]["generator"]["active"]:
        horizontal_conductivity = np.empty(
            (dis.nlay, dis.nrow, dis.ncol), dtype=np.float
        )
        # Generate random fields for hydraulic conductivity (data now based purely
        # on random mean and variance)
        # Get node coordinates and volumes for random field generation.
        _, _, node_z = dis.get_node_coordinates()
        node_vol = dis.get_cell_volumes()
        # node_x = np.tile(x, (len(node_z), 1, 1))
        # node_y = np.tile(y, (len(node_z), 1, 1))

        layers = np.repeat(range(len(n_sublayers)), n_sublayers)
        unique_layers = np.unique(layers, return_index=True)
        srf = [
            gstools.SRF(
                model=gstools.Gaussian(
                    dim=3,
                    var=model_config["flow"]["generator"]["variances"][l],
                    len_scale=model_config["flow"]["generator"]["cor_lengths"][l],
                ),
                mean=np.log10(model_config["flow"]["k_h"]["regional"][li]),
                upscaling="coarse_graining",
            )
            for l, li in zip(*unique_layers)
        ]
        for l, ln in enumerate(layers):
            horizontal_conductivity[l, :, :] = 10 ** np.reshape(
                srf[ln](
                    pos=(x.ravel(), y.ravel(), node_z[l].ravel()),
                    seed=seed(),
                    point_volumes=node_vol[l].ravel(),
                ),
                (dis.nrow, dis.ncol),
            )
        # Set vertical conductivity to one tenth of the horizontal conductivity in
        # every node.
        vertical_conductivity = horizontal_conductivity / 10
    else:
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
    flopy.modflow.ModflowLpf(
        model=mf, hk=horizontal_conductivity, vka=vertical_conductivity
    )

    flopy.modflow.ModflowHfb(model=mf, nphfb=0, nacthfb=0, hfb_data=hfb_data)

    welldata = flopy.modflow.ModflowWel.get_empty(ncells=len(wells) * dis.nlay)
    wellflux = model_config["wells"]["discharge"]
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
    first_year = np.where(nov.index == startjaar)[0][0]
    flopy.modflow.ModflowWel(model=mf, stress_period_data={first_year: welldata})

    # Setup recharge (RCH package) using a time dependent recharge rate.
    # Add recharge only to active cells of the top aquifer
    recharge = {j: ibounds[0] * n / 1000 / 365 for j, n in enumerate(nov["NOV"])}
    flopy.modflow.ModflowRch(model=mf, rech=recharge)

    # Setup the output control (OC package), saving heads and budgets for all
    # timesteps in all stress periods
    oc_spd = {
        (p, sp["n_steps"] - 1): ["SAVE HEAD", "SAVE BUDGET"]
        for p, sp in enumerate(stress_periods_)
    }
    oc = flopy.modflow.ModflowOc(model=mf, stress_period_data=oc_spd)
    oc.reset_budgetunit()

    # Setup link to MT3DMS
    flopy.modflow.ModflowLmt(model=mf)

    # Setup MODFLOW solver (Preconditioned Conjugate-Gradient, PCG package)
    flopy.modflow.ModflowGmg(model=mf)

    # Write MODFLOW input files to disk and run MODFLOW executable
    mf.write_input()
    success, _ = mf.run_model()
    if not success:
        raise Exception("MODFLOW failed")
    return mf


def run_transport(mf, model_config):
    modelname = f"{mf.name}_mt"
    model_workspace = mf.model_ws

    init_conc_cyanide = model_config["transport"]["initial_conc"]["cyanide"]
    init_conc_PAH = model_config["transport"]["initial_conc"]["PAH"]

    mt = flopy.mt3d.Mt3dms(
        modelname=modelname,
        modflowmodel=mf,
        version="mt3d-usgs",
        exe_name=config.mtusgsexe,
        model_ws=model_workspace,
    )

    icbund = np.abs(
        mf.bas6.ibound.array
    )  # Set only active flow cells to active concentration
    Kd_cyanide = 1e-9
    Kd_PAH = (
        10
        ** (
            np.array([3.3, 4.4, 5.2, 5.0, 5.0, 5.6, 5.2, 5.8, 5.9, 5.8, 4.6, 5.1, 4.7])
        ).mean()
        / 1e3
        # / 1e6
    )
    n_print_times = 15
    flopy.mt3d.Mt3dBtn(
        model=mt,
        prsity=0.30,
        ncomp=2,
        mcomp=2,
        icbund=icbund,
        species_names=["Cyanide", "PAH"],
        sconc=init_conc_cyanide,
        sconc2=init_conc_PAH,
        nprs=n_print_times,
        timprs=np.linspace(0, np.sum(mf.dis.perlen.array), num=n_print_times),
    )
    flopy.mt3d.Mt3dAdv(model=mt, mixelm=3, mxpart=8_000_000)
    flopy.mt3d.Mt3dDsp(model=mt, al=0.1, dmcoef=0, dmcoef2=0, trpt=0.1, trpv=0.01)
    flopy.mt3d.Mt3dRct(
        model=mt,
        isothm=1,
        rhob=1800,
        igetsc=0,
        srconc=0,
        srconc2=0,
        sp1=Kd_cyanide,
        sp12=Kd_PAH,
        sp2=0,
        sp22=0,
        rc1=0,
        rc12=0,
        rc2=0,
        rc22=0,
    )
    flopy.mt3d.Mt3dSsm(model=mt, crch=0, crch2=0)

    flopy.mt3d.Mt3dGcg(model=mt)

    for f in model_workspace.glob("MT3D*.UCN"):
        f.unlink()
    for f in model_workspace.glob("MT3D*.MAS"):
        f.unlink()

    mt.write_input()
    mt.run_model()
    return mt
