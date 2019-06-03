import numpy as np
import flopy
import matplotlib.pyplot as plt
import fiona
import fiona.crs


def plot_model(mf, mt, model_output_dir):
    modelname = mf.name
    model_workspace = mf.model_ws

    wall_segments = []
    for hfb in mf.hfb6.hfb_data:
        if hfb[0] == 0:
            v1 = mf.modelgrid.get_cell_vertices(hfb[1], hfb[2])
            v2 = mf.modelgrid.get_cell_vertices(hfb[3], hfb[4])
            v1 = [complex(*v) for v in v1]
            v2 = [complex(*v) for v in v2]
            wall_segments.append(
                tuple([*np.array([(v.real, v.imag) for v in np.intersect1d(v1, v2)]).T])
            )

    hds_file = flopy.utils.HeadFile(model_workspace / f"{modelname}.hds")
    # heads = hds_file.get_data((11, 21))
    heads = hds_file.get_data()

    cbc_file = flopy.utils.CellBudgetFile(model_workspace / f"{modelname}.cbc")
    frf = cbc_file.get_data(text="FLOW RIGHT FACE")[0]
    fff = cbc_file.get_data(text="FLOW FRONT FACE")[0]
    flf = cbc_file.get_data(text="FLOW LOWER FACE")[0]

    conc_file1 = flopy.utils.UcnFile(model_workspace / f"MT3D001.UCN")
    conc_file2 = flopy.utils.UcnFile(model_workspace / f"MT3D002.UCN")
    # sorb_conc_file1 = flopy.utils.UcnFile(model_workspace / f"MT3D001S.UCN")
    # sorb_conc_file2 = flopy.utils.UcnFile(model_workspace / f"MT3D002S.UCN")
    conc1 = conc_file1.get_alldata()
    conc2 = conc_file2.get_alldata()

    for icomp, comp in enumerate([conc1, conc2]):
        comp_name = mt.btn.species_names[icomp]
        for ilay in range(mf.dis.nlay):
            fig, axs = plt.subplots(ncols=2, figsize=(24, 16))
            ax, ax2 = axs
            ax.set_title(f"Original concentration {comp_name} in layer {ilay+1}")
            ax2.set_title(f"Final concentration {comp_name} in layer {ilay+1}")
            ax.set_aspect("equal")
            ax2.set_aspect("equal")
            extent = None
            extent = [137000, 137400, 456525, 457325]  # Wall
            pmv = flopy.plot.PlotMapView(model=mf, ax=ax, layer=ilay, extent=extent)
            pmv_conc = flopy.plot.PlotMapView(
                model=mf, ax=ax2, layer=ilay, extent=extent
            )
            for segment in wall_segments:
                v = ax.plot(*segment, color="grey")
            # c = pmv.plot_array(heads, alpha=0.5, masked_values=[-999.99])
            c = pmv.plot_array(
                comp[0], masked_values=[comp.max()], alpha=0.5, vmin=0, vmax=1
            )
            plt.colorbar(c, ax=ax)
            pmv.plot_discharge(
                frf=frf, fff=fff, flf=flf, head=heads, color="#ff0000", istep=3, jstep=3
            )
            pmv.plot_bc("WEL", plotAll=True)
            pmv.plot_shapefile("data/resistive_wall.shp", alpha=1, facecolor="None")
            # pmv.plot_shapefile("data/PAK/PAK_0_5.shp", alpha=0.5, facecolor="red")
            pmv.plot_shapefile(
                "data/location_wells.shp", radius=1, edgecolor="blue", facecolor="b"
            )

            c = pmv_conc.plot_array(
                comp[-1], masked_values=[comp.max()], alpha=0.5, vmin=0, vmax=1
            )
            plt.colorbar(c, ax=ax2)
            for segment in wall_segments:
                v = ax2.plot(*segment, color="grey")

            fig.tight_layout()
            fig.savefig(
                model_output_dir / f"compare_concentrations_{comp_name}_{ilay}.png"
            )
            plt.close(fig)

    xs_lines = {
        "A-A'": [[137215, 457200], [137215, 456600]],
        "B-B'": [[137000, 457075], [137375, 457075]],
        "C-C'": [[137050, 456800], [137375, 456800]],
    }
    for icomp, comp in enumerate([conc1, conc2]):
        for line_title, xs_line in xs_lines.items():
            comp_name = mt.btn.species_names[icomp]
            fig, axs = plt.subplots(nrows=2, figsize=(16, 16))
            ax, ax2 = axs
            ax.set_title(f"Original concentration {comp_name} on transect {line_title}")
            ax2.set_title(f"Final concentration {comp_name} on transect {line_title}")
            # ax.set_aspect(3)
            # ax2.set_aspect(3)
            # row=72, column=51
            # xs_line = wells
            # xs_line = [[137215, 457200], [137215, 456600]]
            pxs = flopy.plot.PlotCrossSection(
                model=mf, ax=ax, line={"line": xs_line}  # , extent=(600, 800, -80, 0)
            )
            pxs2 = flopy.plot.PlotCrossSection(
                model=mf, ax=ax2, line={"line": xs_line}  # , extent=(600, 800, -80, 0)
            )
            pxs.plot_grid(linewidths=0.5, alpha=0.5)
            pxs2.plot_grid(linewidths=0.5, alpha=0.5)
            c = pxs.plot_array(
                comp[0], head=heads, masked_values=[conc2.max()], vmin=0, vmax=1
            )
            plt.colorbar(c, ax=ax)
            c = pxs2.plot_array(
                comp[-1], head=heads, masked_values=[conc2.max()], vmin=0, vmax=0.5
            )  #
            plt.colorbar(c, ax=ax2)
            # pxs.plot_ibound(color_noflow="#aaaaaa", head=heads)
            # pxs.plot_discharge(
            #     frf=frf,
            #     fff=fff,
            #     flf=flf,
            #     head=heads,
            #     color="#ffffff",
            #     normalize=True,
            #     hstep=3,
            #     kstep=1,
            # )
            fig.tight_layout()
            fig.savefig(
                model_output_dir / f"crosssection_{comp_name}_{line_title[0]}.png"
            )
            plt.close(fig)

    with fiona.open(
        model_output_dir / "crosssections.shp",
        "w",
        driver="ESRI Shapefile",
        crs=fiona.crs.from_epsg(28992),
        schema={"geometry": "LineString", "properties": {"title": "str"}},
    ) as writer:
        for line_title, xs_line in xs_lines.items():
            writer.write(
                {
                    "geometry": {"type": "LineString", "coordinates": xs_line},
                    "properties": {"title": line_title},
                }
            )
