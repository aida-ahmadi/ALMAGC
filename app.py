import os.path

import dash
from dash import dcc
from dash import html

# ===============================
from astropy.io import fits
import pandas as pd
import numpy as np
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy import units as u
from astropy.coordinates import SkyCoord
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.patches import Circle, Polygon
import re
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template


def my_circle(center, radius, n_points=75):
    t=np.linspace(0, 1, n_points)
    x=center[0]+radius*np.cos(2*np.pi*t)
    y=center[1]+radius*np.sin(2*np.pi*t)
    return x, y


def get_footprint_scatter_pts(footprints):
    # --- Check the footprint type ---#
    get_type = re.split('ICRS', footprints)
    # --- Handle circles ---#
    if get_type[0].rstrip() == 'Circle':
        circle_props = re.split('\s+', get_type[1].lstrip())

        # --- convert to galactic coords --#
        gc = SkyCoord(ra=float(circle_props[0]) * u.degree, dec=float(circle_props[1]) * u.degree, frame='icrs')
        galc = gc.transform_to('galactic')
        l = galc.l * u.degree
        b = galc.b * u.degree

        # --- Stupid spherical galaxy ---#
        if float(l.value) > 180.0:
            use_l = float(l.value) - 360.0
        else:
            use_l = float(l.value)
        # circle = Circle((float(use_l), float(b.value)), float(circle_props[2]), linestyle='--')
        return my_circle([float(use_l), float(b.value)], float(circle_props[2]), n_points=75)

    # --- Handle circles ---#
    elif get_type[0].rstrip() == 'Polygon':

        poly_props = re.split('\s+', get_type[1].lstrip())
        xvals = poly_props[::2]  # --- get all even number entries in array
        xvalsf = [float(i) for i in xvals]  # --- list to float ... ugly
        yvals = poly_props[1::2]  # --- get all odd number entries in array
        yvalsf = [float(i) for i in yvals]  # --- list to float ... ugly

        poly_XYs = np.column_stack((xvalsf, yvalsf))
        # --- convert to galactic coords ---#
        # --- again mildly ugly! ---#
        xvalsf_galc = np.array([])
        yvalsf_galc = np.array([])

        for pair in poly_XYs:
            gc = SkyCoord(ra=float(pair[0]) * u.degree, dec=float(pair[1]) * u.degree, frame='icrs')
            galc = gc.transform_to('galactic')
            l = galc.l * u.degree
            b = galc.b * u.degree
            # --- Stupid spherical galaxy ---#
            if float(l.value) > 180.0:
                use_l = float(l.value) - 360.0
            else:
                use_l = float(l.value)

            xvalsf_galc = np.append(xvalsf_galc, float(use_l))
            yvalsf_galc = np.append(yvalsf_galc, float(b.value))
        xvalsf_galc = np.append(xvalsf_galc, xvalsf_galc[0])
        yvalsf_galc = np.append(yvalsf_galc, yvalsf_galc[0])
        return xvalsf_galc, yvalsf_galc


def get_all_moment_fitsname(uid):
    data_path = './assets'
    fits_files = os.listdir(data_path)
    if uid is not np.nan:
        uid_fits_format = uid.replace("://", "___").replace("/", "_")
        return [s for s in fits_files if uid_fits_format in s]


def add_moment_fitsname(uid, name, fits_files):
    if uid is not np.nan:
        uid_fits_format = uid.replace("://", "___").replace("/", "_")
        list_of_fitsnames = [s for s in fits_files if uid_fits_format in s]
        return [s for s in list_of_fitsnames if name in s]


def add_freq_ranges(fits_files):
    for f, current_fits in enumerate(fits_files):
        if current_fits is not None:
            freq_start = current_fits.split("GHz")[0].split("_")[-1]
            freq_end = current_fits.split("GHz")[1].replace("-", "")
            freq_range = "{}-{}".format(freq_start, freq_end)
            return freq_range

# ---------------------------------------------

# Get ATLASGAL as df
ATLASGAL_fits = './assets/ATLASGAL_CMZ-3squaredegree.fits'

with fits.open(ATLASGAL_fits) as atlasgal_data:
    df = pd.DataFrame(atlasgal_data[0].data)
atlasgal_data.close()

# Get ATLASGAL WCS
filename = get_pkg_data_filename(ATLASGAL_fits)
hdu = fits.open(filename)[0]
wcs = WCS(hdu.header)
img = fits.getdata(ATLASGAL_fits, origin='lower')
header=fits.getheader(ATLASGAL_fits)

x_start_pix = 0
x_end_pix = np.shape(img)[1]
y_start_pix = 0
y_end_pix = np.shape(img)[0]
x_array_pix = np.arange(x_start_pix, x_end_pix)
y_array_pix = np.arange(y_start_pix, y_end_pix)

x_dim = header['NAXIS1']
x_ref_pix = header['CRPIX1']
x_ref_val = header['CRVAL1']  # glon
dx_val = header['CDELT1']  # deg
x_0_pix = x_ref_val-x_ref_pix*dx_val
x_end_pix = x_0_pix+dx_val*x_dim

y_ref_pix = header['CRPIX2']
y_ref_val = header['CRVAL2']  # glat
dy_val = header['CDELT2']  # deg
y_0_pix = y_ref_val-y_ref_pix*dy_val

# Plot ATLASGAL
viridis = px.colors.sequential.Viridis
magma=px.colors.sequential.Magma
cividis=px.colors.sequential.Cividis
fig = go.Figure()
fig.add_trace(go.Heatmap(z=df,
                        x0=x_0_pix,
                        dx=dx_val,
                        y0=y_0_pix,
                        dy=dy_val,
                        colorscale=[
                        [0, viridis[0]],        #0
                        [1./100, viridis[3]],   #1000
                        [1./10, viridis[6]],       #10000
                        [1., viridis[9]],             #100000
                        ],
                         name='ATLASGAL',
                        colorbar={"title": "ATLASGAL Intensity (Jy/beam)"},
                         hovertemplate=
                         "<b>Longitude:</b> %{x:.5f} deg<br>" +
                         "<b>Latitude:</b> %{y:.5f} deg<br>" +
                         "<b>Intensity:</b> %{z:.5f} Jy/beam<br>"))
fig.update_layout(xaxis_range=(1.6, -1.6))
fig['layout']['xaxis']['range'] = [1.6, -1.6]

# ---------------------------------------------

mol_name = ['13CO', 'HCOplus', 'HCN', 'HNC', 'N2Hplus']
mol_name_html = ["<sup>13</sup>CO", "HCN", "HCO<sup>+</sup>", "N<sub>2</sub>H+"]
plotly_colours = ['#FECB52', '#EF553B', '#FF97FF', '#19D3F3', '#B6E880', '#AB63FA', '#636EFA', '#00CC96', '#FFA15A', '#FF6692']
for m, mol in enumerate(mol_name):
    data_path = './assets/Moment_maps/{}/all_sources'.format(mol)
    if os.path.isdir(data_path):
        fits_files = os.listdir(data_path)
    line_df = pd.read_csv('./assets/Tables/alma_cmz_2deg_{}.csv'.format(mol))
    line_df['s_region_gal_scatter'] = line_df['s_region'].apply(get_footprint_scatter_pts)
    line_df['s_region_gal_xscatter'] = line_df['s_region_gal_scatter'].str[0]
    line_df['s_region_gal_yscatter'] = line_df['s_region_gal_scatter'].str[1]

    # Check if fits file names contain the target name and MOUS ID and write the fits filename to a new column
    line_df['fits_files'] = line_df.apply(lambda x: add_moment_fitsname(x.MOUS_id, x.target_name, fits_files), axis=1)

    # Check if fits file names contain the target name and MOUS ID and write the fits filename to a new column
    line_df['freq_ranges'] = line_df['fits_files'].apply(add_freq_ranges)

    line_df.to_csv("./assets/Tables/alma_cmz_2deg_{}_fitsinfo.csv".format(mol))

    regions = []
    for s in range(len(line_df['s_region_gal_xscatter'])):
        if line_df['s_region_gal_xscatter'][s] is not None:
            # Plot regions as overlays
            if abs(min(line_df['s_region_gal_xscatter'][s])) < 1.5 and abs(min(line_df['s_region_gal_yscatter'][s])) < 0.5:
                fig.add_trace(go.Scatter(x=line_df['s_region_gal_xscatter'][s], y=line_df['s_region_gal_yscatter'][s],
                                         mode='lines', line =dict(color=plotly_colours[m], width=2), name=line_df['covered_lines'][s],
                                         legendgroup=line_df['covered_lines'][s],
                                         visible='legendonly',
                                         hovertemplate=
                                             "<b>"+line_df['covered_lines'][s]+"</b><br><br>" +
                                             "<b>Longitude:</b> %{x:.5f} deg<br>" +
                                             "<b>Latitude:</b> %{y:.5f} deg<br><br>" +
                                             "<b>Project code:</b> " + str(line_df['proposal_id'][s]) + "<br>" +
                                             "<b>ALMA source name:</b> " + str(line_df['target_name'][s]) + "<br>" +
                                             "<b>Angular resolution:</b> {:.2f}".format(
                                                 line_df['ang_res_arcsec'][s]) + " arcsec<br>" +
                                             "<b>Spectral resolution:</b> {:.2f}".format(
                                                 line_df['vel_res_kms'][s]) + " km/s<br>" +
                                             "<b>MOUS ID:</b> " + str(line_df['MOUS_id'][s]) + "<br>" +
                                             "<b>Archive URL:</b> " + str(line_df['access_url'][s]) + "<br>" +
                                             "<extra></extra>"))
                # fig['layout']['hoverlabel']['bgcolor'] = plotly_colours[m]
            # Plot moment 0 maps as overlays
            current_fits_list = line_df['fits_files'][s]
            for f, current_fitsname in enumerate(current_fits_list):
                current_fits_path = os.path.join(data_path, current_fitsname)
                if current_fitsname is not None:
                    with fits.open(current_fits_path) as data:
                        overlay_df = pd.DataFrame(data[0].data)
                    overlay_header = fits.getheader(current_fits_path)

                    overlay_x_dim = overlay_header['NAXIS1']
                    overlay_x_ref_pix = overlay_header['CRPIX1']
                    overlay_x_ref_val = overlay_header['CRVAL1']  # glon
                    overlay_dx_val = overlay_header['CDELT1']  # deg
                    overlay_x_0_pix = overlay_x_ref_val - overlay_x_ref_pix * overlay_dx_val
                    if overlay_x_0_pix > 180.0:
                        overlay_x_0_pix = overlay_x_0_pix - 360
                    overlay_x_end_pix = overlay_x_0_pix + overlay_dx_val * overlay_x_dim
                    overlay_y_ref_pix = overlay_header['CRPIX2']
                    overlay_y_ref_val = overlay_header['CRVAL2']  # glat
                    overlay_dy_val = overlay_header['CDELT2']  # deg
                    overlay_y_0_pix = overlay_y_ref_val - overlay_y_ref_pix * overlay_dy_val
                    fig.add_trace(go.Heatmap(z=overlay_df,
                                             x0=overlay_x_0_pix,
                                             dx=overlay_dx_val,
                                             y0=overlay_y_0_pix,
                                             dy=overlay_dy_val, showscale=False, name=line_df['covered_lines'][s],
                                             legendgroup=line_df['covered_lines'][s],
                                             visible='legendonly',
                                             hovertemplate=
                                             "<b>"+line_df['covered_lines'][s]+"</b><br><br>" +
                                             "<b>Longitude:</b> %{x:.5f} deg<br>" +
                                             "<b>Latitude:</b> %{y:.5f} deg<br>" +
                                             "<b>Integrated intensity:</b> %{z:.5f} Jy/beam.km/s <br>" +
                                             "<b>Integration range:</b> " + str(line_df['freq_ranges'][s]) + " GHz<br><br>" +
                                             "<b>Project code:</b> " + str(line_df['proposal_id'][s]) + "<br>" +
                                             "<b>ALMA source name:</b> " + str(line_df['target_name'][s]) + "<br>" +
                                             "<b>Angular resolution:</b> {:.2f}".format(
                                                 line_df['ang_res_arcsec'][s]) + " arcsec<br>" +
                                             "<b>Spectral resolution:</b> {:.2f}".format(
                                                 line_df['vel_res_kms'][s]) + " km/s<br>" +
                                             "<b>MOUS ID:</b> " + str(line_df['MOUS_id'][s]) + "<br>" +
                                             "<b>Archive URL:</b> " + str(line_df['access_url'][s]) + "<br>" +
                                             "<extra></extra>",
                                             customdata=np.array(line_df['access_url'][s]),
                                             hoverlabel={'bgcolor': plotly_colours[m]}
                                             ))
                    data.close(current_fitsname)


# Plot only unique legend names (transitions)
names = set()
fig.for_each_trace(
    lambda trace:
        trace.update(showlegend=False)
        if (trace.name in names) else names.add(trace.name))

# Legend position in box:
fig.update_layout(legend=dict(
    yanchor="top",
    y=1.0,
    xanchor="left",
    x=-0.25,
))

fig.update_layout(xaxis_range=(1.5, -1.5))
fig.update_layout(yaxis_scaleanchor="x")
fig.update_layout(template='plotly_white')
# ---------------------------------------------
# set plot parameters
fig['layout']['xaxis']['title'] = "Galactic Longitude (deg)"
fig['layout']['yaxis']['title'] = "Galactic Latitude (deg)"

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
load_figure_template('FLATLY')

# app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.Div([html.H1("ALMA's View of Milky Way's Central Molecular Zone",
            style={
                'background-image': 'linear-gradient(rgba(0, 0, 0, 0.0), rgba(0, 0, 0, 1.0)), url("/assets/alma_cropped.jpg")',
                'textAlign': 'center', 'color': 'white', 'line-height': '3em',
                'background-position': 'center top',
                'background-repeat': 'no-repeat', 'background-size': 'cover'
                })]),
    html.Div([dcc.Graph(id="graph", figure=fig)])
])


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
