from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import brightway2 as bw
import numpy as np

habe_path = Path('/Users/akim/Documents/LCA_files/HABE_2017/')
year09='091011'
year12='121314'
year15='151617'
ausgaben_label = 'Ausgaben'
def get_filepath(rootdir, filelabel, year):
    list_ = [f for f in rootdir.glob('**/*') if f.is_file() and filelabel in f.name and year in f.name]
    assert len(list_)==1
    return list_[0]
filepath09 = get_filepath(habe_path, ausgaben_label, year09)
filepath12 = get_filepath(habe_path, ausgaben_label, year12)
filepath15 = get_filepath(habe_path, ausgaben_label, year15)
df09 = pd.read_csv(filepath09, sep='\t')
df12 = pd.read_csv(filepath12, sep='\t')
df15 = pd.read_csv(filepath15, sep='\t')

bw.projects.set_current('GSA for protocol')
co = bw.Database('CH consumption 1.0')

clothing = [act for act in co if 'Clothing and footwear sector'==act['name']][0]
furnishings = [act for act in co if 'Furnishings, household equipment and routine household maintenance sector'==act['name']][0]
recreation = [act for act in co if 'Recreation and culture sector'==act['name']][0]
miscellaneous = [act for act in co if 'Miscellaneous goods and services sector'==act['name']][0]
health = [act for act in co if 'Health sector'==act['name']][0]

# sector_choice = 'Clothing and footwear'
# sector_choice = 'Furnishings, household equipment and routine household maintenance'
# sector_choice = 'Recreation and culture'
# sector_choice = 'Miscellaneous goods and services'
sector_choice = 'Health'

sectors = {
    'Clothing and footwear': clothing,
    'Furnishings, household equipment and routine household maintenance': furnishings,
    'Recreation and culture': recreation,
    'Miscellaneous goods and services': miscellaneous,
    'Health': health,
}
sector_act = sectors[sector_choice]

codes_names = {}
for exc in list(sector_act.exchanges()):
    if exc['type']=='technosphere':
        raw_code = exc.input.key[1]
        if 'mx' in raw_code:
            code = 'A' + raw_code[2:]
        else:
            code = 'A' + raw_code[1:]
        codes_names[code] = {
            'name': exc.input['name'],
            'unit': exc.input['unit'],
        }
n_acts = len(codes_names)

data_all_years = {
    '2009-2011': df09,
    '2012-2014': df12,
    '2015-2017': df15,
}
colors = {
    '2009-2011': {
        'color': 'red',
        'colorscale': 'Reds'
    },
    '2012-2014': {
        'color': 'blue',
        'colorscale': 'ice'
    },
    '2015-2017': {
        'color': 'green',
        'colorscale': 'Viridis'
    },
}

ncols = 3
titles_ = [
    [
        "{}. {}".format(i+1, val['name'][:60]),
        "histogram w/o high outliers".format(val['name']),
        "histogram w/o zeros".format(val['name']),
    ] for i,val in enumerate(codes_names.values())
]
titles = [item for sublist in titles_ for item in sublist]

fig = make_subplots(
    rows=n_acts,
    cols=ncols,
    subplot_titles=titles,
    horizontal_spacing=0.12,
)

row = 1
for code,name_unit in codes_names.items():
    x_all = np.array([])
    if row == 1:
        showlegend = True
    y_year = 1
    col = 1
    for year, df in data_all_years.items():
        x = df[code].values
        x_all = np.hstack([x_all,x])
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=[y_year]*len(x),
                mode='markers',
                showlegend=False,
                opacity=1,
                marker = dict(color=np.random.randn(len(x)),colorscale=colors[year]['colorscale'], line_width=1),
                legendgroup=year,
            ),
            row=row,
            col=col,
        )
        y_year += 1
    fig.update_xaxes(
        title_text='HH consumption values, {}'.format(name_unit['unit']),
        row=row,
        col=col
    )
    fig.update_yaxes(
        title_text='Years',
        row=row,
        col=col,
        tickmode='array',
        tickvals=[1, 2, 3],
        ticktext=['2009-2011', '2012-2014', '2015-2017', ],
    )

    num_bins = 50
    x_all_no_outliers = x_all[x_all < np.quantile(x_all, .99)]
    x_all_no_zeros = x_all[x_all != 0]
    bins_no_outliers = np.linspace(min(x_all_no_outliers), max(x_all_no_outliers), num_bins, endpoint=True)
    bins_no_zeros = np.linspace(min(x_all_no_zeros), max(x_all_no_zeros), num_bins, endpoint=True)

    for year, df in data_all_years.items():
        x = df[code].values
        # Histogram
        col = 2
        x_no_outliers = x[x < np.quantile(x_all, .99)]
        freq, bins = np.histogram(x_no_outliers, bins=bins_no_outliers)
        fig.add_trace(
            go.Bar(
                x=bins,
                y=freq,
                showlegend=showlegend,
                opacity=0.5,
                name=year,
                legendgroup=year,
                marker = dict(color=colors[year]['color']),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(
            title_text='HH consumption values, {}'.format(name_unit['unit']),
            row=row,
            col=col
        )
        fig.update_yaxes(
            title_text='Frequency',
            row=row,
            col=col
        )
        if year=='2009-2011' and col==3:
            k = (row - 1) * 3 + col-1
            fig.layout.annotations[k]['text'] += ', {:3.1f}% of values'.format(len(x_all_no_outliers)/len(x_all)*100)

        col = 3
        x_no_zeros = x[x > 0]
        freq, bins = np.histogram(x_no_zeros, bins=bins_no_zeros)
        fig.add_trace(
            go.Bar(
                x=bins,
                y=freq,
                showlegend=False,
                opacity=0.5,
                name=year,
                legendgroup=year,
                marker=dict(color=colors[year]['color']),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(
            title_text='HH consumption values, {}'.format(name_unit['unit']),
            row=row,
            col=col
        )
        fig.update_yaxes(
            title_text='Frequency',
            row = row,
            col = col
        )
        if year=='2009-2011' and col==3:
            k = (row - 1) * 3 + col-1
            fig.layout.annotations[k]['text'] += ', {:3.1f}% of values'.format(len(x_all_no_zeros)/len(x_all)*100)
    showlegend = False
    row += 1

fig.add_annotation(
    x=0.5, y=1 + 100/n_acts/300,
    xref='paper',
    yref='paper',
    text="{} sector".format(sector_choice),
    showarrow=False,
    font_size=20,
    bordercolor="#c7c7c7",
    borderwidth=2,
    borderpad=4,
)

fig.update_layout(
    width=500*ncols,
    height=300*n_acts,
    barmode='overlay',
    margin=dict(l=150,b=0,r=0,t=150),
)

fig.show()

fig.write_html("write_files/figures/{}.html".format(sector_choice.lower()).replace(' ', '_'))