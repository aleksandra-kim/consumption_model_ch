import plotly.graph_objects as go
import numpy as np


def plot_archetypes_scores_yearly(archetypes_scores):
    fig = go.Figure()
    scores_arr = np.array(list(archetypes_scores.values()))
    argsort = np.argsort(scores_arr)
    names = np.array(list(archetypes_scores.keys()))[argsort]
    names = [n.replace("archetype_", "").replace("_consumption", "") for n in names]
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(names)),
            y=scores_arr[argsort],
            mode='markers',
        )
    )
    fig.update_xaxes(
        tickmode='array',
        tickvals=np.arange(len(names)),
        ticktext=names,
    )
    fig.update_layout(
        width=700,
        height=400
    )
    fig.update_xaxes(title_text="Archetypes")
    fig.update_yaxes(title_text="Yearly impact of archetype, [kg CO2-eq.]")
    return fig


def plot_archetypes_scores_per_sector(archetypes_scores):
    # Define some known variables
    sectors_dict = {
        "Food": [
            "Food and non-alcoholic beverages sector",
            "Alcoholic beverages and tobacco sector",
        ],
        "Restaurants & hotels": ["Restaurants and hotels sector"],
        "Clothing": ["Clothing and footwear sector"],
        "Housing": ["Housing, water, electricity, gas and other fuels sector"],
        "Furnishings": ["Furnishings, household equipment and routine household maintenance sector"],
        "Health": ["Health sector"],
        "Transport": ["Transport sector"],
        "Communication": ["Communication sector"],
        "Recreation": ["Recreation and culture sector"],
        "Education": ["Education sector"],
        "Other": [
            "Durable goods sector",
            "Fees sector",
            "Miscellaneous goods and services sector",
            "Other insurance premiums sector",
            "Premiums for life insurance sector",
        ]
    }
    num_people_dict = {
        "A": 4.2, "B": 3.7, "C": 3.5, "D": 2.1, "E": 1.6, "F": 3.3, "G": 1.6, "H": 1.0, "I": 1.6,
        "J": 4.2, "K": 3.2, "L": 2.4, "M": 2.2, "N": 1.2, "O": 1.1, "OA": 3.3, "OB": 1.8, "P": 2.2,
        "Q": 1.4, "R": 1.3, "S": 2.6, "T": 2.0, "U": 1.7, "V": 2.0, "W": 1.6, "X": 2.0, "Y": 2.0, "Z": 3.3,
    }
    months_in_year = 12
    names_letters_dict = {
        name: name.replace("archetype_", "").replace("_consumption", "") for name in archetypes_scores.keys()
    }
    # Sort archetypes
    total_scores = []
    for name, sector_scores in archetypes_scores.items():
        total = sum(list(sector_scores.values())) / num_people_dict[names_letters_dict[name]]
        total_scores.append(total)
    archetypes_scores_argsort = np.argsort(total_scores)
    archetypes_sorted = np.array(list(archetypes_scores.keys()))[archetypes_scores_argsort]
    archetypes_names_sorted = [names_letters_dict[name] for name in archetypes_sorted]

    data = []
    for sector_name, sectors in sectors_dict.items():
        x = []
        for name in archetypes_sorted:
            avalue = archetypes_scores[name]
            sector_score = sum([avalue[sector] for sector in sectors]) / num_people_dict[names_letters_dict[name]]
            x.append(sector_score)
        x = np.array(x) * months_in_year
        data.append(
            go.Bar(name=sector_name, y=archetypes_names_sorted, x=x, orientation='h'),
        )
    fig = go.Figure(
        data=data,
    )
    fig.update_layout(
        width=600,
        height=600,
        barmode='stack',
    )
    fig.update_xaxes(title_text="Yearly impact per capita, [kg CO2-eq.]")
    fig.update_yaxes(title_text="Archetypes")
    return fig
