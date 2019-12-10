# Clustering with Hedonic Games
# Detecting communities in networks with cooperative game theory
# A research experiment in colaboration between *UFRJ and ^INRIA
# *Lucas Lopes, *Daniel Sadoc, ^Kostya and ^Giovanni
# October 2019

## Import Dependecies ##########################################################

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import networkx as nx
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
from hedonic import Game
from players import sequential

################################################################################
## Helper Functions ############################################################
################################################################################

game = Game()
game.replay('KAR_0.95_GRE_r0.5_i13_v52.50_e23.43_abs14.75.pickle')

## Get list of Networks ########################################################

def get_existing_networks():
    options = []
    networks = Game.show_networks(get=True)
    for net in networks:
        option = {'label': net, 'value': net[:3].upper()}
        options.append(option)
    return options


def get_iteration_data():
    total_verts = game.infos['verts']
    total_edges = game.infos['edges']
    acc = np.array(game.hist['accumulated'])
    verts_yes = np.array(game.hist['verts_yes'])
    edges_yes = np.array(game.hist['edges_yes'])
    edges_no  = np.array(game.hist['edges_no'])

    pot_yes, pot_no = game.global_potential(verts_yes, edges_yes, edges_no, sum=False)
    potential_prop = pot_yes / (pot_yes + pot_no)

    edges_yes_prop = edges_yes / total_edges
    edges_off_prop = ((total_edges - edges_yes - edges_no) / total_edges) + edges_yes_prop

    verts_yes_prop = verts_yes / total_verts

    iterations, instant = [0], []
    for i in range(len(acc)-1):
        iterations.append(i+1)
        instant.append(acc[i+1]-acc[i])

    return {
        'iterations': iterations,
        'instantaneous' : instant,
        'accumulated' : acc,
        'potential_prop': potential_prop,
        'verts_yes_prop': verts_yes_prop,
        'edges_yes_prop': edges_yes_prop,
        'edges_off_prop': edges_off_prop }

game_data = get_iteration_data()

## Plot a Graph ################################################################

def plot_graph(G = nx.random_geometric_graph(200, 0.125)): # nx.karate_club_graph()): #
    # V=range(N)# list of vertices
    # g=nx.Graph()
    # g.add_nodes_from(V)
    # g.add_edges_from(E)# E is the list of edges
    # pos=nx.fruchterman_reingold_layout(g)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        # print(G.node[edge[0]])
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.node[node]['pos']
        node_x.append(x)
        node_y.append(y)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    # Another option would be to size points by the number of connections i.e. node_trace.marker.size = node_adjacencies
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Visualize the Network',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Verts: V and Edges: E",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

################################################################################
## Dash Divs ###################################################################
################################################################################

## 1. Header ###################################################################

Header = html.Div(children=[
    html.H1('Hedonic Games'),
    html.Div([
        html.P("Detecting communities in networks with cooperative game theory."),
        html.P("A research experiment in colaboration between *UFRJ and ^INRIA."),
        html.P("*Lucas Lopes, *Daniel Sadoc, ^Kostya and ^Giovanni."),
        html.P("October 2019")]),
    html.H2('Run an Experiment')],
    style={'textAlign': 'center'})

## 1.1 Run and Experiments #####################################################

RunExperiments = html.Div(style={'columnCount': 2}, children=[
    html.Label('Choose a Network:'), # Choose a Network
    dcc.Dropdown(
        id='network-selection',
        options=get_existing_networks(),
        multi=False,
        value='DAG'),
    html.Label('Or upload yours:'), # Upload a Network - TO_DO: update dropdown
    dcc.Upload(
        id='upload-network',
        children=html.Div([ # Add a new experiment by
            'Drag and Drop or ',
            html.A('Select .CSV')]),
        style={
            'width': '100%',
            'height': '50px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'},
        multiple=True), # Allow multiple files to be uploaded
    html.Label('Alpha', id='alpha-value'),
    dcc.Slider(
        id='alpha-slider',
        min=0,
        max=1,
        step=0.05,
        value=0.95),
    dcc.Tabs(id="tabs-init-mode", value='select-init-mode', children=[
        dcc.Tab(
            label="Initial mode: 'Select'",
            value='select-init-mode',
            children=[
                html.Label('Select Nodes: ', id='select-nodes-value'),
                dcc.Input(
                    id="nodes-selected",
                    type='text',
                    placeholder="['node', 'name', '...']")]),
        dcc.Tab(
            label="Initial mode: 'Random'",
            value='random-init-mode',
            children=[
                html.Label('Random Classification: ', id='random-value'),
                dcc.Slider(
                    id='init-random-slider',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.5)])]),
    html.Label('Options:'),
    dcc.Checklist(
        options=[
            {'label': 'Verbose', 'value': 'ver'},
            {'label': 'Export', 'value': 'exp'} ],
        value=['ver', 'exp']),
    html.Button('Run!', id='run-button', style={'width': '100%'}),
    html.Textarea( # TODO: only show when running
        id='running-message',
        style={'width':'100%','margin':'10px'}),
    dcc.Graph(
        id='network-preview',
        figure=go.Figure(plot_graph()))])

## 2. Vizualize Experiments ####################################################

## 2.1 Table Results ###########################################################

df = pd.read_csv('experiments/results.csv')

TableResults = html.Div(children=[
    html.H2('Visualize Experiments Results', style={'textAlign': 'center'}), # H1?
    html.H3('Experiments Results', style={'textAlign': 'center'}),
    dash_table.DataTable(
        id='experiments-datatable',
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True} for i in df.columns],
        data=df.to_dict('records'),
        style_cell_conditional=[ {
            'if': {'column_id': c},
            'textAlign': 'left'
        } for c in ['Date', 'Region'] ],
        style_header={
            'backgroundColor': 'light_blue',
            'fontWeight': 'bold' },
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_selectable="single",
        selected_rows=[0],
        style_table={
            'overflowX': 'scroll',
            'overflowY': 'scroll',
            'maxHeight': '450px',
            'minWidth': '100%'},
        fixed_columns={ 'headers': True, 'data': 1 },
        fixed_rows={ 'headers': True, 'data': 0 },
        style_cell={
        # all three widths are needed
            'minWidth': '180px', 'width': '180px', 'maxWidth': '360px',
            'overflow': 'hidden', 'padding': '7px',
            'textOverflow': 'ellipsis'}),
    html.Div(id='datatable-interactivity-container'),
    html.Label('Or upload yours:'), # Upload a Network
    dcc.Upload(
        id='upload-experiment',
        children=html.Div([ # Add a new experiment by
            'Drag and Drop or ',
            html.A('Select .PKL')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'},
        multiple=True)])

## 2.2 Instantaneous Gain ######################################################

InstantGain = html.Div(children=[
    html.H4('Instantaneous Gain', style={'textAlign': 'center'}),
    # dcc.Graph(id="instant-graph")]) # figure=instant
    dcc.Graph(figure=go.Figure(data=go.Scatter(x=game_data['iterations'], y=game_data['instantaneous'])))])

## 2.3 Accumulated Gain ########################################################

AccumulatedGain = html.Div(children=[
    html.H4('Accumulated Gain', style={'textAlign': 'center'}),
    dcc.Graph(figure=go.Figure(data=go.Scatter(x=game_data['iterations'], y=game_data['accumulated'])))])

## 2.4 Potential Proportion ####################################################

PotentialProportion = html.Div(children=[
    html.H4('Potential Proportion', style={'textAlign': 'center'}),
    dcc.Graph(figure=go.Figure(data=go.Scatter(x=game_data['iterations'], y=game_data['potential_prop'])))])

## 2.5 Vertices Proportion #####################################################

VerticesProportion = html.Div(children=[
    html.H4('Vertices Proportion', style={'textAlign': 'center'}),
    dcc.Graph(figure=go.Figure(data=go.Scatter(x=game_data['iterations'], y=game_data['verts_yes_prop'])))])

## 2.5 Edges Proportion ########################################################

EdgesProportion = html.Div(children=[
    html.H4('Edges Proportion', style={'textAlign': 'center'}),
    dcc.Graph(figure=go.Figure(data=[
        go.Scatter(x=game_data['iterations'], y=game_data['edges_yes_prop'], name='Edges Yes Proportion'),
        go.Scatter(x=game_data['iterations'], y=game_data['edges_off_prop'], name='Edges Off Proportion')]))])

## 2.6 Number of moves #########################################################

################################################################################
## App Server ##################################################################
################################################################################

## App Layout ##################################################################

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
        Header,RunExperiments,
        TableResults,InstantGain,AccumulatedGain,PotentialProportion,VerticesProportion,EdgesProportion])
        #Experiments,TableResults,Networks,Iterations])

################################################################################
## Decorators ##################################################################
################################################################################

@app.callback(
    Output(component_id='alpha-value', component_property='children'),
    [Input(component_id='alpha-slider', component_property='value')])
def update_alpha_value(alpha_value):
    return f'Alpha: {alpha_value}'

@app.callback(
    Output(component_id='random-value', component_property='children'),
    [Input(component_id='init-random-slider', component_property='value')])
def update_random_mode_value(random_value):
    return f'Random Classification: {random_value}'


@app.callback(
    Output(component_id='network-preview', component_property='figure'),
    [Input(component_id='network-selection', component_property='value')])
def update_network(network):
    net = nx.read_edgelist(f'networks/{network}.csv')
    return plot_graph(net)

@app.callback(
    Output(component_id='running-message', component_property='children'),
    [Input(component_id='run-button', component_property='n_clicks')],
    [State('network-selection', 'value'), State('alpha-slider', 'value'),
    State('nodes-selected', 'value'), State('init-random-slider', 'value')]) # todo: options checkbox
def run_game(n_clicks, network, alpha, init_mode, random_mode):
    if n_clicks is None:
        raise PreventUpdate
    else:
        game.load_network(network)
        game.set_alpha(alpha)
        if init_mode:
            game.set_initial_state('s', init_mode)
        else:
            game.set_initial_state('r', random_mode)
        game.play(sequential) # todo: better way to pass player
        return "Elephants are the only animal that can't jump"



# todo: importar csv from id=upload-network



# @app.callback(
#     Output('datatable-interactivity', 'style_data_conditional'),
#     [Input('datatable-interactivity', 'selected_columns')] )
# def update_styles(selected_columns):
#     return [{
#         'if': { 'column_id': i },
#         'background_color': '#D2F3FF'
#     } for i in selected_columns]

# @app.callback(
#     Output('instant-graph', 'figure'),
#     [Input('selected-value', 'value'), Input('values-range', 'value')])
# def update_instant_graph(selected, values):
#
#     instant = go.Figure()
    # for i, y in enumerate(ys):
    #     instant.add_trace(go.Scatter(
    #         x=x, y=y, name=f'Karate {i+1}',
    #         line=dict(color=f'rgb({np.random.randint(0,255)},\
    #         {np.random.randint(0,255)},{np.random.randint(0,255)})', width=1)))
    #         x=x+x_rev,
    #         y=y1_upper+y1_lower,
    #         fill='toself',
    #         fillcolor='rgba(0,100,80,0.2)',
    #         line_color='rgba(255,255,255,0)',
    #         showlegend=False,
    #         name='Fair',

    # return instant
    #
    # text = {"Max_TemperatureC": "Maximum Temperature", "Mean_TemperatureC": "Mean Temperature",
    #         "Min_TemperatureC": "Minimum Temperature"}
    # dff = df[(df["values"] >= year[0]) & (df["values"] <= values[1])]
    # trace = []
    # for type in selected:
    #     trace.append(go.Scatter(x=dff["Date"], y=dff[type], name=text[type], mode='lines',
    #                             marker={'size': 8, "opacity": 0.6, "line": {'width': 0.5}}, ))
    #
    # x, y = get_instant_gain()
    # trace = []
    # trace.append(go.Scatter(
    #     x=x,
    #     y=y,
    #     fill='toself',
    #     fillcolor='rgba(0,100,80,0.2)',
    #     line_color='rgba(255,255,255,0)',
    #     showlegend=False,
    #     name='Fair',
    #     mode='lines',
    #     marker={'size': 8, "opacity": 0.6, "line": {'width': 0.5}} ))
    #
    # return {"data": trace,
    #         "layout": go.Layout(title="Instantaneous Gain", colorway=['#fdae61', '#abd9e9', '#2c7bb6'],
    #                             yaxis={"title": "Profit on move"}, xaxis={"title": "Iteration"})}


# @app.callback(
#     Output('datatable-interactivity-container', "children"),
#     [Input('datatable-interactivity', "derived_virtual_data"),
#      Input('datatable-interactivity', "derived_virtual_selected_rows")])
# def update_graphs(rows, derived_virtual_selected_rows):
    # When the table is first rendered, `derived_virtual_data` and
    # `derived_virtual_selected_rows` will be `None`. This is due to an
    # idiosyncracy in Dash (unsupplied properties are always None and Dash
    # calls the dependent callbacks when the component is first rendered).
    # So, if `rows` is `None`, then the component was just rendered
    # and its value will be the same as the component's dataframe.
    # Instead of setting `None` in here, you could also set
    # `derived_virtual_data=df.to_rows('dict')` when you initialize
    # the component.
    #
    #
    # if derived_virtual_selected_rows is None:
    #     derived_virtual_selected_rows = []
    #
    # dff = df if rows is None else pd.DataFrame(rows)
    #
    # colors = ['#7FDBFF' if i in derived_virtual_selected_rows else '#0074D9'
    #           for i in range(len(dff))]

    # return [
    #     dcc.Graph(
    #         id=column,
    #         figure={
    #             "data": [ {
    #                 "x": dff["country"],
    #                 "y": dff[column],
    #                 "type": "bar",
    #                 "marker": {"color": colors} } ],
    #             "layout": {
    #                 "xaxis": {"automargin": True},
    #                 "yaxis": {
    #                     "automargin": True,
    #                     "title": {"text": column} },
    #                 "height": 250,
    #                 "margin": {"t": 10, "l": 10, "r": 10} } } )
    #     # check if column exists - user may have deleted it
    #     # If `column.deletable=False`, then you don't
    #     # need to do this check.
    #     for column in ["pop", "lifeExp", "gdpPercap"] if column in dff ]

## Run Server ##################################################################

if __name__ == '__main__':

    # if 'DYNO' in os.environ:
    #     app_name = os.environ['DASH_APP_NAME']
    # else:
    #     app_name = 'Hedonic Ploting'

    app.run_server(debug=True)
