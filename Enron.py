import pandas as pd

data = pd.read_csv('emails.csv')
data1 = data.iloc[0:1000]
app = data.iloc[517201:517400]
data1=data1.append(app)
data = data1
app.shape
data1.shape
weights = []

for index,row in data.iterrows():
    message = ''
    for i in row['message'].split('\n')[16:]:
        message = message+i
    weights.append(len(message))
len(data.iloc[0]['message'].split('\n')[16])
senders = []
receivers = []

for index,row in data.iterrows():
    sender = row['message'].split('\n')[2][6:]
    receiver = row['message'].split('\n')[3][4:]
    senders.append(sender)
    if receiver.find(','):
        receiver = receiver.split(',')
        receiver = receiver[0]
    receivers.append(receiver)

edgelist = pd.DataFrame({'senders':senders,'receivers':receivers,'weights':weights})
edgelist['weights'] = edgelist['weights']/100
all_nodes = list(set(senders))

for rec in list(set(receivers)):
    if rec in all_nodes:
        continue
    else:
        all_nodes.append(rec)

from random import randint

def gencoordinates(m, n):
    seen = set()

    x, y = randint(m, n), randint(m, n)

    while True:
        seen.add((x, y))
        yield [x, y]
        x, y = randint(m, n), randint(m, n)
        while (x, y) in seen:
            x, y = randint(m, n), randint(m, n)
g = gencoordinates(1,1598)

xs = []
ys = []

for i in range(len(all_nodes)):
    xy = next(g)
    x = xy[0]
    y = xy[1]
    xs.append(x)
    ys.append(y)

nodelist = pd.DataFrame({'nodes':all_nodes,'x':xs,'y':ys})



import networkx as nx

from collections import namedtuple

from math import sqrt

import bokeh

from bokeh.models import HoverTool

from bokeh.plotting import show, figure

from bokeh.colors import RGB

import random
from bokeh.io import curdoc



#corresponding package on pypi is confusingly called python-louvain
import community.community_louvain as community




def create_bokeh_graph(graph):

    

    def gen_edge_coordinates(graph, layout):

        xs = []

        ys = []

        val = namedtuple("edges", "xs ys")

        for edge in graph.edges():

            from_node = layout[edge[0]]

            to_node = layout[edge[1]]

            xs.append([from_node[0],to_node[0]])

            ys.append([from_node[1], to_node[1]])

        return val(xs=xs, ys=ys)



    def gen_node_coordinates(layout):

        names, coords = zip(*layout.items())

        xs, ys = zip(*coords)

        val = namedtuple("nodes", "names xs ys")

        return val(names=names, xs=xs, ys=ys)



    #Calc Layout - Slowest Part

    plot_layout = nx.spring_layout(graph)

    edges = graph.edges()
    
    weights = [graph[u][v]['weight'] for u,v in edges]

    nx.draw(graph, plot_layout, edges=edges, width=weights)

    _nodes = gen_node_coordinates(plot_layout)

    _edges = gen_edge_coordinates(graph, plot_layout)

    

    #Prepare Bokeh-Figure

    hover = HoverTool(tooltips=[('name', '@name'), 

                                ('node_id', '$index'),

                                ('degree', '@degree'),

                                ('cluster_id', '@community_nr')], names=["show_hover"])



    fig = figure(width=800, height=600, 

                 tools=[hover, 'box_zoom', 'resize', 'reset', 'wheel_zoom', 'pan', 'lasso_select'],

                logo = None)

    fig.toolbar.logo = None

    fig.axis.visible = False                            

    fig.xgrid.grid_line_color = None

    fig.ygrid.grid_line_color = None

    

    #Draw Edges

    source_edges = bokeh.models.ColumnDataSource(dict(xs=_edges.xs, ys=_edges.ys))

    fig.multi_line('xs', 'ys', line_color='navy', source=source_edges, alpha=0.17, line_width=weights)

    

    #Calc numbers

    degrees = list(nx.degree(graph).values())

    clustering = list(nx.triangles(graph).values())

    clustering1 = list(nx.clustering(graph).values())

    clustering2 = list(nx.square_clustering(graph).values())

    communs = community.best_partition(graph)

    nodes, communities = zip(*communs.items())

    betw = list(nx.betweenness_centrality(graph).values())

    

    #create Colormaps

    colormap_coms = {x : RGB(random.randrange(0,256),random.randrange(0,256),random.randrange(0,256)) 

                     for x in list(set(communities))}

    community_color_list, community_nr = zip(*[(colormap_coms[communs[node]], communs[node]) for node in nodes])



    

    graph_nodes = graph.number_of_nodes()

    

    colors = ['firebrick' for node in range(graph_nodes)]

    

    #Draw circles

    source_nodes = bokeh.models.ColumnDataSource(dict(xs=_nodes.xs, ys=_nodes.ys, name=_nodes.names, 

                                                      single_color = colors,

                                                      color_by_clusters = community_color_list, 

                                                      degree=degrees, 

                                                      clustering=clustering, community_nr=community_nr,

                                                      betweenness = betw))

    
    source_nodes1 = bokeh.models.ColumnDataSource(dict(xs=_nodes.xs, ys=_nodes.ys, name=_nodes.names, 

                                                      single_color = colors,

                                                      color_by_clusters = community_color_list, 

                                                      degree=degrees, 

                                                      clustering=clustering1, community_nr=community_nr,

                                                      betweenness = betw))

    source_nodes2 = bokeh.models.ColumnDataSource(dict(xs=_nodes.xs, ys=_nodes.ys, name=_nodes.names, 

                                                      single_color = colors,

                                                      color_by_clusters = community_color_list, 

                                                      degree=degrees, 

                                                      clustering=clustering2, community_nr=community_nr,

                                                      betweenness = betw))

    source_nodes3 = bokeh.models.ColumnDataSource(dict(xs=_nodes.xs, ys=_nodes.ys, name=_nodes.names, 

                                                      single_color = colors,

                                                      color_by_clusters = community_color_list, 

                                                      degree=degrees, 

                                                      clustering=clustering, community_nr=community_nr,

                                                      betweenness = betw))

    r_circles = fig.circle('xs', 'ys', fill_color='single_color', line_color='single_color', 

                           source = source_nodes, alpha=0.7, size=9, name="show_hover")

    

    #Create Color-Selector    

    colorcallback1 = bokeh.models.callbacks.CustomJS(args=dict(source=source_nodes, source1 = source_nodes1, source2 = source_nodes2, source3 = source_nodes3), code="""

        var value = cb_obj.value;

        var data = source.data;

	data = source1.data;

        source.change.emit()

    """)

    colorcallback = bokeh.models.callbacks.CustomJS(args=dict(source=source_nodes, circles=r_circles), code="""

        var value = cb_obj.get('value');

        circles.glyph.line_color.field = value;

        circles.glyph.fill_color.field = value;

        source.trigger('change')

    """)   

    

    button1 = bokeh.models.widgets.Select(title="Clustering Algorithm", value="Clustering_By_Triangles", 

                                         options=["Clustering_By_Triangles", "Clustering_By_Weighted_Average_Of_Subgraphs", "Clustering_By_Squares"], 

                                         callback=colorcallback1)

    button = bokeh.models.widgets.Select(title="Color Scheme", value="single_color", 

                                         options=["single_color", "color_by_clusters"], 

                                         callback=colorcallback)    

    #Create grid and save

    layout_plot = bokeh.layouts.gridplot([[fig, button, button1]])

    

    #if file is wanted

    #bokeh.io.output_file(f"graph.html")

    #bokeh.io.save(layout_plot)

    

    show(layout_plot)


    curdoc().add_root(layout_plot)
    curdoc().title = "690V Assignment - Classification"






#working example



g = nx.Graph()

for i, elrow in edgelist.iterrows():
    g.add_edge(elrow['senders'], elrow['receivers'], weight=elrow['weights'])
    
for i, nlrow in nodelist.iterrows():
    g.node[nlrow['nodes']] = nlrow[1:].to_dict()
    
create_bokeh_graph(g)






