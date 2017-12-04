import pandas as pd

data = pd.read_csv('emails.csv')

weights = []

for index,row in data.iterrows():
    message = ''
    for i in row['message'].split('\n')[16:]:
        message = message+i
    weights.append(len(message))

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

g = nx.Graph()

for i, elrow in edgelist.iterrows():
    g.add_edge(elrow['senders'], elrow['receivers'], weight=elrow['weights'])
weight=elrow['weights']

for i, nlrow in nodelist.iterrows():
    g.node[nlrow['nodes']] = nlrow[1:].to_dict()


# Define node positions data structure (dict) for plotting
node_positions = {node[0]: (node[1]['x'], -node[1]['y']) for node in g.nodes(data=True)}


import networkx as nx

from bokeh.io import show, output_file
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool,WheelZoomTool, PanTool, ResetTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Reds, RdGy

plot = Plot(plot_width=1200, plot_height=1200,
            x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))
plot.title.text = "Graph Interaction Demonstration"

plot.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool(), WheelZoomTool(), PanTool(),ResetTool())

graph_renderer = from_networkx(g, nx.circular_layout, scale=1, center=(0,0))

graph_renderer.node_renderer.glyph = Circle(size=10, fill_color='#cb181d')
graph_renderer.node_renderer.selection_glyph = Circle(size=10, fill_color=Spectral4[2])
graph_renderer.node_renderer.hover_glyph = Circle(size=10, fill_color=Spectral4[1])

graph_renderer.edge_renderer.glyph = MultiLine(line_color='#000000', line_alpha=0.8, line_width=2)
graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=2)
graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=2)

graph_renderer.selection_policy = NodesAndLinkedEdges()
graph_renderer.inspection_policy = EdgesAndLinkedNodes()

plot.renderers.append(graph_renderer)

layout = layout([
    [plot]
])


curdoc().add_root(layout)
curdoc().title = "690V Assignment - Classification"
show(layout)

