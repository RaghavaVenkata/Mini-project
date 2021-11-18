import json
a={
  "nodes":[
		{"name":"node1","group":1},
		{"name":"node2","group":2},
		{"name":"node3","group":2},
		{"name":"node4","group":3}
	],
	"links":[
		{"source":2,"target":1,"weight":1},
		{"source":0,"target":2,"weight":3}
	]
}
with open('data.json', 'w') as fp:
    json.dump(a, fp)

!pip install pydot

f = open( "/content/data.txt", "r" )
data = []
for line in f:
    data.append([int(i) for i in line.strip().split()])
nodes=[]
for edge in data:
  for i in edge:
    nodes.append(i)
nodes=list(set(nodes))
edges=data

import networkx as nx
import matplotlib.pyplot as plt

G=nx.Graph()
for node in nodes:
  G.add_node(node)
for edge in edges:
  G.add_edge(*edge)

print(G.nodes())
print(G.edges())

plt.figure(figsize=(15,15))
nx.draw(G, pos = nx.spring_layout(G), node_size=30, \
    node_color='blue', linewidths=0.25, font_size=10, \
    font_weight='bold', with_labels=False)
plt.show()
from pyvis.network import Network
import networkx as nx

G=nx.Graph()
for node in nodes:
  G.add_node(node)
for edge in edges:
  G.add_edge(*edge)
nt = Network('500px', '500px')
# populates the nodes and edges data structures
nt.from_nx(G)
nt.show('nx.html')
G = nx.path_graph(4)
pos = nx.circular_layout(G)
G=nx.cycle_graph(len(nodes))
for node in nodes:
  G.add_node(node)
for edge in edges:
  G.add_edge(*edge)
nt = Network('500px', '500px')
# populates the nodes and edges data structures
nt.from_nx(G)
nt.show('nx_circular.html')
nx_graph = nx._graph(10)
G=nx.circular_layout()
for node in nodes:
  G.add_node(node)
for edge in edges:
  G.add_edge(*edge)
plt.figure(figsize=(20,20))
nx.draw(G, pos = nx.nx_pydot.graphviz_layout(G), node_size=30, \
    node_color='blue', linewidths=0.25, font_size=10, \
    font_weight='bold', with_labels=False)
plt.show()

G=nx.Graph()
for node in nodes:
  G.add_node(node)
for edge in edges:
  G.add_edge(*edge)
plt.figure(figsize=(30,30))
nx.draw(G, pos = nx.circular_layout(G), node_size=5, \
    node_color='blue', linewidths=0.025, font_size=5, \
    font_weight='bold', with_labels=True)
plt.show()

nt = Network('1000px', '1000px')
nt.from_nx(G)
nt.show_buttons(filter_=['physics'])
nt.show('nx_cir.html')
nx.nx_pydot.write_dot(G, '/tmp/graph.dot')
plt.figure(figsize=(15,15))
nx.draw(G,node_size=30,node_color=range(10,526),vmin=0.0,vmax=1.0,with_labels=False)
plt.show()
import networkx as nx
import numpy as np
from matplotlib import pyplot, patches

def draw_adjacency_matrix(G, node_order=None, partitions=[], colors=[]):

    adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=node_order)
    fig = pyplot.figure(figsize=(10, 10)) # in inches
    pyplot.imshow(adjacency_matrix,cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear')
import seaborn as sns   
adjacency_matrix = nx.to_numpy_matrix(G)
plt.figure(figsize=(30,30))
ax = sns.heatmap(adjacency_matrix,square=True) 
from scipy import io
draw_adjacency_matrix(G)

import pandas as pd

df=pd.read_csv("/content/AH_Sickle_Cell_Disease_Provisional_Death_Counts_2019-2021.csv")
df['Age Group'].value_counts()
import plotly.express as px
fig = px.treemap(df, path=[px.Constant("Age Group"),'Date of Death Year','Race or Hispanic Origin','Age Group'], values="SCD_Multi")
fig.update_traces(root_color="lightgrey")
fig.update_layout(margin = dict(t=100, l=25, r=25, b=25))
fig.show()
sns.heatmap(df.corr())
temp=df.groupby('Age Group')[['SCD and COVID-19']].sum()
temp_lst=df.groupby('Age Group')[['SCD and COVID-19']].sum().index.get_level_values(0).tolist()
temp_data=df.groupby('Age Group')[['SCD and COVID-19']].sum().reset_index()['SCD and COVID-19'].values.tolist()
print(temp_data)
temp_lst.remove('15-19 years')
temp_lst.remove('<5 years')
temp_data.remove(0)
print(temp_lst,temp_data)
temp_data.remove(0)
import squarify
squarify.plot(sizes=temp_data, label=temp_lst, alpha=.8)
plt.axis('off')
plt.show()
import plotly.express as px
fig = px.parallel_coordinates(df, color='SCD and COVID-19',
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             color_continuous_midpoint=2)
fig.show()
import plotly.express as px
fig = px.parallel_coordinates(data[['LocationID','YearEnd']], color='LocationID',
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             color_continuous_midpoint=2)
fig.show()

import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv('AH_Sickle_Cell_Disease_Provisional_Death_Counts_2019-2021.csv')

df['Cases'] = df['SCD_Underlying'] + df['SCD_Multi'] + df['SCD and COVID-19']

fig = px.treemap(df, path=[px.Constant("all"), 'Date of Death Year', 'Age Group', 'Race or Hispanic Origin', 'Quarter'], values='Cases')
fig.update_traces(root_color="lightgrey")
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()

