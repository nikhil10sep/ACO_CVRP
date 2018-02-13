import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

G = nx.random_tree(10)
print("Done")
for (u,v) in list(G.edges):
	G[u][v]['weight'] = np.random.randint(1,10)

for n in list(G.nodes):
	G.nodes[n]['done'] = 0

root = max(nx.degree(G), key=itemgetter(1))[0]

print ('root', root)

G_bin = nx.DiGraph()

def make_binary_tree(G_in, p, r, root, G_out):
	G.nodes[r]['done'] = 1
	if G_in.degree[r] == 1:
		# G_out.add_edge(p, r)
		# G_out[p][r]['weight'] = G_in[p][r]['weight']
		# print('1 Added Edge ' + str(p) + ', ' + str(r))
		return

	if G_in.degree[r] <= 2 or (G_in.degree[r] == 3 and r != root):
		for child in G[r]:
			if(G.nodes[child]['done'] == 0):
				G_out.add_edge(r, child)
				G_out[r][child]['weight'] = G_in[r][child]['weight']
				print('2 Added Edge ' + str(r) + ', ' + str(child))
				make_binary_tree(G_in, r, child, root, G_out)	
		return

	i = 1
	for child in G[r]:
		if(G.nodes[child]['done'] == 0):
			dummy = 'dummy ' + str(r) + '_' + str(i)
			parent = 'dummy ' + str(r) + '_' + str(i - 1)
			if i == 1:
				parent = r
			G_out.add_edge(parent, child)
			G_out.add_edge(parent, dummy)
			G_out[parent][child]['weight'] = G_in[r][child]['weight']
			G_out[parent][dummy]['weight'] = 0
			i += 1
			print('3 Added Edge ' + str(parent) + ', ' + str(child))
			print('3 Added Edge ' + str(parent) + ', ' + str(dummy))
			make_binary_tree(G_in, parent, child, root, G_out)
	return


print(nx.degree(G))
print(G.edges)

make_binary_tree(G, None, root, root, G_bin)

pos = nx.spring_layout(G)
pos_bin = nx.spring_layout(G_bin)
plt.figure(1)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G, 'weight'))
nx.draw(G, pos)
plt.figure(2)
nx.draw_networkx_labels(G_bin, pos_bin)
nx.draw_networkx_edge_labels(G_bin, pos_bin, nx.get_edge_attributes(G_bin, 'weight'))
nx.draw(G_bin, pos_bin, with_labels=False, arrows=False)
print(nx.degree(G_bin))
print(G_bin.edges)
plt.show()