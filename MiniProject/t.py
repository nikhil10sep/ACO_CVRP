import ant_colony as ac
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import threading
from networkx.drawing.nx_agraph import graphviz_layout

def make_binary_tree(G_in, p, r, root, G_out):
	G_in.nodes[r]['done'] = 1
	if G_in.degree[r] == 1:
		return

	if G_in.degree[r] <= 2 or (G_in.degree[r] == 3 and r != root):
		for child in G_in[r]:
			if(G_in.nodes[child]['done'] == 0):
				G_out.add_edge(r, child)
				G_out[r][child]['weight'] = G_in[r][child]['weight']
				make_binary_tree(G_in, r, child, root, G_out)	
		return

	i = 1
	for child in G_in[r]:
		if(G_in.nodes[child]['done'] == 0):
			dummy = 'dummy ' + str(r) + '_' + str(i)
			parent = 'dummy ' + str(r) + '_' + str(i - 1)
			if i == 1:
				parent = r
			G_out.add_edge(parent, child)
			G_out.add_edge(parent, dummy)
			G_out[parent][child]['weight'] = G_in[r][child]['weight']
			G_out[parent][dummy]['weight'] = 0
			i += 1
			make_binary_tree(G_in, parent, child, root, G_out)
	return

def edges_sum(G_in, r):
	if G_in.out_degree[r] == 0:
		return

	for child in G_in[r]:
		edges_sum(G_in, child)
		G_in.nodes[r]['edges_sum'] += G_in.nodes[child]['edges_sum'] + G_in[r][child]['weight']

	return

def max_depth(G_in, r):
	if G_in.out_degree[r] == 0:
		return

	for child in G_in[r]:
		max_depth(G_in, child)
		if G_in.nodes[r]['max_depth'] < G_in.nodes[child]['max_depth'] + G_in[r][child]['weight']:
			G_in.nodes[r]['max_depth'] = G_in.nodes[child]['max_depth'] + G_in[r][child]['weight']
			G_in.nodes[r]['max_depth_child'] = child

	return

def distance(G_in, r):
	for child in G_in[r]:
		G_in.nodes[child]['dist'] = G_in.nodes[r]['dist'] + G_in[r][child]['weight']
		distance(G_in, child)

	return

def get_node_labels(G_in):
	labels = {}
	edges_sum = nx.get_node_attributes(G_in, 'edges_sum')
	max_depth = nx.get_node_attributes(G_in, 'max_depth')
	dist = nx.get_node_attributes(G_in, 'dist')
	for n in G_in:
		labels[n] = str(n) + '-' + str(edges_sum[n]) + '-' + str(max_depth[n]) + '-' + str(dist[n]) + '-' + str(2 * edges_sum[n] - max_depth[n] + dist[n])

	return labels

def search(G_in, r, D):
	for child in G_in[r]:
		if G_in.nodes[child]['dist'] + 2 * G_in.nodes[child]['edges_sum'] - G_in.nodes[child]['max_depth'] > D:
			return search(G_in, child, D)

	return r

def update_params(G_in, r, change_start):
	for parent in G_in.predecessors(r):
		G_in.nodes[parent]['edges_sum'] = 0
		G_in.nodes[parent]['max_depth'] = 0
		for child in G_in[parent]:
			if(child != change_start):
				G_in.nodes[parent]['edges_sum'] += G_in.nodes[child]['edges_sum'] + G_in[parent][child]['weight']

				if G_in.nodes[parent]['max_depth'] < G_in.nodes[child]['max_depth'] + G_in[parent][child]['weight']:
					G_in.nodes[parent]['max_depth'] = G_in.nodes[child]['max_depth'] + G_in[parent][child]['weight']
					G_in.nodes[parent]['max_depth_child'] = child

		update_params(G_in, parent, change_start)

	return

def ancestor_path(G_in, root):
	node = root
	path = [node]
	while G_in.in_degree(node) != 0:
		for p in G_in.predecessors(node):
			path.append(p)
		node = p

	path.reverse()
	return (path)

def traverse_tree(G_in, r):
	if G_in.out_degree[r] == 0:
		return [r]

	path = [r]
	for child in G_in[r]:
		if child != G_in.nodes[r]['max_depth_child']:
			path.extend(traverse_tree(G_in, child))
			path.append(r)

	for child in G_in[r]:
		if child == G_in.nodes[r]['max_depth_child']:
			path.extend(traverse_tree(G_in, child))

	return path

def minTVF(G_in, root, D):
	rpaths = []
	T = G_in.copy()
	while True:
		r = search(T, root, D)

		if T.out_degree[r] == 0:
			print("Cannot solve the VRP, D is too small")
			break
		
		if r == root and T.nodes[r]['dist'] + 2 * T.nodes[r]['edges_sum'] - T.nodes[r]['max_depth'] < D:
			rpath_with_dummies = traverse_tree(T, r)
			rpath = [x for x in rpath_with_dummies if isinstance(x, int)]
			rpaths.append(rpath)
			break

		root_path = ancestor_path(T, r)
		for child in T[r]:
			child_path = traverse_tree(T, child)
			rpath_with_dummies = root_path + child_path
			rpath = [x for x in rpath_with_dummies if isinstance(x, int)]
			rpaths.append(rpath)


		if r == root:
			break

		update_params(T, r, r)
		for parent in G_in.predecessors(r):
			T.remove_edge(parent, r)

	return rpaths

def draw_tree(G_in, G_in_bin):
	plt.figure(1)
	pos = nx.spring_layout(G_in)
	nx.draw_networkx_labels(G_in, pos)
	nx.draw_networkx_edge_labels(G_in, pos, nx.get_edge_attributes(G, 'weight'))
	nx.draw(G_in, pos, with_labels=False)

	plt.figure(2)
	plt.title('Binary Tree')
	pos_bin = graphviz_layout(G_in_bin, prog='dot')
	nx.draw_networkx_labels(G_in_bin, pos_bin, get_node_labels(G_in_bin))
	nx.draw_networkx_edge_labels(G_in_bin, pos_bin, nx.get_edge_attributes(G_in_bin, 'weight'))
	nx.draw(G_in_bin, pos_bin, with_labels=False, arrows=False)

	plt.show()

#print(minTVF(G_bin, root, 100))

for i in [20]:
	for j in [100]:
		d=j
		G = nx.random_tree(i)
		# print("Done")
		for (u,v) in list(G.edges):
			G[u][v]['weight'] = np.random.randint(1,10)

		nx.set_node_attributes(G, 0, 'done')

		root = max(nx.degree(G), key=itemgetter(1))[0]

		G_bin = nx.DiGraph()

		make_binary_tree(G, None, root, root, G_bin)

		nx.set_node_attributes(G_bin, 0, 'edges_sum')
		nx.set_node_attributes(G_bin, 0, 'max_depth')
		nx.set_node_attributes(G_bin, 0, 'dist')
		edges_sum(G_bin, root)
		max_depth(G_bin, root)
		distance(G_bin, root)

		# H = nx.Graph()

		# for (u, v) in G_bin.edges():
		# 	H.add_edge(u, v)
		# 	H[u][v]['weight'] = G_bin[u][v]['weight'] 

		nodes = {}
		for node in G.nodes():
			nodes[node] = node

		length = dict(nx.all_pairs_dijkstra_path_length(G))
		path = dict(nx.all_pairs_dijkstra_path(G))

		# print('Path found')

		def distance_dijkstra(start, end):
			return length[start][end]

		def path_dijkstra(start, end):
			return path[start][end]

		answer_tvf = minTVF(G_bin, root, d)
		print('dblp_ravi', i, j, len(answer_tvf))

		thread_draw = threading.Thread(target=draw_tree, args=(G, G_bin))
		thread_draw.start()
		max_possible_depth = G_bin.nodes[root]['max_depth']
		for it in [2000]:
			if d<max_possible_depth:
				print("D is less than max depth: Immpossible input")
			else:
				colony = ac.ant_colony(nodes, distance_dijkstra, path_dijkstra, alpha = 2.0, gamma = 10.0 ,beta = 5.0,ant_count = 20, start=root,D=d, pheromone_constant=10.0, iterations=it, pheromone_evaporation_coefficient=0.4)
				answer, dist, k = colony.mainloop()

				# print(answer)

				# print('Total distance:', dist)
				print(i,j,it, k)

