import ant_colony as ac
import csv
import sys

f = open('input', newline='')
csv_reader = csv.reader(f, delimiter=' ')

nodes = {}

for row in csv_reader:
	nodes[int(row[0])] = (float(row[1]), float(row[2]))

f.close()

f = open('cap', newline='')
csv_reader = csv.reader(f, delimiter=' ')

cap = {}

for row in csv_reader:
	cap[int(row[0])] = (float(row[1]))

f.close()


def distance(start, end):
	x0 = start[0]
	x1 = end[0]
	y0 = start[1]
	y1 = end[1]
	return pow((pow(x0 - x1, 2) + pow(y0 - y1, 2)), 0.5)

def capacity(node):
	return cap[node]

# colony = ac.ant_colony(nodes, distance, capacity, alpha = 3.0, gamma = 9.0 ,beta = 5.0,ant_count = 20, start=1,D=float('inf'), Q=100, pheromone_constant=100.0, iterations=30, pheromone_evaporation_coefficient=0.3)
# colony = ac.ant_colony(nodes, distance, capacity, alpha = 3.0, gamma = 9.0 ,beta = 5.0,ant_count = 20, start=1,D=float('inf'), Q=100, pheromone_constant=100.0, iterations=30, pheromone_evaporation_coefficient=0.3)
print('start')
colony = ac.ant_colony(nodes, distance, capacity, start=1, ant_count=30, alpha=1.0, beta=3.0, gamma=1.0, D=float('inf'), Q=100, pheromone_evaporation_coefficient=0.4, pheromone_constant=100.0, iterations=300)
#colony = ac.ant_colony(nodes, distance, capacity, alpha = a, gamma = g ,beta = b,ant_count = 20, start=1,D=float('inf'), Q=100, pheromone_constant=100.0, iterations=30, pheromone_evaporation_coefficient=s)
answer, dist, c, k = colony.mainloop()
print(k, dist, sum(dist))
print (c)
print (answer)
