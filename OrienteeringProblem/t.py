import ant_colony as ac
import csv
import sys

f = open('input', newline='')
csv_reader = csv.reader(f, delimiter='\t')

nodes = {}
cap = {}

i = 0
sum_cap = 0
for row in csv_reader:
	i += 1
	nodes[i] = (float(row[0]), float(row[1]))
	cap[i] = float(row[2])
	sum_cap += cap[i]

f.close()

print(sum_cap)
def distance(start, end):
	x0 = start[0]
	x1 = end[0]
	y0 = start[1]
	y1 = end[1]
	return pow((pow(x0 - x1, 2) + pow(y0 - y1, 2)), 0.5)

def capacity(node):
	return cap[node]

# print(nodes)
# print(cap)

# colony = ac.ant_colony(nodes, distance, capacity, alpha = 3.0, gamma = 9.0 ,beta = 5.0,ant_count = 20, start=1,D=float('inf'), Q=100, pheromone_constant=100.0, iterations=30, pheromone_evaporation_coefficient=0.3)
# colony = ac.ant_colony(nodes, distance, capacity, alpha = 3.0, gamma = 9.0 ,beta = 5.0,ant_count = 20, start=1,D=float('inf'), Q=100, pheromone_constant=100.0, iterations=30, pheromone_evaporation_coefficient=0.3)
print('start')
colony = ac.ant_colony(nodes, distance, capacity, sum_cap, start=1, end_loc=i, ant_count=50, alpha=1.0, beta=4.0, gamma=1.0, D=50.0, Q=100, pheromone_evaporation_coefficient=0.3, pheromone_constant=1.0, iterations=2000)
#colony = ac.ant_colony(nodes, distance, capacity, alpha = a, gamma = g ,beta = b,ant_count = 20, start=1,D=float('inf'), Q=100, pheromone_constant=100.0, iterations=30, pheromone_evaporation_coefficient=s)
answer, dist, c = colony.mainloop()
print (c)
print (dist)
print (answer)
print (len(answer))
