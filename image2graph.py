import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import scipy.misc as smp
import itertools
import networkx as nx
import sys

def euclid_metric(start,end):
	return np.sqrt((start[0]-end[0])**2 + (start[1]-end[1])**2)


class ImageArray:

	def __init__(self,filename):
		self.img = imread(filename)
		self.img.flags.writeable = True
		self.root = dict([((x,y),(x,y)) for x in xrange(self.width()) for y in xrange(self.height())])
		self.size = dict([((x,y),1) for x in xrange(self.width()) for y in xrange(self.height())])
		self.bg = self.color(0,0)	# set background color as top-left pixel
		self.edges = None
		self.vertices = None

	def show_img(self):
		# Displays the image		
		img = smp.toimage(self.img)    
		img.show()

	def dimensions(self):
		return self.img.shape

	def height(self):
		return self.dimensions()[0]

	def width(self):
		return self.dimensions()[1]

	def bg_color(self):
		return self.bg

	def color(self,x,y):
		if x<0 or y<0:
			return self.bg_color()
		try:
			return tuple(self.img[y][x])
		except IndexError:
			return self.bg_color()

	def get_root(self,key):
		## Two keys have the same root iff they belong to the same component
		parent = self.root.get(key,key)
		while key!=parent:
			key = parent 
			parent = self.root.get(key,key)

		return key


	def union(self,a,b):
		if type(a)!= tuple or type(b)!=tuple:
			raise Exception("Expected types for union must be tuples.")

		if a not in self.root or b not in self.root:
			return

		r1=self.get_root(a)
		r2=self.get_root(b)

		sz1 = self.size.get(r1,1)
		sz2 = self.size.get(r2,1)

		if r1==r2:
			return None
		if sz1 < sz2:
			self.root[r1] = r2
			self.size[r2] = sz2+1
			self.size[r1] = None
		else:
			self.root[r2] = r1
			self.size[r1] = sz1+1
			self.size[r2]=None


	def get_component_roots(self):
		## returns a list of roots by which to identify components
		## (disregards the background component)
		#return dict.fromkeys([self.get_root((x,y)) for (x,y) in self.root.values()]).keys()
		return dict.fromkeys([self.get_root((x,y)) for (x,y) in self.root.values() if self.color(x,y)!=self.bg_color()]).keys()

	def get_component(self,key):
		## returns a set of pixels belonging to the component with root key
		list_of_roots = self.get_component_roots()
		if key not in list_of_roots:
			raise Exception("Key must be one of the following roots: {0}".format(str(list_of_roots)))
		return set([(x,y) for x in xrange(self.width()) for y in xrange(self.height()) if self.get_root((x,y))==key])

	def get_local_square(self,x,y,d=1):
		## Returns set of points in the (2d+1)x(2d+1) square 
		## belonging to the same component as (x,y).

		square = [(x+i,y+j) for i in xrange(-d,d+1) for j in xrange(-d,d+1)]
		return set([(a-x,b-y) for (a,b) in square if self.get_root((x,y))==self.get_root((a,b))])


	def show_local_square(self,x,y,d=100):
		## Displays a (2d+1)x(2d+1) pixel image of the image centered at (x,y)
		x,y = int(x),int(y)

		data = np.zeros((2*d+1,2*d+1,3), dtype=np.uint8)

		for (a,b) in self.get_local_square(x,y,d=d):
			data[b+d,a+d] = [255,0,0]	# colors pixel (a,b) in red

		data[d,d] = [0,0,255]	# colors centre pixel blue
			
		img = smp.toimage(data)    
		img.show()



	def find_vertices(self):
		## Generates list of all vertices and stores in self.vertices

		# list of fingerprints for identifying precorners
		fingerprints = [set([(0,0),(-1,0),(-1,-1),(0,-1)]),
						set([(0,0),(1,0),(1,1),(0,1)]),
						set([(0,0),(1,0),(1,-1),(0,-1)]),
						set([(0,0),(-1,0),(-1,1),(0,1)]),
						set([(0,0),(-1,0),(-1,1)]),
						set([(0,0),(-1,0),(-1,-1)]),
						set([(0,0),(1,0),(1,1)]),
						set([(0,0),(1,0),(1,-1)]),
						set([(0,0),(0,1),(1,1)]),
						set([(0,0),(0,1),(-1,1)]),
						set([(0,0),(0,-1),(1,-1)]),
						set([(0,0),(0,-1),(-1,-1)]),
						set([(0,0),(0,-1),(-1,-1),(1,-1)]),
						set([(0,0),(0,1),(-1,1),(1,1)]),
						set([(0,0),(1,0),(1,1),(1,-1)]),
						set([(0,0),(-1,0),(-1,1),(-1,-1)]),
						set([(0,0),(1,0)]),
						set([(0,0),(-1,0)]),
						set([(0,0),(0,1)]),
						set([(0,0),(0,-1)]),
						set([(1,0)]),
						set([(-1,0)]),
						set([(0,1)]),
						set([(0,-1)]),
						]

		fingerprints += [set([(i,j) for i in [-1,0,1] for j in [-1,0,1]]) - fp for fp in fingerprints]	# include the inverse fingerprints to catch convex corners

		vertices = []
		for key in self.get_component_roots():
			C = self.get_component(key)

			# finds precorners by relative amount of colour
			precorners = set()
			for (x,y) in C:
				if self.get_local_square(x,y,d=1) in fingerprints:
				 	precorners.add((x,y))



			# clusters precorners together based on euclidean distance
			d = dict([(node,set([node])) for node in precorners])
			for (start,end) in itertools.combinations(precorners,2):
				if euclid_metric(start,end) < 10:
					C1 = d.get(start,set([start]))
					C2 = d.get(end,set([end]))

					d[start] = C1|C2
					d[end] = C1|C2

			# gets the midpoint of each cluster
			for S in d.values():
				midpoint = self.midpoint(S)
				if midpoint not in vertices:
					vertices.append(midpoint)

		# store locally
		self.vertices = vertices



	def remove_square(self,x,y,d=5):
		## Removes a square of diameter (2*d+1) centered at (x,y) 
		x,y = int(x),int(y)
		square = [(x+i,y+j) for i in xrange(-d,d+1) for j in xrange(-d,d+1)]
		for (a,b) in square:
			try:
				self.img[b,a] = list(self.bg_color())	# overwrite the corner pixels in whit
			except:
				pass



	def find_components(self):
		## Finds all connected components by applying
		## Quick-union on neighbouring points that share the same colour

		neighbours = [(-1,0), (-1,-1), (0,-1), (-1,1)]	# list of relative positions of upper-left neighbours
														# (lower right are caught symmetrically)

		for x in xrange(self.width()):
			for y in xrange(self.height()):
				c = self.color(x,y)
				for (i,j) in neighbours:		
					if c==self.color(x+i,y+j):
						# join the two components if the share the same colour
						self.union((x,y),(x+i,y+j))


	def find_edges(self,d=5):
		## Generates list of all edges and stores in self.edges

		# remove a square of radius 5 from each vertex
		for (x,y) in self.get_vertices():
			self.remove_square(x,y,d=d)

		# reset root and size
		self.root = dict([((x,y),(x,y)) for x in xrange(self.width()) for y in xrange(self.height())])
		self.size = dict([((x,y),1) for x in xrange(self.width()) for y in xrange(self.height())])

		# find components
		self.find_components()

		# store edge roots locally
		self.edge_roots = self.get_component_roots()

		# each new component is an edge
		edge_list = {}
		for (x,y) in self.get_vertices():
			d = d+1
			square = set([(x+i,y+j) for i in xrange(-d,d+1) for j in xrange(-d,d+1)])

			for (a,b) in square:
				root = self.get_root((a,b))
				if root in self.edge_roots:
					edge_list[(x,y)] = edge_list.get((x,y),set()) | set([root])

		edges = []
		for (u,v) in itertools.combinations (self.get_vertices(),2):
			if len(edge_list[u] & edge_list[v])>0:
				edges.append((u,v))

		# store locally
		self.edges = edges



	def midpoint(self,S):
		## Returns the euclidean midpoint a list/set of nodes
		return tuple(np.average([list(node) for node in S],axis=0).astype(int))

	def find_crossings(self,closeness=20):

		vertices = self.get_vertices()

		for (u,v) in itertools.combinations(vertices,2):
			if euclid_metric(u,v)<closeness:
				midpoint = self.midpoint([u,v])
				d = closeness/2
				square = set([(midpoint[0]+i,midpoint[1]+j) for i in xrange(-d,d+1) for j in xrange(-d,d+1)])

				crossing_edges = set()
				for (x,y) in square:
					root = self.get_root((x,y))
					if root in self.edge_root:
						crossing_edges.add(self.edge_root[root])







	def get_edges(self):
		## Returns list of edges as tuples

		if not self.edges:
			self.find_edges()	# find edges if not exist
		return self.edges

	def get_vertices(self):
		## Returns list of vertices

		if not self.vertices:
			self.find_vertices()	# find edges if not exist
		return self.vertices

	def get_crossings(self):
		## Returns list of crossings

		if not self.crossings:
			self.find_crossings()	# find crossings if not exist
		return self.crossings

	def to_graph(self):
		## Return NetworkX Graph object

		vertices = self.get_vertices() 
		edges = self.get_edges()

		G = nx.Graph()

		for (x,y) in vertices:
			G.add_node((x,y),pos=(x,self.height()-y))
		#G.add_nodes_from(vertices)
		G.add_edges_from(edges)

		return G





##################### Main #####################


if __name__ == '__main__':

	if len(sys.argv)<2:
		raise Exception('Missing input file. Call python image2graph.py input.png')

	filename = sys.argv[1]


	I = ImageArray(filename)

	print("Image dimensions: {0} x {1}".format(I.width(),I.height()))

	print("Finding components...")
	I.find_components()

	components = I.get_component_roots()

	print("Components: {0}".format(str(components)[1:-1]))


	print("Finding vertices...")
	vertices = I.get_vertices()
	print("Vertices: {0}".format(str(vertices)[1:-1]))

	print("Finding edges...")
	edges = I.get_edges()
	print("Edges: {0}".format(str(edges)[1:-1]))

	print("Generating graph...")
	G = I.to_graph()

	pos=nx.get_node_attributes(G,'pos')
	nx.draw(G,pos)

	plt.savefig("output.png") 
	plt.show()

