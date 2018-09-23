class NewVerbInventory:
	
	def __init__(self, clusters_file, verbs_file):
		self.clusters = self.from_files(clusters_file, verbs_file)
	
	def from_files(self, clusters_file, verbs_file):
		clusters_list = []
		with open(clusters_file) as f:
			clusts = f.readlines()
		for clust in clusts:
			c = clust.strip().split('\t')
			clusters_list.append(Cluster(c[0], c[1], '\t'.join(c[2:]), verbs_file))
		return clusters_list
	
	def remove_clusters(removal_condition):
		self.clusters = [c for c in self.getClusters() if removal_condition(c)]
	
	def getClusters(self):
		return self.clusters
		
class Cluster:
	type_2_arg = None
	
	def __init__(self, i, n, args, verbs_file):
		self.id = i
		self.name = n
		allowed_args = ['agent', 'patient', 'experiencer', 'theme']
		
		if '2)' in args:
			type_2_start = args.index('2)')
			type_1 = args[3 : type_2_start]
			type_2 = args[type_2_start + 3:]
			self.type_2_arg = [Argument(t) for t in type_2.strip().split('\t') if t.strip().split('(')[0].lower().strip() in allowed_args]
			self.type_1_arg = [Argument(t) for t in type_1.strip().split('\t') if t.strip().split('(')[0].lower().strip() in allowed_args]
		else:
			#print([t.strip().split('(')[0].lower() for t in args.strip().split('\t')])
			self.type_1_arg = [Argument(t) for t in args.strip().split('\t') if t.strip().split('(')[0].lower().strip() in allowed_args]
		with open(verbs_file) as f:
			vbs = f.readlines()
		self.verbs = [Verb(vb) for vb in vbs if vb.split(',')[7].strip() == self.id]
	
	def getId(self):
		return self.id
	
	def getName(self):
		return self.name
		
	def getType1Arg(self):
		return self.type_1_arg
	
	def getType2Arg(self):
		return self.type_2_arg
		
	def getVerbs(self):
		return self.verbs
		
	def printArgs(self):
		print([str(arg.getRole()) + '(' + str(arg.getPreferences()) + ')' for arg in self.getType1Arg()])
		if self.getType2Arg() != None:
			print([str(arg.getRole()) + '(' + str(arg.getPreferences()) + ')' for arg in self.getType2Arg()])
	
	def printVerbs(self):
		print([v.getPBFrameset() for v in self.getVerbs()])
		
		
class Argument:

	def __init__(self, args):
	
		args = args.strip().split(' ')
		self.role = args[0].lower()
		self.preferences = [p.lower().replace('(', '').replace(')', '').replace('|', '') 
			for p in args[1:] if p.lower().replace('(', '').replace(')', '').replace('|', '') != '']
	
	def getRole(self):
		return self.role
		
	def getPreferences(self):
		return self.preferences
		
class Verb:

	def __init__(self, verb):
		verb = verb.strip().split(',')
		self.PB_frameset = verb[0][3:]
		self.example = verb[2]
		self.gloss = verb[3]
		self.babel_synset = verb[5]
		self.syntactic_deviation = verb[8]
		self.argument_info = verb[9]
		self.prototypicality = verb[10]
		self.lemmas = verb[11]
	
	def getPBFrameset(self):
		return self.PB_frameset
		
	def getExample(self):
		return self.example
		
	def getGloss(self):
		return self.gloss
		
	def getBabelSynset(self):
		return self.babel_synset
		
	def getSyntacticDeviation(self):
		return self.syntactic_deviation
		
	def getArgumentInfo(self):
		return self.argument_info
		
	def getPrototypicality(self):
		return self.prototypicality
		
	def getLemmas(self):
		return self.lemmas
		
	
		