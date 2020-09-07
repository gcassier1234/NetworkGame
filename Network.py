import math
import time
import heapq
import numpy as np
from scipy import optimize

"""
La capacité aporter à un liens (Link) lorsqu'un operateur construit une connection, plusieur operateur peuvent construire leur propre connection sur un même liens,
ainsi si deux operateurs construisent une connection sur un liens la capacité de ce dernier sera de 20.
"""
CAPA =10 


class Zone:
    def __init__(self, zoneId):
        self.zoneId = zoneId
        self.lat = 0
        self.lon = 0
        self.destList = []


class Node:
    '''
    This class has attributes associated with any node
    '''
    def __init__(self, Id):
        self.Id = Id
        self.lat = 0
        self.lon = 0
        self.outLinks = []
        self.inLinks = []
        self.label = float("inf")
        self.pred = ""
        self.inDegree = 0
        self.outDegree = 0
        self.order = 0 # Topological order
        self.wi = 0.0 # Weight of the node in Dial's algorithm
        self.xi = 0.0 # Toal flow crossing through this node in Dial's algorithm


class Link:
    '''
    This class has attributes associated with any link
    '''
    def __init__(self, tailNodeID, headNodeID, capacity=CAPA, length=10, fft=5, beta=1, alpha=1, speedLimit=50):
        self.tailNode = tailNodeID
        self.headNode = headNodeID
        self.capacity = float(capacity) # veh per hour
        self.length = float(length) 
        self.fft = float(fft) # Free flow travel time (min)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.speedLimit = float(speedLimit)
        #self.toll = float(_tmpIn[9])
        #self.linkType = float(_tmpIn[10])
        self.flow = 0.0
        self.cost =  float(fft) #float(fft)*(1 + float(alpha)*math.pow((float(speedLimit)/float(capacity)), float(beta)))
        self.logLike = 0.0
        self.reasonable = True # This is for Dial's stochastic loading
        self.wij = 0.0 # Weight in the Dial's algorithm
        self.xij = 0.0 # Total flow on the link for Dial's algorithm


class Demand:
    def __init__(self, fromZone, toZone, demand):
        self.fromZone = fromZone
        self.toNode = toZone
        self.demand = float(demand)
        
class Network:
    def __init__(self, Nodes, Links, Demands, nOperator):
        self.Nodes        = Nodes # dictionary of Nodes where the index is the Id of the Node
        self.Trips        = Demands  # dictionary where the index of a link is define by the Id of the pair of nodes it connex, first comes the fromZone.
        self.Links        = Links # dictionary where the index of a link is define by the Id of the pair of nodes it connex, first comes the tail.
        
        ID                = np.identity(len(self.Nodes), dtype = "int")
        self.Connections  = np.tile(ID,(nOperator,1,1))
        self.nOperator    = nOperator
        self.Zones        = {}
        self.originZones  = set([k[0] for k in self.Trips])
        
        self.connectNodes()
        self.collectTravelZones()
        
        
    def connectNodes(self):
        for NodePair in self.Links.keys():
            
            self.Connections[NodePair] = 1
            if NodePair[1] not in self.Nodes[NodePair[0]].outLinks:
                self.Nodes[NodePair[0]].outLinks.append(NodePair[1])
            if NodePair[0] not in self.Nodes[NodePair[1]].inLinks:
                self.Nodes[NodePair[1]].inLinks.append(NodePair[0])
    
    def collectTravelZones(self):
    
        for ZonePair in self.Trips.keys():
            
            if ZonePair[0] not in self.Zones:
                self.Zones[ZonePair[0]] = Zone([ZonePair[0]])
            if ZonePair[1] not in self.Zones:
                self.Zones[ZonePair[1]] = Zone([ZonePair[1]])
            if ZonePair[1] not in self.Zones[ZonePair[0]].destList:
                self.Zones[ZonePair[0]].destList.append(ZonePair[1])
    
    
    def addLink(self, link):
        assert not((link.tailNode,link.headNode) in self.Links.keys()), "tentative d'ajout d'un arc déjà existant"
        self.Nodes[link.tailNode].outLinks.append(link.headNode)
        self.Nodes[link.headNode].inLinks.append(link.tailNode)
        self.Links.update({(link.tailNode,link.headNode):link})
        
    def rewardOp(self, Operator):
        reward = 0
        #print("##########"+"Rewardcalculation for operator {}".format(Operator)+"##########")
        for (tailNode, headNode) in zip(list(np.nonzero(self.Connections[Operator])[0]), list(np.nonzero(self.Connections[Operator])[1])): 
            
            if tailNode != headNode :
                
                nOpertorOnLink = np.count_nonzero(self.Connections[:,tailNode,headNode])
                reward += self.Links[(tailNode, headNode)].flow/nOpertorOnLink
                """
                print ("tail : {}".format(tailNode))
                print ("head : {}".format(headNode))
                print ("flow : {}".format(self.Links[(tailNode, headNode)].flow))
                print ("apport : {}".format(self.Links[(tailNode, headNode)].flow/nOpertorOnLink))
                print("-----------------------------------")
                """
        return reward
    
    def rewardAll(self ):
        self.assignment("deterministic", "FW")
        return np.array([self.rewardOp(operator) for operator in range(self.nOperator)])
    
    def addConnection(self, Operator, tailNode, headNode):
        #print("{} is creating a connection form {} to {}".format(Operator, tailNode, headNode))
        assert self.Connections[Operator, tailNode, headNode] == 0, "Un operateur essaye de construire une connection déjà existante"
        self.Connections[Operator, tailNode, headNode] = 1
        if (tailNode, headNode) in self.Links.keys() :
            self.Links[tailNode, headNode].capacity += CAPA
        else :
            self.addLink(Link(tailNode, headNode))
            
    def reset(self):
        ID                = np.identity(len(self.Nodes), dtype = "int")
        self.Connections  = np.tile(ID,(self.nOperator,1,1))
        self.Links        = {}
        
        for node in self.Nodes.values() :
            
            node.outLinks = []
            node.inLinks = []
            
            
    
    
    
            
            
        
        
        
        
        
    def DijkstraHeap(self,origin):
        '''
        Calcualtes shortest path from an origin to all other destinations.
        The labels and preds are stored in node instances.
        '''
        for n in self.Nodes:
            self.Nodes[n].label = float("inf")
            self.Nodes[n].pred = ""
        self.Nodes[origin].label = 0.0
        self.Nodes[origin].pred = "NA"
        SE = [(0, origin)]
        while SE:
            currentNode = heapq.heappop(SE)[1]
            currentLabel = self.Nodes[currentNode].label
            for toNode in self.Nodes[currentNode].outLinks:
                link = (currentNode, toNode)
                newNode = toNode
                newPred =  currentNode
                existingLabel = self.Nodes[newNode].label
                newLabel = currentLabel + self.Links[link].cost
                if newLabel < existingLabel:
                    heapq.heappush(SE, (newLabel, newNode))
                    self.Nodes[newNode].label = newLabel
                    self.Nodes[newNode].pred = newPred
    
    def updateTravelTime(self,):
        '''
        This method updates the travel time on the links with the current flow
        '''
        for l in self.Links:
            self.Links[l].cost = self.Links[l].fft*(1 + self.Links[l].alpha*math.pow((self.Links[l].flow*1.0/self.Links[l].capacity), self.Links[l].beta))
    
    
    def findAlpha(self,x_bar):
        from scipy.optimize import fsolve
        '''
        This uses unconstrained optimization to calculate the optimal step size required
        for Frank-Wolfe Algorithm
    
        ******************* Need to be revised: Currently not working.**********************************************
        '''
        #alpha = 0.0
    
    
        def df(alpha):
            sum_derivative = 0 ## this line is the derivative of the objective function.
            for l in self.Links:
                tmpFlow = (self.Links[l].flow + alpha*(x_bar[l] - self.Links[l].flow))
                #print("tmpFlow", tmpFlow)
                tmpCost = self.Links[l].fft*(1 + self.Links[l].alpha*math.pow((tmpFlow*1.0/self.Links[l].capacity), self.Links[l].beta))
                sum_derivative = sum_derivative + (x_bar[l] - self.Links[l].flow)*tmpCost
            return sum_derivative
        sol = optimize.root(df, np.array([0.1]))
        sol2 = fsolve(df, np.array([0.1]))
        #print(sol.x[0], sol2[0])
        return max(0.1, min(1, sol2[0]))
        '''
        def int(alpha):
            tmpSum = 0
            for l in linkSet:
                tmpFlow = (linkSet[l].flow + alpha*(x_bar[l] - linkSet[l].flow))
                tmpSum = tmpSum + linkSet[l].fft*(tmpFlow + linkSet[l].alpha * (math.pow(tmpFlow, 5) / math.pow(linkSet[l].capacity, 4)))
            return tmpSum
    
        bounds = ((0, 1),)
        init = np.array([0.7])
        sol = optimize.minimize(int, x0=init, method='SLSQP', bounds = bounds)
    
        print(sol.x, sol.success)
        if sol.success == True:
            return sol.x[0]#max(0, min(1, sol[0]))
        else:
            return 0.2
        '''
    
    def tracePreds(self,dest):
        '''
        This method traverses predecessor nodes in order to create a shortest path
        '''
        prevNode = self.Nodes[dest].pred
        spLinks = []
        while self.Nodes[dest].pred != "NA":
            """
            spLinks.append((prevNode, dest))
            dest = prevNode
            prevNode = self.Nodes[dest].pred
            """
            try :
                spLinks.append((prevNode, dest))
                dest = prevNode
                prevNode = self.Nodes[dest].pred
            except :
                spLinks=[]
                break
            
        return spLinks
    
    
    
    
    def loadAON(self):
        '''
        This method produces auxiliary flows for all or nothing loading.
        '''
        x_bar = {l: 0.0 for l in self.Links}
        SPTT = 0.0
        for r in self.originZones:
            self.DijkstraHeap(r)
            for s in self.Zones[r].destList:
                try:
                    dem = self.Trips[r, s].demand
                except KeyError:
                    dem = 0.0
                SPTT = SPTT + self.Nodes[s].label * dem
                if r != s:
                    for spLink in self.tracePreds(s):
                        x_bar[spLink] = x_bar[spLink] + dem
        return SPTT, x_bar
    
    def findReasonableLinks(self):
        for l in self.Links:
            if self.Nodes[l[1]].label > self.Nodes[l[0]].label:
                self.Links[l].reasonable = True
            else:
                self.Links[l].reasonable = False
    
    def computeLogLikelihood(self):
        '''
        This method computes link likelihood for the Dial's algorithm
        '''
        for l in self.Links:
            if self.Links[l].reasonable == True: # If reasonable link
                self.Links[l].logLike = math.exp(self.Nodes[l[1]].label - self.Nodes[l[0]].label - self.Links[l].cost)
    
    
    def topologicalOrdering(self,):
        '''
        * Assigns topological order to the nodes based on the inDegree of the node
        * Note that it only considers reasonable links, otherwise graph will be acyclic
        '''
        for e in self.Links:
            if self.Links[e].reasonable == True:
                    self.Nodes[e[1]].inDegree = self.Nodes[e[1]].inDegree + 1
        order = 0
        SEL = [k for k in self.Nodes if self.Nodes[k].inDegree == 0]
        while SEL:
            i = SEL.pop(0)
            order = order + 1
            self.Nodes[i].order = order
            for j in self.Nodes[i].outLinks:
                if self.Links[i, j].reasonable == True:
                    self.Nodes[j].inDegree = self.Nodes[j].inDegree - 1
                    if self.Nodes[j].inDegree == 0:
                        SEL.append(j)
        if order < len(self.Nodes):
            print("the network has cycle(s)")
    
    def resetDialAttributes(self,):
        for n in self.Nodes:
            self.Nodes[n].inDegree = 0
            self.Nodes[n].outDegree = 0
            self.Nodes[n].order = 0
            self.Nodes[n].wi = 0.0
            self.Nodes[n].xi = 0.0
        for l in self.Links:
            self.Links[l].logLike = 0.0
            self.Links[l].reasonable = True
            self.Links[l].wij = 0.0
            self.Links[l].xij = 0.0
    
    
    
    def DialLoad(self,):
        '''
        This method runs the Dial's algorithm and prepare a stochastic loading.
        '''
        self.resetDialAttributes()
        x_bar = {l: 0.0 for l in self.Links}
        for r in self.originZones:
            self.DijkstraHeap(r)
            self.findReasonableLinks()
            self.topologicalOrdering()
            self.computeLogLikelihood()
    
            '''
            Assigning weights to nodes and links
            '''
            order = 1
            while (order <= len(self.Nodes)):
                i = [k for k in self.Nodes if self.Nodes[k].order == order][0] # Node with order no equal to current order
                if order == 1:
                    self.Nodes[i].wi = 1.0
                else:
                    self.Nodes[i].wi = sum([self.Links[k, i].wij for k in self.Nodes[i].inLinks if self.Links[k, i].reasonable == True])
                for j in self.Nodes[i].outLinks:
                    if self.Links[i, j].reasonable == True:
                        self.Links[i, j].wij = self.Nodes[i].wi*self.Links[i, j].logLike
                order = order + 1
            '''
            Assigning load to nodes and links
            '''
            order = len(self.Nodes) # The loading works in reverse direction
            while (order >= 1):
                j = [k for k in self.Nodes if self.Nodes[k].order == order][0]  # Node with order no equal to current order
                try:
                    dem = self.tripSet[r, j].demand
                except KeyError:
                    dem = 0.0
                self.Nodes[j].xj = dem + sum([self.Links[j, k].xij for k in self.Nodes[j].outLinks if self.Links[j, k].reasonable == True])
                for i in self.Nodes[j].inLinks:
                    if self.Links[i, j].reasonable == True:
                        self.Links[i, j].xij = self.Nodes[j].xj * (self.Links[i, j].wij / self.Nodes[j].wi)
                order = order - 1
            for l in self.Links:
                if self.Links[l].reasonable == True:
                    x_bar[l] = x_bar[l] + self.Links[l].xij
    
        return x_bar
    
    
    
    def assignment(self,loading, algorithm, accuracy = 0.01, maxIter=1000):
        '''
        * Performs traffic assignment
        * Type is either deterministic or stochastic
        * Algorithm can be MSA or FW
        * Accuracy to be given for convergence
        * maxIter to stop if not converged
        '''
        it = 1
        gap = float("inf")
        x_bar = {l: 0.0 for l in self.Links}
        startP = time.time()
        while gap > accuracy:
            if algorithm == "MSA" or it < 2:
                alpha = (1.0/it)
            elif algorithm == "FW":
                alpha = self.findAlpha(x_bar)
                #print("alpha", alpha)
            else:
                print("Terminating the program.....")
                print("The solution algorithm ", algorithm, " does not exist!")
            prevLinkFlow = np.array([self.Links[l].flow for l in self.Links])
            for l in self.Links:
                self.Links[l].flow = alpha*x_bar[l] + (1-alpha)*self.Links[l].flow
            self.updateTravelTime()
            if loading == "deterministic":
                SPTT, x_bar = self.loadAON()
                #print([self.Links[a].flow * self.Links[a].cost for a in self.Links])
                TSTT = round(sum([self.Links[a].flow * self.Links[a].cost for a in self.Links]), 3)
                SPTT = round(SPTT, 3)
                gap = round(abs((TSTT / SPTT) - 1), 5)
                # print(TSTT, SPTT, gap)
                if it == 1:
                    gap = gap + float("inf")
            elif loading == "stochastic":
                x_bar = self.DialLoad()
                currentLinkFlow = np.array([self.Links[l].flow for l in self.Links])
                change = (prevLinkFlow -currentLinkFlow)
                if it < 3:
                    gap = gap + float("inf")
                else:
                    gap = round(np.linalg.norm(np.divide(change, prevLinkFlow,  out=np.zeros_like(change), where=prevLinkFlow!=0)), 2)
    
            else:
                print("Terminating the program.....")
                print("The loading ", loading, " is unknown")
    
            it = it + 1
            if it > maxIter:
                #print("The assignment did not converge with the desired gap and max iterations are reached")
                #print("current gap ", gap)
                break
        #print("Assignment took", time.time() - startP, " seconds")
        #print("assignment converged in ", it, " iterations")