import numpy as np
from numpy import random
from numpy import linalg as LA
import os
import pickle

import matplotlib.pyplot as plt
from pylab import pi, cos, sin
import time

from matplotlib import rc
rc('text', usetex=False)
rc('font',size=20)
rc('font',family='serif')
rc('axes',labelsize=20)


from tqdm import tqdm
import networkx as nx
import random 

from utils.branch_sim_utils import convert_branch_coords_to_graph

from numba import int32, float32
from numba.experimental import jitclass
from numba import types
from numba import njit

from matplotlib.animation import FuncAnimation

np.random.seed(43)



### tissue 1 subroutines as numba methods ###


@njit
def subroutine_1(index_rnr, j, skip, coordinates, node, min_branch, lstep):
    np.random.seed(43)
    for j in range(len(index_rnr)):
        coord_temp = np.append(coordinates,np.array(node),axis=0)
        # highest tree label of existing nodes
        topnr = max(coord_temp[:,-1])

        # elongation happens for the first two entries of the cumulative dist:
        if index_rnr[j] == 0:
            # determine a random elongation angle
            # change the angle and coordinates of active tips
            ang_elong = np.random.uniform(0,min_branch)
            # the first entry of cumulative distribution cooresponds to a forward step, i.e. increase the local angle!
            angle[j+skip] += ang_elong
            node[j+skip] = [node[j+skip][0]+lstep*cos(angle[j+skip]),\
                            node[j+skip][1]+lstep*sin(angle[j+skip]),node[j+skip][2],node[j+skip][3]]   
        elif index_rnr[j] == 1:
            # determine a random elongation angle
            # change the angle and coordinates of active tips
            ang_elong = np.random.uniform(0,min_branch)
            # the second entry of cumulative distribution cooresponds to a backward step, i.e. decrease the local angle!
            angle[j+skip] -= ang_elong
            node[j+skip] = [node[j+skip][0]+lstep*cos(angle[j+skip]),\
                            node[j+skip][1]+lstep*sin(angle[j+skip]),node[j+skip][2],node[j+skip][3]]  
            
        # branching happens for the last two entries of the cumulative dist:
        elif index_rnr[j] >= 2:
            # determine two random angles between pi/10 and pi/2:
            ang_branch1 = np.random.uniform(min_branch,pi/2)
            ang_branch2 = np.random.uniform(min_branch,pi/2)
            # add a new branch changing the coordinates with the random angle ang_branch1: 
            angle = np.insert(angle,j+skip+1,angle[j+skip]+ang_branch1)
            node = np.insert(node,j+skip+1,[node[j+skip][0]+lstep*cos(angle[j+skip+1]), \
                            node[j+skip][1]+lstep*sin(angle[j+skip+1]),node[j+skip][3],topnr+2],axis=0)
            # change the angle and coordinates of the remaining branch with the random angle ang_branch2: 
            angle[j+skip] = angle[j+skip]-ang_branch2
            node[j+skip] = [node[j+skip][0]+lstep*cos(angle[j+skip]), \
                            node[j+skip][1]+lstep*sin(angle[j+skip]),node[j+skip][3],topnr+1]
            skip += 1
    
    return node, angle

@njit
def subroutine_2(node, coordinates, angle, node_temp, radavoid, fav):
    for j in range(len(node)):
        # self-avoidance rules (apply only if there is avoidance potential):
        if fav!=0:
            tip = node[j]

            # determine the distances between the active tip and inactive nodes
            dist = np.add(tip,-coordinates)

            # ignore distances between active tip and parent nodes, as well as within the same duct and sister branches
            for k in range(len(dist)):
                if tip[-2]==coordinates[k,-1] or tip[-1]==coordinates[k,-1] or tip[-2]==coordinates[k,-2]:
                    dist[k] = [0,0,0,0]
                # ignore distances above avoidance potential
                norm = LA.norm(dist[k][:2])
                if norm > radavoid:
                    dist[k] = [0,0,0,0]

            # sum of the distances within radavoid for the active tip
            dist_sum = np.sum(dist[:,:2],axis=0)
            # normalized vector and the final displacement vector weighted by a factor 'fav'
            norm_dis = LA.norm(dist_sum)
            if norm_dis > 0:
                displace = dist_sum/norm_dis
            else:
                displace = np.array([0,0], dtype=np.float32)

            pol = -fav*displace

            tip[0] += pol[0]
            tip[1] += pol[1]

            for k in range(len(node_temp)):
                # filter only displaced nodes
                if pol[0]!=0 and pol[1]!=0:
                    # calculate distance between the displaced node and its previous instance or its parent
                    if tip[-1]==node_temp[k][-1] or tip[-2]==node_temp[k][-1]:
                        displace_more = np.add(tip[:2],-node_temp[k][:2])
                        normalize = LA.norm(displace_more)                    
                        # update node coordinates s.t. normalized distance from previous instance is = 1                    
                        node[j][0] = node_temp[k][0]+displace_more[0]/normalize
                        node[j][1] = node_temp[k][1]+displace_more[1]/normalize
                        # update the angle of the displaced node
                        ydis = node[j][1]-node_temp[k][1]
                        xdis = node[j][0]-node_temp[k][0]
                        if xdis<0:
                            angle[j] = np.pi + np.arctan(ydis/xdis)
                        else:
                            angle[j] = np.arctan(ydis/xdis)

    return node, angle


class MamSimulation:

    def __init__(self, tmax=150):
        np.random.seed(43)
        self.Lx = 200
        self.Lz = 300
        self.lstep = 1
        self.min_branch = pi/10
        self.cutoff = 1.5*self.lstep
        self.radavoid = 10*self.lstep

        self.node = np.array([[0,self.Lz/2,0,1]], dtype=np.float32)
        self.angle = np.array([0.0], dtype=np.float32)
        self.angle_list = np.array([[0,0]], dtype=np.float32)
        self.diff = 0
        self.evolve = np.array([len(self.node)], dtype=np.float32)
        self.coordinates = self.node

        self.tmax = tmax
        
    
    def distance(self, vect, tip):
        diff = np.add(vect[:2],-tip[:2])
        distance = LA.norm(diff)
        return distance

    def prob_list(self,x,f,pb):
        # bias for forward steps
        alpha = np.float32((1-f*np.sin(x))/2)
        if alpha < 0: alpha = np.float32(0)
        if alpha > 1: alpha = np.float32(1)
        # bias for backward steps 
        beta = np.float32((1+f*np.sin(x))/2)
        if beta < 0: beta = np.float32(0)
        if beta > 1: beta = np.float32(1)

        # list of stepping probabilities:
        problist = np.array([(1-pb)*alpha,(1-pb)*beta,pb*alpha,pb*beta])
        # 'cumulative' probabilities of the list
        cum_prob = [0]+[sum(problist[:j+1]) for j in range(len(problist))]
        
        probs = {'problist':problist,'cumulative':np.array(cum_prob)}
        
        return probs
    
    def tissue1(self,prob_branch,fav,fchem):
    
        skip =0 

        # draw random numbers to decide on next jumps:
        rnr = np.random.rand(len(self.angle))
        
        # determine the cumulative distribution of the stepping probabilities:
        # (use angle values from the local angle list to impose parallel field as guidance)
        cumprob = np.array([self.prob_list(np.radians(item),fchem,prob_branch)['cumulative'] for item in self.angle_list[-(len(self.angle)):,0]])
        # determine the first entry of cumprob that is larger than rnr, and take the entry before that:
        index_rnr = np.array([np.where(cumprob[j]>rnr[j])[0][0]-1 for j in range(len(rnr))])

        node_temp = self.coordinates[-int(self.evolve[-1]):]

        for j in range(len(index_rnr)):
            coord_temp = np.append(self.coordinates,np.array(self.node),axis=0)
            # highest tree label of existing nodes
            topnr = max(coord_temp[:,-1])

            # elongation happens for the first two entries of the cumulative dist:
            if index_rnr[j] == 0:
                # determine a random elongation angle
                # change the angle and coordinates of active tips
                ang_elong = np.random.uniform(0,self.min_branch)
                # the first entry of cumulative distribution cooresponds to a forward step, i.e. increase the local angle!
                self.angle[j+skip] += ang_elong
                self.node[j+skip] = [self.node[j+skip][0]+self.lstep*cos(self.angle[j+skip]),\
                                self.node[j+skip][1]+self.lstep*sin(self.angle[j+skip]),self.node[j+skip][2],self.node[j+skip][3]]   
            elif index_rnr[j] == 1:
                # determine a random elongation angle
                # change the angle and coordinates of active tips
                ang_elong = np.random.uniform(0,self.min_branch)
                # the second entry of cumulative distribution cooresponds to a backward step, i.e. decrease the local angle!
                self.angle[j+skip] -= ang_elong
                self.node[j+skip] = [self.node[j+skip][0]+self.lstep*cos(self.angle[j+skip]),\
                                self.node[j+skip][1]+self.lstep*sin(self.angle[j+skip]),self.node[j+skip][2],self.node[j+skip][3]]  
                
            # branching happens for the last two entries of the cumulative dist:
            elif index_rnr[j] >= 2:
                # determine two random angles between pi/10 and pi/2:
                ang_branch1 = np.random.uniform(self.min_branch,pi/2)
                ang_branch2 = np.random.uniform(self.min_branch,pi/2)
                # add a new branch changing the coordinates with the random angle ang_branch1: 
                self.angle = np.insert(self.angle,j+skip+1,self.angle[j+skip]+ang_branch1)
                self.node = np.insert(self.node,j+skip+1,[self.node[j+skip][0]+self.lstep*cos(self.angle[j+skip+1]), \
                                self.node[j+skip][1]+self.lstep*sin(self.angle[j+skip+1]),self.node[j+skip][3],topnr+2],axis=0)
                # change the angle and coordinates of the remaining branch with the random angle ang_branch2: 
                self.angle[j+skip] = self.angle[j+skip]-ang_branch2
                self.node[j+skip] = [self.node[j+skip][0]+self.lstep*cos(self.angle[j+skip]), \
                                self.node[j+skip][1]+self.lstep*sin(self.angle[j+skip]),self.node[j+skip][3],topnr+1]
                skip += 1
        
        # Make a list of local angle (\varphi) values BEFORE avoidance or guidance:
        self.angle = (self.angle+pi) % (2*pi) - pi 
        
        #self.node, self.angle = subroutine_2(self.node,self.coordinates,self.angle,node_temp,self.radavoid,fav)
        for j in range(len(self.node)):
            
            # self-avoidance rules (apply only if there is avoidance potential):
            if fav!=0:
                tip = self.node[j]

                # determine the distances between the active tip and inactive nodes
                dist = np.add(tip,-self.coordinates)

                # ignore distances between active tip and parent nodes, as well as within the same duct and sister branches
                for k in range(len(dist)):
                    if tip[-2]==self.coordinates[k,-1] or tip[-1]==self.coordinates[k,-1] or tip[-2]==self.coordinates[k,-2]:
                        dist[k] = [0,0,0,0]
                    # ignore distances above avoidance potential
                    norm = LA.norm(dist[k][:2])
                    if norm > self.radavoid:
                        dist[k] = [0,0,0,0]

                # sum of the distances within radavoid for the active tip
                dist_sum = np.array(np.sum(dist[:,:2],axis=0))
                # normalized vector and the final displacement vector weighted by a factor 'fav'
                norm_dis = LA.norm(dist_sum)
                if norm_dis > 0:
                    displace = np.array(dist_sum/norm_dis)
                else:
                    displace = np.array([0,0])

                pol = -fav*displace

                tip[0] += pol[0]
                tip[1] += pol[1]

                for k in range(len(node_temp)):
                    # filter only displaced nodes
                    if pol[0]!=0 and pol[1]!=0:
                        # calculate distance between the displaced node and its previous instance or its parent
                        if tip[-1]==node_temp[k][-1] or tip[-2]==node_temp[k][-1]:
                            displace_more = np.add(tip[:2],-node_temp[k][:2])
                            normalize = LA.norm(displace_more)                    
                            # update node coordinates s.t. normalized distance from previous instance is = 1                    
                            self.node[j][0] = node_temp[k][0]+displace_more[0]/normalize
                            self.node[j][1] = node_temp[k][1]+displace_more[1]/normalize
                            # update the angle of the displaced node
                            ydis = self.node[j][1]-node_temp[k][1]
                            xdis = self.node[j][0]-node_temp[k][0]
                            if xdis<0:
                                self.angle[j] = pi + np.arctan(ydis/xdis)
                            else:
                                self.angle[j] = np.arctan(ydis/xdis)

        # tip annihilation condition:
        skipp = 0

        # list of recent nodes generated in the last 2 time steps
        checklist = [list(item) for item in self.coordinates[-int(np.sum(self.evolve[-2:])):]]

        for j in range(len(self.node)):
            tip = np.array(self.node[j-skipp])
            once = 0

            # end loop if there are no active tips left         
            if len(self.node)==0:
                break

            # annihilation due to active tip-passive node contact
            for item in self.coordinates:
                radius = self.distance(item[:2],tip[:2])
                # exclude nodes from the same duct, from the parent ..
                # ..if these nodes were generated in the last 2 time steps
                if list(item) in checklist:
                    if item[-1]!=tip[-1] and item[-1]!=tip[-2]:
                        if 0<radius<self.cutoff:
                            self.node = np.delete(self.node,j-skipp,0)
                            self.angle = np.delete(self.angle,j-skipp,0)
                            once = 1
                            skipp += 1
                elif 0<radius<self.cutoff:
                    self.node = np.delete(self.node,j-skipp,0)
                    self.angle = np.delete(self.angle,j-skipp,0)
                    once = 1
                    skipp += 1
                if once==1:
                    break     

        # save the length of node vector (to track node evolution over time)
        self.evolve = np.append(self.evolve,len(self.node))

        # set angle values to be within [-pi,pi]
        self.angle = (self.angle+pi) % (2*pi) - pi     
        # save the angles of nodes [in degrees!] including the generation number
        self.angle_list = np.append(self.angle_list,np.column_stack((np.degrees(self.angle),self.node[:,-1])),axis=0)

        # save the coordinates of all nodes
        self.coordinates = np.append(self.coordinates,np.array(self.node),axis=0) 

    def simulate(self):
        print("Starting simulation...")
        start_time = time.time()
        t = 0
        for _ in tqdm(range(self.tmax)):
            t+= 1
            if t<5 and len(self.node)>1:
                break
            if len(self.node)!=0:
                self.tissue1(0.03,-0.1,0.6)
            if len(self.node) == 0:
                break
        end_time = time.time()

        # remove duplicate first node
        self.coordinates = self.coordinates[1:]
        # get the final index of the coordinates array
        final_index = int(np.sum(self.evolve[:self.tmax]))
        self.coordinates = self.coordinates[:final_index]

        # cast evolve to int array
        self.evolve = self.evolve.astype(int)
        print(f"Branch growth simulation completed in {end_time - start_time} seconds")

        # create the graph of the final network
        G = convert_branch_coords_to_graph(self.coordinates)

        return self.coordinates, self.evolve, G



if __name__ == "__main__":

    from utils.branch_sim_utils import branch_growth_animation



    tmax = 150
    mam = MamSimulation(tmax=tmax)
    coordinates, evolve, G = mam.simulate()
    #branch_growth_animation(coordinates, evolve, f"branch_growth_{tmax}.gif")

    # print coordinates with x between 15 and 17
    print(coordinates[np.where((coordinates[:,0] > 15) & (coordinates[:,0] < 17))])

    # plot the graph
    fig, ax = plt.subplots(figsize=(15,15))
    nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), with_labels=False, node_size=10, ax=ax)
    plt.show()
