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



def subroutine_1(index_rnr, coordinates, node, min_branch, lstep, fchem, prob_branch):

    np.random.seed(43)
    
    skip = 0 


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
    """
    node, coordinates and angle are all np.ndarrays
    """
    for j in range(node.shape[0]):
        # self-avoidance rules (apply only if there is avoidance potential):
        if fav!=0:
            tip = node[j]

            # determine the distances between the active tip and inactive nodes
            dist = np.empty_like(coordinates, dtype=np.float32)
            for k in range(coordinates.shape[0]):
                for m in range(tip.shape[0]):
                    dist[k,m] = tip[m]-coordinates[k,m]

            # ignore distances between active tip and parent nodes, as well as within the same duct and sister branches
            for k in range(dist.shape[0]):
                if tip[-2]==coordinates[k,-1] or tip[-1]==coordinates[k,-1] or tip[-2]==coordinates[k,-2]:
                    for m in range(dist.shape[1]):
                        dist[k,m] = 0.0

                # ignore distances above avoidance potential
                norm = LA.norm(dist[k][:2])
                if norm > radavoid:
                    for m in range(dist.shape[1]):
                        dist[k,m] = 0.0
            
            # sum of the distances within radavoid for the active tip
            dist_sum = np.sum(dist[:,:2],axis=0)
            # normalized vector and the final displacement vector weighted by a factor 'fav'
            norm_dis = LA.norm(dist_sum)
            displace = np.zeros(2, dtype=np.float32)
            if norm_dis > 0:
                displace[0] = dist_sum[0]/norm_dis
                displace[1] = dist_sum[1]/norm_dis

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

@njit
def get_recent_coords(coordinates, evolve):
    """
    Get the indices of the recent coordinates in the coordinates array
    """
    n_elems = np.int32(np.sum(evolve[-2:]))
    start_idx = max(0, len(coordinates) - n_elems)
    
    # Initialize array of indices
    indices = np.arange(start_idx, len(coordinates), dtype=np.int32)
    
    return indices

@njit
def subroutine_3(node,coordinates,recent_indices, angle, cutoff):
    
    skipp = 0
    
    for j in range(len(node)):
        #tip = np.array(node[j-skipp])
        tip = node[j-skipp]
        once = 0

        # end loop if there are no active tips left         
        if len(node)==0:
            break
        
        # annihilation due to active tip-passive node contact
        for i, item in enumerate(coordinates):
            radius = distance(item[:2],tip[:2])
            # exclude nodes from the same duct, from the parent ..
            # ..if these nodes were generated in the last 2 time steps

            ## check if the item is in the checklist
            is_recent = False
            for idx in recent_indices:
                if idx == i:
                    is_recent = True
                    break

            if is_recent:
                if item[-1]!=tip[-1] and item[-1]!=tip[-2]:
                    if 0<radius<cutoff:
                        # create new arrays instead of using np.delete
                        node = np.concatenate((node[:j-skipp], node[j-skipp+1:]))
                        angle = np.concatenate((angle[:j-skipp], angle[j-skipp+1:]))
                        #node = np.delete(node,j-skipp,0)
                        #angle = np.delete(angle,j-skipp,0)
                        once = 1
                        skipp += 1
            elif 0<radius<cutoff:
                node = np.concatenate((node[:j-skipp], node[j-skipp+1:]))
                angle = np.concatenate((angle[:j-skipp], angle[j-skipp+1:]))    
                #node = np.delete(node,j-skipp,0)
                #angle = np.delete(angle,j-skipp,0)
                once = 1
                skipp += 1
            if once==1:
                break     
    
    return node, angle 

@njit(fastmath=True)
def distance(vect, tip):
    diff = np.add(vect[:2],-tip[:2])
    distance = LA.norm(diff)
    return distance

@njit(fastmath=True)
def prob_list(x,f,pb):
    # bias for forward steps
    alpha = np.float32(max(0, min(1, (1-f*np.sin(x))/2)))
    # bias for backward steps 
    beta = np.float32(max(0, min(1, (1+f*np.sin(x))/2)))

    # list of stepping probabilities:
    problist = np.array([(1-pb)*alpha,(1-pb)*beta,pb*alpha,pb*beta], dtype=np.float32)
    # 'cumulative' probabilities of the list
    cum_prob = np.zeros(len(problist)+1, dtype=np.float64)
    for j in range(len(problist)):
        cum_prob[j+1] = cum_prob[j] + problist[j]
    
    return cum_prob

@njit(fastmath=True)
def compute_probabilities(angle_values, fchem, prob_branch):
    radians = np.radians(angle_values)
    result = np.empty((len(radians), 5), dtype=np.float64)
    for i in range(radians.shape[0]):
        result[i] = prob_list(radians[i], fchem, prob_branch)
    return result


class MamSimulation:

    def __init__(self, tmax=150, prob_branch=0.03, fav=-0.1, fchem=0.6, graph_output=True, seed=43):
        np.random.seed(seed)
        self.Lx = 200
        self.Lz = 300
        self.lstep = 1
        self.min_branch = pi/10
        self.cutoff = 1.5*self.lstep
        self.radavoid = 10*self.lstep

        self.prob_branch = prob_branch
        self.fav = fav
        self.fchem = fchem

        self.node = np.array([[0,self.Lz/2,0,1]], dtype=np.float32)
        self.angle = np.array([0.0], dtype=np.float32)
        self.angle_list = np.array([[0,0]], dtype=np.float32)
        self.diff = 0
        self.evolve = np.array([len(self.node)], dtype=np.float32)
        self.coordinates = self.node

        self.tmax = tmax
        self.graph_output = graph_output
    # prob_branch: probability of branching
    # fav: favorability of self-avoidance
    # fchem: favorability of chemical guidance -> influences directional bias of branch growth
    def tissue1(self,prob_branch:float,fav:float,fchem:float):
        
        skip = 0 

        # draw random numbers to decide on next jumps:
        rnr = np.random.rand(len(self.angle))
        
        # determine the cumulative distribution of the stepping probabilities:
        # (use angle values from the local angle list to impose parallel field as guidance)
        angle_slice = self.angle_list[-(len(self.angle)):,0]
        cumprob = compute_probabilities(angle_slice, fchem, prob_branch)
        ###cumprob = np.array([prob_list(np.radians(item),fchem,prob_branch)for item in self.angle_list[-(len(self.angle)):,0]])
        
        # determine the first entry of cumprob that is larger than rnr, and take the entry before that:
        index_rnr = np.array([np.where(cumprob[j]>rnr[j])[0][0]-1 for j in range(len(rnr))])

        node_temp = self.coordinates[-int(self.evolve[-1]):]

        #self.node, self.angle = subroutine_1(index_rnr, self.coordinates, self.node, self.min_branch, self.lstep, fchem, prob_branch)
        
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
        
        assert isinstance(self.node, np.ndarray)
        assert isinstance(self.angle, np.ndarray)
        assert isinstance(self.coordinates, np.ndarray)
        self.node, self.angle = subroutine_2(self.node.astype(np.float32),self.coordinates.astype(np.float32),self.angle.astype(np.float32),node_temp.astype(np.float32),self.radavoid,fav)
        # for j in range(len(self.node)):
            
        #     # self-avoidance rules (apply only if there is avoidance potential):
        #     if fav!=0:
        #         tip = self.node[j]

        #         # determine the distances between the active tip and inactive nodes
        #         dist = np.add(tip,-self.coordinates)

        #         # ignore distances between active tip and parent nodes, as well as within the same duct and sister branches
        #         for k in range(len(dist)):
        #             if tip[-2]==self.coordinates[k,-1] or tip[-1]==self.coordinates[k,-1] or tip[-2]==self.coordinates[k,-2]:
        #                 dist[k] = [0,0,0,0]
        #             # ignore distances above avoidance potential
        #             norm = LA.norm(dist[k][:2])
        #             if norm > self.radavoid:
        #                 dist[k] = [0,0,0,0]

        #         # sum of the distances within radavoid for the active tip
        #         dist_sum = np.array(np.sum(dist[:,:2],axis=0))
        #         # normalized vector and the final displacement vector weighted by a factor 'fav'
        #         norm_dis = LA.norm(dist_sum)
        #         if norm_dis > 0:
        #             displace = np.array(dist_sum/norm_dis)
        #         else:
        #             displace = np.array([0,0])

        #         pol = -fav*displace

        #         tip[0] += pol[0]
        #         tip[1] += pol[1]

        #         for k in range(len(node_temp)):
        #             # filter only displaced nodes
        #             if pol[0]!=0 and pol[1]!=0:
        #                 # calculate distance between the displaced node and its previous instance or its parent
        #                 if tip[-1]==node_temp[k][-1] or tip[-2]==node_temp[k][-1]:
        #                     displace_more = np.add(tip[:2],-node_temp[k][:2])
        #                     normalize = LA.norm(displace_more)                    
        #                     # update node coordinates s.t. normalized distance from previous instance is = 1                    
        #                     self.node[j][0] = node_temp[k][0]+displace_more[0]/normalize
        #                     self.node[j][1] = node_temp[k][1]+displace_more[1]/normalize
        #                     # update the angle of the displaced node
        #                     ydis = self.node[j][1]-node_temp[k][1]
        #                     xdis = self.node[j][0]-node_temp[k][0]
        #                     if xdis<0:
        #                         self.angle[j] = pi + np.arctan(ydis/xdis)
        #                     else:
        #                         self.angle[j] = np.arctan(ydis/xdis)


        # list of recent nodes generated in the last 2 time steps
        recent_indices = get_recent_coords(self.coordinates, self.evolve)
        self.node, self.angle = subroutine_3(self.node,self.coordinates,recent_indices,self.angle,self.cutoff)

        # # tip annihilation condition:
        # skipp = 0
        
        # for j in range(len(self.node)):
        #     tip = np.array(self.node[j-skipp])
        #     once = 0

        #     # end loop if there are no active tips left         
        #     if len(self.node)==0:
        #         break

        #     # annihilation due to active tip-passive node contact
        #     for item in self.coordinates:
        #         radius = distance(item[:2],tip[:2])
        #         # exclude nodes from the same duct, from the parent ..
        #         # ..if these nodes were generated in the last 2 time steps
        #         if list(item) in checklist:
        #             if item[-1]!=tip[-1] and item[-1]!=tip[-2]:
        #                 if 0<radius<self.cutoff:
        #                     self.node = np.delete(self.node,j-skipp,0)
        #                     self.angle = np.delete(self.angle,j-skipp,0)
        #                     once = 1
        #                     skipp += 1
        #         elif 0<radius<self.cutoff:
        #             self.node = np.delete(self.node,j-skipp,0)
        #             self.angle = np.delete(self.angle,j-skipp,0)
        #             once = 1
        #             skipp += 1
        #         if once==1:
        #             break     

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
                self.tissue1(self.prob_branch,self.fav,self.fchem)
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
        G = convert_branch_coords_to_graph(self.coordinates) if self.graph_output else None

        return self.coordinates, self.evolve, G



def generate_branches(tmax, seeds: list[int], prob_branch:float, fav:float, fchem:float):
    output_folder = "output/branch_structures"
    os.makedirs(output_folder, exist_ok=True)

    # create the subfolder for the current parameters
    subfolder = os.path.join(output_folder, f"prob_branch={prob_branch}_fav={fav}_fchem={fchem}_tmax={tmax}")
    os.makedirs(subfolder, exist_ok=True)

    for seed in tqdm(seeds):
        mam = MamSimulation(tmax=tmax, prob_branch=prob_branch, fav=fav, fchem=fchem, graph_output=False, seed=seed)
        coordinates, evolve, G = mam.simulate()
        np.save(os.path.join(subfolder, f"coordinates_{seed}.npy"), coordinates)
        np.save(os.path.join(subfolder, f"evolve_{seed}.npy"), evolve)
    
    print(f"Generated {len(seeds)} branches with parameters: prob_branch={prob_branch}, fav={fav}, fchem={fchem}, tmax={tmax}")
    return



if __name__ == "__main__":


    seeds = list(range(50))
    generate_branches(tmax=500, seeds=seeds, prob_branch=0.03, fav=-0.1, fchem=0.6)
    # from utils.branch_sim_utils import branch_growth_animation, plot_branch_network

    # tmax = 150
    # mam = MamSimulation(tmax=tmax, graph_output=False)
    # coordinates, evolve, G = mam.simulate()


    # # I want to define a subgraph of the final graph based on a square region drawn in space
    # # the square should be centered in the middle of the final graph
    # # in the middle of x direction and about 0.5 of width
    # # in the middle of y direction and about 0.8

    # width = np.max(coordinates[:,0]) - np.min(coordinates[:,0])
    # height = np.max(coordinates[:,1]) - np.min(coordinates[:,1])

    # # get the center of the graph
    # center_x = np.min(coordinates[:,0]) + width/2
    # center_y = np.min(coordinates[:,1]) + height/2

    # # define the square
    # square_width = 0.5*width
    # square_height = 0.8*height

    # square_x_min = center_x - square_width/2
    # square_x_max = center_x + square_width/2
    # square_y_min = center_y - square_height/2
    # square_y_max = center_y + square_height/2



    # ax = plot_branch_network(coordinates, evolve, end_time=1000, show=False)
    # ax.plot([square_x_min, square_x_max, square_x_max, square_x_min, square_x_min],
    #          [square_y_min, square_y_min, square_y_max, square_y_max, square_y_min],
    #          color='red')
    # plt.show()
    

    # plot_branch_network(coordinates, evolve, end_time=1000)
    # if tmax == 150:
    #     # load the test coordinates and evolve
    #     coordinates_test = np.load("coordinates_150_test.npy")
    #     evolve_test = np.load("evolve_150_test.npy")

    #     # compare the coordinates and evolve, check if all elements are the same
    #     print(f"Coordinates match: {np.allclose(coordinates, coordinates_test)}")
    #     print(f"Largest difference: {np.max(np.abs(coordinates - coordinates_test))}")
    #     print("Difference distributions:")
    #     print(np.histogram(np.abs(coordinates - coordinates_test), bins=10))
    #     print(f"Evolve match: {np.allclose(evolve, evolve_test)}")

    #     # check the shapes 
    #     print(f"Coordinates shape match: {coordinates.shape == coordinates_test.shape}")
    #     print(f"Branch naming match: {np.all(coordinates[:, 2:] == coordinates_test[:,2:])}")

    


