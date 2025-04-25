import numpy as np
from numpy import random
from numpy import linalg as LA
import os

import matplotlib.pyplot as plt
from pylab import *

from matplotlib import rc
rc('text', usetex=False)
rc('font',size=20)
rc('font',family='serif')
rc('axes',labelsize=20)

from tqdm import tqdm
import networkx as nx
import random 

from numba import int32, float32
from numba.experimental import jitclass
from numba import types


class MamSimulation:

    def __init__(self, tmax=150):
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
        print(f"Simulation completed in {end_time - start_time} seconds")
    

    def plot_network(self, save_path=None):
        fig, ax = plt.subplots(figsize=(9,9))
        ms = 1.5

        def step(till,evolve):
            step = np.sum(evolve[:till])
            return int(step)
        
        # choose time1 to plot until a certain timepoint (time1=tmax for the complete network)
        time1 = self.tmax-1
        time2 = time1+1

        self.x = [item[0] for item in self.coordinates[:step(time2,self.evolve)]]
        self.y = [item[1] for item in self.coordinates[:step(time2,self.evolve)]]


        ax.plot(self.x,self.y,'o', color='steelblue', markersize=ms)    
        ax.plot(self.x[0],self.y[0],'x',color='firebrick',markersize=8)

        # plot active tips with different color
        ax.plot(self.x[step(time1,self.evolve):step(time2,self.evolve)],self.y[step(time1,self.evolve):step(time2,self.evolve)],'o',color='C1',markersize=ms+1.5)


        plt.tick_params(    
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,   
            left=False,
            labelleft=False,
            labelbottom=False) 
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
    
    def convert_to_graph(self, save_path=None, distance_threshold=1.5):
        """
        Add edges between nodes that are within a certain distance of each other
        """
        node_coords = [(xi,yi) for xi,yi in zip(self.x, self.y)]
        # remove the duplicate first node
        node_coords = node_coords[1:]

        indices = [i for i in range(len(node_coords))]
        G = nx.Graph()
        for i in indices:
            G.add_node(i, pos=node_coords[i])
        for i in indices:
            for j in indices:
                if i != j:
                    if LA.norm(np.array(node_coords[i]) - np.array(node_coords[j])) < distance_threshold:
                        G.add_edge(i, j)
        
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos=pos, with_labels=False, node_size=10)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
        self.G = G
        return G
    


if __name__ == "__main__":

    np.random.seed(43)
    
    output_dir = "output_mam_final"
    os.makedirs(output_dir, exist_ok=True)
    mam = MamSimulation(tmax=150)
    mam.simulate()
    mam.plot_network(os.path.join(output_dir, "network_structure.png"))
    mam.convert_to_graph(os.path.join(output_dir, "network_graph.png"))

    from utils import graph_to_model_format
    nodes, neighbors = graph_to_model_format(mam.G)
    print(nodes.size)
    pos = nx.get_node_attributes(mam.G, 'pos')
    from main_tree import IsingModel, simulate_ising_model, animate_ising_model

    n_iter = 10_000
    J = 2.0
    
    # Create a denser sampling around the critical region
    temps_low = np.linspace(0.1, 2.0, 30)  # More points in the lower temperature region
    temps_high = np.linspace(2.1, 5.0, 20)  # Fewer points in the higher temperature region
    temps_specific = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0,3.5,4.0])
    temps = np.sort(np.unique(np.concatenate([temps_low, temps_high, temps_specific])))
    
    # create a figure with a subplot for each temperature in a grid 4x4
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs_index = 0
    magnetization = []
    energy = []
    models = []
    for T in temps:
        model = IsingModel(nodes, neighbors, temperature=T, J=J, pos=pos)
        model = simulate_ising_model(model, n_iterations=n_iter)
        spins_final = model.spins_final
        animate_ising_model(model, output_dir=output_dir, T=round(T, 2))
        magnetization.append(model.magnetization_final)
        energy.append(model.energy_final)
        models.append(model)
        
        # plot the final graph with the spins as colors
        if T in temps_specific:
            pos = nx.get_node_attributes(mam.G, 'pos')
            color_map = ["blue" if spin == 1 else "red" for spin in spins_final]
            nx.draw(mam.G, pos=pos, with_labels=False, node_size=10, node_color=color_map, ax=axs[axs_index//4, axs_index%4])
            axs[axs_index//4, axs_index%4].set_title(f"T={T}")
            axs[axs_index//4, axs_index%4].axis('off')
            axs_index += 1
        

    # add a title to the figue 
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"n_iter{n_iter}_J{J}_spins.png"))
    #plt.show()


    plt.figure(figsize=(10, 5))
    sp = plt.subplot(1, 2, 1)
    sp.scatter(temps, energy, label='energy', marker='o', color="IndianRed")
    sp.set_xlabel("Temperature")
    sp.set_ylabel("Energy")
    sp = plt.subplot(1, 2, 2)
    sp.scatter(temps, magnetization, label='magnetization', marker='o', color="RoyalBlue")
    sp.set_xlabel("Temperature")
    sp.set_ylabel("Magnetization")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"n_iter{n_iter}_J{J}_magnetization_energy.png"))
    plt.show()


