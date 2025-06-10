import os
import sys
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))



from main_tree import calc_neighbor_sum
def test_calc_neighbor_sum():
    spins = np.array([1, 1, 1, 1, 1])
    neighbors = np.array([[1, 2], [0, 2], [0, 1], [0, 4], [3, 0]])
    i = 0
    J = 1.0
    beta = 0.2
    n_sum = calc_neighbor_sum(i, spins, neighbors)
    assert n_sum == 2

    neighbors[i,:] = -1
    n_sum = calc_neighbor_sum(i, spins, neighbors)
    assert n_sum == 0


from main_tree import calc_hamiltonian, calc_energy_diff

# def test_calc_energy_diff():
#     spins = np.array([1, 1, 1, 1, 1])
#     neighbors = np.array([[1, 2], [0, 2], [0, 1], [0, 4], [3, 0]])
#     J = 1.0
#     beta = 0.2
#     i = 0
#     energy_diff_method = calc_energy_diff(spins, neighbors, J, i)
#     hamil_before = calc_hamiltonian(spins, neighbors, J)
#     spins[i] *= -1
#     hamil_after = calc_hamiltonian(spins, neighbors, J)
#     energy_diff_method_2 = hamil_after - hamil_before
#     assert energy_diff_method == energy_diff_method_2


from main_tree import calc_hamiltonian
def test_calc_hamiltonian():
    spins = np.array([1, 1, 1, 1, 1])
    neighbors = np.array([[1, 2], [0, 2], [0, 1], [0, 4], [3, 0]])
    J = 1.0
    beta = 0.2
    hamil = calc_hamiltonian(spins, neighbors, J)

    H = 0
    for i in range(0,spins.shape[0]):
        neighs_sum = calc_neighbor_sum(i,spins, neighbors=neighbors)
        if np.isnan(neighs_sum):
            print(f"NaN neighbor sum at node {i}")
            continue
        H += -J * spins[i] * neighs_sum
    H /= 2

    assert hamil == H
    
    ### check that the hamiltonian is the same for a certain config and the same config with a column of -1 appended
    spins_2 = np.array([1, 1, 1, 1, 1])
    neighbors_2 = np.array([[1, 2], [0, 2], [0, 1], [0, 4], [3, 0]])
    hamil_2 = calc_hamiltonian(spins_2, neighbors_2, J)
    spins_2 = np.array([1, 1, 1, 1, 1])
    neighbors_2 = np.array([[1, 2,-1], [0, 2,-1], [0, 1,-1], [0, 4,-1], [3, 0,-1]])
    hamil_2_2 = calc_hamiltonian(spins_2, neighbors_2, J)
    assert hamil_2 == hamil_2_2


def calc_hamiltonian_lattice(spins, J):
    H = 0
    for i in range(0,spins.shape[0]):
        for j in range(0,spins.shape[1]):
            neighs_sum = (
                spins[i, (j+1) % spins.shape[1]] +
                spins[i, (j-1) % spins.shape[1]] +
                spins[(i+1) % spins.shape[0], j] +
                spins[(i-1) % spins.shape[0], j]
            )
            H += -J * spins[i,j] * neighs_sum
    return H / 2


from utils.gen_utils import create_lattice
def test_data_structure():
    size = 10
    J = 1.0

    # test the hamiltonian calculation on identical structures but differen representations
    lattice = np.ones((size, size))

    nodes, neighbors = create_lattice(size)

    # calculate the hamiltonian
    hamil_lattice = calc_hamiltonian(nodes, neighbors, J)
    hamil_array = calc_hamiltonian_lattice(lattice, J)



    assert hamil_lattice == hamil_array


from main_tree import wolff_step
def test_wolff_step():
    # test if the wolf step is the same as the wolff step with -1 columns appended
    import numpy as np
    np.random.seed(42)
    spins = np.array([1, 1, 1, 1, 1])
    neighbors = np.array([[1, 2], [0, 2], [0, 1], [0, 4], [3, 0]])
    J = 1.0
    beta = 0.2
    spins_wolff_original = wolff_step(spins, neighbors, J, beta)
    print("spins_wolff after 1", spins_wolff_original)
    spins_wolff_2 = wolff_step(spins, neighbors, J, beta)
    print("spins_wolff after 2", spins_wolff_2)
    assert spins_wolff_original.tolist() == spins_wolff_2.tolist()  
    
    # test if adding a column of -1 to the neighbors matrix changes the result
    np.random.seed(42)
    spins_2 = np.array([1, 1, 1, 1, 1])
    neighbors_2 = np.array([[1, 2,-1], [0, 2,-1], [0, 1,-1], [0, 4,-1], [3, 0,-1]])
    spins_2 = wolff_step(spins_2, neighbors_2, J, beta)
    print("spins_wolff no -1", spins_wolff_original)
    print("spins_wolff with -1", spins_2)
    assert spins_wolff_original.tolist() == spins_2.tolist()