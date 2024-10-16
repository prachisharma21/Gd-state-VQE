# Essential imports for the calculations
from multiprocessing.pool import ThreadPool
from qiskit.providers.fake_provider import FakeQuitoV2, FakeGuadalupeV2
import scipy
import numpy as np
from circuit_builder import CircuitBuilder
from model_hamiltonian import *
from exact_diagonalization import *
from helper import QSimulator
import matplotlib.pyplot as plt
import concurrent.futures
import matplotlib.colors as mcolors


# Here VQE is done with SV simulator 
def VQE_parameterized_circuit_sv(params =[],backend = FakeQuitoV2(),initial_layout = [0, 1, 2, 3, 4], num_layers = 1):
    """
    Builds a parameterized circuits to perform the VQE step using HVA ansatz
    """
    Vcircuit= CircuitBuilder(params, backend, initial_layout, nlayers = num_layers)
    # here circuits are built without any measurement since State-vector simulator is used for VQE optimization
    circ_w_no_meas = [Vcircuit.makevqeCircuit(measure = False)]
    return circ_w_no_meas

def VQE_energy(params =[],backend = FakeQuitoV2(),initial_layout = [0, 1, 2, 3, 4],bonds = [], num_layers = 1, model_input= ()):
    """
    Cost function for the optimizer, i.e., energy of the hamiltonian 
    """
    circ_w_no_meas = VQE_parameterized_circuit_sv(params, backend, initial_layout, num_layers )
    qsim = QSimulator(backend)
    SV_counts = qsim.State_Vector_Simulator(circ_w_no_meas) 
    res = SV_counts.expectation_value(Hamiltonian_MFIM(model_input = model_input, num_qubits = backend.num_qubits, bonds = bonds))
    res = np.real(res)
    return res



def optimizer(init_params,backend = FakeQuitoV2(), initial_layout = [0, 1, 2, 3, 4], bonds = [],num_layers = 1, model_input= ()):
    """
    Classical optimization step of VQE
    """
    res = scipy.optimize.minimize(VQE_energy, init_params, args = ( backend, initial_layout, bonds,num_layers,model_input),method ="BFGS")
    return res

def VQE_optimization_Step(init_params,backend = FakeGuadalupeV2(), initial_layout = [i for i in range(16)], bonds = [],num_layers = 1, model_input=()):
    res_vqe_sv = optimizer(init_params,backend,initial_layout, bonds, num_layers, model_input)
    if res_vqe_sv.success:
        optimum_params = res_vqe_sv.x
        lowest_fun = res_vqe_sv.fun
        
    else:
        print(res_vqe_sv.success)    
    
    return optimum_params, lowest_fun

def exact_gd_state_energy(model_input, num_qubits,bonds):
    """
    function to evaluate the MFIM's exact energy using qutip
    """

    exact_ham = map_hamiltonian(complete_Hamiltonian_str(model_input,num_qubits,bonds))

    return get_lowest_states(exact_ham, 1)[0][0]

def main():
    # Benutzerabfrage für die Anzahl der Schichten (num_layers)
    while True:
        try:
            num_layers = int(input("Please enter the number of layers: "))
            if num_layers > 0:
                break
            else:
                print("The number of layers must be positive.")
        except ValueError:
            print("The number of layers must be an integer.")
    
    print(f"Start optimizing with layer={num_layers}")
    
    # Update the phase space parameter below 
    J = -1.0
    hx = -1.0
    hz = 0.5
    model_input = (J, hx, hz)
    print(f"Considered MFIM model parameters (J = {J}, hx = {hx}, hz = {hz})")

    # native connectivity of the Guadalupe backend
    bonds_Guad = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 9), (8, 11), (11, 14), (14, 13),
                   (13, 12), (12, 15), (1, 4), (4, 7), (7, 6), (7, 10), (10, 12)]
    # native connectivity of Quito backend
    bonds_Quito = [[0, 1],[1, 2],[1, 3],[3, 4]]
    
    # Es gibt drei Parameter pro Schicht in diesem VQE HVA Ansatz
    num_params = 3 

    # Generating random Parameter for VQE
    init_params =  np.random.uniform(-np.pi/3, np.pi/3, num_layers * num_params)
        
    # Choosing Backend 
    backend = FakeQuitoV2()  # FakeGuadalupeV2()
    bonds = bonds_Quito  # bonds_Guad 

    init_VQE_circuit = VQE_parameterized_circuit_sv(params = init_params, backend = backend,
                                       initial_layout = [i for i in range(backend.num_qubits)], 
                                       num_layers=num_layers)[0]
    init_VQE_energy = VQE_energy(params = init_params, backend = backend,
                                       initial_layout = [i for i in range(backend.num_qubits)], 
                                       bonds = bonds, num_layers=num_layers, model_input=model_input)
    
    # Initialen VQE-Zustand und Energie anzeigen
    print(f"Quantum circuit for randomly sampled initial parameters: {init_params} with the corresponding VQE energy: {init_VQE_energy} is \n ", init_VQE_circuit)

    # Optimierungsschritt
    optimal_params, lowest_energy = VQE_optimization_Step(init_params, backend = backend, 
                                                          initial_layout = [i for i in range(backend.num_qubits)], 
                                                          bonds=bonds, num_layers=num_layers, 
                                                          model_input=model_input)

    # Optimierte Parameter und niedrigste Energie anzeigen
    print(f"Optimized parameters are {optimal_params} and obtained lowest energy is {lowest_energy}")  
        
    # Genau berechnete Grundzustandsenergie anzeigen
    print(f"Exact ground state energy: {exact_gd_state_energy(model_input, backend.num_qubits, bonds)}")
    
    # Differenz zwischen berechneter und exakter Energie berechnen und anzeigen
    energy_difference = lowest_energy - exact_gd_state_energy(model_input, backend.num_qubits, bonds)  
    print(f"The difference between the exact ground state energy and the calculated lowest energy is: {energy_difference}")

    # Optional: Aufruf der scan_hx_hz-Funktion
    scan_hx_hz(backend, bonds, num_layers, num_params)






def perform_vqe_multiple_times(backend, num_layers, num_params, model_input, bonds):
    # Repeat the VQE optimization 10 times and return the smallest energy
    min_lowest_energy = None
    for repeat in range(10):
        # Initial random parameters for VQE
        init_params = np.random.uniform(-np.pi/3, np.pi/3, num_layers * num_params)

        # Perform the VQE optimization
        optimal_params, lowest_energy = VQE_optimization_Step(init_params, backend=backend, 
                                                              initial_layout=[i for i in range(backend.num_qubits)], 
                                                              bonds=bonds, num_layers=num_layers, 
                                                              model_input=model_input)
        # Track the minimum lowest_energy
        if min_lowest_energy is None or lowest_energy < min_lowest_energy:
            min_lowest_energy = lowest_energy
    
    return min_lowest_energy



def scan_hx_hz(backend, bonds, num_layers, num_params):
    # Define the ranges for hx and hz 
    hx_values = np.linspace(0, 2, 10)  # Values for hx
    hz_values = np.linspace(0, 2, 10)  # Values for hz

    # Create a grid to store the energy differences
    energy_diff_grid = np.zeros((len(hz_values), len(hx_values)))
    
    # Create another grid to store the exact energies
    exact_energy_grid = np.zeros((len(hz_values), len(hx_values)))

    J = -1.0  # J is kept constant

    # First, calculate the exact energies for all hx and hz combinations
    for i, hx in enumerate(hx_values):
        for j, hz in enumerate(hz_values):
            model_input = (J, hx, hz)
            exact_energy_grid[j, i] = exact_gd_state_energy(model_input, backend.num_qubits, bonds)

    # Now perform the VQE optimization in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Iterate over hx and hz in parallel
        futures = []
        for i, hx in enumerate(hx_values):
            for j, hz in enumerate(hz_values):
                model_input = (J, hx, hz)
                # Submit each task for parallel execution
                futures.append((i, j, executor.submit(perform_vqe_multiple_times, backend, num_layers, num_params, model_input, bonds)))

        # Collect the results and compute the energy difference using the precomputed exact energy
        for i, j, future in futures:
            min_lowest_energy = future.result()
            exact_energy = exact_energy_grid[j, i]  # Fetch the precomputed exact energy
            energy_diff = min_lowest_energy - exact_energy 

            # Replace non-positive values with a small positive value for LogNorm
            if energy_diff <= 0:
                energy_diff = 1e-10  # Small positive value

            energy_diff_grid[j, i] = energy_diff

            # Ausgabe der Werte für jede Schleife
            print(f"hx={hx_values[i]}, hz={hz_values[j]}, energy_diff={energy_diff}")

    # Create the plot with a logarithmic color scale
    plt.figure(figsize=(6, 6))
    plt.imshow(energy_diff_grid, extent=[hx_values.min(), hx_values.max(), hz_values.min(), hz_values.max()],
               origin='lower', cmap='jet', aspect='auto', norm=mcolors.LogNorm(vmin=1e-4, vmax=1e1))

    # Add color bar and labels
    plt.colorbar(label='Energy Difference (log scale)')
    plt.xlabel('hx')
    plt.ylabel('hz')
    plt.title('Logarithmic Energy Difference as function of hx and hz')

    # Show the plot
    plt.show()




if __name__ == "__main__":
    main()


