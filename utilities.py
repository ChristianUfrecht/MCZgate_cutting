from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram
from numpy import pi
import numpy as np
from qiskit.providers.aer import StatevectorSimulator
from qiskit.compiler import transpile


def observable(n_qubits):
    '''Creates the observable consisting of a Pauli Z
       observable on n_qubits in numpy matrix representation'''
    Z=np.array([[1,0],[0,-1]])
    for i in range(n_qubits):
        if i==0:
            O=Z
        else:
            O=np.kron(O,Z)
    return O


def expectation(O,state):
    '''Function evaluates the expectation value of
       a given observable O with respect to a state.'''
    v=np.array([state]).T
    return (np.conj(v.T)@O@v)[0,0]


def evaluateCircuit(qc, intermediate_measurement):
    '''Function evaluates the Qiskit circuits. The function transpiles the circuit, runs
        it and returns the state vector.
        Input: If intermediate_measurements=False:
                    qc: quantum circuit
        Output: state vector after application of the circuit
        Input: If intermediate_measurement=True:
                    qc=[left part before measurement, right_part after measurement]
        Output: List of tuples with (probability for intermediate measurement, state vector at the
                end of circuit after observing this measurement for all bit strings'''
    if not intermediate_measurement:
        gates = Aer.get_backend('aer_simulator').configuration().basis_gates
        qc=transpile(qc, basis_gates=gates,optimization_level=0)
        sim=StatevectorSimulator()
        result=sim.run(qc, shots=1).result()
        state_vec=result.get_statevector()
        return state_vec
    
    if intermediate_measurement:
        qc_l=qc[0]
        qc_r=qc[1]
        qubits=qc_l.num_qubits
        gates = Aer.get_backend('aer_simulator').configuration().basis_gates
        # transpile left and right part of upper/lower circuit
        qc_l=transpile(qc_l, basis_gates=gates,optimization_level=0)
        qc_r=transpile(qc_r, basis_gates=gates,optimization_level=0)
        sim=StatevectorSimulator()
        # calculate state before the intermediate measurement
        result=sim.run(qc_l, shots=1).result()
        state_before_measure=result.get_statevector()

        state_prob_list=[]
        for index in range(2**qubits):
            # calculate probability for observing the bit string with index 
            # given by variable index
            p=np.abs(state_before_measure[index])**2
            new_state=np.zeros(2**qubits)
            # initialize this bit string as new state
            new_state[index]=1
            # initialize the new state and act on it
            # with the right part of the circuit
            qc_fresh=QuantumCircuit(qubits)
            qc_fresh.initialize(new_state)
            qc_fresh=qc_fresh.compose(qc_r,list(range(qubits)))
            # calculate final state
            result=sim.run(qc_fresh, shots=1).result()
            final_state=result.get_statevector()
            state_prob_list.append((p,final_state))
        return state_prob_list

   
   
