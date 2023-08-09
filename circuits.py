from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram
from numpy import pi
import numpy as np
from qiskit.providers.aer import StatevectorSimulator
from qiskit.compiler import transpile  
from qiskit.circuit.library import MCXGate, MCMT
from itertools import product


def mC(circuit,cgate):
    '''Create multi-control gate'''
    qubits=circuit.num_qubits
    # If only one qubit the multi-control gate just becomes an uncontroled gate
    if qubits==1:
        if cgate=='s':
            circuit.s(0)
        if cgate=='sdg':
            circuit.sdg(0)
        if cgate=='z':
            circuit.z(0)
    else:
        circuit=circuit.compose(MCMT(cgate,qubits-1,1))
    return circuit  


def randomQC(qubits=3, gate_num=20, seed=4):
    '''This function initializes a random circuit from the gate set
       ['CNOT', 'CNOT','RX', 'RY', 'RZ']. This list contains two times the CNOT gate
       to guarantee enough CNOTs in the circuit.
       Input: qubits: integer for the number of qubits in the circuit
              gate_num: integer specifying the number of gates in the circuit
              seed: random seed for sampling
       Output: qc: random Qiskit circuit'''
    #No CNOT gates if there is only one qubit
    if qubits==1:
        gate_set=['RX', 'RY', 'RZ']
    else:
        gate_set=['CNOT', 'CNOT','RX', 'RY', 'RZ']
    qc = QuantumCircuit(qubits)   # first number: n_qubits, second number n_classical bits
    np.random.seed(seed=seed)
    for i in range(gate_num):
        gate=np.random.choice(gate_set)
        if gate=='CNOT':
            t,c=np.random.choice(qubits,2)
            if not (t==c):
                qc.cx(t,c)
        if gate=='RX':
            t=np.random.choice(qubits,1)
            theta=int(2*np.random.rand()*100)/100*np.pi
            qc.rx(theta,t[0])
        if gate=='RY':
            t=np.random.choice(qubits,1)
            theta=int(2*np.random.rand()*100)/100*np.pi
            qc.ry(theta,t[0])
        if gate=='RZ':
            t=np.random.choice(qubits,1)
            theta=int(2*np.random.rand()*100)/100*np.pi
            qc.rz(theta,t[0])
    return  qc   
    
    
def create_circ(upper_qubits=3, lower_qubits=2, gate_num=20, seed=4, measure_all=True, instance='Full'):
    '''This function creates a full random circuit with a multi control Z gate in the 
       center that connects all qubits.  
       Input: upper_qubits: integer denoting the number of qubits above the cut
              lower_qubits: integer denoting the qubits below the cut
              gate_num: integer denoting the (approximately) total number of gates
              seed: random seed for the random circuit sampling
              measure_all: boolean determining whether the circuits should be measured
              instance: string denoting which pair of circuits is returned. Which string
              labels which circuit is defined in the figure in the jupyter notebook.'''
    # determine total number of qubits and how many gates there should be
    # in the upper and lower part of the circuit 
    n_qubits=upper_qubits+lower_qubits
    gates_up=int(upper_qubits/n_qubits*gate_num/2)
    gates_low=int(lower_qubits/n_qubits*gate_num/2)
    
    # create left-bottom (qc_lb) part of the cut
    qc_lb=randomQC(qubits=lower_qubits, gate_num=gates_low, seed=seed)
    # create right-bottom (qc_rb) part of the cut
    qc_rb=randomQC(qubits=lower_qubits, gate_num=gates_low, seed=seed+1)
    # create left-upper (qc_lu) part of the cut
    qc_lu=randomQC(qubits=upper_qubits, gate_num=gates_up, seed=seed+2)
    # create right-upper (qc_ru) part of the cut
    qc_ru=randomQC(qubits=upper_qubits, gate_num=gates_up, seed=seed+3)    
    
    # in the following the upper parts of the cut circuits and the lower part of the cut
    # circuits are composed, depending on the variable instance passed to the function
    
    if instance=='Full':
        qc=QuantumCircuit(n_qubits)
        qc=qc.compose(qc_lu,list(range(upper_qubits)))
        qc=qc.compose(qc_lb,list(range(upper_qubits,n_qubits)))
        qc=mC(qc,'z')
        qc=qc.compose(qc_ru,list(range(upper_qubits)))
        qc=qc.compose(qc_rb,list(range(upper_qubits,n_qubits)))
        if measure_all:
            qc.measure_all()
        return qc
        
    elif instance=='No_control':
        qc=QuantumCircuit(n_qubits)
        qc=qc.compose(qc_lu,list(range(upper_qubits)))
        qc=qc.compose(qc_lb,list(range(upper_qubits,n_qubits)))
        qc=qc.compose(qc_ru,list(range(upper_qubits)))
        qc=qc.compose(qc_rb,list(range(upper_qubits,n_qubits)))
        if measure_all:
            qc.measure_all()
        return qc
               
    elif instance=='a1':
        qc_u=mC(qc_lu,'s')
        qc_u=qc_u.compose(qc_ru)
        if measure_all:
            qc_u.measure_all()
        return qc_u
    
    elif instance=='a2':
        qc_l=mC(qc_lb,'s')
        qc_l=qc_l.compose(qc_rb)
        if measure_all:
            qc_l.measure_all()
        return qc_l
    
    elif instance=='b1':
        qc_u=mC(qc_lu,'sdg')
        qc_u=qc_u.compose(qc_ru)
        if measure_all:
            qc_u.measure_all()
        return qc_u
    
    elif instance=='b2':
        qc_l=mC(qc_lb,'sdg')
        qc_l=qc_l.compose(qc_rb)
        if measure_all:
            qc_l.measure_all()
        return qc_l
    
    elif instance=='d1_1':
        qc_u=mC(qc_lu,'z')
        qc_u=qc_u.compose(qc_ru)
        
        if measure_all:
            qc_u.measure_all()
        return qc_u
    
    elif instance=='c2_1':
        qc_l=mC(qc_lb,'z')
        qc_l=qc_l.compose(qc_rb)
        if measure_all:
            qc_l.measure_all()
        return qc_l
    
    elif instance=='d1_0':
        qc_u=qc_lu.compose(qc_ru)
        if measure_all:
            qc_u.measure_all()
        return qc_u
    
    elif instance=='c2_0':     
        qc_l=qc_lb.compose(qc_rb)
        if measure_all:
            qc_l.measure_all()
        return qc_l
    
    elif instance=='e2':
        circuit_list=[]
        # creates a list of circuits with all combinations
        # of a Z gate or no Z gate on the qubits
        for indices in product(*([[0,1]]*lower_qubits)):  
            qc = qc_lb.copy()
            for index,value in enumerate(indices):
                if value==1:
                    qc.z(index) 
            qc=qc.compose(qc_rb)
            if measure_all:
                qc.measure_all()
            circuit_list.append(qc)  
        return circuit_list

    elif instance=='e1':
        circuit_list=[]
        # creates a list of circuits with all combinations
        # of a Z gate or no Z gate on the qubits
        for indices in product(*([[0,1]]*upper_qubits)):  
            qc = qc_lu.copy()
            for index,value in enumerate(indices):
                if value==1:
                    qc.z(index) 
            qc=qc.compose(qc_ru)
            if measure_all:
                qc.measure_all()
            circuit_list.append(qc)  
        return circuit_list
    
    # Intermediate measurement circuit
    elif instance=='c1':
        return qc_lu, qc_ru
    
    # Intermediate measurement circuit
    elif instance=='d2':
        return qc_lb, qc_rb
        
    else:
        assert False, 'circuit instance: ' + str(instance) + ' not available'
        

