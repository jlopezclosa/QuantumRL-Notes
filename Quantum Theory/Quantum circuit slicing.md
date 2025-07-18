Splits the circuit to be mapped in slices that contain only gates that can be executed in parallel, i.e., gates that are not sharing any of the logical qubits they act on. 

![[circuitslicing.png]]

## Algorithm 

To obtain the partitioning of a quantum circuit in time slices the iterative Algorithm 1 can be considered. The algorithm operates on a given list $G$ of 2-qubit gates, represented as pairs of logical qubit indices. 
1) For each gate ( $q_1, q_2$ ), the process entails starting from the last time slice, moving backward until a slice $t$ is encountered that contains gates involving either qubit $q_1$ or $q_2$. 
2) Subsequently, the gate is appended to slice $t+1$. 
3) If no such slice is identified, the gate is added to the first slice. 
4) If slice $t+1$ does not exist, it is created and appended to the output list of slices denoted as $S$.

![[circuitslicing-algorithm.png]]

## An example:

$G = [(q_1, q_2), (q_3, q_4), (q_2, q_5), (q_1, q_3), (q_5, q_6)]$

The process works as follows:

1) Start with an empty list of time slices $S = [\ ]$. 
2) For the first gate $(q_1, q_2)$: There are no existing slices, so you create slice 1 and add the gate to it. $S = [[(q1, q2)]]$
3) For the second gate $(q_3, q_4)$: Again, no existing slice contains qubits $q_3$ or $q_4$, so you create slice 2 and add the gate to it.  $S  = [[(q_1, q_2)], [(q_3, q_4)]].$
4) For the third gate $(q_2, q_5)$: Slice 1 contains $q_2$, so you move this gate to slice 2 (the next slice after slice 1). $S = [[(q1, q2)], [(q3, q4), (q2, q5)]]$
5) For the fourth gate $(q_1, q_3)$: Slice 2 contains $q_3$, so you move the gate to slice 3. $S = [[(q_1, q_2)], [(q_3, q_4), (q_2, q_5)], [(q_1, q_3)]].$
6) For the fifth gate $(q_5, q_6)$: Slice 2 contains $q_5$, so you move the gate to slice 3. $S = [[(q_1, q_2)], [(q_3, q_4), (q_2, q_5)], [(q_1, q_3), (q_5, q_6)]]$


