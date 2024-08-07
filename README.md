# EV-GNN: Scalable Reinforcement Learning for Dynamic Electric Vehicle Charging Optimization Using Graph Neural Networks: 

![image](https://github.com/user-attachments/assets/01fee857-e2f3-4f5c-a5c0-690444242def)

**Concept and Architecture of EV-GNN**
  - **EV Charging Problems**: Visualized as a network of charging stations connected to the grid. The problem is formulated as a mathematical optimization with objectives such as minimizing load peaks or maximizing profits, subject to constraints like power limits and EV State of Charge (SoC) requirements.
  - **Graph Modeling**: EV charging optimization is represented as a graph with nodes for EVs, chargers, transformers, and Charging Point Operators (CPOs), each with unique features.
  - **Graph Simplification**: Nodes with no active connections, such as certain idle charging stations, are removed from the graph.
  - **Node Feature Processing**: Node-specific Multi-Layer Perceptrons (MLPs) convert heterogeneous features into uniform, higher-dimensional embeddings.
  - **Actor Network**: Uses $L$ GCN layers to process the graph, reducing node features to a fixed size suitable for continuous or discrete action spaces. Outputs are mapped to a standard action vector.
  - **Critic Network**: Combines action features with state features, processes them through $K$ GCN layers, and aggregates into a fixed-size graph-wide embedding. This embedding is used in an MLP to compute the Q-value.

# Results

## Scalability
![image](https://github.com/user-attachments/assets/7441e9dc-796b-4bce-b3df-f71cbf24782f)
 **Optimality Gap as a function of RL algorithm and experiment scale**

![image](https://github.com/user-attachments/assets/aa196882-633c-4e90-8460-b811ffb4c803)

**Explainability analysis for EV-GNN in the 25 EVSE case**


## Application to Multi-Discrete domains

<img align="center" src="https://github.com/user-attachments/assets/b8820d0d-1435-48ff-9ce6-e8c9c8a013ec" width="55%"/>

## Application to V2G Profit Maximization
![image](https://github.com/user-attachments/assets/2766bd16-d5d2-4cc1-a955-0b29e3779cda)

V2G profit maximization with loads, PV, and demand response events. Training performance of baseline and enhanced RL algorithms for 25 EVSEs,
showing the best and average rewards achieved by various methods, including GNN-enhanced approaches, in a smaller and larger-scale scenario.

# Cite

```
TBA
```
