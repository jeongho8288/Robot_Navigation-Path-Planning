# Robot Navigation - Path Planning Algorithm

### This project was conducted using the following github:  
- https://github.com/AtsushiSakai/PythonRobotics  
<hr style="border-top: 3px solid #bbb;">

## Local Path Planning

- Bug Algorithm  
- Artificial Potential Field
- Dynamic Window Approach
- Follow the Gap  
<hr style="border-top: 3px solid #bbb;">

## Global Path Planning

### Graph Based Apporoach

- Dijkstra's Algorithm  
- A* Algorithm  

### Sampling Based Approach

- Rapidly-Exploring Random Tree (RRT)  
- Probabilistic Roadmap (RPM)  
- Visiblity Planning  
- Smoothing Planned Paths  
<hr style="border-top: 3px solid #bbb;">


## 1. BUG Algorithm

- This is an algorithm where the vehicle moves in a straight line towards the goal point, but when it encounters an obstacle, it moves around the obstacle's perimeter and then resumes moving straight towards the goal.  
- It starts with prior knowledge of the final destination, but without any prior information about obstacles or the map.  
- There are various versions of the BUG algorithm, such as BUG0, BUG1, BUG2, and Tangent Bug.  
- It is simple to implement and requires low computation, but it does not always guarantee the optimal path.  

<img src="https://github.com/user-attachments/assets/5a6f25dd-657a-46d1-9141-fc07b150a543" alt="Description2" style="width: 60%; height: 500px;">  

### Discussion

- After running the three algorithms, the BUG0 algorithm visually appeared to provide the most optimal path and took the least amount of time.  
- In the case of the BUG1 algorithm, it showed an inefficient path, rotating around the obstacle more than once.  
- Both the BUG0 and BUG2 algorithms appeared more efficient compared to BUG1, but there was a difference in that BUG0 moved along the optimal distance, whereas BUG2 followed the edges of the obstacle.  
<hr style="border-top: 3px solid #bbb;">

## 2. Dynamic Window Approach Algorithm


- The algorithm explores possible combinations of velocity (linear velocity/acceleration) in the velocity and angular velocity space.  
- It derives the optimal linear velocity and angular velocity based on speed, direction, and the distance to obstacles to avoid collisions and reach the destination.  
- The process of calculating the cost involves sampling the velocity space and simulating, which can result in high computational costs.  
- Since it focuses on local path planning, it does not guarantee the optimal path after collision avoidance.  

<img src="https://github.com/user-attachments/assets/8830fb87-f014-49e8-9800-acc7ee964685" alt="Description2" style="width: 60%; height: 350px;">  
<hr style="border-top: 3px solid #bbb;">

## 3. Dijkstra & A* Algorithm

### Dijkstra

- Starting from the initial node, the algorithm visits the nearest node first and computes the shortest distance information for all other nodes.  
- Since it calculates for all nodes, it can be computationally expensive and slow.  

### A* 

- The modified Dijkstra algorithm uses a heuristic function between the starting point and the goal point to calculate scores, and based on the calculated scores, it searches for the optimal path.  
- This reduces the computational load, making it faster and generally providing the optimal path.  
- The performance and accuracy of the path depend on the heuristic function used.  


The A* algorithm has the advantage of more efficiently searching for the path to the goal by visiting only certain nodes based on priority, allowing it to find the optimal route more quickly.  


<img src="https://github.com/user-attachments/assets/4ca2d510-93de-4327-a685-b574025b8585" alt="Description2" style="width: 60%; height: 250px;">  

### Discussion

It was observed that the A* algorithm, with its smaller search space compared to the Dijkstra algorithm, is significantly superior in terms of time and computational efficiency.  
<hr style="border-top: 3px solid #bbb;">

## 4. Potential field Planning Algorithm

- This algorithm generates an attractive force toward the goal point and a repulsive force around obstacles, causing the robot to move based on a physical force field.  
- When the robot approaches an obstacle, the repulsive force changes abruptly, leading to oscillation.  
- A local minima problem can occur when the repulsive force from obstacles and the attractive force toward the goal point become equal, causing the robot to stop mid-path.  

### Creating a Failure Scenario & Reduced Repulsive Force

<div style="display: flex; justify-content: center;">
  <img src="https://github.com/user-attachments/assets/a82bf21f-5c17-4686-8cdb-b4ac75199677" alt="Description1" style="width: 30%; height: 200px; margin-right: 2%;">
  <img src="https://github.com/user-attachments/assets/19ffa4bf-cbea-4223-9fee-4ed2229bcac7" alt="Description2" style="width: 30%; height: 200px;">
</div>
<hr style="border-top: 3px solid #bbb;">

## Optional

Integrating the Dynamic Window Approach, BUG, and Potential Field algorithms into a single code.  

<img src="https://github.com/user-attachments/assets/e14bd9f1-7fa5-420b-8d90-7016b81b8784" alt="Description2" style="width: 50%; height: 500px;">  
<hr style="border-top: 3px solid #bbb;">

## 5. RRT Algorithm

- An algorithm that generates a path to reach the target point by expanding a tree through a Random Sampling process.  
- It efficiently searches for paths in unstructured or high-dimensional spaces.  
- A drawback is that the tree may expand unnecessarily due to the random sampling process.  
- It operates efficiently with a simple structure and has the advantage of being applicable to high-dimensional spaces.  

![image](https://github.com/user-attachments/assets/b03029d8-a8e7-48f5-92b3-17d12ef27905)

<img src="https://github.com/user-attachments/assets/b03029d8-a8e7-48f5-92b3-17d12ef27905" alt="Description2" style="width: 50%; height: 500px;">  

### Discussion

Since it operates through random sampling, the time taken to reach the destination varies each time it is executed, and ultimately, different paths are generated.
<hr style="border-top: 3px solid #bbb;">

## Optional

I thought the RRT algorithm was not very efficient, so I considered ways to improve its performance and applied some of those methods.

### 1. Generate Multiple Tree

A method where multiple trees are generated simultaneously from the starting point, and the path of the tree that reaches the destination first is selected.  
→ Increases efficiency in terms of time.  
→ I wrote a code : rrt_many_tree_generate.py.  
  
<img src="https://github.com/user-attachments/assets/d3ca1751-4e2f-4b2e-a511-3f92d8f68c7f" alt="Description2" style="width: 40%; height: 400px;">  
  
### 2. Sampling Close Points

By performing random sampling anywhere, the process can take a long time due to randomness.  
To prevent this, random sampling is conducted within a small distance radius only around the node closest to the destination, avoiding inefficient random sampling.  
→ Increases efficiency in terms of time and computation.  
→ I wrote a code : rrt_sampling_short.py.  
  
<img src="https://github.com/user-attachments/assets/9a0a83e2-640b-4fd2-80c6-7410dc3f7cc1" alt="Description2" style="width: 40%; height: 400px;">  
  
### 3. Simultaneous generating trees from the starting point and the goal point.

By generating and expanding two trees simultaneously from the starting point and the goal point, and selecting the path when the two trees meet within a certain distance.  
→ Increases efficiency in terms of time.  
→ I wrote a code called rrt_both_path.py.  
  
<img src="https://github.com/user-attachments/assets/d00b33c4-dc47-4a71-81f1-787c462769e2" alt="Description2" style="width: 40%; height: 400px;">  
  
This approach finds the path relatively faster compared to the previously improved algorithms.
