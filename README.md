# Robot Navigation - Path Planning Algorithm

### This project was conducted using the following github:  
- https://github.com/AtsushiSakai/PythonRobotics  
<hr style="border-top: 3px solid #bbb;">

## Local Path Planning

- Bug Algorithm  
- Artificial Potential Field
- Dynamic Window Approach
- Follow the Gap  

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


### 1. Pure Pursuit Algorithm

The Pure-pursuit algorithm is a simple and efficient path-tracking algorithm used to determine the steering angle of a vehicle as it follows a specific path.  
This is based on the vehicle's current position and a target point on the path.  
In this algorithm, the distance to the target point that the vehicle aims to track is defined by a parameter called the Look-ahead distance,  
represented as the variable Lfc in the code. Below are the observed results when varying the Lfc parameter.  

<img src="https://github.com/user-attachments/assets/41af6fcc-2699-41bb-b90a-7da55bfacf08" alt="Description2" style="width: 60%; height: 600px;">  

#### Discussion

In the results above, it was observed that setting the parameter value to 2.0 produced the smoothest path-following behavior.  
As the value was gradually increased beyond 2.0, the tracking trajectory was examined further,  
revealing that larger values caused the vehicle to exhibit increasingly unnecessary wide turns at corners while following the path.
<hr style="border-top: 3px solid #bbb;">

### 2. Stanley Controller Algorithm

The Stanley Controller algorithm adjusts the vehicle's position and errors based on its front axle as the reference point to follow a given path.  
The Stanley Controller corrects two primary types of errors to track the path: Crosstrack error and Heading error.  
Crosstrack error refers to the lateral error between the vehicle's current position and the path,  
while Heading error is the angular difference between the vehicle's heading and the direction of the path.
  
Upon examining the code, it was found that the adjustable parameter, control gain, is defined as the variable k,  
and it is associated with the crosstrack error. The results of varying the k parameter were observed as described below.

<img src="https://github.com/user-attachments/assets/d8fe5f07-34db-41a6-bc4a-9cf231093842" alt="Description2" style="width: 60%; height: 450px;">  
<img src="https://github.com/user-attachments/assets/50129580-bdf7-460d-ac79-79cb9b49234f" alt="Description2" style="width: 60%; height: 250px;">  

#### Discussion

The results were analyzed by gradually increasing the control gain parameter starting from a small value of 0.01.  
At 0.01, the tracking path showed a similar direction and shape to the original path, but the distance error remained unchanged.  
As the control gain value was increased, the tracking path progressively aligned more closely with the course.  
When the control gain k was set to 1.0, the course and the tracking path overlapped perfectly. 
Further increasing the value revealed that the vehicle approached the course at increasingly steeper angles during the initial correction phase.  
<hr style="border-top: 3px solid #bbb;">

### 3. LQR_Steer_Control Algorithm

LQR steering control is a control method designed to optimize the behavior of linear systems,  
addressing the limitations of PID control.  
It employs an algorithm that predicts the next state based on the current state and control input.  
The maximum steering angle parameter is defined in the code as the variable max_steer.  
By adjusting the value of max_steer, the upper limit of the steering angle was modified to analyze its impact on the algorithm.  

<img src="https://github.com/user-attachments/assets/a0362ca3-3baa-492e-a2d4-81b7df9cc7d4" alt="Description2" style="width: 60%; height: 600px;">  
<img src="https://github.com/user-attachments/assets/3d6265ff-964d-472a-8a34-be79b074fef5" alt="Description2" style="width: 60%; height: 400px;">  

#### Discussion

Initially, the code was executed with the default setting of max_steer = 45°, and the vehicle closely followed the given path with high accuracy.  
When the value was set to 5° and the code was executed, the limited maximum steering angle of 5° resulted in insufficient turning,  
causing the vehicle to deviate significantly from the path and trace a large elliptical trajectory before reaching the destination.  
When the value was set to 25° or higher, the vehicle effectively followed the given path, except for slight deviations at corners.
<hr style="border-top: 3px solid #bbb;">

### Optional

Comparison of Pure Pursuit Algorithm and Stanley Controller Algorithm on a Single Screen  

<img src="https://github.com/user-attachments/assets/7dbc1a70-9c63-4767-b16b-e627fe80b5c0" alt="Description2" style="width: 60%; height: 400px;">  
<hr style="border-top: 3px solid #bbb;">
