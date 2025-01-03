"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

"""

import math                         # 수학 함수를 사용하기 위한 임포트
from enum import Enum               # Enum 클래스 사용을 위한 임포트

import matplotlib.pyplot as plt     # 그래프 시각화를 위한 임포트
import numpy as np                  # 수치 계산을 위한 임포트

show_animation = True               # 애니메이션을 보여줄지에 대한 여부 설정


def dwa_control(x, config, goal, ob):                                       # 현재까지의 경로, 목적지, 장애물 정보들을 매겨변수로 받음
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)                                     # 현재 상태와 로봇 설정을 토대로 동적 창을 계산함 -> 가능한 window에 대한 값을 얻음

    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)    # 현재까지의 경로, dw, 세부 설정, 목적지, 장애물 위치를 토대로 경로를 계산함

    return u, trajectory                                                    # 최적의 제어 입력, 예측된 경로 반환


class RobotType(Enum):
    circle = 0                  # 원형 로봇 사용
    rectangle = 1               # 사각형 로봇 사용


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter 설정
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check
        # obstacles [x(m) y(m), ....]   # 장애물의 위치 설정
        self.ob = np.array([[-1, -1],
                            [0, 2],
                            [4.0, 2.0],
                            [5.0, 4.0],
                            [5.0, 5.0],
                            [5.0, 6.0],
                            [5.0, 9.0],
                            [8.0, 9.0],
                            [7.0, 9.0],
                            [8.0, 10.0],
                            [9.0, 11.0],
                            [12.0, 13.0],
                            [12.0, 12.0],
                            [15.0, 15.0],
                            [13.0, 13.0]
                            ])

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


config = Config()


def motion(x, u, dt):                       # 현재 상태의 선속도, 각속도, dt를 매개변수로 받음
    """
    motion model
    """

    x[2] += u[1] * dt                       # 각속도에 시간을 곱하여 이를 현재 위치에 더함
    x[0] += u[0] * math.cos(x[2]) * dt      # 선속도를 이용해 x 좌표를 업데이트
    x[1] += u[0] * math.sin(x[2]) * dt      # 선속도를 이용해 y 좌표를 업데이트
    x[3] = u[0]                             # 현재 선속도 업데이트
    x[4] = u[1]                             # 현재 각속도 업데이트

    return x                                # 새롭게 업데이트 된 현재 로봇의 상태 반환


def calc_dynamic_window(x, config):                             # 현재까지의 경로, 세부 설정 값을 파라미터로 받음
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification                   
    Vs = [config.min_speed, config.max_speed,                   # 최소/최고 선속도, 최소/최고 각속도에 대한 값을 원소로 하는 리스트 구성
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model                          
    Vd = [x[3] - config.max_accel * config.dt,                  # 현재 속도에서 최대 가속도를 이용한 일정 시간 동안의 감속할 때의 최소 속도를 의미
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]               # 최종적인 동적 창
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),                 # 로봇의 가용한 최소 속도와 해당 속도에서 나타낼 수 있는 최소 속도를 구한 후 더 큰 갑을 석택
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]                 # 로봇의 가용한 최대 속도와 현재 속도에서 나타낼 수 있는 최대 속도를 구한 후 더 작은 값을 선택

    return dw                                                   # 최종 동적 창 반환


def predict_trajectory(x_init, v, y, config):                   # 현재까지의 경로, 선속도, 각속도, config를 파라미터로 받음
    """
    predict trajectory with an input
    """

    x = np.array(x_init)                                        # 현재까지의 경로를 x에 할당
    trajectory = np.array(x)                                    # x를 넘파이 배열로 하여 trajectory 변수에 초기화
    time = 0                                                    # 시간을 초기화
    while time <= config.predict_time:                          # 예측 시간 보다 작은 경우 반복
        x = motion(x, [v, y], config.dt)                        # 주어진 속도, 각속도를 통해 x 업데이트
        trajectory = np.vstack((trajectory, x))                 # 새롭게 업데이트 된 x를 trajectory에 추가하여 업데이트
        time += config.dt                                       # 시간 증가

    return trajectory                                           # 최종적으로 예측된 경로를 반환함


def calc_control_and_trajectory(x, dw, config, goal, ob):       # 현재까지의 경로, dw, 세부 설정 값, 목적지, 장애물에 대한 정보를 파라미터로 받음
    """
    calculation final input with dynamic window                 # 동적 창을 이용하여 최종 입력을 계산함
    """

    x_init = x[:]                                               # 현재까지의 경로를 x_init에 복사함
    min_cost = float("inf")                                     # 최소 비용을 설정
    best_u = [0.0, 0.0]                                         # 최적의 속도, 각속도 초기화
    best_trajectory = np.array([x])                             # 최적의 경로를 초기화함

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):                                      # 최소 속도부터 최대 속도까지 해당 간격만큼 증가분을 가지는 배열을 만들어 반복
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):                           # 최소 각속도부터 최대 각속도까지 해당 간격만큼 증가분을 가지는 배열을 만들어 반복

            trajectory = predict_trajectory(x_init, v, y, config)                               # 현재까지의 경로, 선속도, 각속도, config 값을 통해 경로를 예측함
            # calc cost
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)       # 목적지까지에 대한 목표 비용 계산
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])        # 정해진 최대 속도와 현재 속도의 차이에 비례하는 속도 비용 계산
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)    # 경로와 장애물에 대한 정보를 통해 장애물 회피 비용 계산

            final_cost = to_goal_cost + speed_cost + ob_cost                                    # 각 비용에 대한 합을 구하여 최종 비용 계산

            # search minimum trajectory                                         # 최소 비용을 가지는 경로를 탐색함
            if min_cost >= final_cost:                                         
                min_cost = final_cost                                           # 최소 비용을 업데이트
                best_u = [v, y]                                                 # 최적의 속도와 각속도 업데이트
                best_trajectory = trajectory                                    # 최적의 경로를 업데이트
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:           
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)                                    # 로봇이 갇힌 상태에서의 최선의 선택을 설정함
                    best_u[1] = -config.max_delta_yaw_rate                      # 로봇이 장애물 앞에서 멈추는 것을 방지하기 위하여 회전 각속도를 설정함
    return best_u, best_trajectory                                              # 최적의 제어 입력, 경로를 반환


def calc_obstacle_cost(trajectory, ob, config):                                     # 경로, 장애물 정보를 파라미터로 받음
    """
    calc obstacle cost inf: collision
    """     
    ox = ob[:, 0]                                                                   # 장애물의 x좌표
    oy = ob[:, 1]                                                                   # 장애물의 y좌표
    dx = trajectory[:, 0] - ox[:, None]                                             # 각 장애물과 현재 로봇의 경로 사이의 x값 계산
    dy = trajectory[:, 1] - oy[:, None]                                             # 각 장애물과 현재 로봇의 경로 사이의 y값 계산
    r = np.hypot(dx, dy)                                                            # 로봇과 장애물과의 거리 계산

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]                                                      # 로봇의 현재 각도
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])   # 회전 행렬 설정
        rot = np.transpose(rot, [2, 0, 1])                                          # 회전 행렬 전치
        local_ob = ob[:, None] - trajectory[:, 0:2]                                 # 현재 로봇의 위치로 부터 장애물의 상대적인 위치를 구함
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])                            # 회전 행렬로 로봇의 각도에 따른 상대적인 회전을 함
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        # 회전시킨 로봇의 경계 상자에 대한 네 변의 충돌 여부를 판단함
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():        # 로봇의 경계 상자와 장애물과 충돌하였는지 판단
            return float("Inf")                                                     # 충돌을 감지하면 무한대의 값을 반환하여 경로를 사용하지 않도록 함
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")                                                     # 충돌 발생 시 무한대의 비용을 반환

    min_r = np.min(r)                                                               # 최소 거리를 계산함
    return 1.0 / min_r  # OK                                                        # 거리에 대한 역수 반환 ( 거리의 증가 -> 비용 감소 )


def calc_to_goal_cost(trajectory, goal):                                    # 경로와 목적지의 값을 파라미터로 받음
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]                                        # 목표점과 로봇의 위치 간의 x 거리
    dy = goal[1] - trajectory[-1, 1]                                        # 목표점과 로봇의 위치 간의 y 거리
    error_angle = math.atan2(dy, dx)                                        # 목표점까지의 상대적인 각도 계산
    cost_angle = error_angle - trajectory[-1, 2]                            # 로봇의 현재 방향과 목표 각도 간의 차이 계산
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))      # 최종적인 각도 차이 비용 계산

    return cost     # 최종 구한 cost 반환


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover   # 애니메이션에서 로봇의 방향을 화살표로 표시하는 함수
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):  # pragma: no cover                  # 애니메이션에서 로봇의 현재 위치를 시각화 해주는 함수
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")


def main(gx=10.0, gy=10.0, robot_type=RobotType.circle):                # 목적지 좌표와 로봇의 형태를 파라미터로 받음
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])                   # 로봇의 초기 상태에 대한 설정
    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])

    # input [forward speed, yaw_rate]

    config.robot_type = robot_type                                      # 로봇의 모양 설정
    trajectory = np.array(x)                                            # 현재까지의 로봇의 경로 초기화
    ob = config.ob                                                      # 장애물 위치 정보를 저장
    while True:
        u, predicted_trajectory = dwa_control(x, config, goal, ob)      # DWA 를 이용하여 제어 입력을 계산함
        x = motion(x, u, config.dt)  # simulate robot                   # 로봇의 상태 업데이트
        trajectory = np.vstack((trajectory, x))  # store state history  # 경로에 현재 상태를 추가함

        if show_animation:
            plt.cla()                                                   # 현재 그림 지우기
            # for stopping simulation with the esc key.함
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")  # 예측 경로 표시
            plt.plot(x[0], x[1], "xr")                                  # 현재 위치 표시
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")                          # 장애물 위치 표시
            plot_robot(x[0], x[1], x[2], config)                    
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # check reaching goal                                           # 목표점에 도착했는지 확인함
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            print("Goal!!")
            break

    print("Done")
    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")              # 최종 경로를 시각화
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main(robot_type=RobotType.rectangle)
    # main(robot_type=RobotType.circle)
