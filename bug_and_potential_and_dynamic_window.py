""
"""
Integration of DWA and Bug Algorithms
author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı, Sarim Mehdi
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from collections import deque

show_animation = True               # 애니메이션을 보여줄지에 대한 여부 설정

# 기존 DWA 코드
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
        self.robot_type = RobotType.rectangle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 0.5  # [m] for collision check

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


# Bug 알고리즘 추가 구현
class BugPlanner:
    def __init__(self, start_x, start_y, goal_x, goal_y, obs_x, obs_y):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.obs_x = obs_x
        self.obs_y = obs_y
        self.r_x = [start_x]
        self.r_y = [start_y]
        self.out_x = []
        self.out_y = []
        for o_x, o_y in zip(obs_x, obs_y):
            for add_x, add_y in zip([1, 0, -1, -1, -1, 0, 1, 1],
                                    [1, 1, 1, 0, -1, -1, -1, 0]):
                cand_x, cand_y = o_x + add_x, o_y + add_y
                valid_point = True
                for _x, _y in zip(obs_x, obs_y):
                    if cand_x == _x and cand_y == _y:
                        valid_point = False
                        break
                if valid_point:
                    self.out_x.append(cand_x), self.out_y.append(cand_y)

    def mov_normal(self):
        return self.r_x[-1] + np.sign(self.goal_x - self.r_x[-1]), \
               self.r_y[-1] + np.sign(self.goal_y - self.r_y[-1])

    def mov_to_next_obs(self, visited_x, visited_y):
        for add_x, add_y in zip([1, 0, -1, 0], [0, 1, 0, -1]):
            c_x, c_y = self.r_x[-1] + add_x, self.r_y[-1] + add_y
            for _x, _y in zip(self.out_x, self.out_y):
                use_pt = True
                if c_x == _x and c_y == _y:
                    for v_x, v_y in zip(visited_x, visited_y):
                        if c_x == v_x and c_y == v_y:
                            use_pt = False
                            break
                    if use_pt:
                        return c_x, c_y, False
                if not use_pt:
                    break
        return self.r_x[-1], self.r_y[-1], True

    def bug0(self, ax):
        print("Bug Start")
        mov_dir = 'normal'
        cand_x, cand_y = -np.inf, -np.inf
        visited_x, visited_y = [], []

        # 기존 플롯(ax)에 장애물, 시작점, 목표점 추가
        ax.plot(self.obs_x, self.obs_y, ".k")
        ax.plot(self.r_x[-1], self.r_y[-1], "og")
        ax.plot(self.goal_x, self.goal_y, "xb")
        ax.grid(True)
        ax.set_title('DWA and BUG Algorithm Combined')

        while True:
            if self.r_x[-1] == self.goal_x and self.r_y[-1] == self.goal_y:
                print("Goal!")
                break
            if mov_dir == 'normal':
                cand_x, cand_y = self.mov_normal()
            if mov_dir == 'obs':
                cand_x, cand_y, _ = self.mov_to_next_obs(visited_x, visited_y)
            if mov_dir == 'normal':
                found_boundary = False
                for x_ob, y_ob in zip(self.out_x, self.out_y):
                    if cand_x == x_ob and cand_y == y_ob:
                        self.r_x.append(cand_x), self.r_y.append(cand_y)
                        visited_x[:], visited_y[:] = [], []
                        visited_x.append(cand_x), visited_y.append(cand_y)
                        mov_dir = 'obs'
                        found_boundary = True
                        break
                if not found_boundary:
                    self.r_x.append(cand_x), self.r_y.append(cand_y)
            elif mov_dir == 'obs':
                can_go_normal = True
                for x_ob, y_ob in zip(self.obs_x, self.obs_y):
                    if self.mov_normal()[0] == x_ob and self.mov_normal()[1] == y_ob:
                        can_go_normal = False
                        break
                if can_go_normal:
                    mov_dir = 'normal'
                else:
                    self.r_x.append(cand_x), self.r_y.append(cand_y)
                    visited_x.append(cand_x), visited_y.append(cand_y)

            # 기존 플롯(ax)에 경로 업데이트
            ax.plot(self.r_x, self.r_y, "-r")
            plt.pause(0.001)

        ax.plot(self.r_x, self.r_y, "-r", label = "Bug Path")
        # 최종 경로 표시 (중복 레이블 방지)
        ax.legend(loc="upper left")


# potential field algorithm
# Parameters
KP = 5.0  # attractive potential gain
ETA = 10.0  # repulsive potential gain
AREA_WIDTH = 50.0  # potential area width [m]
OSCILLATIONS_DETECTION_LENGTH = 5

show_animation = True


def calc_potential_field(gx, gy, ox, oy, rr, sx, sy):
    minx = min(min(ox), sx, gx) - AREA_WIDTH / 2.0
    miny = min(min(oy), sy, gy) - AREA_WIDTH / 2.0
    maxx = max(max(ox), sx, gx) + AREA_WIDTH / 2.0
    maxy = max(max(oy), sy, gy) + AREA_WIDTH / 2.0

    x_range = np.arange(minx, maxx, 1.0)
    y_range = np.arange(miny, maxy, 1.0)

    pmap = np.zeros((len(x_range), len(y_range)))

    for ix, x in enumerate(x_range):
        for iy, y in enumerate(y_range):
            ug = calc_attractive_potential(x, y, gx, gy)
            uo = calc_repulsive_potential(x, y, ox, oy, rr)
            pmap[ix, iy] = ug + uo

    return pmap, minx, miny, x_range, y_range


def calc_attractive_potential(x, y, gx, gy):
    return 0.5 * KP * np.hypot(x - gx, y - gy)


def calc_repulsive_potential(x, y, ox, oy, rr):
    dmin = float("inf")
    for i in range(len(ox)):
        d = np.hypot(x - ox[i], y - oy[i])
        dmin = min(dmin, d)

    if dmin <= rr:
        if dmin <= 0.1:
            dmin = 0.1
        return 0.5 * ETA * (1.0 / dmin - 1.0 / rr) ** 2
    else:
        return 0.0


def get_motion_model():
    return [[1, 0], [0, 1], [-1, 0], [0, -1],
            [-1, -1], [-1, 1], [1, -1], [1, 1]]


def oscillations_detection(previous_ids, ix, iy):
    previous_ids.append((ix, iy))
    if len(previous_ids) > OSCILLATIONS_DETECTION_LENGTH:
        previous_ids.popleft()

    return len(previous_ids) != len(set(previous_ids))


def potential_field_planning(sx, sy, gx, gy, ox, oy, rr, ax):
    """
    Potential Field 알고리즘 실행
    :param sx: 시작 위치 x
    :param sy: 시작 위치 y
    :param gx: 목표 위치 x
    :param gy: 목표 위치 y
    :param ox: 장애물 위치 x 리스트
    :param oy: 장애물 위치 y 리스트
    :param rr: 로봇 반경
    :param ax: 기존 플롯 창의 Axes 객체
    """
    pmap, minx, miny, x_range, y_range = calc_potential_field(gx, gy, ox, oy, rr, sx, sy)

    d = np.hypot(sx - gx, sy - gy)
    ix, iy = np.argmin(np.abs(x_range - sx)), np.argmin(np.abs(y_range - sy))
    gix, giy = np.argmin(np.abs(x_range - gx)), np.argmin(np.abs(y_range - gy))

    rx, ry = [sx], [sy]
    motion = get_motion_model()
    previous_ids = deque()

    # 기존 플롯에 장애물, 시작점, 목표점 추가
    # ax.scatter(ox, oy, color="black", label="Obstacles")
    # ax.scatter(sx, sy, color="blue", label="Start")
    # ax.scatter(gx, gy, color="red", label="Goal")
    # ax.legend(loc="upper left")

    while d >= 1.0:
        minp = float("inf")
        minix, miniy = -1, -1
        for mx, my in motion:
            inx, iny = ix + mx, iy + my
            if 0 <= inx < len(pmap) and 0 <= iny < len(pmap[0]):
                p = pmap[inx][iny]
                if p < minp:
                    minp = p
                    minix, miniy = inx, iny

        if minix == -1 or miniy == -1:
            print("Path blocked!")
            break

        ix, iy = minix, miniy
        xp, yp = x_range[ix], y_range[iy]
        d = np.hypot(gx - xp, gy - yp)
        rx.append(xp)
        ry.append(yp)

        if oscillations_detection(previous_ids, ix, iy):
            print(f"Oscillation detected at ({ix}, {iy})!")
            break

        # 기존 플롯(ax)에 경로 업데이트
        ax.plot(rx, ry, "-y")
        plt.pause(0.1)

    print("Goal!!")
    return rx, ry



def dwa_main(config, gx, gy, ax):
    """
    DWA 알고리즘 실행 (단일 플롯)
    """
    print("DWA Algorithm Start!")
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])  # 로봇 초기 상태
    goal = np.array([gx, gy])
    trajectory = np.array([x])  # 최종 경로를 저장할 배열
    ob = config.ob

    while True:
        u, predicted_trajectory = dwa_control(x, config, goal, ob)  # DWA 제어 계산
        x = motion(x, u, config.dt)  # 로봇 상태 업데이트
        trajectory = np.vstack((trajectory, x))  # 경로 업데이트

        ax.cla()
        ax.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g", label="DWA Predicted Path")
        ax.plot(x[0], x[1], "xr")
        ax.plot(goal[0], goal[1], "xb")
        ax.plot(ob[:, 0], ob[:, 1], "ok")
        plot_robot(x[0], x[1], x[2], config)
        ax.axis("equal")
        ax.grid(True)
        ax.set_title('DWA Algorithm')
        plt.pause(0.0001)

        # 목표 지점 도달 여부 확인
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            print("Goal Reached by DWA!")
            break

    # 최종 경로 시각화
    ax.plot(trajectory[:, 0], trajectory[:, 1], "-b", label="DWA Path")
    ax.legend(loc="upper left")

def bug_main(config, gx, gy, ax):
    """
    Bug 알고리즘 실행 (단일 플롯)
    """
    print("Bug Algorithm Start!")
    obs_x, obs_y = config.ob[:, 0], config.ob[:, 1]
    s_x, s_y = 0.0, 0.0
    bug_planner = BugPlanner(s_x, s_y, gx, gy, obs_x, obs_y)

    # `ax`를 전달하여 기존 플롯에서 실행
    bug_planner.bug0(ax)

def potential_field_main(config, gx, gy, ax):
    """
    Potential Field 알고리즘 실행 (단일 플롯)
    """
    print("Running Potential Field Algorithm...")
    sx, sy = 0.0, 0.0  # 시작 위치
    robot_radius = 5.0

    # 장애물 정보
    ox = config.ob[:, 0]
    oy = config.ob[:, 1]

    # Potential Field 알고리즘 실행
    rx, ry = potential_field_planning(sx, sy, gx, gy, ox, oy, robot_radius, ax)

    # 경로 시각화
    ax.plot(rx, ry, "-y", label="Potential Field Path")
    ax.legend(loc="upper left")

def main():
    # 전체 플롯 설정
    fig, ax = plt.subplots()

    # DWA 실행
    config = Config()
    gx, gy = 16.0, 16.0  # 목표 지점

    # DWA 실행
    dwa_main(config, gx, gy, ax)

    # Bug 알고리즘 실행
    bug_main(config, gx, gy, ax)

    # Potential Field 알고리즘 실행
    potential_field_main(config, gx, gy, ax)

    plt.title('DWA, Bug, and Potential Field Algorithms')
    plt.show()

if __name__ == '__main__':
    main()