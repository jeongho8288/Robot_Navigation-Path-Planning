"""
Bug Planning
author: Sarim Mehdi(muhammadsarim.mehdi@studio.unibo.it)
Source: https://sites.google.com/site/ece452bugalgorithms/
"""

import numpy as np
import matplotlib.pyplot as plt

show_animation = True

# BugPlanner 클래스 정의
class BugPlanner:
    def __init__(self, start_x, start_y, goal_x, goal_y, obs_x, obs_y):
        self.goal_x = goal_x                                                # 목적지의 좌표 저장
        self.goal_y = goal_y
        self.obs_x = obs_x                                                  # 장애물에 대한 좌표 목록 저장
        self.obs_y = obs_y
        self.r_x = [start_x]                                                # 로봇의 현재 위치를 기록할 리스트 생성 (시작 지점으로 초기화)
        self.r_y = [start_y]
        self.out_x = []                                                     # 장애물 경계에 대한 포인트들 저장할 리스트 생성
        self.out_y = []
        
        for o_x, o_y in zip(obs_x, obs_y):
            for add_x, add_y in zip([1, 0, -1, -1, -1, 0, 1, 1],            # 각 경계 지점에 대해 8-neighbor으로 확장한 경계 포인트 저장
                                    [1, 1, 1, 0, -1, -1, -1, 0]):
                cand_x, cand_y = o_x + add_x, o_y + add_y
                valid_point = True
                for _x, _y in zip(obs_x, obs_y):                            # 이미 장애물 포인트와 겹치면 valid_point를 false로 설정
                    if cand_x == _x and cand_y == _y:
                        valid_point = False
                        break
                if valid_point:                                             # 겹치지 않는 지점인 경우 경계 목록 리스트에 추가
                    self.out_x.append(cand_x), self.out_y.append(cand_y)

    def mov_normal(self):                                                   # 목적지 방향으로 한 단계 이동하는 함수
        # 목적지의 x좌표와 y좌표 그리고 현재 로봇의 x좌표 y좌표간의 차이를 계산하여 양수면 1 음수면 -1 같으면 0 을 더한 좌표를 반환
        return self.r_x[-1] + np.sign(self.goal_x - self.r_x[-1]), \
               self.r_y[-1] + np.sign(self.goal_y - self.r_y[-1])

    def mov_to_next_obs(self, visited_x, visited_y):                        # 로봇이 장애물을 따라 이동할 때 다음 이동 포인트를 찾는 함수
        for add_x, add_y in zip([1, 0, -1, 0], [0, 1, 0, -1]):              # 현재 위치에서 4방향으로 탐색
            c_x, c_y = self.r_x[-1] + add_x, self.r_y[-1] + add_y           # 현재 좌표에서 상하좌우로 움직인 좌표를 추가
            for _x, _y in zip(self.out_x, self.out_y):
                use_pt = True
                if c_x == _x and c_y == _y:                                 # 추가한 좌표가 장애물의 경계 포인트와 일치하는지 확인
                    for v_x, v_y in zip(visited_x, visited_y):              # 이미 방문한 포인트인지에 대한 확인
                        if c_x == v_x and c_y == v_y:
                            use_pt = False
                            break
                    if use_pt:                                              # 새로운 탐색 좌표가 방문하지 않은 포인트인 경우 
                        return c_x, c_y, False                              # 해당 좌표를 이동 포인트로 선택하여 반환
                if not use_pt:
                    break
        return self.r_x[-1], self.r_y[-1], True                             # 유효한 이동 포인트가 없다면, 현재 로봇의 위치를 그대로 반환함

    # Bug 0 알고리즘 실행
    def bug0(self):
        """
        목표를 향해 탐욕적으로 이동하다가 장애물을 만나면 장애물을 우회.
        다시 목표 방향으로 이동 가능한 지점이 나오면 다시 탐욕적으로 이동.
        """
        mov_dir = 'normal'                                                           # 현재 이동 상태로서 'normal' or 'obs'로 결정
        cand_x, cand_y = -np.inf, -np.inf                                            # 후보 지점의 좌표를 -inf 값으로 초기화
        if show_animation:                                                           # 장애물, 시작 지점, 목표 지점 시각화
            plt.plot(self.obs_x, self.obs_y, ".k")                                   # 장애물 위치 표시
            plt.plot(self.r_x[-1], self.r_y[-1], "og")                               # 로봇의 시작점 표시
            plt.plot(self.goal_x, self.goal_y, "xb")                                 # 목적지 위치 표시
            plt.plot(self.out_x, self.out_y, ".")
            plt.grid(True)
            plt.title('BUG 0')

        for x_ob, y_ob in zip(self.out_x, self.out_y):
            if self.r_x[-1] == x_ob and self.r_y[-1] == y_ob:                        # 현재 위치가 장애물의 경계 포인트에 있는지 확인
                mov_dir = 'obs'                                                      # 위치가 장애물의 경계라면 이동 상태를 변경
                break

        visited_x, visited_y = [], []                                                # 방문한 지점들 저장할 리스트 생성
        while True:
            if self.r_x[-1] == self.goal_x and self.r_y[-1] == self.goal_y:          # 목적지에 도달한 경우 종료
                break
            if mov_dir == 'normal':                                                  # 이동 상태가 'normal'인 경우
                cand_x, cand_y = self.mov_normal()                                   # 목적지를 향해서 (1,1)만큼 이동하는 함수 실행
            if mov_dir == 'obs':                                                     # 이동 상태가 'obs'인 경우
                cand_x, cand_y, _ = self.mov_to_next_obs(visited_x, visited_y)       # 장애물의 경로로 이동하는 함수 실행
            if mov_dir == 'normal':                                                  # 이동 상태가 'normal'인 경우
                found_boundary = False
                for x_ob, y_ob in zip(self.out_x, self.out_y):
                    if cand_x == x_ob and cand_y == y_ob:                            # 후보 좌표가 장애물의 외각의 좌표와 일치하는 경우
                        self.r_x.append(cand_x), self.r_y.append(cand_y)             # 후보 좌표를 로봇의 이동경로 리스트에 저장
                        visited_x[:], visited_y[:] = [], []                          # 방문한 리스트를 초기화
                        visited_x.append(cand_x), visited_y.append(cand_y)           # 방문한 리스트에 후보 좌표를 추가
                        mov_dir = 'obs'                                              # 이제 장애물을 따라 이동하므로 이동 상태를 'obs'로 변경
                        found_boundary = True                                        # 경계를 찾았으므로 True로 변경
                        break
                if not found_boundary:                                               # 로봇이 장애물의 경계에 도달하지 못한 경우
                    self.r_x.append(cand_x), self.r_y.append(cand_y)                 # 후보 좌표를 추가하기만 함 -> 직진
            elif mov_dir == 'obs':                                                   # 이동 상태가 'obs'인 경우
                can_go_normal = True
                for x_ob, y_ob in zip(self.obs_x, self.obs_y):
                    if self.mov_normal()[0] == x_ob and self.mov_normal()[1] == y_ob:   # 목표 방향으로 이동 가능한지 확인
                        can_go_normal = False                                        # 이동 불가능하다면 False로 변경
                        break
                if can_go_normal:
                    mov_dir = 'normal'                                               # 이동 가능한 경우 탐욕적으로 (1,1) 이동
                else:                                                                # 이동 불가능한 경우
                    self.r_x.append(cand_x), self.r_y.append(cand_y)                 # 장애물의 경계를 따라가는 후보 좌표를 로봇의 경로 리스트에 추가 
                    visited_x.append(cand_x), visited_y.append(cand_y)
            if show_animation:                                                       # 애니메이션으로 경로 시각화
                plt.plot(self.r_x, self.r_y, "-r")                                   # 로봇의 위치를 빨간색 실선으로 표시
                plt.pause(0.001)
        if show_animation:
            plt.show()                                                               # 최종 결과 출력

    # Bug 1 알고리즘 실행
    def bug1(self):
        """
        장애물을 만나면 장애물의 처음 만난 지점으로 돌아간 후,
        목표에 가장 가까운 지점을 찾아 다시 탐욕적으로 이동.
        """
        mov_dir = 'normal'                                                           # 로봇이 목표점을 향해 가도록 초기 이동 상태를 'normal'로 설정
        cand_x, cand_y = -np.inf, -np.inf                                            # 후보 좌표를 초기화
        exit_x, exit_y = -np.inf, -np.inf                                            # 출구 변수에 대한 좌표를 초기화
        dist = np.inf                                                                # 거리를 초기화
        back_to_start = False                                                        # 장애물 처음 만난 지점으로 돌아왔는지에 대한 플래그
        second_round = False                                                         # 두 번째 우회 여부에 대한 플래그

        if show_animation:
            plt.plot(self.obs_x, self.obs_y, ".k")                                   # 장애물, 시작 지점, 목표 지점 시각화
            plt.plot(self.r_x[-1], self.r_y[-1], "og")
            plt.plot(self.goal_x, self.goal_y, "xb")
            plt.plot(self.out_x, self.out_y, ".")
            plt.grid(True)
            plt.title('BUG 1')

        # BUG0과 동일
        for xob, yob in zip(self.out_x, self.out_y):
            if self.r_x[-1] == xob and self.r_y[-1] == yob:
                mov_dir = 'obs'
                break

        visited_x, visited_y = [], []  # BUG0과 동일

        while True:
            if self.r_x[-1] == self.goal_x and self.r_y[-1] == self.goal_y:          # 목표에 도달한 경우 종료
                break
            if mov_dir == 'normal':                                                  # 이동 상태에 따라 다음 후보 포인트 결정
                cand_x, cand_y = self.mov_normal()                                   # 목표점으로 이동
            if mov_dir == 'obs':
                cand_x, cand_y, back_to_start = self.mov_to_next_obs(visited_x, visited_y)  # 장애물 우회하는 함수 실행
            if mov_dir == 'normal':                                                  # 경계에 도달한 경우 우회 시작
                found_boundary = False
                for x_ob, y_ob in zip(self.out_x, self.out_y):
                    if cand_x == x_ob and cand_y == y_ob:                            # BUG0과 동일 후보 좌표가 장애물의 경계인 경우
                        self.r_x.append(cand_x), self.r_y.append(cand_y)
                        visited_x[:], visited_y[:] = [], []                          # 방문한 지점 초기화
                        visited_x.append(cand_x), visited_y.append(cand_y)
                        mov_dir = 'obs'                                              # 이동 상태를 'obs'로 변경
                        dist = np.inf                                                # 새로운 장애물을 만난 경우이므로 거리를 초기화
                        back_to_start = False                                        # 출발지로 돌아가는 경우가 아니므로 False 설정
                        second_round = False                                         # 로봇이 두 번째 우회를 하는 경우가 아니므로 False설정
                        found_boundary = True                                        # 새로운 장애물의 경계를 만났으므로 True설정
                        break
                if not found_boundary:                                               # 경계에 도달하지 않은 경우 계속 이동
                    self.r_x.append(cand_x), self.r_y.append(cand_y)
            elif mov_dir == 'obs':
                # 목적지와 현재 로봇의 좌표를 통해 목적지와의 거리 계산
                d = np.linalg.norm(np.array([cand_x, cand_y]) - np.array([self.goal_x, self.goal_y]))
                if d < dist and not second_round:                                    # 목표에 더 가까운 지점이고 두 번째 우회가 아니면 갱신
                    exit_x, exit_y = cand_x, cand_y                                  # 출구 좌표를 후보 좌표로 갱신
                    dist = d                                                         # 구한 d의 거리를 dist로 갱신
                if back_to_start and not second_round:                               # 로봇이 처음 만난 지점으로 돌아가는데 두 번째 우회가 아닌 경우
                    second_round = True                                              # 두 번째 우회를 True로 설정
                    del self.r_x[-len(visited_x):]                                   # 방문했던 경로들의 기록 삭제
                    del self.r_y[-len(visited_y):]
                    visited_x[:], visited_y[:] = [], []                              # 방문했던 지점 초기화
                self.r_x.append(cand_x), self.r_y.append(cand_y)                     # BUG0과 동일 - 각 리스트에 후보 좌표를 추가
                visited_x.append(cand_x), visited_y.append(cand_y)
                if cand_x == exit_x and cand_y == exit_y and second_round:           # 목표에 가장 가까운 지점에서 다시 이동 시작
                    mov_dir = 'normal'                                               # 이동 상태를 'normal'로 변경하여 목표를 향해 이동
            if show_animation:                                                       # 애니메이션을 통한 경로 시각화 
                plt.plot(self.r_x, self.r_y, "-r")
                plt.pause(0.001)
        if show_animation:
            plt.show()

    # Bug 2 알고리즘 실행 함수
    def bug2(self):
        """
        목표에 가까워지는 동안 장애물의 경계를 따라가다가,
        목표에서 멀어지기 시작하면 그 지점에서 다시 목표를 향해 이동.
        """
        mov_dir = 'normal'                                                           # 로봇이 목표점을 향해 가도록 초기 이동 상태를 'normal'로 설정
        cand_x, cand_y = -np.inf, -np.inf                                            # 후보 좌표를 초기화
        if show_animation:                                                           # 장애물, 시작 지점, 목표 지점 시각화
            plt.plot(self.obs_x, self.obs_y, ".k")
            plt.plot(self.r_x[-1], self.r_y[-1], "og")
            plt.plot(self.goal_x, self.goal_y, "xb")
            plt.plot(self.out_x, self.out_y, ".")

        straight_x, straight_y = [self.r_x[-1]], [self.r_y[-1]]                      # 직진 경로에 시작 좌표를 추가
        hit_x, hit_y = [], []                                                        # 로봇이 장애물과 부딪히는 지점을 기록하는 리스트 생성

        while True:                                                                  # 목적지를 향해 직진하면서 장애물 만날 때까지 이동
            if straight_x[-1] == self.goal_x and straight_y[-1] == self.goal_y:      # 목표에 도달한 경우 종료
                break
            c_x = straight_x[-1] + np.sign(self.goal_x - straight_x[-1])             # 목적지를 향해 직진하여 다음 후보 좌표 계산
            c_y = straight_y[-1] + np.sign(self.goal_y - straight_y[-1])
    
            for x_ob, y_ob in zip(self.out_x, self.out_y):                           # 장애물과의 충돌 여부 확인
                if c_x == x_ob and c_y == y_ob:
                    hit_x.append(c_x), hit_y.append(c_y)                             # 만약 충돌했다면 충돌한 지점을 기록
                    break
            straight_x.append(c_x), straight_y.append(c_y)                           # 직진 경로를 갱신
        if show_animation:                                                           # 직진 경로와 충돌 지점을 시각화
            plt.plot(straight_x, straight_y, ",")
            plt.plot(hit_x, hit_y, "d")
            plt.grid(True)
            plt.title('BUG 2')

        for x_ob, y_ob in zip(self.out_x, self.out_y):                               # 현재 위치가 장애물의 경계에 있는지 확인하여 이동 상태 변경
            if self.r_x[-1] == x_ob and self.r_y[-1] == y_ob:
                mov_dir = 'obs'                                                      # 이동 상태를 'obs'로 변경
                break

        visited_x, visited_y = [], []                                                # 방문한 지점들 저장
        while True:
            if self.r_x[-1] == self.goal_x and self.r_y[-1] == self.goal_y:          # 목표에 도달한 경우 종료
                break
            if mov_dir == 'normal':                                                  # 이동 상태가 'normal'인 경우
                cand_x, cand_y = self.mov_normal()                                   # 목적지를 향해서 (1,1)만큼 이동하는 함수 실행
            if mov_dir == 'obs':                                                     # 이동 상태가 'obs'인 경우
                cand_x, cand_y, _ = self.mov_to_next_obs(visited_x, visited_y)       # 장애물 우회하는 함수 실행
            if mov_dir == 'normal':
                found_boundary = False
                for x_ob, y_ob in zip(self.out_x, self.out_y):
                    if cand_x == x_ob and cand_y == y_ob:                            # 장애물의 경계 포인트에 도달한 경우
                        self.r_x.append(cand_x), self.r_y.append(cand_y)
                        visited_x[:], visited_y[:] = [], []                          # 방문한 지점을 초기화
                        visited_x.append(cand_x), visited_y.append(cand_y)
                        del hit_x[0]                                                 # 충돌 지점 삭제
                        del hit_y[0]
                        mov_dir = 'obs'                                              # 이동 상태를 'obs'로 변경
                        found_boundary = True                                        # 장애물의 경계를 찾았으므로 True로 변경
                        break
                if not found_boundary:                                               # 장애물의 경계에 도달하지 못한 경우 계속 이동
                    self.r_x.append(cand_x), self.r_y.append(cand_y)                 # 이동 경로에 후보 좌표를 추가
            elif mov_dir == 'obs':
                self.r_x.append(cand_x), self.r_y.append(cand_y)                     # 경계를 따라 이동하며 목적지에 가까워지는 지점을 확인
                visited_x.append(cand_x), visited_y.append(cand_y)
                for i_x, i_y in zip(range(len(hit_x)), range(len(hit_y))):
                    if cand_x == hit_x[i_x] and cand_y == hit_y[i_y]:                # 충돌 지점에 도달한 경우 다시 목적지를 향해 이동 시작
                        del hit_x[i_x]                                               # 해당 충돌 지점을 삭제
                        del hit_y[i_y]
                        mov_dir = 'normal'                                           # 이동 상태 'normal'로 변경
                        break
            if show_animation:
                plt.plot(self.r_x, self.r_y, "-r")
                plt.pause(0.001)
        if show_animation:                                                           # 최종 경로 시각화
            plt.show()

# main 함수 - 알고리즘 실행
def main(bug_0, bug_1, bug_2):  # bug_0, bug_1, bug_2 중 실행할 알고리즘 선택
   
    o_x, o_y = [], []           # 장애물 위치 설정
    
    s_x = 0.0                   # 시작 지점과 목표 지점 설정
    s_y = 0.0
    g_x = 167.0
    g_y = 50.0

    for i in range(20, 40):      # 장애물 설정 (여러 사각형 형태의 장애물 생성)
        for j in range(20, 40):
            o_x.append(i)
            o_y.append(j)

    for i in range(60, 100):
        for j in range(40, 80):
            o_x.append(i)
            o_y.append(j)

    for i in range(120, 140):
        for j in range(80, 100):
            o_x.append(i)
            o_y.append(j)

    for i in range(80, 140):
        for j in range(0, 20):
            o_x.append(i)
            o_y.append(j)

    for i in range(0, 20):
        for j in range(60, 100):
            o_x.append(i)
            o_y.append(j)

    for i in range(20, 40):
        for j in range(80, 100):
            o_x.append(i)
            o_y.append(j)

    for i in range(120, 160):
        for j in range(40, 60):
            o_x.append(i)
            o_y.append(j)

    # 선택한 알고리즘을 실행
    if bug_0:
        my_Bug = BugPlanner(s_x, s_y, g_x, g_y, o_x, o_y)
        my_Bug.bug0()
    if bug_1:
        my_Bug = BugPlanner(s_x, s_y, g_x, g_y, o_x, o_y)
        my_Bug.bug1()
    if bug_2:
        my_Bug = BugPlanner(s_x, s_y, g_x, g_y, o_x, o_y)
        my_Bug.bug2()

if __name__ == '__main__':
    main(bug_0=True, bug_1=False, bug_2=False)  # bug_0, bug_1, bug_2 중 bug_0 실행하도록 선택
