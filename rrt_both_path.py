import math
import random
import matplotlib.pyplot as plt
import numpy as np


class RRTBothPath:
    """
    RRT_Both_Path algorithm
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    class AreaBounds:

        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=500,
                 play_area=None,
                 robot_radius=0.0):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        play_area:stay inside this area [xmin,xmax,ymin,ymax]
        robot_radius: robot body modeled as circle with given radius

        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.robot_radius = robot_radius

    def planning(self, animation=True):
        self.node_list_start = [self.start]  # 시작점으로 부터 확장하는 트리
        self.node_list_goal = [self.end]     # 목표점으로 부터 확장하는 트리

        for i in range(self.max_iter):
            rnd_node = self.get_random_node()

            # 시작 트리 확장
            nearest_ind_start = self.get_nearest_node_index(self.node_list_start, rnd_node)
            nearest_node_start = self.node_list_start[nearest_ind_start]
            new_node_start = self.steer(nearest_node_start, rnd_node, self.expand_dis)

            if self.check_if_outside_play_area(new_node_start, self.play_area) and \
                    self.check_collision(new_node_start, self.obstacle_list, self.robot_radius):
                self.node_list_start.append(new_node_start)

            # 목표 트리 확장
            nearest_ind_goal = self.get_nearest_node_index(self.node_list_goal, rnd_node)
            nearest_node_goal = self.node_list_goal[nearest_ind_goal]
            new_node_goal = self.steer(nearest_node_goal, rnd_node, self.expand_dis)

            if self.check_if_outside_play_area(new_node_goal, self.play_area) and \
                    self.check_collision(new_node_goal, self.obstacle_list, self.robot_radius):
                self.node_list_goal.append(new_node_goal)

            # 두 트리가 연결 가능한지 확인
            if self.connect_trees(new_node_start, self.node_list_goal):
                if animation:
                    self.draw_graph(rnd_node)
                return self.generate_final_course(new_node_start, self.node_list_goal)

            if self.connect_trees(new_node_goal, self.node_list_start):
                if animation:
                    self.draw_graph(rnd_node)
                return self.generate_final_course(new_node_goal, self.node_list_start)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

        return None  # 경로를 찾지 못함

    def connect_trees(self, new_node, target_tree):
        for target_node in target_tree:
            d, _ = self.calc_distance_and_angle(new_node, target_node)
            if d <= self.expand_dis:
                new_node.path_x += target_node.path_x
                new_node.path_y += target_node.path_y
                return True
        return False

    def generate_final_course(self, connecting_node, target_tree):
        path = [[connecting_node.x, connecting_node.y]]
        node = connecting_node
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.reverse()  # 시작점에서 중간 노드까지

        # 목표 트리에서 중간 노드부터 목표점까지 추가
        nearest_to_connecting = self.get_nearest_node_index(target_tree, connecting_node)
        node = target_tree[nearest_to_connecting]
        while node is not None:
            path.append([node.x, node.y])
            node = node.parent

        return path

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node
        return new_node

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list_start:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")
        for node in self.node_list_goal:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-b")
        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_circle(x, y, size, color="-k"):
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
                 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):
        if play_area is None:
            return True
        if node.x < play_area.xmin or node.x > play_area.xmax or \
                node.y < play_area.ymin or node.y > play_area.ymax:
            return False
        return True

    @staticmethod
    def check_collision(node, obstacleList, robot_radius):
        if node is None:
            return False
        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= (size + robot_radius) ** 2:
                return False  # collision
        return True

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


def main(gx=6.0, gy=10.0):
    print("RRT_Both_Path Planning")

    # ====Search Path with RRT====
    obstacleList = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2),
        (8, 10, 1)
    ]  # [x, y, radius]

    # Set Initial parameters
    rrt_connect = RRTBothPath(
        start=[0, 0],
        goal=[gx, gy],
        rand_area=[-2, 15],
        obstacle_list=obstacleList,
        robot_radius=0.8
    )
    path = rrt_connect.planning(animation=True)

    if path is None:
        print("Cannot find path")
    else:
        print("Found path!!")

        # Draw final path
        rrt_connect.draw_graph()
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()
