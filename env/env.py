import numpy as np

import queue
import sys
import random
import math
import os



MAX_QUEUE_SIZE = 40

class HISTORICAL_INFO:

    def __init__(self, rl_policy=False):

        self.rl_policy = rl_policy

        if rl_policy:
            self.average_cross_block = []
            self.average_delay = []
            self.num_of_car = []
            self.traffic_signal = []
            self.rewards = []
            self.e_greedy = []
            self.rtf = []

    def clear(self):

        if self.rl_policy:
            self.average_cross_block = []
            self.average_delay = []
            self.num_of_car = []
            self.traffic_signal = []
            self.rewards = []
            self.e_greedy = []
            self.rtf = []


class VEHICLE_MOVE:

    def __init__(self):
        self.list = []

    def run(self):
        for t in self.list:
            # add 1 time step when vehicle move to next lane
            t[1].capacity.put_nowait(t[0].capacity.get_nowait() + 1)
        self.list = [] # empty the task list


class LANE:

    def __init__(self):

        self.capacity = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.next = []


    def get_total_delay(self, time_step):

        delay_sum = 0
        temp_q = queue.Queue()

        while(not self.capacity.empty()):
            delay = self.capacity.get_nowait()
            delay_sum += (time_step - delay)
            temp_q.put_nowait(delay)

        while(not temp_q.empty()):
            self.capacity.put_nowait(temp_q.get_nowait())

        return delay_sum

    def add_delay(self, add_val):

        temp_q  = queue.Queue()

        while(not self.capacity.empty()):
            delay = self.capacity.get_nowait() - add_val
            temp_q.put_nowait(delay)

        while(not temp_q.empty()):
            self.capacity.put_nowait(temp_q.get_nowait())


    def get_num_of_car(self):

        return self.capacity.qsize()


    def clear(self):

        self.capacity = queue.Queue(maxsize=MAX_QUEUE_SIZE)


class EDGE:

    def __init__(self, aar=0.1):
        self.START = []
        self.average_arrival_rate = aar

    def vehicle_generate(self, time_step):

        for e in self.START:
            p = self.poisson_probability(1, self.average_arrival_rate)
            if random.random() < p and e.capacity.qsize() < MAX_QUEUE_SIZE:
            # if e.capacity.qsize() < MAX_QUEUE_SIZE:
                e.capacity.put_nowait(time_step)

    def poisson_probability(self, actual, mean):

        # return math.exp(-mean) * mean**actual / math.factorial(actual)
        p = math.exp(-mean)
        for i in range(actual):
            p *= mean
            p /= i+1

        return p


class INTERSECTION:

    def __init__(self, edge, task, tct=20, rl_policy=False):

        # neighbor intersections
        self.neighbor = {'up': None, 'down': None, 'left': None, 'right': None}
        self.edge = edge

        # choose traffic policy True: RL method, False: LQF
        self.rl_policy = rl_policy

        # lane the number from 1 to 8
        self.road = [LANE(), LANE(), LANE(), LANE(), LANE(), LANE(), LANE(), LANE()]

        # available traffic signal set
        self.traffic_light_set = [(0,4), (0,5), (1,4), (1,5), (2,6), (2,7), (3,6), (3,7)]
        self.current_traffic_light = random.randint(0, 7) # initial value

        # task list: use when update the state of the intersection
        self.task = task

        # cross blocking count
        self.cross_blocking_cnt = 0.0
        self.average_corss_blocking = 0.0

        # relative traffic flow
        self.rtf = np.zeros(8)

        # reward
        self.reward = 0

        # delay of current intersection
        self.current_delay = 0.0
        self.last_delay = 0.0

        # number of car in the intersection
        self.num_of_car = 0

        # episode is over or not
        self.done = False

        self.h_info = HISTORICAL_INFO(rl_policy=rl_policy)

        self.traffic_change_time = tct


    def clear_road(self):

        for lane in self.road:
            lane.clear()

        self.cross_blocking_cnt = 0.0
        self.average_corss_blocking = 0.0
        self.rtf = np.zeros(8)
        self.reward = 0
        self.current_delay = 0.0
        self.last_delay = 0.0
        self.num_of_car = 0
        self.done = False


    def intersection_concatenater(self, neighbor_I, f, t1, t2):

        self.road[f].next.append(neighbor_I.road[t1])
        self.road[f].next.append(neighbor_I.road[t2])


    def intersection_combiner(self, neighbor_I, f, t1, t2, start):

        if isinstance(neighbor_I, INTERSECTION):
            for f_num in f:
                self.intersection_concatenater(neighbor_I, f_num, t1, t2)

        elif neighbor_I == None:
            for s in start:
                self.edge.START.append(self.road[s])

        else:
            sys.exit("wrong intersection type: INTERSECTION or EDGE")


    def get_total_delay(self, time_step):

        total_delay = 0

        for lane in self.road:
            total_delay += lane.get_total_delay(time_step)

        return total_delay


    def add_delay(self, add_val):

        for lane in self.road:
            lane.add_delay(add_val)


    def relative_traffic_flow(self, time_step):
        '''
        The relative traffic flow is defined as the total delay of vehicles
        in a lane divided by the average delay at all lanes in the intersection.
        '''

        total_delay_of_vehicles_in_a_lane = []
        average_delay = 0.0
        rtf = []

        for lane in self.road:
            total_delay = lane.get_total_delay(time_step)
            total_delay_of_vehicles_in_a_lane.append(total_delay)
            average_delay += total_delay

        average_delay = average_delay/8

        for t in total_delay_of_vehicles_in_a_lane:
            rtf.append(t/average_delay if average_delay > 0 else 0)

        self.rtf = np.array(rtf)


    def get_state(self): # for CRL

        state = np.array(self.rtf)
        for n in self.neighbor:
            if self.neighbor[n] != None:
                state = np.hstack((state, self.neighbor[n].rtf))

        return state


    def get_num_of_car(self):

        num = 0
        for lane in self.road:
            num += lane.get_num_of_car()

        return num


    def LQF(self):

        longest_lane = 0
        traffic_light = 0
        for i, t in enumerate(self.traffic_light_set):
            len_sum = self.road[t[0]].capacity.qsize() + self.road[t[1]].capacity.qsize()
            if longest_lane < len_sum:
                longest_lane = len_sum
                traffic_light = i

        return traffic_light


    def intersection_initializer(self):

        # from lane 5,6,8 -> to up lane 3,8
        self.intersection_combiner(self.neighbor['up'], (4, 5, 7), 2, 7, (3, 6))
        # from lane 1,2,4 -> to down lane 4,7
        self.intersection_combiner(self.neighbor['down'], (0, 1, 3), 3, 6, (2, 7))
        # from lane 3,4,6 -> to left lane 6,1
        self.intersection_combiner(self.neighbor['left'], (2, 3, 5), 5, 0, (4, 1))
        # from lane 7,8,2 -> to right lane 5,2
        self.intersection_combiner(self.neighbor['right'], (6, 7, 1), 4, 1, (5, 0))



    def run(self, time_step, action):

        '''
        the run_setting() function identify the vehicle that move to next intersection.

        -- current --
        randomly chosen a next lane to move, that is does not follow any rules.
        follow the LQF policy that choose the traffic light.
        '''

        if self.rl_policy:
            self.current_traffic_light = action
        elif not self.rl_policy and time_step % self.traffic_change_time == 0:
            self.current_traffic_light = self.LQF()


        if time_step % self.traffic_change_time == 0:
            self.average_corss_blocking = self.cross_blocking_cnt / self.traffic_change_time
            self.cross_blocking_cnt = 0.0


        # for lane in self.road: # modify to have follow policy
        for t in self.traffic_light_set[self.current_traffic_light]:
            lane = self.road[t]
            if lane.capacity.qsize() > 0 and lane.capacity.qsize() <= MAX_QUEUE_SIZE:
                if lane.next != []:
                    rand_num = random.randint(0, len(lane.next)-1)
                    if lane.next[rand_num].capacity.qsize() <= MAX_QUEUE_SIZE -1:
                        self.task.list.append((lane, lane.next[rand_num]))
                    else:
                        self.cross_blocking_cnt += 1 # count cross-blocking
                else: # escape condition
                    lane.capacity.get_nowait()
            else: # queue is empty.
                pass

        # calculate the relative traffic flow
        self.relative_traffic_flow(time_step)

        # calculate delay of current state
        self.last_delay = self.current_delay
        self.current_delay = self.get_total_delay(time_step)


        self.reward = self.last_delay - self.current_delay

        self.num_of_car = self.get_num_of_car()
        # if self.num_of_car > MAX_QUEUE_SIZE * 6: # done condition
        #     self.done = True
        #     self.reward = -1


        if self.rl_policy:
            # save timeseries information
            self.h_info.average_cross_block.append(self.average_corss_blocking)
            self.h_info.average_delay.append(self.current_delay/np.max((self.num_of_car, 1)))
            self.h_info.num_of_car.append(self.num_of_car)
            self.h_info.traffic_signal.append(self.current_traffic_light)
            self.h_info.rewards.append(self.reward)
            self.h_info.rtf.append(self.rtf)


    def print_status(self):

        print("relative traffic flow:", np.around(self.rtf, 2))
        l1, l2 = self.traffic_light_set[self.current_traffic_light]
        print("current delay: ", self.current_delay)
        print("last delay: ", self.last_delay)
        print("average cross-blocking: ", self.cross_blocking_cnt)
        print("average delay per vehicle: ", self.current_delay/np.max((self.num_of_car, 1)))
        if self.rl_policy:
            print("reward: ", self.reward)
        print("traffic signal:", l1+1, l2+1 )
        for i, r in enumerate(self.road):
            len = r.capacity.qsize()
            print("lane" + str(i+1) + ":", "#"*len, "-", len)
