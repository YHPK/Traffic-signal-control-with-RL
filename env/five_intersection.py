from .env import EDGE, VEHICLE_MOVE, INTERSECTION

from time import sleep
import os

import matplotlib.pyplot as plt
import numpy as np
import csv


class FiveIntersection:
    def __init__(self, args):
        self.time_step = 0

        self.observation_space_size = 40
        self.action_space_size = 8
        self.tct = args.traffic_change_time
        self.render_type = args.render_type

        self.edge = EDGE(aar=args.average_arrival_time)
        self.move = VEHICLE_MOVE() # TASK_LIST updates the state of intersection

        self.central_intersection = INTERSECTION(self.edge, self.move, tct=args.traffic_change_time, rl_policy=True)
        self.out_up_intersection = INTERSECTION(self.edge, self.move, tct=args.traffic_change_time)
        self.out_down_intersection = INTERSECTION(self.edge, self.move, tct=args.traffic_change_time)
        self.out_left_intersection = INTERSECTION(self.edge, self.move, tct=args.traffic_change_time)
        self.out_right_intersection = INTERSECTION(self.edge, self.move, tct=args.traffic_change_time)


        self.central_intersection.neighbor['up'] = self.out_up_intersection
        self.central_intersection.neighbor['down'] = self.out_down_intersection
        self.central_intersection.neighbor['left'] = self.out_left_intersection
        self.central_intersection.neighbor['right'] = self.out_right_intersection

        self.out_up_intersection.neighbor['down'] = self.central_intersection
        self.out_down_intersection.neighbor['up'] = self.central_intersection
        self.out_left_intersection.neighbor['right'] = self.central_intersection
        self.out_right_intersection.neighbor['left'] = self.central_intersection


        self.central_intersection.intersection_initializer()
        self.out_up_intersection.intersection_initializer()
        self.out_down_intersection.intersection_initializer()
        self.out_left_intersection.intersection_initializer()
        self.out_right_intersection.intersection_initializer()


    def reset(self):
        self.time_step = 0

        self.central_intersection.clear_road()
        self.out_up_intersection.clear_road()
        self.out_down_intersection.clear_road()
        self.out_left_intersection.clear_road()
        self.out_right_intersection.clear_road()
        self.central_intersection.h_info.clear()

        return self.central_intersection.get_state()


    def printer(self):
        print("central_intersection")
        self.central_intersection.print_status()
        print("out_up_intersection")
        self.out_up_intersection.print_status()
        print("out_down_intersection")
        self.out_down_intersection.print_status()
        print("out_left_intersection")
        self.out_left_intersection.print_status()
        print("out_right_intersection")
        self.out_right_intersection.print_status()


    def step(self, action):
        # for itr in range(self.tct):
        self.edge.vehicle_generate(self.time_step)

        self.central_intersection.run(self.time_step, action)
        self.out_up_intersection.run(self.time_step, action)
        self.out_down_intersection.run(self.time_step, action)
        self.out_left_intersection.run(self.time_step, action)
        self.out_right_intersection.run(self.time_step, action)

        self.move.run()

        self.time_step += 1

        if self.render_type == 'print':
            self.show_status()
            sleep(0.3)

        return self.central_intersection.get_state(), self.central_intersection.reward, self.central_intersection.done


    def get_lqf_action(self):
        return self.central_intersection.LQF()


    def get_relative_traffic_flow(self):
        return self.central_intersection.rtf


    def get_avg_cross_block(self):
        return self.central_intersection.average_corss_blocking


    def show_status(self):
        print ("time step:", self.time_step)
        self.printer()


    def plot_show(self, intersection):

        plot_len = np.arange(len(intersection.h_info.average_delay))

        fig1 = plt.figure()
        fig2 = plt.figure()

        ax1 = fig1.add_subplot(2,2,1)
        ax1.plot(plot_len, intersection.h_info.average_delay)
        ax1.set_title("average delay")

        ax2 = fig1.add_subplot(2,2,2)
        ax2.plot(plot_len, intersection.h_info.average_cross_block)
        ax2.set_title("average cross block")

        ax3 = fig1.add_subplot(2,2,3)
        ax3.plot(plot_len, intersection.h_info.num_of_car)
        ax3.set_title("number of car in the intersection")

        ax4 = fig1.add_subplot(2,2,4)
        ax4.plot(plot_len, intersection.h_info.traffic_signal)
        ax4.set_title("traffic signal")

        if intersection.rl_policy:
            ax5 = fig2.add_subplot(2,1,1)
            ax5.plot(plot_len, intersection.h_info.rewards)
            ax5.set_title("rewards")

            ax6 = fig2.add_subplot(2,1,2)
            ax6.plot(np.arange(len(intersection.h_info.e_greedy)), intersection.h_info.e_greedy)
            ax6.set_title("e_greedy")

        plt.show()


    # def writer(self, intersection):
    #     f2 = open('data/average_cross_block.csv', 'a', encoding='utf-8', newline='')
    #     f3 = open('data/number_of_car_in_the_intersection.csv', 'a', encoding='utf-8', newline='')
    #     f4 = open('data/traffic_signal.csv', 'a', encoding='utf-8', newline='')
    #     f5 = open('data/rewards.csv', 'a', encoding='utf-8', newline='')
    #     f6 = open('data/e_greedy.csv', 'a', encoding='utf-8', newline='')
    #
    #     # wr1 = csv.writer(f1)
    #     wr2 = csv.writer(f2)
    #     wr3 = csv.writer(f3)
    #     wr4 = csv.writer(f4)
    #     wr5 = csv.writer(f5)
    #     wr6 = csv.writer(f6)
    #
    #
    #     # wr1.writerow(intersection.h_info.average_delay)
    #     wr2.writerow(intersection.h_info.average_cross_block)
    #     wr3.writerow(intersection.h_info.num_of_car)
    #     wr4.writerow(intersection.h_info.traffic_signal)
    #     wr5.writerow(intersection.h_info.rewards)
    #     wr6.writerow(intersection.h_info.e_greedy)
    #
    #     # f1.close()
    #     f2.close()
    #     f3.close()
    #     f4.close()
    #     f5.close()
    #     f6.close()

    # def logger(self, intersection):
        # intersection.h_info.average_cross_block
        # intersection.h_info.average_delay
