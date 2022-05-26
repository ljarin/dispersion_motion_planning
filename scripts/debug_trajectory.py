from kr_planning_msgs.msg import SplineTrajectory
import rospy
import matplotlib.pyplot as plt
import numpy as np
from nav_msgs.msg import Odometry
from matplotlib.animation import FuncAnimation
from kr_planning_msgs.msg import PlanTwoPointActionGoal
from matplotlib.collections import LineCollection
import matplotlib as mpl
from visualization_msgs.msg import MarkerArray


class CheckTrajectory():
    def __init__(self):
        # self.fig, self.ax[0] = plt.subplots(3, 1)
        self.fig, self.ax = plt.subplots(1, 3)
        self.ax[1].set_xlim(0, 15)
        self.ax[1].set_ylim(-6, 6)
        self.ax[2].set_xlim(0, 15)
        self.ax[2].set_ylim(-6, 6)


        self.seg_number = -1
        self.time_elapsed = -1
        self.time_remaining = 0

        odom_ln, = self.ax[0].plot([], [], 'bo', label="Odom")
        self.cmap = mpl.cm.Set1
        traj_ln = self.ax[0].add_collection(LineCollection([], cmap=self.cmap, alpha=.5))
        traj_ln.set_linewidth(3)
        bounds = np.arange(self.cmap.N-1)
        norm = mpl.colors.BoundaryNorm(bounds, self.cmap.N, extend='both')
        traj_ln.set_norm(norm)
        traj_ln2 = self.ax[0].add_collection(LineCollection([], cmap=self.cmap))
        traj_ln2.set_linewidth(2)
        traj_ln2.set_norm(norm)

        self.fig.colorbar(traj_ln2, label="Planner trajectory")
        # self.fig.colorbar(traj_ln, label="Tracker trajectory")
        start_ln, = self.ax[0].plot([], [], 'go', label="Planner start")
        goal_ln, = self.ax[0].plot([], [], 'ro', label="Planner goal")
        seg_num_ln = self.ax[0].text(1.1, 1.1, '', fontsize=15, horizontalalignment='center',
                                     verticalalignment='center', transform=self.ax[0].transAxes)
        traj_feedback_ln, = self.ax[0].plot([], [])
        # traj_feedback_ln, = self.ax[0].plot([], [], 'k*', label="Calculated Position from Feedback")
        query_ln, = self.ax[0].plot([], [], 'yo', label="Planner query start")

        velocity_ln = self.ax[1].add_collection(LineCollection([], cmap=self.cmap))
        velocity_ln.set_linewidth(2)
        velocity_ln.set_norm(norm)
        velocity_ln2 = self.ax[1].add_collection(LineCollection([], cmap=self.cmap))
        velocity_ln2.set_linewidth(2)
        velocity_ln2.set_norm(norm)

        accel_ln = self.ax[2].add_collection(LineCollection([], cmap=self.cmap))
        accel_ln.set_linewidth(2)
        accel_ln.set_norm(norm)
        accel_ln2 = self.ax[2].add_collection(LineCollection([], cmap=self.cmap))
        accel_ln2.set_linewidth(2)
        accel_ln2.set_norm(norm)

        self.ln = [odom_ln, traj_ln, start_ln, goal_ln, seg_num_ln, traj_feedback_ln,
                   traj_ln2, query_ln, velocity_ln, velocity_ln2, accel_ln, accel_ln2]
        self.ax[0].legend()

        self.odom_data = np.zeros(3)
        self.odom_history = []
        # self.tracker_pos_data = []
        # self.tracker_vel_data = []
        self.planner_pos_data = []
        self.planner_vel_data = []
        self.planner_acc_data = []
        self.planner_ts = []
        self.time_data = np.zeros(1)
        self.first_pos = True
        self.traj_feedback = np.zeros(3)
        self.start = None
        self.goal = None
        self.planning_query_start = None

        rospy.Subscriber("/quadrotor/local_plan_server/trajectory", SplineTrajectory, self.planner_spline_traj_cb, queue_size=100)
        # rospy.Subscriber("/quadrotor/quadrotor_manager_control/trajectory",
        #                  SplineTrajectory, self.traj_tracker_spline_traj_cb, queue_size=100)
        # rospy.Subscriber("/quadrotor/trackers_manager/execute_trajectory/feedback",
        #                  RunTrajectoryActionFeedback, self.feedback_callback, queue_size=1)
        rospy.Subscriber("/quadrotor/odom", Odometry, self.odom_callback, queue_size=100)
        rospy.Subscriber("/quadrotor/local_plan_server/start_and_goal", MarkerArray, self.start_goal_callback, queue_size=100)
        rospy.Subscriber("/quadrotor/local_plan_server/plan_local_trajectory/goal",
                         PlanTwoPointActionGoal, self.action_goal_callback, queue_size=100)

    def spline_traj_process(self, msg):
        positions = [[], [], []]
        velocities = [[], [], []]
        accelerations = [[], [], []]
        ts = [[], [], []]
        for dim, spline in enumerate(msg.data):
            total_t = 0.
            # seg_num = 0
            for seg in spline.segs:
                dts = []
                position = []
                velocity = []
                acceleration = []
                for dt in np.arange(0, seg.dt, .02):
                    dts.append(dt + total_t)
                    position.append(np.polyval(seg.coeffs[::-1], dt/seg.dt))
                    vratio = (1/seg.dt)
                    velocity.append(vratio*np.polyval(np.polyder(seg.coeffs[::-1]), dt/seg.dt))
                    aratio = (1/seg.dt)**2
                    acceleration.append(aratio*np.polyval(np.polyder(seg.coeffs[::-1], m=2), dt/seg.dt))
                # self.ax[0][dim].plot(total_t + np.array(dts), position, '-k')
                # self.ax[0][dim].plot(total_t + np.array(dts), velocity, '-b')
                total_t += seg.dt
                ts[dim].append(np.array(dts))
                positions[dim].append(np.array(position))
                velocities[dim].append(np.array(velocity))
                accelerations[dim].append(np.array(acceleration))
        return positions, velocities, accelerations, ts

    # def traj_tracker_spline_traj_cb(self, msg):
    #     positions, velocities, ts = self.spline_traj_process(msg)
    #     self.tracker_traj = msg
    #     self.tracker_pos_data = positions
    #     self.tracker_vel_data = velocities
    #     self.eval_traj(msg)

    def planner_spline_traj_cb(self, msg):
        positions, velocities, accelerations, ts = self.spline_traj_process(msg)
        self.planner_traj = msg
        self.planner_pos_data = positions
        self.planner_vel_data = velocities
        self.planner_acc_data = accelerations
        self.planner_ts = ts

    def odom_callback(self, msg):
        self.odom_data[0] = msg.pose.pose.position.x
        self.odom_data[1] = msg.pose.pose.position.y
        self.odom_data[2] = msg.pose.pose.position.z
        self.ax[0].set_xlim(msg.pose.pose.position.x-20, msg.pose.pose.position.x+20)
        self.ax[0].set_ylim(msg.pose.pose.position.y-20, msg.pose.pose.position.y+20)

    def action_goal_callback(self, msg):
        self.planning_query_start = [msg.goal.p_init.position.x, msg.goal.p_init.position.y, msg.goal.p_init.position.z]

    def start_goal_callback(self, msg):
        self.start = [msg.markers[0].pose.position.x, msg.markers[0].pose.position.y, msg.markers[0].pose.position.z]
        self.goal = [msg.markers[1].pose.position.x, msg.markers[1].pose.position.y, msg.markers[1].pose.position.z]

    def feedback_callback(self, msg):
        self.seg_number = msg.feedback.seg_number
        self.time_elapsed = msg.feedback.time_elapsed
        self.time_remaining = msg.feedback.time_remaining

    def update_plot(self, frame):
        if len(self.planner_pos_data) == 0:
            # print("waiting for data")
            return self.ln,
        self.odom_history.append(self.odom_data)
        odom_history = np.array(self.odom_history)
        self.ln[0].set_data(odom_history[:, 0], odom_history[:, 1])

        # num_segs = len(self.tracker_pos_data[0])
        # segs = [np.column_stack([self.tracker_pos_data[0][i], self.tracker_pos_data[1][i]]) for i in range(num_segs)]
        # self.ln[1].set_segments(segs)
        # self.ln[1].set_array(np.arange(num_segs))

        if self.start is not None:
            self.ln[2].set_data(self.start[0], self.start[1])
            self.ln[3].set_data(self.goal[0], self.goal[1])

        # poly_start_time = 0
        # if self.seg_number < len(self.planner_traj.data[0].segs):
        #     for seg_num in range(self.seg_number):
        #         poly_start_time += self.planner_traj.data[0].segs[seg_num].dt
        # poly_time_elapsed = self.time_elapsed - poly_start_time
        # self.ln[4].set_text(f'{self.seg_number}, {self.time_elapsed:.2f}, {poly_time_elapsed: .2f}')
        # self.ln[5].set_data(self.traj_feedback[0], self.traj_feedback[1])

        num_segs = len(self.planner_pos_data[0])
        segs = [np.column_stack([self.planner_pos_data[0][i], self.planner_pos_data[1][i]]) for i in range(num_segs)]
        self.ln[6].set_segments(segs)
        self.ln[6].set_array(np.arange(num_segs))

        if self.planning_query_start is not None:
            self.ln[7].set_data(self.planning_query_start[0], self.planning_query_start[1])

        num_segs = len(self.planner_vel_data[0])
        segs_x = [np.column_stack([self.planner_ts[0][i], self.planner_vel_data[0][i]]) for i in range(num_segs)]
        segs_y = [np.column_stack([self.planner_ts[1][i], self.planner_vel_data[1][i]]) for i in range(num_segs)]
        self.ln[8].set_segments(segs_x)
        self.ln[8].set_array(np.arange(num_segs))
        self.ln[9].set_segments(segs_y)
        self.ln[9].set_array(np.arange(num_segs))

        num_segs = len(self.planner_acc_data[0])
        segs_x = [np.column_stack([self.planner_ts[0][i], self.planner_acc_data[0][i]]) for i in range(num_segs)]
        segs_y = [np.column_stack([self.planner_ts[1][i], self.planner_acc_data[1][i]]) for i in range(num_segs)]
        self.ln[10].set_segments(segs_x)
        self.ln[10].set_array(np.arange(num_segs))
        self.ln[11].set_segments(segs_y)
        self.ln[11].set_array(np.arange(num_segs))
        return self.ln,

    def eval_traj(self, msg):
        t = 0
        if self.time_elapsed >= 0:
            for seg_num, seg in enumerate(msg.data[0].segs):
                t += seg.dt
                if t > self.time_elapsed:
                    t -= seg.dt
                    break
            self.traj_feedback = np.zeros(3)
            for dim in range(3):
                coeffs = msg.data[dim].segs[seg_num].coeffs
                self.traj_feedback[dim] = np.polyval(np.flip(coeffs), self.time_elapsed - t)


if __name__ == "__main__":
    rospy.init_node('check_trajectory', anonymous=True)
    ct = CheckTrajectory()
    ani = FuncAnimation(ct.fig, ct.update_plot, interval=50)
    plt.show(block=True)
    if not rospy.is_shutdown():
        rospy.spin()
