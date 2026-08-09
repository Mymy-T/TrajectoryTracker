import copy
from data_models import BallState, BallProperties, BounceProperties, CourtProperties, EnvironmentProperties
import math
import matplotlib.pyplot as plt
import numpy as np

class PhysicsSimulation():
    """
    Class for simulating ball trajectories. The simulator's copy of the initial ball state will be modified in place.
    """
    def __init__(self, initial_state: BallState, simulation_timestep: float, ball_properties = BallProperties(), court_properties = CourtProperties(), bounce_properties = BounceProperties(), environment_properties = EnvironmentProperties()):
        self.ball_state = copy.deepcopy(initial_state)
        self.ball_properties = ball_properties
        self.court_properties = court_properties
        self.bounce_properties = bounce_properties
        self.environment_properties = environment_properties
        self.timestep = simulation_timestep

    def compute_3d_trajectory(self, bounces=True):
        # TEMP: simulate for 10 seconds, break if y < 0 bounce physics and out of bounds TODO
        n_bounces = 0
        trajectory = []
        times = []
        max_num_steps = int(10 / self.timestep)
        for i in range(max_num_steps):
            trajectory.append(self.ball_state.position.copy())
            times.append(i * self.timestep)
            self._rk4_step()
            y = self.ball_state.position[1]
            if y <= 0:
                if bounces:
                    prev_x = self.ball_state.position[0]
                    prev_z = self.ball_state.position[2]
                    start_pos = np.array([prev_x, 0, prev_z])
                    self.ball_state.position = start_pos
                    vf, wf = self._compute_bounce()
                    n_bounces += 1
                    self.ball_state.velocity = vf
                    self.ball_state.spin = wf
                else:
                    break
            

            out_of_bounds = (abs(self.ball_state.position[0]) > self.court_properties.court_length/2 + 3 or abs(self.ball_state.position[2]) > self.court_properties.court_width/2 + 3)
            if n_bounces == 4 or out_of_bounds:
                break
            
        return np.array(trajectory), np.array(times)


    def _compute_accelerations(self, v, w):
        m = self.ball_properties.m
        A = self.ball_properties.A
        I = self.ball_properties.I
        C_d = self.ball_properties.C_d
        C_l = self.ball_properties.C_l
        C_m = self.ball_properties.C_m
        g_scalar = self.environment_properties.g
        rho = self.environment_properties.rho
        mu = self.environment_properties.mu

        F_g = m * np.array([0, -g_scalar, 0])
        F_d = -0.5 * C_d * rho * A * np.linalg.norm(v) * v
        F_m = 0.5 * C_l * rho * A * np.linalg.norm(v)**2 * (np.cross(w, v) / np.linalg.norm(np.cross(w, v)))
        tau_d = -0.5 * C_m * rho * A * np.linalg.norm(v)**2 * (w / np.linalg.norm(w))
        
        a_net = (F_g + F_d + F_m) / m
        alpha_net = tau_d / I
        return a_net, alpha_net
    

    def _compute_bounce(self):
        """
        This bounce model assumes:
        - Tennis ball is rigid body
        - Ratio |v_slip_z| / |v_slip_x| remains approximately constant
        - Normal force as a function of time is of the form N = asin(bt) where b = pi/T_c and T_c is the time of contact
        - Moment of inertia is of the general form 2/3 * m * R^2

        This function doesn't modify ball state in place.
        """
        ball_properties = BallProperties()
        e = self.bounce_properties.e
        mu = self.bounce_properties.mu
        R = ball_properties.R
        v = self.ball_state.velocity
        w = self.ball_state.spin

        v_ox = v[0]
        w_oz = w[2]

        v_oz = v[2]
        w_ox = w[0]

        v_oy = v[1]
        w_oy = w[1]

        # Compute effective mu for x and z directions so that at least the initial frictional force magnitude is mu*N and not sqrt(2)*mu*N
        v_slip = np.array([v_ox + R*w_oz, v_oz - R*w_ox])
        v_slip_norm = np.linalg.norm(v_slip)
        mu_x = mu * abs(v_slip[0]) / v_slip_norm
        mu_z = mu * abs(v_slip[1]) / v_slip_norm

        # Check if rolling will occur or if slipping occurs throughout the entire motion
        rolling_in_x = (0.4 * abs(v_slip[0]) <= mu_x * (1 + e) * abs(v_oy))
        rolling_in_z = (0.4 * abs(v_slip[1]) <= mu_z * (1 + e) * abs(v_oy))
        
        # Compute final state

        v_fy = -e * v_oy
        w_fy = w_oy

        if rolling_in_x:
            v_fx = (3*v_ox - 2*R*w_oz) / 5
            w_fz = (2*R*w_oz - 3*v_ox) / (5*R)
        else:
            v_fx = v_ox + mu_x * math.copysign(1, v_slip[0]) * v_oy * (1 + e)
            w_fz = w_oz + 3 * mu_x * math.copysign(1, v_slip[0]) * v_oy * (1 + e) / (2 * R)

        if rolling_in_z:
            v_fz = (3*v_oz + 2*R*w_ox) / 5
            w_fx = (2*R*w_ox + 3*v_oz) / (5*R)
        else:
            v_fz = v_oz + mu_z * math.copysign(1, v_slip[1]) * v_oy * (1 + e)
            w_fx = w_ox - 3 * mu_z * math.copysign(1, v_slip[1]) * v_oy * (1 + e) / (2 * R)
        
        v_f = np.array([v_fx, v_fy, v_fz])
        w_f = np.array([w_fx, w_fy, w_fz])

        return v_f, w_f


    
    def _forward_euler_integration_step(self): # Obsolete due to RK4
        """
        Ball state is modified in place.
        """
        state = self.ball_state
        dt = self.timestep

        a, alpha = self._compute_accelerations(state.velocity, state.spin)

        state.position += state.velocity * dt
        state.velocity += a * dt

        state.spin += alpha * dt
    
    def _rk4_step(self):
        """
        Ball state is modified in place.
        """
        state = self.ball_state
        dt = self.timestep
        x1 = state.position
        v1 = state.velocity
        w1 = state.spin

        k1_x = state.velocity
        k1_v, k1_w = self._compute_accelerations(v1, w1)

        v2 = v1 + 0.5 * dt * k1_v
        w2 = w1 + 0.5 * dt * k1_w
        k2_x = v2
        k2_v, k2_w = self._compute_accelerations(v2, w2)

        v3 = v1 + 0.5 * dt * k2_v
        w3 = w1 + 0.5 * dt * k2_w
        k3_x = v3
        k3_v, k3_w = self._compute_accelerations(v3, w3)

        v4 = v1 + dt * k3_v
        w4 = w1 + dt * k3_w
        k4_x = v4
        k4_v, k4_w = self._compute_accelerations(v4, w4)

        x_next = x1 + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        v_next = v1 + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        w_next = w1 + (dt / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)

        state.position = x_next
        state.velocity = v_next
        state.spin = w_next



def generate_sample_trajectory(timestep=0.001, bounces=True):
    court_properties = CourtProperties()
    l = court_properties.court_length
    w = court_properties.court_width
    x_i = np.array([-l/2, 1, -w/4]).astype(float)
    v_i = np.array([20, 7, 5]).astype(float)
    spin_i = np.array([0, 300, -300]).astype(float) # 300 rad/s topspin, 300 rad/s left sidespin; exaggerated to show spin effects
    ball_state = BallState(position=x_i, velocity=v_i, spin=spin_i)
    simulation = PhysicsSimulation(initial_state=ball_state, simulation_timestep=timestep)
    return simulation.compute_3d_trajectory(bounces=bounces) # trajectory, end_time


def generate_random_trajectory(rng: np.random.Generator, timestep=0.001, bounces=True, position_bounds=None, velocity_bounds=None, spin_bounds=None):
    """
    Same as generate_sample_trajectory, but the initial ball state is sampled at random
    (via rng) within position_bounds/velocity_bounds/spin_bounds instead of being hardcoded.

    - Each bounds argument is a pair (low, high) where low and high are length-3 ndarrays
    - Defaults are centered around the same values used by generate_sample_trajectory.
    """
    court_properties = CourtProperties()
    l = court_properties.court_length
    w = court_properties.court_width

    if position_bounds is None:
        position_bounds = ([-l/2, 0.5, -w/2], [-l/2, 1.5, w/2])
    if velocity_bounds is None:
        velocity_bounds = ([10, 3, -8], [30, 10, 8])
    if spin_bounds is None:
        spin_bounds = ([-400, -400, -400], [400, 400, 400])

    x_i = rng.uniform(position_bounds[0], position_bounds[1]).astype(float)
    v_i = rng.uniform(velocity_bounds[0], velocity_bounds[1]).astype(float)
    spin_i = rng.uniform(spin_bounds[0], spin_bounds[1]).astype(float)

    ball_state = BallState(position=x_i, velocity=v_i, spin=spin_i)
    simulation = PhysicsSimulation(initial_state=ball_state, simulation_timestep=timestep)
    
    return simulation.compute_3d_trajectory(bounces=bounces) # trajectory, times


def plot_3d_trajectory(trajectory, ax=None):
    """
    If ax is None, creates a new figure/3D axes and calls plt.show() at the end (standalone use);
    else, draws onto it and leaves creating/showing the figure to the caller (eg for embedding in GUI)
    """
    court_properties = CourtProperties()
    l = court_properties.court_length
    w = court_properties.court_width
    sw = court_properties.singles_width
    sld = court_properties.service_line_dist_from_net

    # t = np.linspace(0, end_time, int(10/0.001))
    x = []
    y = []
    z = []
    for point in trajectory:
        x.append(point[0])
        y.append(point[1])
        z.append(point[2])

    # Plotting
    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")


    # Outer court lines
    ax.plot(
        [-l/2, l/2, l/2, -l/2, -l/2],
        [-w/2, -w/2, w/2, w/2, -w/2],
        [0, 0, 0, 0, 0],
        color="red",
        linewidth=1,
        label="Court Lines"
    )

    # Singles lines
    ax.plot(
        [-l/2, l/2],
        [-sw/2, -sw/2],
        [0, 0],
        color="red",
        linewidth=1,
        label="Court Lines"
    )

    ax.plot(
        [-l/2, l/2],
        [sw/2, sw/2],
        [0, 0],
        color="red",
        linewidth=1,
        label="Court Lines"
    )

    # Net line (across X = 0)
    ax.plot([0, 0], [-w/2, w/2], [0, 0], color="red", linewidth=1.5)

    # Service lines
    ax.plot(
        [-sld, -sld],
        [-w/2, w/2],
        [0, 0],
        color="red",
        linewidth=1,
    )
    ax.plot(
        [sld, sld],
        [-w/2, w/2],
        [0, 0],
        color="red",
        linewidth=1,
    )

    # Center Service Line (Connects the two service lines down the middle at Z = 0)
    ax.plot(
        [-sld, sld],
        [0, 0],
        [0, 0],
        color="red",
        linewidth=1,
    )


    # Plot trajectory
    ax.plot(x, z, y, label="3D Trajectory", color="b", linewidth=2)

    # Highlight start and end points
    ax.scatter(x[0], z[0], y[0], color="green", s=50, label="Start")
    ax.scatter(x[-1], z[-1], y[-1], color="red", s=50, label="End")

    ax.set_xlim([-l/2 - 1, l/2 + 1])    
    ax.set_ylim([-w/2 - 1, w/2 + 1])
    ax.invert_yaxis() # This way negative values of z axis are towards the left
    ax.set_zlim([0, 4]) 
    ax.set_box_aspect([l, w, 4])

    # NOTE: In matplotlib, +z is up; in the coordinates of the trajectory, +y is up
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Z Axis")
    ax.set_zlabel("Y Axis")
    ax.set_title("3D Path Trajectory")
    ax.grid(False)
    ax.legend()

    if standalone:
        plt.show()

if __name__ == "__main__":
    trajectory, end_time = generate_sample_trajectory()

    print(f"Num points in trajectory: {len(trajectory)}")

    plot_3d_trajectory(trajectory)
