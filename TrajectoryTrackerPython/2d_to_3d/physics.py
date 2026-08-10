import copy
from data_models import BallState, BallProperties, BounceProperties, CourtProperties, EnvironmentProperties
from enum import Enum
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


class TrajectoryMode(Enum):
    """
    Modes supported by PhysicsSimulation.compute_3d_trajectory.
    """
    NO_BOUNCE = "no_bounce"        # stop as soon as the ball would touch the ground
    BOUNCE_ONLY = "bounce_only"    # a single shot with up to max_bounces ground bounces
    BOUNCE_AND_HIT = "bounce_and_hit"  # a multi-hit rally (ground bounces + racket hits)


class PhysicsSimulation():
    """
    Class for simulating ball trajectories. The simulator's copy of the initial ball state will be modified in place.
    """
    def __init__(self, initial_state: BallState, simulation_timestep: float, ball_properties = BallProperties(), court_properties = CourtProperties(), bounce_properties = BounceProperties(), environment_properties = EnvironmentProperties(), rng: np.random.Generator = None):
        self.ball_state = copy.deepcopy(initial_state)
        self.ball_properties = ball_properties
        self.court_properties = court_properties
        self.bounce_properties = bounce_properties
        self.environment_properties = environment_properties
        self.timestep = simulation_timestep
        self.rng = rng if rng is not None else np.random.default_rng()


    def compute_3d_trajectory(self, mode=TrajectoryMode.BOUNCE_ONLY, num_hits_range=(2, 5), hit_zone_bounds=(3/8, 1/2), bounce_trigger_cap=3, max_bounces=4, oob_buffer=3):
        """
        - num_hits_range, hit_zone_bounds, bounce_trigger_cap only apply to TrajectoryMode.BOUNCE_AND_HIT
        - max_bounces only applies to TrajectoryMode.BOUNCE_ONLY.
        - oob_buffer is the extra length in meters added to length and width of court before ball is considered out of bounds; applied to all modes
        """
        if mode == TrajectoryMode.NO_BOUNCE:
            return self._compute_trajectory_no_bounce(oob_buffer=oob_buffer)
        elif mode == TrajectoryMode.BOUNCE_ONLY:
            return self._compute_trajectory_bounce_only(max_bounces=max_bounces, oob_buffer=oob_buffer)
        elif mode == TrajectoryMode.BOUNCE_AND_HIT:
            return self._compute_trajectory_bounce_and_hit(num_hits_range=num_hits_range, hit_zone_bounds=hit_zone_bounds, bounce_trigger_cap=bounce_trigger_cap, oob_buffer=oob_buffer)


    def _compute_trajectory_no_bounce(self, oob_buffer):
        """
        Stops when ball touches the ground or when it leaves the court bounds
        """
        trajectory = []
        times = []
        i = 0
        while True:

            trajectory.append(self.ball_state.position.copy())
            times.append(i * self.timestep)

            i += 1
            self._rk4_step()

            x, y, z = self.ball_state.position
            if y <= 0:
                break

            out_of_bounds = (abs(x) > self.court_properties.court_length/2 + oob_buffer or abs(z) > self.court_properties.court_width/2 + oob_buffer)
            if out_of_bounds:
                break

        return np.array(trajectory), np.array(times)


    def _compute_trajectory_bounce_only(self, max_bounces, oob_buffer):
        """
        Stops after max_bounces bounces or when ball leaves the court bounds
        """
        trajectory = []
        times = []
        n_bounces = 0
        i = 0
        while True:

            trajectory.append(self.ball_state.position.copy())
            times.append(i * self.timestep)

            i += 1
            self._rk4_step()

            x, y, z = self.ball_state.position
            if y <= 0:
                self.ball_state.position = np.array([x, 0, z])
                vf, wf = self._compute_bounce()
                self.ball_state.velocity = vf
                self.ball_state.spin = wf
                n_bounces += 1

            out_of_bounds = (abs(x) > self.court_properties.court_length/2 + oob_buffer or abs(z) > self.court_properties.court_width/2 + oob_buffer)
            if n_bounces == max_bounces or out_of_bounds:
                break

        return np.array(trajectory), np.array(times)


    def _compute_trajectory_bounce_and_hit(self, num_hits_range, hit_zone_bounds, bounce_trigger_cap, oob_buffer):
        """
        - Ball hit once the its position reaches the hit_zone_bounds fraction of the court length, or after bounce_trigger_cap bounces without reaching that zone
        - Num hits of the rally is sampled once from num_hits_range; stops after that num of hits has occurred, once the ball lands
        - Also stops if ball goes out of bounds
        """
        target_hits = int(self.rng.integers(num_hits_range[0], num_hits_range[1] + 1))
        num_hits = 0
        bounces_since_last_hit = 0
        l = self.court_properties.court_length

        start_x = self.ball_state.position[0]
        leg_start_sign = math.copysign(1.0, start_x) if start_x != 0 else 1.0

        trajectory = []
        times = []
        i = 0
        while True:

            trajectory.append(self.ball_state.position.copy())
            times.append(i * self.timestep)

            i += 1
            self._rk4_step()

            x, y, z = self.ball_state.position

            landed = (y <= 0)
            if landed:
                self.ball_state.position = np.array([x, 0, z])

            if num_hits < target_hits:

                crossed_into_far_side = (x != 0 and math.copysign(1.0, x) == -leg_start_sign)
                in_hit_zone = (crossed_into_far_side and (hit_zone_bounds[0] * l <= abs(x) <= hit_zone_bounds[1] * l))

                bounce_trigger_cap_reached = (landed and bounces_since_last_hit >= bounce_trigger_cap)

                if in_hit_zone or bounce_trigger_cap_reached:
                    vf, wf = self._simulate_racket_hit()
                    self.ball_state.velocity = vf
                    self.ball_state.spin = wf

                    num_hits += 1
                    bounces_since_last_hit = 0

                    leg_start_sign = math.copysign(1.0, x) if x != 0 else leg_start_sign

                elif landed:
                    vf, wf = self._compute_bounce()
                    self.ball_state.velocity = vf
                    self.ball_state.spin = wf

                    bounces_since_last_hit += 1

            elif landed:
                vf, wf = self._compute_bounce()
                self.ball_state.velocity = vf
                self.ball_state.spin = wf

            out_of_bounds = (abs(x) > self.court_properties.court_length/2 + oob_buffer or abs(z) > self.court_properties.court_width/2 + oob_buffer)
            if num_hits >= target_hits and landed:
                break
            if out_of_bounds:
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

        eps = 1e-9 # Prevent division by 0

        F_g = m * np.array([0, -g_scalar, 0])
        F_d = -0.5 * C_d * rho * A * np.linalg.norm(v) * v

        cross_wv = np.cross(w, v)
        cross_wv_norm = np.linalg.norm(cross_wv)
        if cross_wv_norm > eps:
            F_m = 0.5 * C_l * rho * A * np.linalg.norm(v)**2 * (cross_wv / cross_wv_norm)
        else:
            F_m = np.zeros(3)

        w_norm = np.linalg.norm(w)
        if w_norm > eps:
            tau_d = -0.5 * C_m * rho * A * np.linalg.norm(v)**2 * (w / w_norm)
        else:
            tau_d = np.zeros(3)

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
    

    def _predict_landing(self, position, velocity, spin, dt=0.01, max_time=10.0):
        """
        - Cheap forward-Euler simulation starting from some state to estimate where ball will land
        - Returns x, z (coords of landing) if ball landed within max_time else None
        """
        pos = np.array(position, dtype=float)
        v = np.array(velocity, dtype=float)
        w = np.array(spin, dtype=float)
        t = 0.0
        while t < max_time:
            prev_pos = pos
            a, alpha = self._compute_accelerations(v, w)

            pos += v * dt
            v += a * dt
            w += alpha * dt
            t += dt

            if pos[1] <= 0.0 and prev_pos[1] > 0.0: # Linear interpolation to estimate exact bounce position
                frac = prev_pos[1] / (prev_pos[1] - pos[1])
                x_land = prev_pos[0] + frac * (pos[0] - prev_pos[0])
                z_land = prev_pos[2] + frac * (pos[2] - prev_pos[2])
                return x_land, z_land
            
        return None


    def _simulate_racket_hit(self, max_attempts=20):
        """
        - Samples a random v and w that sends the ball back toward the other side of the court, rejecting candidates that wouldn't land in bounds (validated via a rough forward-Euler rollout); falls back to a deterministic safe shot ignoring drag/spin if no random candidate is accepted
        - Returns v_f, w_f
        """
        court_properties = self.court_properties
        l = court_properties.court_length
        w_court = court_properties.court_width

        hit_pos = self.ball_state.position
        hit_v = self.ball_state.velocity

        if hit_pos[0] != 0:
            target_sign = -math.copysign(1.0, hit_pos[0])
        elif hit_v[0] != 0:
            target_sign = -math.copysign(1.0, hit_v[0])
        else:
            target_sign = 1.0

        v_low = np.array([10.0, 3.0, -8.0]) # TODO: Use same bounds as used in generate random trajectory
        v_high = np.array([30.0, 10.0, 8.0])

        w_low = np.array([-400.0, -400.0, -400.0])
        w_high = np.array([400.0, 400.0, 400.0])

        for _ in range(max_attempts):

            v_candidate = self.rng.uniform(v_low, v_high)
            v_candidate[0] = target_sign * v_candidate[0]

            w_candidate = self.rng.uniform(w_low, w_high)

            landing = self._predict_landing(hit_pos, v_candidate, w_candidate)
            if landing is None:
                continue
            x_land, z_land = landing

            in_bounds = (abs(x_land) <= l / 2 and abs(z_land) <= w_court / 2)
            correct_side = ((x_land == 0) or (math.copysign(1.0, x_land) == target_sign))

            if in_bounds and correct_side:
                return v_candidate, w_candidate

        # Simplified deterministic fallback; aim well inside the court on the target side so land pos is highly likely to be in bounds when spin & drag are taken into account
        x_target = target_sign * (l / 4)
        z_target = 0.0
        flight_time = 1.0 # Arbitrarily choosen
        g = self.environment_properties.g

        # f denotes the final velocity after the hit, not the final landing velocity; it's actually initial velocity for the quadratic equation
        v_fx = (x_target - hit_pos[0]) / flight_time
        v_fz = (z_target - hit_pos[2]) / flight_time
        v_fy = 0.5 * g * flight_time  # y(t) = v_fy*t - 0.5*g*t^2 = 0

        v_f = np.array([v_fx, v_fy, v_fz])
        w_f = np.array([0.0, 1.0, 0.0])

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



def _resolve_trajectory_mode(bounces, racket_hits):
    if not bounces:
        return TrajectoryMode.NO_BOUNCE
    
    return TrajectoryMode.BOUNCE_AND_HIT if racket_hits else TrajectoryMode.BOUNCE_ONLY


def generate_sample_trajectory(timestep=0.001, bounces=True, racket_hits=False, num_hits_range=(2, 5), hit_zone_bounds=(3/8, 1/2), bounce_trigger_cap=3, max_bounces=4, rng: np.random.Generator = None):
    court_properties = CourtProperties()
    l = court_properties.court_length
    w = court_properties.court_width
    x_i = np.array([-l/2, 1, -w/4]).astype(float)
    v_i = np.array([20, 7, 5]).astype(float)
    spin_i = np.array([0, 300, -300]).astype(float) # 300 rad/s topspin, 300 rad/s left sidespin; exaggerated to show spin effects
    ball_state = BallState(position=x_i, velocity=v_i, spin=spin_i)
    simulation = PhysicsSimulation(initial_state=ball_state, simulation_timestep=timestep, rng=rng)
    mode = _resolve_trajectory_mode(bounces, racket_hits)
    return simulation.compute_3d_trajectory(mode=mode, num_hits_range=num_hits_range, hit_zone_bounds=hit_zone_bounds, bounce_trigger_cap=bounce_trigger_cap, max_bounces=max_bounces) # trajectory, end_time


def generate_random_trajectory(rng: np.random.Generator, timestep=0.001, bounces=True, position_bounds=None, velocity_bounds=None, spin_bounds=None, racket_hits=False, num_hits_range=(2, 5), hit_zone_bounds=(3/8, 1/2), bounce_trigger_cap=3, max_bounces=4):
    """
    Same as generate_sample_trajectory, but the initial ball state is sampled at random

    - Each bounds argument is a pair (low, high) where low and high are length-3 ndarrays
    - Defaults are centered around the same values used by generate_sample_trajectory
    - The same rng drives both the initial state sampling here and the in-rally racket hit sampling inside PhysicsSimulation (if racket_hits=True)
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
    simulation = PhysicsSimulation(initial_state=ball_state, simulation_timestep=timestep, rng=rng)

    mode = _resolve_trajectory_mode(bounces, racket_hits)
    return simulation.compute_3d_trajectory(mode=mode, num_hits_range=num_hits_range, hit_zone_bounds=hit_zone_bounds, bounce_trigger_cap=bounce_trigger_cap, max_bounces=max_bounces) # trajectory, times


def _plot_fading_line_3d(ax, plot_x, plot_y, plot_z, fade_window, label=None, color=(0.0, 0.0, 1.0)):
    """
    The newest segment is fully opaque; opacity ramps down linearly over the trailing fade_window points, and anything older than fade_window points back is omitted entirely
    """
    n_points = len(plot_x)
    start_index = max(0, n_points - 1 - int(round(fade_window)))

    x = np.asarray(plot_x[start_index:])
    y = np.asarray(plot_y[start_index:])
    z = np.asarray(plot_z[start_index:])
    if len(x) < 2:
        return

    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # age (in points) of each segment's newer endpoint; 0 = the current/newest frame
    newest_index = n_points - 1
    segment_end_indices = start_index + np.arange(1, len(segments) + 1)
    ages = newest_index - segment_end_indices
    alphas = np.clip(1.0 - ages / fade_window, 0.0, 1.0) if fade_window > 0 else np.ones(len(segments))

    colors = [(*color, a) for a in alphas]
    ax.add_collection3d(Line3DCollection(segments, colors=colors, linewidth=2, label=label))


def plot_3d_trajectory(trajectory, ax=None, fade_window=None):
    """
    If ax is None, creates a new figure/3D axes and calls plt.show() at the end (standalone use);
    else, draws onto it and leaves creating/showing the figure to the caller (eg for embedding in GUI)
    """
    court_properties = CourtProperties()
    l = court_properties.court_length
    w = court_properties.court_width
    sw = court_properties.singles_width
    sld = court_properties.service_line_dist_from_net

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
    if fade_window is not None and len(x) > 1:
        _plot_fading_line_3d(ax, x, z, y, fade_window, label="3D Trajectory")
    else:
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
