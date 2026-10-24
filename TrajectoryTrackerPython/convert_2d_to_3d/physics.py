from convert_2d_to_3d.data_models import BallState, BallProperties, BounceProperties, CourtProperties, EnvironmentProperties, TrajectoryGenerationSettings, ShotType, TrajectoryMode, ShotDistribution
from convert_2d_to_3d.settings import SyntheticDataGeneration
from convert_2d_to_3d.visualizer import plot_3d_trajectory
import copy
import math
import numpy as np


# MARK: Physics Simulator
class PhysicsSimulation():
    """
    Class for simulating ball trajectories. The simulator's copy of the initial ball state will be modified in place.
    """
    def __init__(self, initial_state: BallState, data_gen_settings: SyntheticDataGeneration):

        self.ball_state = copy.deepcopy(initial_state)
        self.rng = data_gen_settings.physics_rng
        self.gen_settings = data_gen_settings.trajectory_generation_settings

        self.ball_properties = BallProperties()
        self.bounce_properties = BounceProperties()
        self.court_properties = CourtProperties()
        self.environment_properties = EnvironmentProperties()


    # MARK: Simulate Trajectory
    def compute_3d_trajectory(self):
        """
        Returns trajectory, times, eot
        """
        mode = self.gen_settings.mode
        if mode == TrajectoryMode.NO_BOUNCE:
            return self._compute_trajectory_no_bounce()
        elif mode == TrajectoryMode.BOUNCE_ONLY:
            return self._compute_trajectory_bounce_only()
        elif mode == TrajectoryMode.BOUNCE_AND_HIT:
            return self._compute_trajectory_bounce_and_hit()


    def _compute_trajectory_no_bounce(self):
        """
        Stops when ball touches the ground or when it leaves the court bounds
        """
        trajectory = []
        times = []
        i = 0
        while True:

            trajectory.append(self.ball_state.position.copy())
            times.append(i * self.gen_settings.simulation_timestep)

            i += 1
            self._rk4_step()

            x, y, z = self.ball_state.position
            if y <= 0:
                break
            
            oob_buffer = self.gen_settings.oob_buffer
            out_of_bounds = (abs(x) > self.court_properties.court_length/2 + oob_buffer or abs(z) > self.court_properties.court_width/2 + oob_buffer)
            if out_of_bounds:
                break

        bounce_tokens = [0.0] * (len(trajectory) - 1)
        bounce_tokens.append(1.0)

        return np.array(trajectory, dtype=np.float64), np.array(times, dtype=np.float64), np.zeros(len(trajectory), dtype=np.float64), np.array(bounce_tokens, dtype=np.float64) # Always saved at full precision; downcast happens only when loaded (see dataset.py)


    def _compute_trajectory_bounce_only(self):
        """
        Stops after max_bounces bounces or when ball leaves the court bounds
        """
        trajectory = []
        times = []
        bounce_tokens = []
        n_bounces = 0
        i = 0
        while True:

            trajectory.append(self.ball_state.position.copy())
            times.append(i * self.gen_settings.simulation_timestep)

            if len(bounce_tokens) == i: # Prevent double counting if a bounce adds a 1.0 token
                bounce_tokens.append(0.0)

            i += 1
            self._rk4_step()

            x, y, z = self.ball_state.position
            if y <= 0:
                self.ball_state.position = np.array([x, 0, z])
                vf, wf = self._compute_bounce()
                self.ball_state.velocity = vf
                self.ball_state.spin = wf
                n_bounces += 1
                bounce_tokens.append(1.0)

            oob_buffer = self.gen_settings.oob_buffer
            out_of_bounds = (abs(x) > self.court_properties.court_length/2 + oob_buffer or abs(z) > self.court_properties.court_width/2 + oob_buffer)
            if n_bounces == self.gen_settings.max_bounces or out_of_bounds:
                trajectory.append(self.ball_state.position.copy())
                times.append(i * self.gen_settings.simulation_timestep)
                if y > 0: # 1.0 tokens would have already been added
                    bounce_tokens.append(0.0)
                break

        return np.array(trajectory, dtype=np.float64), np.array(times, dtype=np.float64), np.zeros(len(trajectory), dtype=np.float64), np.array(bounce_tokens, dtype=np.float64) # Always saved at full precision; downcast happens only when loaded (see dataset.py)


    def _compute_trajectory_bounce_and_hit(self):
        """
        - Ball hit once the its position reaches the hit_zone_bounds fraction of the court length, or after bounce_trigger_cap bounces without reaching that zone
        - Num hits of the rally is sampled once from num_hits_range; stops after that num of hits has occurred, once the ball lands
        - Also stops if ball goes out of bounds
        """
        l = self.court_properties.court_length

        hit_zone_bounds = self.gen_settings.hit_zone_bounds
        hit_depth_threshold = self.rng.uniform(hit_zone_bounds[0] * l, hit_zone_bounds[1] * l) # Varied each time

        num_hits_range = self.gen_settings.num_hits_range
        target_hits = int(self.rng.integers(num_hits_range[0], num_hits_range[1] + 1))

        start_x = self.ball_state.position[0]
        leg_start_sign = math.copysign(1.0, start_x) if start_x != 0 else 1.0

        n_delay_timesteps = round(self.gen_settings.delayed_hit_secs_after_bounce / self.gen_settings.simulation_timestep)

        trajectory = []
        times = []
        eot = []
        bounce_tokens = []
        i = 0
        num_hits = 0
        bounces_since_last_hit = 0
        local_max_bounce_height = 0.0
        delayed_hit_pending = False
        delayed_hit_bounce_occured = False
        delayed_timesteps_since_bounce = 0

        
        while True:

            trajectory.append(self.ball_state.position.copy())
            local_max_bounce_height = max(self.ball_state.position[1], local_max_bounce_height)
            times.append(i * self.gen_settings.simulation_timestep)
            eot.append(0.0)

            if delayed_hit_bounce_occured and delayed_hit_pending:
                delayed_timesteps_since_bounce += 1

            if len(bounce_tokens) == i: # Prevent double counting if a bounce already added a 1.0 token
                bounce_tokens.append(0.0)

            i += 1
            self._rk4_step()

            x, y, z = self.ball_state.position

            landed = (y <= 0)
            if landed:
                self.ball_state.position = np.array([x, 0, z])
                bounce_tokens.append(1.0)


            perform_hit = False
            if num_hits < target_hits:

                if not delayed_hit_pending: # Check zone/bounce cap triggers only if not already awaiting for a hit after a bounce

                    crossed_into_far_side = (x != 0 and math.copysign(1.0, x) == -leg_start_sign)
                    in_possible_hit_zone = (crossed_into_far_side and (hit_zone_bounds[0] * l <= abs(x)))
                    in_hit_zone = (in_possible_hit_zone and (abs(x) >= hit_depth_threshold)) # > since ball is moving towards court ends when crossed into far side

                    bounce_trigger_cap_reached = (landed and bounces_since_last_hit >= self.gen_settings.bounce_trigger_cap)
                    
                    if in_hit_zone or bounce_trigger_cap_reached:

                        if not bounce_trigger_cap_reached and self.ball_state.position[1] < (self.gen_settings.min_hit_height_frac * local_max_bounce_height):
                            delayed_hit_pending = True 
                            
                        else:
                            perform_hit = True
                    

                if delayed_hit_pending and delayed_timesteps_since_bounce == n_delay_timesteps:
                    perform_hit = True
                
                
                if perform_hit:

                    eot[i - 1] = 1.0 # Set EOT flag to true for the timestep right before the racket hit; i is always >= 1 here due to already being incremented

                    vf, wf = self._simulate_racket_hit()
                    self.ball_state.velocity = vf
                    self.ball_state.spin = wf

                    num_hits += 1
                    bounces_since_last_hit = 0

                    leg_start_sign = math.copysign(1.0, x) if x != 0 else leg_start_sign
                
                    delayed_hit_pending = False
                    delayed_hit_bounce_occured = False
                    delayed_timesteps_since_bounce = 0
                    local_max_bounce_height = 0

                    hit_depth_threshold = self.rng.uniform(hit_zone_bounds[0] * l, hit_zone_bounds[1] * l) # Varied each time

                elif landed:
                    vf, wf = self._compute_bounce()
                    self.ball_state.velocity = vf
                    self.ball_state.spin = wf

                    local_max_bounce_height = 0

                    if delayed_hit_pending:
                        delayed_hit_bounce_occured = True
                        delayed_timesteps_since_bounce = 0
                    
                    bounces_since_last_hit += 1

            elif landed: # Bounce after target hits is achieved
                vf, wf = self._compute_bounce()
                self.ball_state.velocity = vf
                self.ball_state.spin = wf

            oob_buffer = self.gen_settings.oob_buffer
            out_of_bounds = (abs(x) > self.court_properties.court_length/2 + oob_buffer or abs(z) > self.court_properties.court_width/2 + oob_buffer)

            if num_hits >= target_hits and landed:
                trajectory.append(self.ball_state.position.copy()) # Normally, the loop waits until the next iteration before adding the current position; there is no next iteration upon final break, so it's added here instead
                eot.append(0.0)
                times.append(i * self.gen_settings.simulation_timestep)
                break

            if out_of_bounds:
                trajectory.append(self.ball_state.position.copy()) # Same as above
                eot.append(0.0)
                if not landed:
                    bounce_tokens.append(0.0)
                times.append(i * self.gen_settings.simulation_timestep)
                break

        return np.array(trajectory, dtype=np.float64), np.array(times, dtype=np.float64), np.array(eot, dtype=np.float64), np.array(bounce_tokens, dtype=np.float64) # Always saved at full precision; downcast happens only when loaded (see dataset.py)


    # MARK: Compute Accelerations
    def compute_accelerations(self, v, w):
        m = self.ball_properties.m
        A = self.ball_properties.A
        I = self.ball_properties.I
        C_d = self.ball_properties.calculate_C_d(v, w)
        C_l = self.ball_properties.calculate_C_l(v, w)
        C_m = self.ball_properties.C_m
        g_scalar = self.environment_properties.g
        rho = self.environment_properties.rho

        eps = 1e-9 # Prevent division by 0

        F_g = m * np.array([0, -g_scalar, 0])
        F_d = -0.5 * C_d * rho * A * np.linalg.norm(v) * v

        cross_wv = np.cross(w, v)
        cross_wv_norm = np.linalg.norm(cross_wv)
        if cross_wv_norm > eps:
            F_m = 0.5 * C_l * rho * A * np.linalg.norm(v)**2 * (cross_wv / cross_wv_norm)
        else:
            F_m = np.zeros(3)

        if self.gen_settings.spin_decay:
            w_norm = np.linalg.norm(w)
            if w_norm > eps: # Prevent division by 0
                tau_d = -0.5 * C_m * rho * A * np.linalg.norm(v)**2 * (w / w_norm)
            else:
                tau_d = np.zeros(3)
            alpha_net = tau_d / I
        else:
            alpha_net = np.zeros(3)

        a_net = (F_g + F_d + F_m) / m
        return a_net, alpha_net
    

    # MARK: Compute Bounce
    def _compute_bounce(self):
        """
        This bounce model assumes:
        - Tennis ball is rigid body
        - Ratio |v_slip_z| / |v_slip_x| remains approximately constant
        - Normal force as a function of time is of the form N = asin(bt) where b = pi/T_c and T_c is the time of contact
        - Moment of inertia is of the general form 2/3 * m * R^2

        This function doesn't modify ball state in place.
        """
        e = self.bounce_properties.e
        mu = self.bounce_properties.mu
        R = self.ball_properties.R
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


    # MARK: Simulate Racket Hit
    def _simulate_racket_hit(self):
        """
        - Directly calls sample_shot(...)
        - Picks the FH/BH distribution from the hit position, then samples a shot back toward the other side of the court
        - Returns v_f, w_f
        """
        shot_dist = _get_shot_distribution(self.gen_settings, self._classify_shot_type())
        return self.sample_shot(shot_dist)


    def _classify_shot_type(self):
        """
        Assumes right-handed players; the -x side player faces +x so their FH side is +z, and the +x side player's FH side is -z; used as a helper for _simulate_racket_hit to determine the shot for sample_shot
        """
        hit_pos = self.ball_state.position
        on_near_side = (hit_pos[0] <= 0)
        on_fh_side = (hit_pos[2] >= 0) if on_near_side else (hit_pos[2] <= 0)
        return ShotType.FH if on_fh_side else ShotType.BH


    # MARK: Sample a Shot
    def sample_shot(self, shot_dist: ShotDistribution, max_attempts=20):
        """
        - Samples v and w from shot_dist for a shot hit from the current ball position toward the target_sign side of the court, rejecting candidates that wouldn't land in bounds; falls back to a deterministic safe shot ignoring drag/spin if no random candidate is accepted
        - shot_dist is defined for a shot toward +x; for target_sign < 0 the sample is rotated 180 deg about the y axis (v_x, w_x, w_z flipped) to preserve spin in correct orientation
        - Returns v_f, w_f
        """

        l = self.court_properties.court_length
        w_court = self.court_properties.court_width

        hit_pos = self.ball_state.position
        hit_v = self.ball_state.velocity

        # Find the sign for v_x
        if hit_pos[0] != 0:
            target_sign = -math.copysign(1.0, hit_pos[0])
        elif hit_v[0] != 0:
            target_sign = -math.copysign(1.0, hit_v[0])
        else:
            target_sign = 1.0

        for _ in range(max_attempts):

            v_candidate = self._sample_velocity(self.rng, shot_dist.v_mu, shot_dist.v_stdev, shot_dist.v_z_positional_bias, hit_pos)
            v_candidate[0] = target_sign * v_candidate[0]

            w_candidate = self._sample_spin(self.rng, shot_dist.w_mu, shot_dist.w_stdev)
            w_candidate[0] = target_sign * w_candidate[0]
            w_candidate[2] = target_sign * w_candidate[2]

            landing = self._predict_landing(hit_pos, v_candidate, w_candidate)
            if landing is None:
                continue

            x_land, z_land = landing

            in_bounds = (abs(x_land) <= l / 2 and abs(z_land) <= w_court / 2)
            correct_side = ((x_land == 0) or (math.copysign(1.0, x_land) == target_sign))

            if in_bounds and correct_side:
                return v_candidate, w_candidate

        # Simplified deterministic fallback; aim well inside the court on the target side so land pos is highly likely to be in bounds when spin & drag are taken into account
        x_t = target_sign * (l / 4)
        z_t = 1.0
        g = self.environment_properties.g
        h = self.court_properties.net_height_at_posts # Min height to pass net; set to net posts to be on safe side

        x_0 = hit_pos[0]
        y_0 = hit_pos[1]
        z_0 = hit_pos[2]

        denom = 2 * ( (y_0 / (x_t - x_0)) + ((y_0 - h) / x_0) )
        radicand = g * x_t / denom
        if radicand < 0:
            v_x = target_sign * 20.0 # No constraint in this case, the ball is already above net
        else:
            v_x = target_sign * math.sqrt(g * x_t / denom) * 3/4 # Stay within the constraints for vx to hit the target AND pass the net height
        v_y = (g * (x_t - x_0) / (2 * v_x)) - (y_0 * v_x / (x_t - x_0))
        v_z = ((z_t - z_0) / (x_t - x_0)) * v_x

        v_f = np.array([v_x, v_y, v_z])
        w_f = np.array([0.0, 1.0, 0.0])

        return v_f, w_f
    

    def _sample_velocity(self, rng: np.random.Generator, v_mu, v_stdev, v_bias, x):
        """
        Helper that will sample a velocity for an initial hit or a subsequent racket hit
        """
        w = CourtProperties().court_width

        # Adjust v_z_mu based on z position
        x_z = x[2]
        v_mu_adjusted = np.array(v_mu, dtype=float) # Copy so the shared distribution parameters are never modified
        normalized_scale_factor = -x_z / (w/2)
        v_mu_adjusted[2] = v_mu_adjusted[2] + normalized_scale_factor * v_bias

        return rng.normal(loc=v_mu_adjusted, scale=v_stdev).astype(float) # (3,)
    

    def _sample_spin(self, rng: np.random.Generator, spin_mu, spin_stdev):
        """
        Helper that will sample a spin for an initial hit or a subsequent racket hit
        """
        return rng.normal(loc=spin_mu, scale=spin_stdev).astype(float) # (3,)
    

    def _predict_landing(self, position, velocity, spin, dt=0.01, max_time=10.0):
        """
        - Cheap forward-Euler simulation starting from some state to estimate where ball will land
        - Returns x, z (coords of landing) if ball landed within max_time else None
        - Helper function for sample_shot
        """
        pos = np.array(position, dtype=float)
        v = np.array(velocity, dtype=float)
        w = np.array(spin, dtype=float)
        t = 0.0
        while t < max_time:
            prev_pos = pos.copy() # Copy needed; pos is updated in place below
            a, alpha = self.compute_accelerations(v, w)

            pos += v * dt
            v += a * dt
            w += alpha * dt
            t += dt

            # Ensure that it passes the net
            h = self.court_properties.net_height_at_posts
            passed_net_x = ((pos[0] > 0 and prev_pos[0] < 0) or (pos[0] < 0 and prev_pos[0] > 0))
            if passed_net_x:
                y_min = min(pos[1], prev_pos[1])
                if y_min <= h:
                    return None # Ball hit net; regenerate
                

            if pos[1] <= 0.0 and prev_pos[1] > 0.0: # Linear interpolation to estimate exact bounce position
                frac = prev_pos[1] / (prev_pos[1] - pos[1])
                x_land = prev_pos[0] + frac * (pos[0] - prev_pos[0])
                z_land = prev_pos[2] + frac * (pos[2] - prev_pos[2])
                return x_land, z_land
            
        return None


    # MARK: Integrators
    def _forward_euler_integration_step(self): # Obsolete due to RK4
        """
        Ball state is modified in place.
        """
        state = self.ball_state
        dt = self.gen_settings.simulation_timestep

        a, alpha = self.compute_accelerations(state.velocity, state.spin)

        state.position += state.velocity * dt
        state.velocity += a * dt

        state.spin += alpha * dt
    

    def _rk4_step(self):
        """
        Ball state is modified in place.
        """
        state = self.ball_state
        dt = self.gen_settings.simulation_timestep
        x1 = state.position
        v1 = state.velocity
        w1 = state.spin

        k1_x = state.velocity
        k1_v, k1_w = self.compute_accelerations(v1, w1)

        v2 = v1 + 0.5 * dt * k1_v
        w2 = w1 + 0.5 * dt * k1_w
        k2_x = v2
        k2_v, k2_w = self.compute_accelerations(v2, w2)

        v3 = v1 + 0.5 * dt * k2_v
        w3 = w1 + 0.5 * dt * k2_w
        k3_x = v3
        k3_v, k3_w = self.compute_accelerations(v3, w3)

        v4 = v1 + dt * k3_v
        w4 = w1 + dt * k3_w
        k4_x = v4
        k4_v, k4_w = self.compute_accelerations(v4, w4)

        x_next = x1 + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        v_next = v1 + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        w_next = w1 + (dt / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)

        state.position = x_next
        state.velocity = v_next
        state.spin = w_next


def _get_shot_distribution(generation_settings: TrajectoryGenerationSettings, shot_type: ShotType):
    match shot_type:
        case ShotType.FH:
            return generation_settings.fh_dist
        case ShotType.BH:
            return generation_settings.bh_dist


# MARK: Example Trajectory
def generate_sample_trajectory():
    l = CourtProperties().court_length
    w = CourtProperties().court_width
    x_i = np.array([-l/2, 1, -w/4]).astype(float)
    v_i = np.array([20, 7, 5]).astype(float)
    spin_i = np.array([0, 300, -300]).astype(float) # 300 rad/s topspin, 300 rad/s left sidespin; exaggerated to show spin effects
    ball_state = BallState(position=x_i, velocity=v_i, spin=spin_i)
    simulation = PhysicsSimulation(initial_state=ball_state, data_gen_settings=SyntheticDataGeneration()) # trajectory, times, eot, bounce_tokens
    return simulation.compute_3d_trajectory()


# MARK: Random Trajectory
def generate_random_trajectory(data_gen_settings: SyntheticDataGeneration):
    """
    Same as generate_sample_trajectory, but the initial ball state is sampled at random

    - Each bounds argument is a pair (low, high) where low and high are length-3 ndarrays
    - Defaults are centered around the same values used by generate_sample_trajectory
    - The same rng drives both the initial state sampling here and the in-rally racket hit sampling inside PhysicsSimulation (if racket_hits=True)
    """
    rng = data_gen_settings.physics_rng
    gen_settings = data_gen_settings.trajectory_generation_settings

    shot_type = rng.choice(list(ShotType))
    shot_gen_settings = _get_shot_distribution(gen_settings, shot_type)

    position_bounds = (shot_gen_settings.x_lower_bound, shot_gen_settings.x_upper_bound)
    x_i = rng.uniform(position_bounds[0], position_bounds[1]).astype(float)

    ball_state = BallState(position=x_i, velocity=np.zeros(3), spin=np.zeros(3))
    simulation = PhysicsSimulation(initial_state=ball_state, data_gen_settings=data_gen_settings)

    # Initial shot uses the same in-bounds rejection sampling as racket hits; x_i is always on the -x side so the shot goes toward +x
    v_i, spin_i = simulation.sample_shot(shot_gen_settings)
    simulation.ball_state.velocity = v_i
    simulation.ball_state.spin = spin_i

    return simulation.compute_3d_trajectory() # trajectory, times, eot, bounce_tokens


# MARK: Timestep Interpolation
def interpolate_to_new_timestep(trajectory: np.ndarray, eot: np.ndarray, bounce_tokens: np.ndarray, dt_0: float, dt_1: float):
    if dt_0 > dt_1:
        raise ValueError("dt_1 must be greater than or equal to dt_0.")
    
    # Trajectory
    T = (len(trajectory) - 1) * dt_0
    original_times = np.arange(start=0, stop=len(trajectory)) * dt_0 # (N,)
    new_times = np.arange(start=0, stop=math.floor(T / dt_1) + 1) * dt_1 # Flooring prevents overflowing past last endpoint of original trajectory; (M,)

    N = len(original_times)
    M = len(new_times)

    new_times_matrix = np.tile(new_times, (N, 1)) # (N, M)
    original_times_matrix = np.expand_dims(original_times, axis=1) # (N, 1)

    diff_matrix = new_times_matrix - original_times_matrix # (N, M); each col shows the differences for each timestep of the new times with every single one of the old times

    min_diff_indices = np.argmin(abs(diff_matrix), axis=0) # (M,)

    new_trajectory = []
    for i in range(M):
        min_diff = diff_matrix[min_diff_indices[i]][i]

        if min_diff > 0:
            point = _linearly_interpolate(trajectory[min_diff_indices[i]], trajectory[min_diff_indices[i] + 1], min_diff / dt_0)
        elif min_diff < 0:
            point = _linearly_interpolate(trajectory[min_diff_indices[i] - 1], trajectory[min_diff_indices[i]], (dt_0 + min_diff) / dt_0) # min_diff < 0
        else:
            point = trajectory[min_diff_indices[i]]
        
        new_trajectory.append(point)
    
    new_trajectory = np.stack(new_trajectory, axis=0, dtype=np.float64)

    # EOT - floored to be timestep before event
    eot_indices = np.nonzero(eot)[0]
    new_eot_indices = np.floor(eot_indices * dt_0 / dt_1).astype(int)
    new_eot = np.zeros((M,), dtype=np.float64)
    new_eot[new_eot_indices] = 1.0

    # Bounce Tokens - rounded to nearest timestep
    bounce_token_indicies = np.nonzero(bounce_tokens)[0]
    new_bounce_token_indicies = np.round(bounce_token_indicies * dt_0 / dt_1).astype(int)
    new_bounce_token_indicies = new_bounce_token_indicies[new_bounce_token_indicies < M] # Last bounce would be rounded to a timestep after the interpolated trajectory ended
    new_bounce_tokens = np.zeros((M,), dtype=np.float64)
    new_bounce_tokens[new_bounce_token_indicies] = 1.0

    return new_trajectory, new_eot, new_bounce_tokens


def _linearly_interpolate(x_0: np.ndarray, x_1: np.ndarray, t: float):
    return (1 - t) * x_0 + t * x_1


if __name__ == "__main__":

    trajectory, _, _ = generate_sample_trajectory()

    print(f"Num points in trajectory: {len(trajectory)}")

    plot_3d_trajectory(trajectory)
