import copy
from data_models import BallState, BallProperties, CourtProperties
import matplotlib.pyplot as plt
import numpy as np

class PhysicsSimulation():
    """
    Class for simulating ball trajectories. The simulator's copy of the initial ball state will be modified in place.
    """
    def __init__(self, initial_state: BallState, simulation_timestep: float, ball_properties = BallProperties(), court_properties = CourtProperties()):
        self.ball_state = copy.deepcopy(initial_state)
        self.ball_properties = ball_properties
        self.court_properties = court_properties
        self.timestep = simulation_timestep

    def compute_3d_trajectory(self):
        # TEMP: simulate for 10 seconds, break if y < 0 bounce physics and out of bounds TODO
        trajectory = []
        num_steps = int(10 / self.timestep)
        for i in range(num_steps):
            trajectory.append(self.ball_state.position.copy())
            a = self._compute_acceleration()
            self._forward_euler_integration_step(a)
            y = self.ball_state.position[1]
            if y <= 0:
                end_time = i * self.timestep
                break
        
        return trajectory, end_time


    def _compute_acceleration(self):
        g = np.array([0, -9.81, 0])
        v = self.ball_state.velocity
        w = self.ball_state.spin
        a_drag = -self.ball_properties.cd * np.linalg.norm(v) * v
        a_magnus = self.ball_properties.cm * np.cross(w, v)
        return g + a_drag + a_magnus # Net acceleration vector
    
    def _forward_euler_integration_step(self, a):
        state = self.ball_state
        t = self.timestep
        # NOTE: Used only for early testing purposes; will be replaced by a more precise algorithm, eg runge-kutta
        state.position += state.velocity * t
        state.velocity += a * t


def generate_sample_trajectory():
    court_properties = CourtProperties()
    l = court_properties.court_length
    x_i = np.array([-l/2, 1, 0]).astype(float)
    v_i = np.array([30, 7, 0]).astype(float)
    spin_i = np.array([0, -300, -300]).astype(float) # 300 rad/s topspin, 300 rad/s right sidespin; exaggerated to show spin effects
    ball_state = BallState(position=x_i, velocity=v_i, spin=spin_i)
    simulation = PhysicsSimulation(initial_state=ball_state, simulation_timestep=0.001)
    return simulation.compute_3d_trajectory()


def plot_3d_trajectory(trajectory):
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
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')


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
    ax.plot(x, z, y, label='3D Trajectory', color='b', linewidth=2)

    # Highlight start and end points
    ax.scatter(x[0], z[0], y[0], color='green', s=50, label='Start')
    ax.scatter(x[-1], z[-1], y[-1], color='red', s=50, label='End')

    ax.set_xlim([-l/2 - 1, l/2 + 1])    
    ax.set_ylim([-w/2 - 1, w/2 + 1])
    ax.invert_yaxis() # This way negative values of z axis are towards the left
    ax.set_zlim([0, 4]) 
    ax.set_box_aspect([l, w, 4])

    # NOTE: In matplotlib, +z is up; in the coordinates of the trajectory, +y is up
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Z Axis')
    ax.set_zlabel('Y Axis')
    ax.set_title('3D Path Trajectory')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    trajectory, end_time = generate_sample_trajectory()

    print(f"Num points in trajectory: {len(trajectory)}")

    plot_3d_trajectory(trajectory)

