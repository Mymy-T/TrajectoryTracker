import copy
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class BallProperties:
    """
    - mass is measured in kg
    - radius is measured in m
    - cd is measured in m^-1
    - cm is measured in rad^-1
    """
    mass: float = 0.057
    radius: float = 0.033
    cd: float = 0.02
    cm: float = 1.2e-3

@dataclass
class CourtProperties:
    """
    - court_length is measured in m
    - court_width is measured in m
    """
    court_length: float = 23.77
    court_width: float = 10.97

@dataclass
class BallState:
    """
    Coordinate frame convention: +x is along the court length, +y is upward, +z is along court width; origin is at center of court
    - position is a vector of length 3, measured in m
    - velocity is a vector of length 3, measured in m/s
    - spin is a vector of length 3, measured in rad/s
    """
    position: np.ndarray
    velocity: np.ndarray
    spin: np.ndarray


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
        # TEMP: simulate for 10 seconds, break if y < 0
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
        """
        Returns the net acceleration vector
        """
        g = np.array([0, -9.81, 0])
        v = self.ball_state.velocity
        w = self.ball_state.spin
        a_drag = -self.ball_properties.cd * np.linalg.norm(v) * v
        a_magnus = self.ball_properties.cm * np.cross(w, v)
        return g + a_drag + a_magnus
    
    def _forward_euler_integration_step(self, a):
        state = self.ball_state
        t = self.timestep
        # NOTE: Used only for early testing purposes; will be replaced by a more precise algorithm, eg runge-kutta
        state.position += state.velocity * t
        state.velocity += a * t


if __name__ == "__main__":
    court_properties = CourtProperties()
    l = court_properties.court_length
    w = court_properties.court_width
    x_i = np.array([-l/2, 1, 0]).astype(float)
    v_i = np.array([30, 7, 0]).astype(float)
    spin_i = np.array([0, -300, -300]).astype(float) # 300 rad/s topspin, 300 rad/s left sidespin; exaggerated to show spin effects
    ball_state = BallState(position=x_i, velocity=v_i, spin=spin_i)
    simulation = PhysicsSimulation(initial_state=ball_state, simulation_timestep=0.001)
    trajectory, end_time = simulation.compute_3d_trajectory()

    print(f"Num points in trajectory: {len(trajectory)}")

    t = np.linspace(0, end_time, int(10/0.001))
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
    ax.plot(x, z, y, label='3D Trajectory', color='b', linewidth=2)

    # Highlight start and end points
    ax.scatter(x[0], z[0], y[0], color='green', s=50, label='Start')
    ax.scatter(x[-1], z[-1], y[-1], color='red', s=50, label='End')

    ax.set_xlim([-l/2, l/2])    
    ax.set_ylim([-w/2, w/2])
    ax.set_zlim([0, 4]) 
    ax.set_box_aspect([l, w, 4])

    # NOTE: In matplotlib, +z is up; in the coordinates of the trajectory, +y is up
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Z Axis')
    ax.set_zlabel('Y Axis')
    ax.set_title('3D Path Trajectory')
    ax.legend()

    plt.show()

