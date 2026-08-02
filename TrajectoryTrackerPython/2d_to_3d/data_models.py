from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class CourtProperties:
    """
    All variables are measured in meters.
    """
    court_length: float = 23.77
    court_width: float = 10.97
    singles_width: float = 8.23
    service_line_dist_from_net: float = 6.40
    net_height_at_center: float = 0.914
    net_height_at_posts: float = 1.07

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
