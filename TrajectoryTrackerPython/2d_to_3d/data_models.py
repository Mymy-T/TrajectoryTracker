from dataclasses import dataclass, field
import math
import numpy as np

@dataclass(frozen=True)
class EnvironmentProperties:
    """
    g - gravitational acceleration (m/s^2)
    rho - air density @ 20 degrees C (kg/m^3)
    mu - dynamic visocisty of air @ 20 degrees C and 1 atm (Pa * s)
    """
    g: float = 9.81
    rho: float = 1.205
    mu: float = 1.81e-5

@dataclass
class BallProperties:
    """
    m - mass (kg)
    R - ball radius (m)
    A - ball cross sectional area (m^2); automatically computed
    I - moment of inertia (kg * m^2)
    C_d - drag coefficient (dimensionless)
    C_l - lift coefficient associated with Magnus effect (dimensionless)
    C_m - drag aerodynamic moment coefficient
    """
    m: float = 0.057
    R: float = 0.0325
    A: float = field(init=False) 
    I: float = 3.155e-5 # NOTE: This value may differ from the value used in the bounce model, as the bounce model assumed I = 2/3 * m * R^2; fixing this discrepancy could be a TODO
    C_d: float = 0.53 # TODO: Change as needed, do more research on this
    C_l: float = 0.4 # TODO: Placeholder value
    C_m: float = 0.01 # TODO: Figure out if this is in units of m or dimensionless and if there's a typo in the paper

    def __post_init__(self):
        self.A = math.pi * self.R**2

@dataclass(frozen=True)
class BounceProperties:
    """
    e - coefficient of resitution (dimensionless)
    mu - coefficient of friction between ball and court (dimensionless)
    """
    e: float = 0.75 # Dignall, 2004
    mu: float = 0.55 # Dignall, 2004

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
