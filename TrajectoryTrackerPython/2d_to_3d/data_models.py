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
    # C_d: float = 0.53 # TODO: Change as needed, do more research on this
    # C_l: float = 0.4 # TODO: Placeholder value
    C_m: float = 0.01 # TODO: Figure out if this is in units of m or dimensionless and if there's a typo in the paper

    def __post_init__(self):
        self.A = math.pi * self.R**2

    # See Dignall (2004) (or Chadwick (2003) for more detailed results) for more detail
    def calculate_C_d(self, v, w):
        s = np.linalg.norm(w) * self.R / np.linalg.norm(v)
        return 0.5365 + (1.9980 * (s) ** -2.1887 + 2.8619) ** -0.7069

    def calculate_C_l(self, v, w):
        s = np.linalg.norm(w) * self.R / np.linalg.norm(v)
        return (2.591 * (s) ** -1.877 + 4.809) ** -0.529
    


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
class TrajectoryGenerationSettings:
    """
    bounces - Whether bounces will be incorporated into ball trajectory
    racket_hits - Whether racket hits will be incorprated into ball trajectory
    position_bounds - Tuple of position vectors that bounds where the position can be sampled from after a racket hit
    velocity_bounds - Tuple of velocity vectors that bounds where the velocity can be sampled from after a racket hit
    spin_bounds - Tuple of angular velocity vectors that bounds where the angular velocity can be sampled from after a racket hit
    num_hits_range - The min and max number of racket hits in a trajectory is racket hits are enabled
    hit_zone_bounds - Used as bounds to define the hit area if racket hits are enabled; each bound is fractional, ranging from 0 (net) to 1/2 (baseline)
    """
    bounces: bool = True
    racket_hits: bool = False
    position_bounds: tuple[list, list] = None
    velocity_bounds: tuple[list, list] = None
    spin_bounds: tuple[list, list] = None
    num_hits_range: tuple[int, int] = (2, 5)
    hit_zone_bounds: tuple[float, float] = (3/8, 1/2)


@dataclass
class CameraConfigSamplingSettings:
    intrinsics_jitter: float = 5.0
    translation_jitter: float = 0.3
    rotation_jitter_deg: float = 2.0