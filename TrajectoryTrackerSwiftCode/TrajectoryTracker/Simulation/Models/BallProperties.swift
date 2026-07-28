//
//  BallProperties.swift
//  TrajectoryTracker
//
//  Created by Milo on 7/22/26.
//

import Foundation

/// Physical constants for the simulated ball. Regulation values per ITF tennis ball specs.
struct BallProperties {
    var mass: Double = 0.057 // kg
    var radius: Double = 0.033 // m
    
    // For a standard tennis ball with a = g - cd*|v|*v + cm*(w x v) where g, v, w, and a are vectors, the effective drag parameter is cd ~ 0.020 m^-1, and cm ~ 10^-4 to 10^-3 s/m (varying with spin rate and surface fuzz)
    var cd: Double = 0.02
    var cm: Double = 5e-4

    static let regulation = BallProperties()
}
