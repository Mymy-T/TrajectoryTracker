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

    static let regulation = BallProperties()
}
