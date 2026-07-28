//
//  BallState.swift
//  TrajectoryTracker
//
//  Created by Milo on 7/22/26.
//

import Foundation
import simd

/// Coordinate convention used throughout the simulation: SIMD3<Double>, meters, x = downcourt distance, y = height (up), z = lateral offset.
struct BallState {
    var time: TimeInterval
    var position: SIMD3<Double>
    var velocity: SIMD3<Double>
    var angularVelocity: SIMD3<Double>
}

/// A single sample of a simulated trajectory.
typealias TrajectoryPoint = BallState

extension SIMD3 where Scalar == Double {
    var asFloat: SIMD3<Float> {
        SIMD3<Float>(Float(x), Float(y), Float(z))
    }
}
