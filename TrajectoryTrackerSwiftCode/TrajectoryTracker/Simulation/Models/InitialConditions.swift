//
//  InitialConditions.swift
//  TrajectoryTracker
//
//  Created by Milo on 7/22/26.
//

import Foundation
import simd

/// User-specified launch parameters for a simulated shot.
struct InitialConditions {
    var launchPosition: SIMD3<Double>
    var speed: Double // m/s
    var launchAngleDegrees: Double // vertical angle above horizontal
    var azimuthDegrees: Double // 0 when facing net; angle from the normal (of the baseline) is measured, positive if going towards the right
    var spin: SpinInput

    /// Resolves speed/launchAngle/azimuth into a single velocity vector.
    var initialVelocity: SIMD3<Double> {
        let launchAngle = launchAngleDegrees * .pi / 180 // radians
        let azimuth = azimuthDegrees * .pi / 180
        return SIMD3<Double>(
            speed * cos(launchAngle) * cos(azimuth), // vx; +x direction from baseline towards net
            speed * sin(launchAngle), // vy; up/down movement; +y direction upwards
            speed * cos(launchAngle) * sin(azimuth) // vz; +z direction to the right
        )
    }
}
