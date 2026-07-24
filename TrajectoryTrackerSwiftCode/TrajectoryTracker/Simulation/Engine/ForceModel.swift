//
//  ForceModel.swift
//  TrajectoryTracker
//
//  Created by Milo on 7/22/26.
//

import Foundation
import simd

/// Computes net acceleration on the ball at a given state. This is the seam where gravity, drag,
/// and Magnus (spin) forces will be implemented — nothing else in the app needs to change when they are.
protocol ForceModel {
    func netAcceleration(for state: BallState, ball: BallProperties) -> SIMD3<Double>
}

/// Placeholder used until real physics is implemented. Produces zero acceleration, so a
/// `TrajectorySimulator` using this model traces a straight, constant-velocity path.
struct ZeroForceModel: ForceModel {
    func netAcceleration(for state: BallState, ball: BallProperties) -> SIMD3<Double> {
        .zero
    }
}
