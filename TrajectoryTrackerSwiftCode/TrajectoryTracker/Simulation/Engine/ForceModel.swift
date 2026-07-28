//
//  ForceModel.swift
//  TrajectoryTracker
//
//  Created by Milo on 7/22/26.
//

import Foundation
import simd

/// Computes net acceleration on the ball at a given state.
protocol ForceModel {
    func netAcceleration(for state: BallState, ball: BallProperties) -> SIMD3<Double>
}

/// Model where no force act upon the ball; travels in a straight line
struct ZeroForceModel: ForceModel {
    func netAcceleration(for state: BallState, ball: BallProperties) -> SIMD3<Double> {
        return .zero
    }
}

/// Model where the only force acting on the ball is gravity; ball travels in a parabola
struct GravityForceModel: ForceModel {
    func netAcceleration(for state: BallState, ball: BallProperties) -> SIMD3<Double> {
        return SIMD3<Double>(0, -9.8, 0)
    }
}

struct MagnusDragGravityForceModel: ForceModel {
    func netAcceleration(for state: BallState, ball: BallProperties) -> SIMD3<Double> {
        let dragAcceleration: SIMD3<Double> = -ball.cd * simd_length(state.velocity) * state.velocity
        let magnusAcceleration: SIMD3<Double> = ball.cm * cross(state.angularVelocity, state.velocity)
        let gravitationalAcceleration = SIMD3<Double>(0, -9.8, 0)
        return gravitationalAcceleration + dragAcceleration + magnusAcceleration
    }
}
