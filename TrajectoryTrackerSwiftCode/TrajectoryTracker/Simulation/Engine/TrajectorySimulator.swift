//
//  TrajectorySimulator.swift
//  TrajectoryTracker
//
//  Created by Milo on 7/22/26.
//

import Foundation
import simd

/// Numerically integrates a ball's motion forward in time under a `ForceModel`. Currently uses Euler method
struct TrajectorySimulator {
    var forceModel: ForceModel = ZeroForceModel()
    var timeStep: TimeInterval = 1.0 / 120
    var maxDuration: TimeInterval = 5.0

    func run(_ initialConditions: InitialConditions, ball: BallProperties = .regulation) -> [TrajectoryPoint] {
        var state = BallState(
            time: 0,
            position: initialConditions.launchPosition,
            velocity: initialConditions.initialVelocity,
            angularVelocity: initialConditions.spin.angularVelocity
        )
        var states: [TrajectoryPoint] = [state]

        while state.time < maxDuration {
            let acceleration = forceModel.netAcceleration(for: state, ball: ball)
            let velocity = state.velocity + acceleration * timeStep
            let position = state.position + velocity * timeStep

            state = BallState(
                time: state.time + timeStep,
                position: position,
                velocity: velocity,
                angularVelocity: state.angularVelocity
            )
            states.append(state)

            if state.position.y <= 0 {
                // Placeholder ground-contact stop condition; no bounce yet.
                break
            }
        }

        return states
    }
}
