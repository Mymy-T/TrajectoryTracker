//
//  SimulatorViewModel.swift
//  TrajectoryTracker
//
//  Created by Milo on 7/22/26.
//

import Foundation
import RealityKit
import simd

enum SpinInputMode: String, CaseIterable, Identifiable {
    case simple = "Simple"
    case advanced = "Advanced"

    var id: Self { self }
}

@Observable
final class SimulatorViewModel {
    var launchPosition = SIMD3<Double>(Double(-CourtEntity.courtLength / 2), 1.5, 0) // Only takes into affect after starting simulation
    var speed: Double = 20 // m/s
    var launchAngleDegrees: Double = 15
    var azimuthDegrees: Double = 0
    
    var spinInputMode: SpinInputMode = .simple
    var spinType: SpinType = .flat
    var spinRPM: Double = 0
    var spinAxis = SIMD3<Double>(0, 0, 1) // Defaults to topspin or backspin

    private(set) var trajectory: [TrajectoryPoint] = []
    var playbackTime: TimeInterval = 0
    var isPlaying: Bool = false
    
    var cameraControlMode: CameraControls = .pan

    private let simulator = TrajectorySimulator()
    

    var initialConditions: InitialConditions {
        let spin: SpinInput = (spinInputMode == .simple) ? .simple(type: spinType, rpm: spinRPM) : .vector(axis: spinAxis, rpm: spinRPM)

        return InitialConditions(
            launchPosition: launchPosition,
            speed: speed,
            launchAngleDegrees: launchAngleDegrees,
            azimuthDegrees: azimuthDegrees,
            spin: spin
        )
    }

    var duration: TimeInterval { trajectory.last?.time ?? 0 }

    func runSimulation() {
        trajectory = simulator.run(initialConditions)
        playbackTime = 0
        isPlaying = false
    }

    func play() {
        isPlaying = true
    }
    
    func pause() {
        isPlaying = false
    }

    func reset() {
        playbackTime = 0
        isPlaying = false
    }

    func scrub(to time: TimeInterval) {
        playbackTime = min(max(time, 0), duration)
    }

    func advancePlayback(by delta: TimeInterval) {
        guard isPlaying, duration > 0 else { return }
        let newTime = playbackTime + delta
        if newTime >= duration {
            playbackTime = duration
            isPlaying = false
        } else {
            playbackTime = newTime
        }
    }

    /// Linearly interpolates between adjacent samples for smooth playback between fixed simulation steps.
    func state(at time: TimeInterval) -> TrajectoryPoint? {
        guard let first = trajectory.first, let last = trajectory.last else { return nil }
        if time <= first.time { return first }
        if time >= last.time { return last }

        guard let upperIndex = trajectory.firstIndex(where: { $0.time > time }), upperIndex > 0 else {
            return last
        }
        let upper = trajectory[upperIndex]
        let lower = trajectory[upperIndex - 1]
        let span = upper.time - lower.time
        let t = span > 0 ? (time - lower.time) / span : 0

        return BallState(
            time: time,
            position: lower.position + (upper.position - lower.position) * t,
            velocity: lower.velocity + (upper.velocity - lower.velocity) * t,
            angularVelocity: lower.angularVelocity
        )
    }
}
