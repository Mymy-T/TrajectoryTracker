//
//  SpinInput.swift
//  TrajectoryTracker
//
//  Created by Milo on 7/22/26.
//

import Foundation
import simd

/// Qualitive spin direction options
enum SpinType: String, CaseIterable, Identifiable {
    case flat
    case topspin
    case backspin
    case sidespinLeft
    case sidespinRight

    var id: Self { self }

    var displayName: String {
        switch self {
        case .flat: return "Flat"
        case .topspin: return "Topspin"
        case .backspin: return "Backspin"
        case .sidespinLeft: return "Sidespin (Left)"
        case .sidespinRight: return "Sidespin (Right)"
        }
    }
}

/// How the user specified spin: a simplified named type + rate, or a full 3D axis + rate.
enum SpinInput {
    case simple(type: SpinType, rpm: Double) // Direction is qualitative
    case vector(axis: SIMD3<Double>, rpm: Double) // Manually input 3D spin vector

    /// Normalizes either input mode into a single angular-velocity vector (rad/s) for the engine to consume.
    var angularVelocity: SIMD3<Double> {
        switch self {
        case .simple:
            // TODO: map each SpinType to a rotation axis (relative to travel direction) once the
            // Magnus-force model is implemented. Left at zero for now — this is UI/engine plumbing only.
            return .zero
        case .vector(let axis, let rpm):
            let normalizedAxis = simd_length(axis) > 0 ? simd_normalize(axis) : .zero
            let radiansPerSecond = rpm * 2 * .pi / 60
            return normalizedAxis * radiansPerSecond
        }
    }
}
