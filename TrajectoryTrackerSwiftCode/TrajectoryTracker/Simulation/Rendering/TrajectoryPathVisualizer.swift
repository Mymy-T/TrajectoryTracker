//
//  TrajectoryPathVisualizer.swift
//  TrajectoryTracker
//
//  Created by Milo on 7/22/26.
//

import RealityKit
import UIKit

/// Builds the ball entity and the sampled path markers used to visualize a simulated trajectory.
enum TrajectoryPathVisualizer {
    static func makeBallEntity() -> Entity {
        let entity = ModelEntity(
            mesh: .generateSphere(radius: 0.0335), // Regulation tennis ball radius between 3.27 and 3.43 cm -> avg is 3.35 cm, or 0.0335 m
            materials: [SimpleMaterial(color: .yellow, isMetallic: false)]
        )
        entity.name = "Ball"
        return entity
    }

    static func projectPath(in container: Entity, from trajectory: [TrajectoryPoint]) {
        container.children.removeAll()
        guard trajectory.count > 1 else { return }

        let trajectoryPointMaterial = SimpleMaterial(color: .white.withAlphaComponent(0.5), isMetallic: false)
        let sampleStride = max(trajectory.count / 60, 1)

        for index in stride(from: 0, to: trajectory.count, by: sampleStride) {
            let marker = ModelEntity(mesh: .generateSphere(radius: 0.01), materials: [trajectoryPointMaterial])
            marker.position = trajectory[index].position.asFloat
            container.addChild(marker)
        }
    }
}
