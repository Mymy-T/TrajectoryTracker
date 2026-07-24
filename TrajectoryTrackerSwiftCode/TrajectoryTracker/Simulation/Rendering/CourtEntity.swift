//
//  CourtEntity.swift
//  TrajectoryTracker
//
//  Created by Milo on 7/22/26.
//

import RealityKit
import UIKit

/// Builds the static tennis court geometry used as a scale reference in the simulator.
enum CourtEntity {
    private static let courtLength: Float = 23.77 // downcourt distance (x), regulation doubles court
    private static let courtWidth: Float = 10.97 // lateral extent (z)
    private static let lineThickness: Float = 0.05

    static func make() -> Entity {
        let court = Entity()
        court.name = "Court"

        let surfaceMaterial = SimpleMaterial(color: UIColor(red: 0.11, green: 0.35, blue: 0.55, alpha: 1), isMetallic: false)
        let surface = ModelEntity(
            mesh: .generatePlane(width: courtLength, depth: courtWidth),
            materials: [surfaceMaterial]
        )
        court.addChild(surface)

        let lineMaterial = SimpleMaterial(color: .white, isMetallic: false)
        for line in perimeterLines() {
            let box = ModelEntity(mesh: .generateBox(size: line.size), materials: [lineMaterial])
            box.position = line.position
            court.addChild(box)
        }

        return court
    }

    private static func perimeterLines() -> [(size: SIMD3<Float>, position: SIMD3<Float>)] {
        let baselineSize = SIMD3<Float>(lineThickness, 0.01, courtWidth)
        let sidelineSize = SIMD3<Float>(courtLength, 0.01, lineThickness)
        return [
            (baselineSize, SIMD3<Float>(courtLength / 2, 0.01, 0)),
            (baselineSize, SIMD3<Float>(-courtLength / 2, 0.01, 0)),
            (sidelineSize, SIMD3<Float>(0, 0.01, courtWidth / 2)),
            (sidelineSize, SIMD3<Float>(0, 0.01, -courtWidth / 2))
        ]
    }
}
