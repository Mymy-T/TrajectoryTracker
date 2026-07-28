//
//  CourtEntity.swift
//  TrajectoryTracker
//

import RealityKit
import UIKit

/// Builds the static tennis court geometry used as a scale reference in the simulator.
enum CourtEntity {
    static let courtLength: Float = 23.77 // downcourt distance (x), regulation doubles court
    static let courtWidth: Float = 10.97 // lateral extent (z)
    private static let margin: Float = 2.0
    private static let slabHeight: Float = 0.05

    static func make() -> Entity {
        let court = Entity()
        court.name = "Court"

        let totalLength = courtLength + 2 * margin
        let totalWidth = courtWidth + 2 * margin

        let mesh = MeshResource.generateBox(size: SIMD3<Float>(totalLength, slabHeight, totalWidth))
        let material = getCourtMaterial()

        let slab = ModelEntity(mesh: mesh, materials: [material])
        slab.position.y = -slabHeight / 2 // Offset slab center so that y=0 when ball hits top surface (no clipping through court, used for bounce checks)
        court.addChild(slab)

        return court
    }

    private static func getCourtMaterial() -> RealityKit.Material {
        let image = CourtTextureRenderer.makeImage(courtLength: courtLength, courtWidth: courtWidth, margin: margin)

        guard let cgImage = image.cgImage,
              let textureResource = try? TextureResource(image: cgImage, options: .init(semantic: .color)) else {
            return SimpleMaterial(color: UIColor(red: 0.11, green: 0.35, blue: 0.55, alpha: 1), isMetallic: false)
        }
        
        // TODO: Make back face a single color (UIColor(red: 0.11, green: 0.35, blue: 0.55, alpha: 1))

        return UnlitMaterial(texture: textureResource)
    }
}
