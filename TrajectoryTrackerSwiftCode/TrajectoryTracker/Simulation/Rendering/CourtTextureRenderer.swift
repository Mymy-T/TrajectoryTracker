//
//  CourtTextureRenderer.swift
//  TrajectoryTracker
//
//  Created by Milo on 7/26/26.
//

import UIKit

/// Draws a top-down regulation tennis court diagram, used as a texture for `CourtEntity`.
enum CourtTextureRenderer {
    /// - Parameters:
    ///   - courtLength: Length of the doubles court (x axis), baseline to baseline, in meters.
    ///   - courtWidth: Width of the doubles court (z axis), sideline to sideline, in meters.
    ///   - margin: Extra run-off space drawn around the court lines, in meters.
    /// - Returns:
    ///   - UIImage of the court
    static func makeImage(courtLength: Float, courtWidth: Float, margin: Float = 2.0, pixelsPerMeter: CGFloat = 40) -> UIImage {
        let totalLength = (CGFloat(courtLength) + 2 * CGFloat(margin)) * pixelsPerMeter
        let totalWidth = (CGFloat(courtWidth) + 2 * CGFloat(margin)) * pixelsPerMeter
        let size = CGSize(width: totalLength, height: totalWidth) // Rectangle is longer on horizontal side

        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { context in
            let ctx = context.cgContext

            let marginColor = UIColor(red: 0.09, green: 0.28, blue: 0.45, alpha: 1)
            ctx.setFillColor(marginColor.cgColor)
            ctx.fill(CGRect(origin: .zero, size: size))

            let courtOrigin = CGPoint(x: CGFloat(margin) * pixelsPerMeter, y: CGFloat(margin) * pixelsPerMeter) // Coords of top left coordinate
            let courtRect = CGRect(origin: courtOrigin, size: CGSize(width: CGFloat(courtLength) * pixelsPerMeter, height: CGFloat(courtWidth) * pixelsPerMeter))
            let surfaceColor = UIColor(red: 0.11, green: 0.35, blue: 0.55, alpha: 1)
            ctx.setFillColor(surfaceColor.cgColor)
            ctx.fill(courtRect)

            // Converts a point in meters (x = length axis, z = width axis, origin at center of the court) into image pixel coordinates; x > 0 -> right half, z > 0 -> bottom half; simplifes line drawing later on
            func point(x: Float, z: Float) -> CGPoint { // courtOrigin has already been converted to pixels
                return CGPoint(x: courtOrigin.x + (CGFloat(courtLength) / 2 + CGFloat(x)) * pixelsPerMeter, y: courtOrigin.y + (CGFloat(z) + CGFloat(courtWidth) / 2) * pixelsPerMeter)
            }

            func stroke(_ from: CGPoint, _ to: CGPoint, color: UIColor, width: CGFloat) {
                ctx.setStrokeColor(color.cgColor)
                ctx.setLineWidth(width)
                ctx.beginPath()
                ctx.move(to: from)
                ctx.addLine(to: to)
                ctx.strokePath()
            }

            let lineColor = UIColor.white
            let lineWidth = max(pixelsPerMeter / 20 , 1) // Regulation tennis court lines range from 2.5 to 10 cm in width depending on the line; 5 cm assumed for simplicity, which is 1/20 of a meter; minimum of 1 pixel to ensure line visibility
            
             // Units of meters
            let halfLength = courtLength / 2
            let halfDoublesWidth = courtWidth / 2
            let singlesWidth: Float = 8.23
            let halfSinglesWidth = singlesWidth / 2
            let serviceLineDistanceFromNet: Float = 6.4
            let centerMarkLength: Float = 0.15

            // Doubles sidelines + baselines (outer court boundary)
            stroke(point(x: -halfLength, z: -halfDoublesWidth), point(x: halfLength, z: -halfDoublesWidth), color: lineColor, width: lineWidth) // Top edge
            stroke(point(x: -halfLength, z: halfDoublesWidth), point(x: halfLength, z: halfDoublesWidth), color: lineColor, width: lineWidth)  // Bottom edge
            stroke(point(x: -halfLength, z: -halfDoublesWidth), point(x: -halfLength, z: halfDoublesWidth), color: lineColor, width: lineWidth) // Left edge
            stroke(point(x: halfLength, z: -halfDoublesWidth), point(x: halfLength, z: halfDoublesWidth), color: lineColor, width: lineWidth) // Right edge

            // Singles sidelines
            stroke(point(x: -halfLength, z: -halfSinglesWidth), point(x: halfLength, z: -halfSinglesWidth), color: lineColor, width: lineWidth) // Top edge
            stroke(point(x: -halfLength, z: halfSinglesWidth), point(x: halfLength, z: halfSinglesWidth), color: lineColor, width: lineWidth) // Bottom edge

            // Service lines
            stroke(point(x: -serviceLineDistanceFromNet, z: -halfSinglesWidth), point(x: -serviceLineDistanceFromNet, z: halfSinglesWidth), color: lineColor, width: lineWidth) // Left edge
            stroke(point(x: serviceLineDistanceFromNet, z: -halfSinglesWidth), point(x: serviceLineDistanceFromNet, z: halfSinglesWidth), color: lineColor, width: lineWidth) // Right edge

            // Center service line (divides ad and deuce boxes)
            stroke(point(x: -serviceLineDistanceFromNet, z: 0), point(x: serviceLineDistanceFromNet, z: 0), color: lineColor, width: lineWidth)

            // Center marks at the midpoint of each baseline
            stroke(point(x: -halfLength, z: 0), point(x: -halfLength + centerMarkLength, z: 0), color: lineColor, width: lineWidth) // Left baseline center mark
            stroke(point(x: halfLength, z: 0), point(x: halfLength - centerMarkLength, z: 0), color: lineColor, width: lineWidth) // Right baseline center mark

            // Net line
            let netColor = UIColor.black.withAlphaComponent(0.6)
            stroke(point(x: 0, z: -halfDoublesWidth), point(x: 0, z: halfDoublesWidth), color: netColor, width: lineWidth * 1.5)
        }
    }
}
