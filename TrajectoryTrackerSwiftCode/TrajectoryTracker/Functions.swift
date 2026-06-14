//
//  Functions.swift
//  TrajectoryTracker
//
//  Created by Milo on 6/14/26.
//

import AVFoundation
import Foundation
import UIKit

func generateThumbnail(for videoURL: URL) async -> UIImage {
    let asset = AVURLAsset(url: videoURL)
    let generator = AVAssetImageGenerator(asset: asset)
    
    // Get exact frame
    generator.requestedTimeToleranceBefore = .zero
    generator.requestedTimeToleranceAfter = .zero
    generator.appliesPreferredTrackTransform = true // Match original orientation
    
    do {
        let (cgImage, _) = try await generator.image(at: .zero) // First frame
        return UIImage(cgImage: cgImage)
    } catch {
        print("Failed to generate thumbnail: \(error.localizedDescription)")
        return UIImage(systemName: "exclamationmark.triangle.fill") ?? UIImage()
    }
}
