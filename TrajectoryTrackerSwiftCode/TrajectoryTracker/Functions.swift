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

enum ExportVideoError: Error {
    case videoTrackLoadingFailure(message: String)
    case generatorError(message: String)
    case imageConversionError(message: String)
    case folderCreationError(message: String)
    case imageSavingError(message: String)
}

// TODO: Fix cannot decode error

func exportVideoFrames(videoURL: URL) async throws {
    // Prepare asset and generator
    let asset = AVURLAsset(url: videoURL)
    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    
    // Generate times (get all frames of video)
    let duration = try await asset.load(.duration)
    guard let videoTrack = try await asset.load(.tracks).first(where: { $0.mediaType == .video }) else {
        throw ExportVideoError.videoTrackLoadingFailure(message: "No video track found.")
    }
    let nominalFrameRate = try await videoTrack.load(.nominalFrameRate)
    let timescale = duration.timescale
    let frameDuration = 1.0 / Double(nominalFrameRate)
    let totalDuration = CMTimeGetSeconds(duration)
    let totalFrames = Int(round(totalDuration * Double(nominalFrameRate)))
//    let toleranceTime = CMTime(seconds: frameDuration / 2.0, preferredTimescale: timescale)
//    generator.requestedTimeToleranceBefore = toleranceTime
//    generator.requestedTimeToleranceAfter = toleranceTime
    generator.requestedTimeToleranceBefore = .zero
    generator.requestedTimeToleranceAfter = .zero
    var times: [CMTime] = []
    let nth_frame = 10 // Save every nth frame
    for frameIndex in 0..<totalFrames {
        if frameIndex % nth_frame == 0 {
            let seconds = Double(frameIndex) * frameDuration
            let time = CMTime(seconds: seconds, preferredTimescale: timescale)
            times.append(time)
        }
    }
    
    // Prepare the save directory
    let folderURL = URL.documentsDirectory.appending(path: "ExportedImages")
    do {
        try FileManager.default.createDirectory(at: folderURL, withIntermediateDirectories: true)
    } catch {
        throw ExportVideoError.folderCreationError(message: "Unable to create folder for storing exported images: \(error.localizedDescription)")
    }
    
    // Get the images
    for await result in generator.images(for: times) {
        switch result {
        case .success(_, let image, _):
            let uiImage = UIImage(cgImage: image)
            guard let pngData = uiImage.jpegData(compressionQuality: 0.8) else {
                throw ExportVideoError.imageConversionError(message: "Unable to convert UIImage to JPEG.")
            }
            let filename = UUID().uuidString
            let saveURL = folderURL.appending(path: filename).appendingPathExtension("png")
            
            do {
                try pngData.write(to: saveURL)
            } catch {
                throw ExportVideoError.imageSavingError(message: "Unable to save image: \(error.localizedDescription)")
            }
            
        case .failure(_, let error):
//            throw ExportVideoError.generatorError(message: "Generator unable to create image: \(error.localizedDescription)")
            print("Generator unable to create image: \(error.localizedDescription)")
        }
    }
}

func deleteVideoFromURL(_ url: URL) {
    do {
        try FileManager.default.removeItem(at: url)
    } catch {
        print("Error in deleting video from url: \(error.localizedDescription)")
    }
}

