//
//  ReviewVideoView.swift
//  TrajectoryTracker
//
//  Created by Milo on 6/12/26.
//

import AVKit
import SwiftData
import SwiftUI
import simd

private struct CameraIntrinsics: Encodable {
    let fx: Float
    let fy: Float
    let cx: Float
    let cy: Float
    let referenceWidth: Int32
    let referenceHeight: Int32

    init(matrix: matrix_float3x3, dimensions: CMVideoDimensions) {
        fx = matrix.columns.0.x
        fy = matrix.columns.1.y
        cx = matrix.columns.2.x
        cy = matrix.columns.2.y
        referenceWidth = dimensions.width
        referenceHeight = dimensions.height
    }
}

struct ReviewVideoView: View {

    let path: URL? // Temporary stored path, not final save path
    let intrinsicMatrix: matrix_float3x3?
    let intrinsicFormatDimensions: CMVideoDimensions?

    @Environment(\.modelContext) var modelContext

    @State private var player: AVPlayer?

    @Binding var showCamera: Bool
    @Binding var reviewMode: Bool
    
    var body: some View {
        ZStack(alignment: .bottom) {
            VStack {
                if let player = player {
                    VideoPlayer(player: player)
                        .onAppear {
                            player.play()
                        }
                        .onDisappear {
                            player.pause()
                        }
                        .ignoresSafeArea()
                } else {
                    // Loading state while the URL is being resolved
                    ProgressView("Loading Video...")
                        .onAppear {
                            if let filePath = path {
                                player = AVPlayer(url: filePath)
                            }
                        }
                }
            }
            
            HStack {
                Button {
                    reviewMode = false
                } label: {
                    Image(systemName: "xmark")
                        .padding()
                        .tint(.red)
                }.glassEffect()
                
                Spacer()
                
                Button {
                    if let filePath = path {
                        let videoID = UUID().uuidString
                        let filename = videoID + "." + filePath.pathExtension
                        let savePath = URL.documentsDirectory.appendingPathComponent(filename)
                        do {
                            try FileManager.default.moveItem(at: filePath, to: savePath)
                            let newVideo = Video(filename: filename)
                            modelContext.insert(newVideo)
                            try modelContext.save()
                            saveIntrinsicsSidecar(videoID: videoID)
                            print("Video saved successfully")
                        } catch {
                            print("Failed to save video: \(error.localizedDescription)")
                        }
                    }
                    reviewMode = false
                    showCamera = false
                } label: {
                    Image(systemName: "checkmark")
                        .padding()
                        .tint(.green)
                }.glassEffect()
                
            }

        }


    }

    private func saveIntrinsicsSidecar(videoID: String) {
        guard let intrinsicMatrix, let intrinsicFormatDimensions else { return }
        let intrinsics = CameraIntrinsics(matrix: intrinsicMatrix, dimensions: intrinsicFormatDimensions)
        let sidecarPath = URL.documentsDirectory.appendingPathComponent(videoID + "_intrinsics.json")
        do {
            let data = try JSONEncoder().encode(intrinsics)
            try data.write(to: sidecarPath)
        } catch {
            print("Failed to write intrinsics sidecar: \(error.localizedDescription)")
        }
    }
}
