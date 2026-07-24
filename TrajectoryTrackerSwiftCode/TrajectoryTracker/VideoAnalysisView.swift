//
//  VideoAnalysisView.swift
//  TrajectoryTracker
//
//  Created by Milo on 7/22/26.
//

import CoreML
import SwiftUI
import Vision

struct VideoAnalysisView: View {

    let videoPath: URL

    @State private var presentingExportFrameConfirmation = false

    var body: some View {
        VStack {
            VideoPlayerView(videoPath: videoPath)

            Text("TODO")

        }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    presentingExportFrameConfirmation = true
                } label: {
                    Image(systemName: "square.and.arrow.up")
                        .foregroundStyle(.green)
                }
            }
        }
        .alert("Export & Save?", isPresented: $presentingExportFrameConfirmation) {
            Button("Cancel", role: .cancel) { }
            Button("Yes", role: .confirm) {
                Task {
                    do {
                        try await exportVideoFrames(videoURL: videoPath)
                    } catch ExportVideoError.generatorError(message: let message) {
                        print("Generator Image Creation Error - \(message)")
                    } catch ExportVideoError.imageConversionError(message: let message) {
                        print("Image Conversion Error - \(message)")
                    } catch ExportVideoError.imageSavingError(message: let message) {
                        print("Image Saving Error - \(message)")
                    } catch ExportVideoError.videoTrackLoadingFailure(message: let message) {
                        print("Video Track Loading Error - \(message)")
                    } catch ExportVideoError.folderCreationError(message: let message) {
                        print("Folder Creation Error - \(message)")
                    } catch {
                        print("Unknown error: \(error.localizedDescription)")
                    }
                }
            }
        }
    }
}
