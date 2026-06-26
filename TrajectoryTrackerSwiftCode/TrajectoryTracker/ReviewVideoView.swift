//
//  ReviewVideoView.swift
//  TrajectoryTracker
//
//  Created by Milo on 6/12/26.
//

import AVKit
import SwiftData
import SwiftUI

struct ReviewVideoView: View {
    
    let path: URL?
    
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
                        let filename = UUID().uuidString + "." + filePath.pathExtension
                        let savePath = URL.documentsDirectory.appendingPathComponent(filename)
                        do {
                            try FileManager.default.moveItem(at: filePath, to: savePath)
                            let newVideo = Video(filename: filename)
                            modelContext.insert(newVideo)
                            try modelContext.save()
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
}
