//
//  AnalysisView.swift
//  TrajectoryTracker
//
//  Created by Milo on 6/14/26.
//

import AVKit
import SwiftUI

struct AnalysisView: View {
    
    let videoPath: URL
    
    var body: some View {
        VStack {
            VideoPlayerView(videoPath: videoPath)
            
            Text("TODO")
            
        }.navigationTitle("Analysis")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct VideoPlayerView: View {
    
    let videoPath: URL
    @State private var player: AVPlayer?
    
    var body: some View {
        if let player = player {
            VideoPlayer(player: player)
                .onAppear {
                    player.play()
                }
                .onDisappear {
                    player.pause()
                }
        } else {
            ProgressView("Loading Video...")
                .onAppear {
                    player = AVPlayer(url: videoPath)
                }
        }
    }
    
}
