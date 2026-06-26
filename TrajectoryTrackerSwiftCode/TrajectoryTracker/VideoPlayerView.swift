//
//  VideoPlayerView.swift
//  TrajectoryTracker
//
//  Created by Milo on 6/25/26.
//

import AVKit
import SwiftUI

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
