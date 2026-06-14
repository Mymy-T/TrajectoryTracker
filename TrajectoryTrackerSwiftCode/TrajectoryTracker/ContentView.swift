//
//  ContentView.swift
//  TrajectoryTracker
//
//  Created by Milo on 5/28/26.
//

import AVFoundation
import SwiftData
import SwiftUI


struct ContentView: View {
    
    @Query var videos: [Video]
    
    @State private var showCamera: Bool = false
    @State private var showAccessError: Bool = false // MARK: TODO
    @State private var thumbnails: [(URL, UIImage)] = []
    
    let columns = [GridItem(.fixed(400))]

    var body: some View {
        NavigationStack {
            ScrollView {
                LazyVGrid(columns: columns) {
                    ForEach(thumbnails, id: \.self.0) { thumbnail in
                        NavigationLink {
                            AnalysisView(videoPath: thumbnail.0)
                        } label: {
                            Image(uiImage: thumbnail.1)
                                .resizable()
                                .scaledToFit()
                                .cornerRadius(10)
                        }
                    }
                }
            }.task {
                var thumbnail = UIImage()
                for video in videos {
                    thumbnail = await generateThumbnail(for: video.savePath)
                    thumbnails.append((video.savePath, thumbnail))
                }
            }
            .navigationTitle("Videos")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    // New video button
                    Button {
                        showCamera = true
                    } label: {
                        Image(systemName: "plus")
                    }.tint(.green)
                    .fullScreenCover(isPresented: $showCamera) {
                        CameraUI(showCamera: $showCamera, showAccessError: $showAccessError)
                    }
                }
            }
        }
    }
    
}
