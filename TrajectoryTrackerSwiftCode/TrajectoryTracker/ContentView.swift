//
//  ContentView.swift
//  TrajectoryTracker
//
//  Created by Milo on 5/28/26.
//

import AVFoundation
import SwiftData
import SwiftUI

// TODO: Refresh content view every time a video is added or deleted

struct ContentView: View {
    
    @Query var videos: [Video]
    @Environment(\.modelContext) var modelContext
    
    @State private var showCamera: Bool = false
    @State private var showAccessError: Bool = false // MARK: TODO
    @State private var thumbnails: [(URL, UIImage)] = []
    
    @State private var presentingDeleteAlert = false
    @State private var videoURLToDelete: URL?
    
    let columns = [GridItem(.fixed(400))]

    var body: some View {
        NavigationStack {
            ScrollView {
                LazyVGrid(columns: columns) {
                    ForEach(thumbnails, id: \.self.0) { thumbnail in
                        // thumbnail is of the form (video url, ui image thumbnail)
                        NavigationLink {
                            AnalysisView(videoPath: thumbnail.0)
                        } label: {
                            Image(uiImage: thumbnail.1)
                                .resizable()
                                .scaledToFit()
                                .cornerRadius(10)
                                .onLongPressGesture {
                                    presentingDeleteAlert = true
                                    videoURLToDelete = thumbnail.0
                                }
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
            .alert("Camera Permission Denied", isPresented: $showAccessError) {
                Button("Ok") { }
            } message: {
                Text("Please go to settings and allow TrajectoryTracker to access the camera.")
            }
            .alert("Delete Video?", isPresented: $presentingDeleteAlert) {
                Button("Cancel", role: .cancel) {
                    presentingDeleteAlert = false
                    videoURLToDelete = nil
                }
                
                Button("Delete", role: .destructive) {
                    deleteVideo(from: videoURLToDelete!)
                    videoURLToDelete = nil
                    presentingDeleteAlert = false
                }
            }
            
        }
    }
    
    private func deleteVideo(from url: URL) {
        deleteVideoFromURL(url)
        guard let videoToDelete = videos.first(where: { $0.savePath == url }) else {
            print("Error in deleting video")
            return
        }
        modelContext.delete(videoToDelete)
    }
    
}
