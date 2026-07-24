//
//  AnalysisView.swift
//  TrajectoryTracker
//
//  Created by Milo on 6/14/26.
//

import SwiftUI

enum AnalysisTab: String, CaseIterable, Identifiable {
    case video = "Video"
    case simulator = "Simulator"

    var id: Self { self }
}

struct AnalysisView: View {

    let videoPath: URL

    @State private var selectedTab: AnalysisTab = .video

    var body: some View {
        VStack {
            Picker("Tab", selection: $selectedTab) {
                ForEach(AnalysisTab.allCases) { tab in
                    Text(tab.rawValue).tag(tab)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)

            switch selectedTab {
            case .video:
                VideoAnalysisView(videoPath: videoPath)
            case .simulator:
                SimulatorView()
            }
        }
        .navigationTitle("Analysis")
        .navigationBarTitleDisplayMode(.inline)
    }
}
