//
//  SimulatorView.swift
//  TrajectoryTracker
//
//  Created by Milo on 7/22/26.
//

import SwiftUI

struct SimulatorView: View {
    @State private var viewModel = SimulatorViewModel()
    @State private var showingInputSheet = false

    var body: some View {
        VStack(spacing: 0) {
            TrajectoryRealityView(viewModel: viewModel)
            //CameraButtonsView(viewModel: viewModel)
            PlaybackControlsView(viewModel: viewModel)
        }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    showingInputSheet = true
                } label: {
                    Image(systemName: "slider.horizontal.3")
                }
            }
        }
        .sheet(isPresented: $showingInputSheet) {
            SimulatorInputView(viewModel: viewModel)
        }
    }
}
