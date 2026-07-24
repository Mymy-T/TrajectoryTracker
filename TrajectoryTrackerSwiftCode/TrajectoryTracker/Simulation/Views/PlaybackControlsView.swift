//
//  PlaybackControlsView.swift
//  TrajectoryTracker
//
//  Created by Milo on 7/22/26.
//

import SwiftUI

struct PlaybackControlsView: View {
    @Bindable var viewModel: SimulatorViewModel

    var body: some View {
        VStack(spacing: 8) {
            Slider(
                value: Binding(
                    get: { viewModel.playbackTime },
                    set: { viewModel.scrub(to: $0) }
                ),
                in: 0...max(viewModel.duration, 0.01)
            )
            .disabled(viewModel.trajectory.isEmpty)

            HStack {
                Button {
                    viewModel.reset()
                } label: {
                    Image(systemName: "backward.end.fill")
                }

                Button {
                    viewModel.isPlaying ? viewModel.pause() : viewModel.play()
                } label: {
                    Image(systemName: viewModel.isPlaying ? "pause.fill" : "play.fill")
                }
                .disabled(viewModel.trajectory.isEmpty)

                Spacer()

                Text("\(viewModel.playbackTime, specifier: "%.2f")s / \(viewModel.duration, specifier: "%.2f")s")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
    }
}
