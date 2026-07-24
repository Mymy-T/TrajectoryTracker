//
//  TrajectoryRealityView.swift
//  TrajectoryTracker
//
//  Created by Milo on 7/22/26.
//

import Combine
import RealityKit
import SwiftUI

struct TrajectoryRealityView: View {
    @Bindable var viewModel: SimulatorViewModel

    @State private var ballEntity = TrajectoryPathVisualizer.makeBallEntity()
    @State private var pathContainer = Entity()
    @State private var lastRenderedSampleCount = 0

    private let playbackClock = Timer.publish(every: 1.0 / 60.0, on: .main, in: .common).autoconnect()

    var body: some View {
        RealityView { content in
            content.add(CourtEntity.make())
            content.add(pathContainer)
            content.add(ballEntity)
        } update: { _ in
            if viewModel.trajectory.count != lastRenderedSampleCount {
                TrajectoryPathVisualizer.rebuildPath(in: pathContainer, from: viewModel.trajectory)
                lastRenderedSampleCount = viewModel.trajectory.count
            }
            if let state = viewModel.state(at: viewModel.playbackTime) {
                ballEntity.position = state.position.asFloat
            }
        }
        .realityViewCameraControls(.orbit)
        .onReceive(playbackClock) { _ in
            viewModel.advancePlayback(by: 1.0 / 60.0)
        }
    }
}
