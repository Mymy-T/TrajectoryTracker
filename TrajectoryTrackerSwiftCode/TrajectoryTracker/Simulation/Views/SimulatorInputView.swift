//
//  SimulatorInputView.swift
//  TrajectoryTracker
//
//  Created by Milo on 7/22/26.
//

import SwiftUI

struct SimulatorInputView: View {
    @Bindable var viewModel: SimulatorViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Launch") {
                    VStack(alignment: .leading) {
                        Text("Speed: \(viewModel.speed, specifier: "%.1f") m/s")
                        Slider(value: $viewModel.speed, in: 1...50)
                    }
                    VStack(alignment: .leading) {
                        Text("Launch Angle: \(viewModel.launchAngleDegrees, specifier: "%.0f")\u{00B0}")
                        Slider(value: $viewModel.launchAngleDegrees, in: -45...45)
                    }
                    VStack(alignment: .leading) {
                        Text("Azimuth: \(viewModel.azimuthDegrees, specifier: "%.0f")\u{00B0}")
                        Slider(value: $viewModel.azimuthDegrees, in: -90...90)
                    }
                }

                Section("Spin") {
                    Picker("Mode", selection: $viewModel.spinInputMode) {
                        ForEach(SpinInputMode.allCases) { mode in
                            Text(mode.rawValue)
                                .tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)

                    switch viewModel.spinInputMode {
                    case .simple:
                        Picker("Spin Type", selection: $viewModel.spinType) {
                            ForEach(SpinType.allCases) { type in
                                Text(type.displayName)
                                    .tag(type)
                            }
                        }
                        VStack(alignment: .leading) {
                            Text("Spin Rate: \(Int(viewModel.spinRPM)) rpm")
                            Slider(value: $viewModel.spinRPM, in: 0...5000)
                        }

                    case .advanced:
                        Stepper("Axis X: \(viewModel.spinAxis.x, specifier: "%.2f")", value: $viewModel.spinAxis.x, in: -1...1, step: 0.1)
                        Stepper("Axis Y: \(viewModel.spinAxis.y, specifier: "%.2f")", value: $viewModel.spinAxis.y, in: -1...1, step: 0.1)
                        Stepper("Axis Z: \(viewModel.spinAxis.z, specifier: "%.2f")", value: $viewModel.spinAxis.z, in: -1...1, step: 0.1)
                        VStack(alignment: .leading) {
                            Text("Spin Rate: \(Int(viewModel.spinRPM)) rpm")
                            Slider(value: $viewModel.spinRPM, in: 0...5000)
                        }
                    }
                }
            }
            .navigationTitle("Simulation Parameters")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Run") {
                        viewModel.runSimulation()
                        dismiss()
                    }
                }
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
        }
    }
}
