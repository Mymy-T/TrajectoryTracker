//
//  CameraUI.swift
//  TrajectoryTracker
//
//  Created by Milo on 6/12/26.
//

import AVFoundation
import Combine
import CoreMotion
import Foundation
import SwiftUI
import simd

@Observable
class DeviceOrientationMonitor {
    private let motionManager = CMMotionManager()
    var isLandscape: Bool = false
    
    init() {
        motionManager.deviceMotionUpdateInterval = 0.2
    }
    
    func startMonitoring() {
        guard motionManager.isDeviceMotionAvailable else { return }
        
        motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, error in
            guard let self = self,
                  let gravity = motion?.gravity else { return }
            let absX = abs(gravity.x)
            let absY = abs(gravity.y)
            
            // If x-axis force dominates y-axis force, the device is horizontal
            DispatchQueue.main.async {
                self.isLandscape = (absX > 0.8 && absY < 0.2)
            }
        }
    }
    
    func stopMonitoring() {
        motionManager.stopDeviceMotionUpdates()
    }
}


struct CameraUI: View {
    
    @State private var videoPath: URL?
    @State private var camera = Camera()
    @State var didSetup = Bool()
    @State private var recording: Bool = false
    @State private var reviewMode: Bool = false
    @State private var deviceOrientationMonitor = DeviceOrientationMonitor()
    @State private var intrinsicMatrix: matrix_float3x3?
    @State private var intrinsicFormatDimensions: CMVideoDimensions?
    
    @Binding var showCamera: Bool
    @Binding var showAccessError: Bool
    
    var body: some View {
        ZStack(alignment: .bottom) {
            CameraPreview(camera: $camera)
                .task {
                    
//                    try? await Task.sleep(for: .milliseconds(300))
                    
                    if await camera.checkCameraAuthorization() {
                        didSetup = camera.setup()
                    } else {
                        // Failed to access camera
                        showAccessError = true
                        showCamera = false
                    }
                    
                    if !didSetup {
                        print("Camera setup failed.")
                        showCamera = false
                    }
                }
                .ignoresSafeArea()
            
            if !deviceOrientationMonitor.isLandscape {
                Color.black.opacity(0.75)
                    .edgesIgnoringSafeArea(.all)
                
                VStack(spacing: 20) {
                    Spacer()
                    Image(systemName: "iphone.radiowaves.left.and.right")
                        .font(.system(size: 60))
                        .foregroundColor(.white)
                        .rotationEffect(.degrees(90)) // Visual indicator turned horizontal
                    
                    Text("Please Rotate Your Device")
                        .font(.title2)
                        .bold()
                        .foregroundColor(.white)
                    
                    Text("Mount your phone horizontally on the fence or tripod to ensure accurate court tracking.")
                        .font(.body)
                        .foregroundColor(.gray)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 40)
                    Spacer()
                    Spacer()
                    Spacer()
                }
                .transition(.opacity)
            }
            
            
            
            cameraControls()
            
        }.fullScreenCover(isPresented: $reviewMode) {
            ReviewVideoView(path: videoPath, intrinsicMatrix: intrinsicMatrix, intrinsicFormatDimensions: intrinsicFormatDimensions, showCamera: $showCamera, reviewMode: $reviewMode)
        }
        .onDisappear {
            camera.session.stopRunning()
            deviceOrientationMonitor.stopMonitoring()
        }
        .onAppear {
            camera.onRecordingFinished = { filepath, intrinsics, formatDimensions in
                self.videoPath = filepath // Temporary path
                self.intrinsicMatrix = intrinsics
                self.intrinsicFormatDimensions = formatDimensions
                self.reviewMode = true
            }
            deviceOrientationMonitor.startMonitoring()
        }
    }
    
    
    private func cameraControls() -> some View {
        return HStack(spacing: 50) {
            
            // Dismiss Button
            Button {
                showCamera = false
            } label: {
                Image(systemName: "xmark")
                    .tint(.red)
            }.frame(width: 65, height: 65)
                .glassEffect()
            
            // Record button
            Button {
                if !recording {
                    camera.startRecordingVideo()
                    recording = true
                } else {
                    recording = false
                    videoPath = camera.tempFilePath
                    camera.stopRecording()
                }
            } label: {
                ZStack {
                    Circle()
                        .stroke(.white, lineWidth: 2)
                        .frame(width: 70)
                    
                    Circle()
                        .fill(.red)
                        .frame(width: recording ? 40 : 60)
                }
            }
            
            
        }
        
    }
    
}
