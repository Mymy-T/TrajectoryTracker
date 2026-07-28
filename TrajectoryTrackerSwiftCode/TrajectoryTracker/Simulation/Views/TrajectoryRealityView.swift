//
//  TrajectoryRealityView.swift
//  TrajectoryTracker
//
//  Created by Milo on 7/22/26.
//

import Combine
import RealityKit
import SwiftUI

enum CameraMode {
    case loading, panXZ, panY, orbit, zoom
}

struct TrajectoryRealityView: View {
    @Bindable var viewModel: SimulatorViewModel
        
    @State private var courtEntity = CourtEntity.make()
    @State private var ballEntity = TrajectoryPathVisualizer.makeBallEntity()
    @State private var pathContainer = Entity()
    @State private var lastRenderedSampleCount = 0
    
    // Camera stuff
    static let originCameraPosition = SIMD3<Float>(-CourtEntity.courtLength / 2, 2, 0)
    
    
    @State private var basePosition: SIMD3<Float> = originCameraPosition
    @State private var savedPosition: SIMD3<Float> = originCameraPosition
    @State private var pivotRotation = simd_quatf(angle: -Float.pi / 2, axis: SIMD3<Float>(0, 1, 0))
    @State private var yaw: Float = -Float.pi / 2   // Left/Right rotation
    @State private var pitch: Float = 0 // Up/Down rotation
    @State private var zoomDistance: Float = 5.0
    @State private var savedZoomDistance: Float = 5.0
    
    @State private var baseEntity: Entity? // Responsible for translational motion; its axes always remain aligned with world coordinate system
    @State private var pivotEntity: Entity? // Responsible for rotational motion (rotating the camera)
    @State private var cameraEntity: PerspectiveCamera? // Responsible for being the actual camera and for zooming
    
    let metersPerPixel = Float(0.005) // Converts from drag translation in pixels to meters
    let radiansPerPixel = Float(0.005) // Converts from drag translation in pixels to radians
    @State private var cameraMode = CameraMode.loading // Change in onAppear to refresh dragGesture computed variable
    // MARK: Gestures
    var dragGesture: AnyGesture<DragGesture.Value>? {
        if cameraMode == .panXZ {
            let gesture = DragGesture(minimumDistance: 0)
                .onChanged { value in
                    let dx = value.translation.width
                    let dy = value.translation.height
                    guard let cameraMatrix = pivotEntity?.transform.matrix else { return }
                    var tempCol = cameraMatrix[0] // Unit vector in camera's right direction
                    let iHat = normalize(SIMD3<Float>(tempCol.x, 0, tempCol.z))
                    tempCol = cameraMatrix[2] // Unit vector in camera's forwards direction
                    let kHat = normalize(SIMD3<Float>(tempCol.x, 0, tempCol.z))
                    let cameraRightDisplacement = -Float(dx) * metersPerPixel * iHat
                    let cameraFrontalDisplacment = -Float(dy) * metersPerPixel * kHat
                    basePosition = savedPosition + cameraRightDisplacement + cameraFrontalDisplacment
                }
                .onEnded { _ in
                    savedPosition = basePosition
                }
            return AnyGesture(gesture)
        } else if cameraMode == .panY {
            let gesture = DragGesture(minimumDistance: 0)
                .onChanged { value in
                    let dy = value.translation.height
                    guard let cameraMatrix = pivotEntity?.transform.matrix else { return }
                    let tempCol = cameraMatrix[1] // Unit vector in camera's upward direction
                    let jHat = normalize(SIMD3<Float>(tempCol.x, tempCol.y, tempCol.z))
                    let cameraUpwardsDisplacement = -Float(dy) * metersPerPixel * jHat
                    basePosition = savedPosition + cameraUpwardsDisplacement
                }
                .onEnded { _ in
                    savedPosition = basePosition
                }
            return AnyGesture(gesture)
        }
        else if cameraMode == .orbit {
            let gesture = DragGesture(minimumDistance: 0)
                .onChanged { value in
                    let dYaw = -Float(value.translation.width) * radiansPerPixel
                    let dPitch = -Float(value.translation.height) * radiansPerPixel

                    let finalYaw = yaw + dYaw
                    let finalPitch = max(-Float.pi/2 + 0.1, min(pitch + dPitch, Float.pi/2 - 0.1)) // Clamp the pitch so the camera doesn't flip completely upside down buffer of 0.1

                    // Create quaternions for each axis and multiply
                    let yawRotation = simd_quatf(angle: finalYaw, axis: SIMD3<Float>(0, 1, 0)) // Rotation about y axis
                    let pitchRotation = simd_quatf(angle: finalPitch, axis: SIMD3<Float>(1, 0, 0)) // Rotation about x axis

                    pivotRotation = yawRotation * pitchRotation
                }
                .onEnded { value in
                    let dYaw = -Float(value.translation.width) * radiansPerPixel
                    let dPitch = -Float(value.translation.height) * radiansPerPixel

                    let finalYaw = yaw + dYaw
                    let finalPitch = max(-Float.pi/2 + 0.1, min(pitch + dPitch, Float.pi/2 - 0.1)) // Clamp the pitch so the camera doesn't flip completely upside down buffer of 0.1
                    
                    yaw = finalYaw
                    pitch = finalPitch
                }
            return AnyGesture(gesture)
        }
        else {
            return nil
        }
    }
    var zoomGesture: AnyGesture<MagnifyGesture.Value>? {
        if cameraMode == .zoom {
            let gesture = MagnifyGesture()
                .onChanged { value in
                    let newDistance = savedZoomDistance / Float(value.magnification) // Invert scale since larger finger pinch camera should come closer
                    zoomDistance = max(1.5, min(newDistance, 20.0)) // Clamp the distance so user can't zoom inside the floor or fly too far away
                }
                .onEnded { _ in
                    savedZoomDistance = zoomDistance
                }
            return AnyGesture(gesture)
        } else {
            return nil
        }
    }
    
    private let playbackClock = Timer.publish(every: 1.0 / 60.0, on: .main, in: .common).autoconnect()
    
    var body: some View {
        
        VStack {
            
            // MARK: Camera Buttons
            HStack {
                Button {
                    cameraMode = .panXZ
                } label: {
                    Image(systemName: "arrow.up.and.down.and.arrow.left.and.right")
                }
                .buttonStyle(.borderedProminent)
                .tint(cameraMode == .panXZ ? .blue : .gray)
                
                Button {
                    cameraMode = .panY
                } label: {
                    Image(systemName: "arrow.up.and.down")
                }
                .buttonStyle(.borderedProminent)
                .tint(cameraMode == .panY ? .blue : .gray)
                
                
                Button {
                    cameraMode = .orbit
                } label: {
                    Image(systemName: "rotate.3d")
                }
                .buttonStyle(.borderedProminent)
                .tint(cameraMode == .orbit ? .blue : .gray)
                
                Button {
                    cameraMode = .zoom
                } label: {
                    Image(systemName: "arrow.up.left.and.down.right.magnifyingglass")
                }
                .buttonStyle(.borderedProminent)
                .tint(cameraMode == .zoom ? .blue : .gray)
                
                Button {
                    basePosition = TrajectoryRealityView.originCameraPosition
                    savedPosition = basePosition
                    pivotRotation = simd_quatf(angle: -Float.pi / 2, axis: SIMD3<Float>(0, 1, 0))
                    yaw = -Float.pi / 2
                    pitch = 0
                    zoomDistance = 5.0
                    savedZoomDistance = zoomDistance
                } label: {
                    Image(systemName: "house")
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
                
            }.padding()
            
            // MARK: RealityView
            RealityView { content in
                content.add(courtEntity)
                content.add(pathContainer)
                content.add(ballEntity)
                
                // Setup camera rig; imagine a base on wheels with a vertical lifting pole, with a pivot mounted on the base, and a camera moounted on the pivot
                let base = Entity()
                let pivot = Entity()
                let camera = PerspectiveCamera()
                
                base.addChild(pivot)
                pivot.addChild(camera)
                content.add(base)
                
                base.position = basePosition
                pivot.orientation = pivotRotation
                camera.position = SIMD3<Float>(0, 0, zoomDistance)
                
                self.baseEntity = base
                self.pivotEntity = pivot
                self.cameraEntity = camera
                
            } update: { _ in
                if viewModel.trajectory.count != lastRenderedSampleCount { // Re-project path if simulation params changed
                    TrajectoryPathVisualizer.projectPath(in: pathContainer, from: viewModel.trajectory)
                    lastRenderedSampleCount = viewModel.trajectory.count
                }
                if let state = viewModel.state(at: viewModel.playbackTime) { // Update ball position when scrubbing
                    ballEntity.position = state.position.asFloat
                }
                
                baseEntity?.position = basePosition
                pivotEntity?.orientation = pivotRotation
                cameraEntity?.position = SIMD3<Float>(0, 0, zoomDistance) // Local coord system has +x right, +y up, and +z forward
            }
            .realityViewCameraControls(.none)
            .onReceive(playbackClock) { _ in
                viewModel.advancePlayback(by: 1.0 / 60.0)
            }
            .gesture(dragGesture)
            .gesture(zoomGesture)
            .onAppear {
                cameraMode = .panXZ
            }
        }
    }
    
}
