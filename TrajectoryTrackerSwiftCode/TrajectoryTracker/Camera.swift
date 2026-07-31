//
//  Camera.swift
//  TrajectoryTracker
//
//  Created by Milo on 6/12/26.
//

import AVFoundation
import Foundation
import simd


class Camera: NSObject, AVCaptureFileOutputRecordingDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    var session = AVCaptureSession()
    var preview = AVCaptureVideoPreviewLayer()
    var output = AVCaptureMovieFileOutput()
    var intrinsicsOutput = AVCaptureVideoDataOutput()
    private let intrinsicsQueue = DispatchQueue(label: "camera.intrinsics")
    private var device: AVCaptureDevice?

    // Continuously updated while the session runs; only meaningful as a snapshot once focus is locked.
    private(set) var currentIntrinsicMatrix: matrix_float3x3?
    private var recordingIntrinsicMatrix: matrix_float3x3?
    private var recordingFormatDimensions: CMVideoDimensions?

    var tempFilePath: URL? = nil
    var onRecordingFinished: ((URL?, matrix_float3x3?, CMVideoDimensions?) -> Void)?

    
    func checkCameraAuthorization() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return true
        case .notDetermined:
            let status = await AVCaptureDevice.requestAccess(for: .video)
            return status
        case .denied:
            return false
        case .restricted:
            return false
        @unknown default:
            return false
        }
    }
    
    // MARK: setup()
    /// Setup the capture session.
    func setup() -> Bool {
        session.beginConfiguration()
        
        session.sessionPreset = .inputPriority
        
        guard let device = AVCaptureDevice.default(for: .video) else {
            print("Unable to obtain device")
            return false
        }
        
        guard let deviceInput = try? AVCaptureDeviceInput(device: device) else {
            print("Unable to obtain device input")
            return false
        }
        
        guard session.canAddInput(deviceInput) else {
            print("Unable to add device input to session")
            return false
        }
        
        guard session.canAddOutput(output) else {
            print("Unable to add output to session")
            return false
        }

        guard session.canAddOutput(intrinsicsOutput) else {
            print("Unable to add intrinsics output to session")
            return false
        }

        self.device = device
        intrinsicsOutput.setSampleBufferDelegate(self, queue: intrinsicsQueue)
        intrinsicsOutput.alwaysDiscardsLateVideoFrames = true

        session.addInput(deviceInput)
        session.addOutput(output)
        session.addOutput(intrinsicsOutput)
        configure120FPS(for: device)
        session.commitConfiguration()

        configureCameraOrientation(for: output)
        configureIntrinsicsDelivery(for: intrinsicsOutput)
        DispatchQueue.global().async {
            self.session.startRunning()
        }

        return true

    }
    
    // MARK: configure120FPS
    /// Prepares camera to record in 120 fps.
    private func configure120FPS(for device: AVCaptureDevice) {
        let bestFormat = device.formats
            .filter { format in
                // Keep only formats that support 120 FPS
                format.videoSupportedFrameRateRanges.contains { range in
                    range.minFrameRate <= 120.0 && range.maxFrameRate >= 120.0
                }
            }
            .max { formatA, formatB in
                let dimensionsA = CMVideoFormatDescriptionGetDimensions(formatA.formatDescription)
                let dimensionsB = CMVideoFormatDescriptionGetDimensions(formatB.formatDescription)
                // Find the larger resolution (width * height)
                return (dimensionsA.width * dimensionsA.height) < (dimensionsB.width * dimensionsB.height)
            }
        
        guard let format = bestFormat else {
            print("120 FPS format not supported on this device.")
            return
        }
        
        do {
            try device.lockForConfiguration()
            device.activeFormat = format
            
            let frameDuration = CMTime(value: 1, timescale: 120)
            device.activeVideoMinFrameDuration = frameDuration
            device.activeVideoMaxFrameDuration = frameDuration
            
            device.unlockForConfiguration()
        } catch {
            print("Failed to lock device for 120fps setup: \(error)")
        }
    }
    
    // MARK: configureCameraOrientation
    /// Rotate camera captured output to landscape mode, and disable stabilization so it can't introduce cropping/scaling that would perturb the intrinsic matrix mid-recording.
    private func configureCameraOrientation(for output: AVCaptureMovieFileOutput) {
        if let connection = output.connection(with: .video) {
            connection.videoRotationAngle = 0
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .off
            }
        }
    }

    // MARK: configureIntrinsicsDelivery
    /// Enables per-frame camera intrinsic matrix delivery on the intrinsics tap output, matching the movie output's orientation/stabilization so the two stay consistent.
    private func configureIntrinsicsDelivery(for output: AVCaptureVideoDataOutput) {
        guard let connection = output.connection(with: .video) else { return }
        connection.videoRotationAngle = 0
        if connection.isVideoStabilizationSupported {
            connection.preferredVideoStabilizationMode = .off
        }
        if connection.isCameraIntrinsicMatrixDeliverySupported {
            connection.isCameraIntrinsicMatrixDeliveryEnabled = true
        }
    }

    // MARK: lockFocus
    /// Freezes the lens at its current position so the intrinsic matrix stays constant for the duration of a recording. Continuous autofocus is left running beforehand (during preview) so the lens has already settled on the framed scene by the time this is called.
    private func lockFocus() {
        guard let device, device.isFocusModeSupported(.locked) else { return }
        do {
            try device.lockForConfiguration()
            device.setFocusModeLocked(lensPosition: AVCaptureDevice.currentLensPosition, completionHandler: nil)
            device.unlockForConfiguration()
        } catch {
            print("Failed to lock focus: \(error)")
        }
    }

    // MARK: startRecordingVideo
    /// Start movie capture.
    func startRecordingVideo() {
        guard !output.isRecording else {
            return
        }
        lockFocus()
        recordingIntrinsicMatrix = currentIntrinsicMatrix
        recordingFormatDimensions = device.map { CMVideoFormatDescriptionGetDimensions($0.activeFormat.formatDescription) }
        tempFilePath = URL.temporaryDirectory.appending(component: UUID().uuidString).appendingPathExtension(for: .quickTimeMovie)
        output.startRecording(to: tempFilePath!, recordingDelegate: self)
    }
    
    // MARK: stopRecording
    /// Stops movie recording.
    func stopRecording() {
        if output.isRecording {
            output.stopRecording()
        }
    }
    
    // MARK: fileOutput
    // Conformance to AVCaptureFileOutputRecordingDelegate
    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL, from connections: [AVCaptureConnection], error: Error?) {
        if let error = error {
            print("Recording failed with error: \(error.localizedDescription)")
            return
        }
        
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.onRecordingFinished?(self.tempFilePath, self.recordingIntrinsicMatrix, self.recordingFormatDimensions)
        }
    }
    
    // MARK: captureOutput
    // Conformance to AVCaptureVideoDataOutputSampleBufferDelegate; read the intrinsics from the buffer
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let data = CMGetAttachment(sampleBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil) as? Data else {
            return
        }
        currentIntrinsicMatrix = data.withUnsafeBytes { $0.load(as: matrix_float3x3.self) }
    }


}
