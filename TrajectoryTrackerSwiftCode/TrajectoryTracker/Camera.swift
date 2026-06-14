//
//  Camera.swift
//  TrajectoryTracker
//
//  Created by Milo on 6/12/26.
//

import AVFoundation
import Foundation


class Camera: NSObject, AVCaptureFileOutputRecordingDelegate {
    var session = AVCaptureSession()
    var preview = AVCaptureVideoPreviewLayer()
    var output = AVCaptureMovieFileOutput()
    
    var tempFilePath: URL? = nil
    var onRecordingFinished: ((URL?) -> Void)?
    
    
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
        
        session.addInput(deviceInput)
        session.addOutput(output)
        configure120FPS(for: device)
        session.commitConfiguration()
        
        configureCameraOrientation(for: output)
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
    /// Rotate camera captured output to landscape mode
    private func configureCameraOrientation(for output: AVCaptureMovieFileOutput) {
        if let connection = output.connection(with: .video) {
            connection.videoRotationAngle = 0
        }
    }
    
    
    
    // MARK: startRecordingVideo
    /// Start movie capture.
    func startRecordingVideo() {
        guard !output.isRecording else {
            return
        }
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
    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL, from connections: [AVCaptureConnection], error: Error?) {
        if let error = error {
            print("Recording failed with error: \(error.localizedDescription)")
            return
        }
        
        DispatchQueue.main.async { [weak self] in
            self?.onRecordingFinished?(self?.tempFilePath)
        }
    }
    
    
}
