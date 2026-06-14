//
//  TrajectoryTrackerApp.swift
//  TrajectoryTracker
//
//  Created by Milo on 5/28/26.
//

import SwiftData
import SwiftUI

@main
struct TrajectoryTrackerApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }.modelContainer(for: [Video.self])
    }
}
