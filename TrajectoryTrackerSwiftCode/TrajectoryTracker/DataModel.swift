//
//  DataModel.swift
//  TrajectoryTracker
//
//  Created by Milo on 6/12/26.
//

import Foundation
import SwiftData

@Model
class Video: Identifiable {
    var id = UUID()
    var filename: String
    var savePath: URL {
        URL.documentsDirectory.appendingPathComponent(self.filename)
    }
    
    init(filename: String) {
        self.filename = filename
    }
    
}
