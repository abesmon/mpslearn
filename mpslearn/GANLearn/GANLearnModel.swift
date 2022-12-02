//
//  GANLearnModel.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 20.11.2022.
//

import SwiftUI

class GANLearnModel: ObservableObject {
    @Published var realImage: CGImage?
    @Published var runningImage: CGImage?
    @Published var stableImage: CGImage?

    @Published var isRunning = false
    @Published var progress: CGFloat = 0
    @Published var epochesRunned: Int = 0
    @Published var itemsProcessed: Int = 0

    @Published var ganLossHistory: [Float32] = []
    @Published var discrLossHistory: [Float32] = []

    @Published var gan: DCGAN?
    @Published var dataset: Dataset?
    @Published var stableLatent: [Float32] = []

    @Published var batchSize: Int = 16
    @Published var learningSteps: Int = 1000
}
