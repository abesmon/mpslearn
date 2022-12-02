//
//  GraphLearnModel.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 16.11.2022.
//

import SwiftUI

class GraphLearnModel: ObservableObject {
    let realData: [CGPoint]
    @Published var predictedData: [CGPoint] = []
    
    init(realData: [CGPoint]) {
        self.realData = realData
    }
    
    static func sinInit() -> GraphLearnModel {
        var data = [CGPoint]()
        for i in stride(from: 0, through: 3.14 * 4, by: 0.1) {
            let x = CGFloat(i)
            data.append(.init(x: x, y: sin(x)))
        }
        return GraphLearnModel(realData: data)
    }
}
