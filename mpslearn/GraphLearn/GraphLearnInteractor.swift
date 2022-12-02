//
//  GraphLearnInteractor.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 16.11.2022.
//

import Foundation

class GraphLearnInteractor {
    private let model: GraphLearnModel
    private let predictor = SequencePredictor()
    
    init(model: GraphLearnModel) {
        self.model = model
    }
    
    func startLearning(iterations: Int) {
        let data = model.realData
        DispatchQueue.global(qos: .background).async { [predictor] in
            predictor.train(data: data, iterations: iterations)
        }
    }
    
    func predict(predictNext itemsToPredict: Int) {
        model.predictedData = predictor.predict(startSequence: Array(model.realData.prefix(40)), predictNext: itemsToPredict)
    }
}
