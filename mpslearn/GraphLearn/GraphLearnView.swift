//
//  GraphLearnView.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 16.11.2022.
//

import SwiftUI
import Charts

struct GraphLearnView: View {
    private let interactor: GraphLearnInteractor?
    @ObservedObject private var model: GraphLearnModel
    
    @State private var trainIterations: Int = 1000
    @State private var itemsToPredict: Int = 1000
    
    init(interactor: GraphLearnInteractor?, model: GraphLearnModel) {
        self.interactor = interactor
        self.model = model
    }
     
    var body: some View {
        VStack {
            Chart {
                ForEach(0..<model.realData.count, id: \.self) { idx in
                    let model = model.realData[idx]
                    PointMark(
                        x: .value("x", Double(model.x)),
                        y: .value("y", Double(model.y))
                    )
                    .foregroundStyle(by: .value("Data", "Real"))
                }
                
                ForEach(0..<model.predictedData.count, id: \.self) { idx in
                    let model = model.predictedData[idx]
                    PointMark(
                        x: .value("x", Double(model.x)),
                        y: .value("y", Double(model.y))
                    )
                    .foregroundStyle(by: .value("Data", "Predicted"))
                }
            }
            
            HStack {
                VStack {
                    Stepper(
                        value: $trainIterations,
                        step: 1,
                        label: {
                            TextField(
                                value: $trainIterations,
                                format: .number,
                                label: {
                                    Text("Train iterations")
                                }
                            )
                        }
                    )
                    
                    Button("Start Learning") {
                        interactor?.startLearning(iterations: trainIterations)
                    }
                }
                
                VStack {
                    Stepper(
                        value: $itemsToPredict,
                        step: 1,
                        label: {
                            TextField(
                                value: $itemsToPredict,
                                format: .number,
                                label: {
                                    Text("Items to predict")
                                }
                            )
                        }
                    )
                    
                    Button("predict") {
                        interactor?.predict(predictNext: itemsToPredict)
                    }
                }
                
                Spacer()
            }
            

            
        }
    }
}

struct GraphLearnView_Preview: PreviewProvider {
    static var previews: some View {
        GraphLearnView(interactor: nil, model: .sinInit())
    }
}
