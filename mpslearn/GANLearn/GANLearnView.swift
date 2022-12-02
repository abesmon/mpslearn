//
//  GANLearnView.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 20.11.2022.
//

import SwiftUI
import Charts

struct GANLearnView: View {
    @ObservedObject private var model: GANLearnModel
    private let interactor: GANLearnInteractor?
    
    init(model: GANLearnModel, interactor: GANLearnInteractor?) {
        self.model = model
        self.interactor = interactor
    }
    
    var body: some View {
        if model.gan == nil {
            Button("Setup the network") {
                interactor?.setupNet(model: model)
            }
        } else {
            VStack {
                Text("It's really takes time to learn, but it would")
                    .font(.callout)

                Text("epoches runned: \(model.epochesRunned)")

                Text("items processed: \(model.itemsProcessed)")

                Chart {
                    ForEach(0..<model.discrLossHistory.count, id: \.self) { idx in
                        let point = model.discrLossHistory[idx]
                        LineMark(
                            x: .value("x", Double(idx)),
                            y: .value("y", Double(point))
                        )
                        .foregroundStyle(by: .value("lossType", "dLoss"))
                    }

                    ForEach(0..<model.ganLossHistory.count, id: \.self) { idx in
                        let point = model.ganLossHistory[idx]
                        LineMark(
                            x: .value("x", Double(idx)),
                            y: .value("y", Double(point))
                        )
                        .foregroundStyle(by: .value("lossType", "gLoss"))
                    }
                }

                HStack {
                    URLDropTile(color: .blue,
                                text: "Drag dataset folder here",
                                iconSystemName: "square.grid.3x1.folder.fill.badge.plus") { url in
                        interactor?.setupImageDataset(folderURL: url, model: model)
                    }


                    URLDropTile(color: .orange,
                                text: "Drop checkpoint folder here",
                                iconSystemName: "flag") { url in
                        interactor?.load(from: url, model: model)
                    }
                }


                HStack {
                    OptionalImage(cgImage: model.realImage,
                                  width: 64, height: 64)

                    OptionalImage(cgImage: model.runningImage,
                                  width: 64, height: 64)

                    OptionalImage(
                        cgImage: model.stableImage,
                        width: 64, height: 64
                    )
                    .onTapGesture {
                        interactor?.generateNewImage(model: model)
                    }
                }
                VStack {
                    if !model.isRunning {
                        Button("Start Learning") {
                            interactor?.startLearning(model: model)
                        }
                    } else {
                        Button("Stop") {
                            interactor?.stop(model: model)
                        }

                        ProgressView(value: model.progress, total: 1)
                    }
                    HStack {
                        Button("Save model") {
                            interactor?.save(model: model)
                        }

                        Button {
                            interactor?.openCheckpointsFolder()
                        } label: {
                            Image(systemName: "folder")
                        }

                    }
                }
            }
        }
    }
}

struct GANLearnView_Previews: PreviewProvider {
    static var previews: some View {
        GANLearnView(model: GANLearnModel(), interactor: nil)
    }
}
