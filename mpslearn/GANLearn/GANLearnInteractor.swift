//
//  GANLearnInteractor.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 20.11.2022.
//

import Foundation
import AppKit

class GANLearnInteractor {
    func setupNet(model: GANLearnModel) {
        let batchSize = model.batchSize
        let gan = DCGAN(batchSize: batchSize)
        let stableLatent = (0..<gan.genInput.featuresCount()).map { _ in Float32.random(in: -1...1) }
        let dataset = ImagesDataset()

        DispatchQueue.main.async {
            model.gan = gan
            model.stableLatent = stableLatent
            model.dataset = dataset
        }
    }

    func startLearning(model: GANLearnModel) {
        guard let gan = model.gan, let dataset = model.dataset else { return }

        model.isRunning = true
        let learningSteps = model.learningSteps
        let batchSize = model.batchSize
        let stableLatent = model.stableLatent
        
        DispatchQueue.global(qos: .background)
            .async { [dataset, batchSize, gan, stableLatent] in
            epochs: for i in 0..<learningSteps {
                DispatchQueue.main.async {
                    model.progress = CGFloat(i) / CGFloat(learningSteps)
                }

                var iter = 0
                for image in dataset {
                    guard model.isRunning else { break epochs }
                    DispatchQueue.main.async { [model] in
                        model.realImage = image
                            .item(index: 0, fromBatchSized: batchSize)
                            .makeCGImage(width: 64, height: 64, balance: (1, 0.5))
                    }
                    let (imageBuffer, dLoss, gLoss) = gan.train(data: image)
                    DispatchQueue.main.async { [model] in
                        model.itemsProcessed += batchSize
                        model.discrLossHistory += dLoss
                        model.ganLossHistory += gLoss
                    }
                    if iter % 10 == 0 {
                        let stableImage = gan.generate(latent: stableLatent)
                        DispatchQueue.main.async { [model] in
                            model.runningImage = imageBuffer
                                .item(index: 0, fromBatchSized: batchSize)
                                .makeCGImage(width: 64, height: 64, balance: (1, 0.5))
                            model.stableImage = stableImage
                                .item(index: 0, fromBatchSized: batchSize)
                                .makeCGImage(width: 64, height: 64, balance: (1, 0.5))
                        }
                    }
                    iter += 1
                }

                DispatchQueue.main.async {
                    model.epochesRunned += 1
                }
            }

                if model.isRunning {
                    DispatchQueue.main.async {
                        model.isRunning = false
                    }
                }
            }
    }

    func stop(model: GANLearnModel) {
        model.isRunning = false
    }

    func setupImageDataset(folderURL: URL, model: GANLearnModel) {
        guard let gan = model.gan else { return }
        model.dataset = FolderDataset(folder: folderURL, batchSize: gan.batchSize)
    }

    func save(model: GANLearnModel) {
        guard let gan = model.gan else { return }
        _ = try? gan.save()
    }

    func load(from checkpointFolderURL: URL, model: GANLearnModel) {
        guard let gan = model.gan else { return }

        do {
            try gan.load(from: checkpointFolderURL)
        } catch {
            print(error.localizedDescription)
        }
    }

    func openCheckpointsFolder() {
        guard let checkpointsFolder = try? FileManager.default.urlForCheckpintsDirectory() else { return }
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: checkpointsFolder.path)
    }

    func updateStableLatent(model: GANLearnModel) {
        guard let gan = model.gan else { return }

        model.stableLatent = (0..<gan.genInput.featuresCount()).map { _ in Float32.random(in: -1...1) }
    }

    func generateNewImage(model: GANLearnModel) {
        guard let gan = model.gan else { return }

        updateStableLatent(model: model)
        let stableLatent = model.stableLatent
        let batchSize = model.batchSize
        DispatchQueue.global(qos: .background).async { [gan, stableLatent, batchSize] in
            let stableImage = gan.generate(latent: stableLatent)
            DispatchQueue.main.async {
                model.stableImage = stableImage
                    .item(index: 0, fromBatchSized: batchSize)
                    .makeCGImage(width: 64, height: 64, balance: (1, 0.5))
            }
        }
    }
}
