//
//  DCGAN.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 27.11.2022.
//

import Foundation
import MetalPerformanceShadersGraph

// Learn more at https://www.tensorflow.org/tutorials/generative/dcgan

class DCGAN {
    typealias Seq = [Float32]

    let batchSize: Int
    let latentLength: Int
    private let mtlDevice = MTLCreateSystemDefaultDevice()!
    private let graph = MPSGraph()
    
    let genInput: MPSGraphTensor
    private let imageGenOut: MPSGraphTensor
    private let noiseForOnes: MPSGraphTensor
    private let noiseForZeros: MPSGraphTensor
    private let discInput: MPSGraphTensor

    private let gLoss: MPSGraphTensor
    private let genOptOperations: [MPSGraphOperation]
    
    private let dLoss: MPSGraphTensor
    private let discOptOperations: [MPSGraphOperation]

    private let genTrainableVariables: [String: VariablesPair]
    private let discrTrainableVariables: [String: VariablesPair]
    
    init(batchSize: Int = 1, latentLength: Int = 1000) {
        self.batchSize = batchSize
        self.latentLength = latentLength
        
        let lr: Double = 2e-4
        let beta1: Double = 0.5
        // GEN
        genInput = graph.placeholder(shape: [batchSize as NSNumber, 1, 1, latentLength as NSNumber], name: "genInput")
        
        let generator = graph.makeGen(input: genInput)
        genTrainableVariables = generator.vars
        imageGenOut = generator.out
        
        // Discr
        discInput = graph.placeholder(shape: imageGenOut.shape!, name: nil)
        let discrReals = graph.makeDisc(input: discInput)
        discrTrainableVariables = discrReals.vars
        let discrFakes = graph.makeDisc(input: imageGenOut,
                                        vars: discrReals.vars)

        let ones = graph.constant(1, shape: discrReals.out.shape!, dataType: discrReals.out.dataType)
        let zeros = graph.constant(0, shape: discrFakes.out.shape!, dataType: discrFakes.out.dataType)

        noiseForOnes = graph.placeholder(shape: ones.shape!, name: nil)
        noiseForZeros = graph.placeholder(shape: zeros.shape!, name: nil)

        let lossWReals = graph.BCELoss(input: discrReals.out, target: graph.subtraction(ones, noiseForOnes, name: nil))
        let lossWFakes = graph.BCELoss(input: discrFakes.out, target: graph.addition(zeros, noiseForZeros, name: nil))
        dLoss = graph.addition(lossWReals, lossWFakes, name: nil)
        
        let discGradients = graph.gradients(of: dLoss, with: discrReals.vars.trainableVariables, name: nil)
        discOptOperations = graph.adamDynamicOptimiseOperations(gradients: discGradients, lr: lr, beta1: beta1)
        
        // Generator loss
        gLoss = graph.BCELoss(input: discrFakes.out, target: ones)

        let genVars = generator.vars.trainableVariables
        let discrRealsVariables = discrReals.vars.trainableVariables

        let genGradients = graph.gradients(of: gLoss, with: genVars + discrRealsVariables, name: nil)
        let genVarsGradients = genGradients.filter { gr in genVars.contains { genVar in gr.key === genVar } }
        genOptOperations = graph.adamDynamicOptimiseOperations(gradients: genVarsGradients, lr: lr, beta1: beta1)
    }
    
    func train(data: Seq) -> (imageData: Seq, dLoss: Seq, gLoss: Seq) {
        autoreleasepool {
            // train Discr
            let genInputData = MPSGraphTensorData.withRandomValues(in: -1...1, device: mtlDevice, shape: genInput.shape!)
            let res = graph.run(
                feeds: [
                    noiseForOnes: MPSGraphTensorData.withRandomValues(in: 0...0.1, device: mtlDevice, shape: noiseForOnes.shape!),
                    noiseForZeros: MPSGraphTensorData.withRandomValues(in: 0...0.1, device: mtlDevice, shape: noiseForZeros.shape!),
                    genInput: genInputData,
                    discInput: data.asMPSGraphTensorData(on: mtlDevice, shape: discInput.shape!)!
                ],
                targetTensors: [dLoss, gLoss, imageGenOut],
                targetOperations: discOptOperations + genOptOperations
            )

            let dLoss: Seq? = res[dLoss]?.exportData()
            let gLoss: Seq? = res[gLoss]?.exportData()

            return (res[imageGenOut]?.exportData() ?? [], dLoss ?? [], gLoss ?? [])
        }
    }
    
    func generate(latent: Seq) -> Seq {
        autoreleasepool {
            let res = graph.run(
                feeds: [
                    genInput: latent.asMPSGraphTensorData(on: mtlDevice, shape: genInput.shape!)!
                ],
                targetTensors: [imageGenOut],
                targetOperations: nil
            )
            return res[imageGenOut]!.exportData()
        }
    }

    func save() throws -> URL {
        try autoreleasepool {
            let allTrainableVars = genTrainableVariables.trainableVariables + discrTrainableVariables.trainableVariables
            let res = graph.run(feeds: [:], targetTensors: allTrainableVars, targetOperations: nil)
            
            var datas: [String: Data] = [:]
            for (k, v) in res {
                datas[k.operation.name] = v.exportRawData(itemType: Float32.self)
            }
            
            let fileManager = FileManager.default
            let checkpoinstFolder = try fileManager.urlForCheckpintsDirectory()
            let checkpointName = UUID().uuidString
            let checkpointFolder = checkpoinstFolder
                .appending(path: checkpointName, directoryHint: .isDirectory)
            
            try fileManager.createDirectory(at: checkpointFolder, withIntermediateDirectories: true)
            for (dataName, data) in datas {
                let filePath = checkpointFolder.appending(path: dataName).appendingPathExtension("bin")
                try data.write(to: filePath)
            }
            
            return checkpointFolder
        }
    }

    func load(from checkpointFolderURL: URL) throws {
        try autoreleasepool {
            let fileManager = FileManager.default
            let dataFiles = try fileManager.contentsOfDirectory(atPath: checkpointFolderURL.path)

            guard dataFiles.contains(where: { filename in filename.hasSuffix("bin") }) else { throw NSError(domain: "bad checkpoint", code: 1) }

            var tensorsByName: [String: MPSGraphTensor] = [:]
            var assignOperations: [MPSGraphOperation] = []

            for tensor in genTrainableVariables.trainableVariables + discrTrainableVariables.trainableVariables {
                tensorsByName[tensor.operation.name] = tensor
            }

            for dataFile in dataFiles {
                let dataFilePath = checkpointFolderURL.appending(path: dataFile)

                let tensorName = dataFile.components(separatedBy: ".").first!
                let data = try Data(contentsOf: dataFilePath)
                guard let correspondingTensor = tensorsByName[tensorName] else { throw NSError(domain: "buba", code: 0) }

                let tensorWithData = graph.constant(data, shape: correspondingTensor.shape!, dataType: correspondingTensor.dataType)
                let assignOp = graph.assign(correspondingTensor, tensor: tensorWithData, name: nil)
                assignOperations.append(assignOp)
            }

            graph.run(feeds: [:], targetTensors: [], targetOperations: assignOperations)
        }
    }
}
