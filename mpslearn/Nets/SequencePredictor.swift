//
//  SequencePredictor.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 20.11.2022.
//

import Foundation
import MetalPerformanceShadersGraph

class SequencePredictor {
    private let batchSize: Int = 1
    
    private let mtlDevice: MTLDevice
    private let commandQueue: MTLCommandQueue
    
    private let graph = MPSGraph()
    private let input: MPSGraphTensor
    private let label: MPSGraphTensor
    
    private let out: MPSGraphTensor
    private let loss: MPSGraphTensor
    
    private let assignOps: [MPSGraphOperation]
    
    init() {
        self.mtlDevice = MTLCreateSystemDefaultDevice()!
        self.commandQueue = mtlDevice.makeCommandQueue()!
        
        self.input = graph.placeholder(shape: [NSNumber(value: batchSize), 40], name: "input")
        self.label = graph.placeholder(shape: [NSNumber(value: batchSize), 1], name: "label")
        
        var variables: [MPSGraphTensor] = []
        
        let linear1 = graph.makeTrainableLinear(input: input, out: 100, name: "linear1")
        variables.append(linear1.weightsVariable)
        variables.append(linear1.biasVariable)
        let relu1 = graph.leakyReLU(with: linear1.out, alpha: 0.03, name: "relu1")

        let linear2 = graph.makeTrainableLinear(input: relu1, out: 50, name: "linear2")
        variables.append(linear2.weightsVariable)
        variables.append(linear2.biasVariable)
        let relu2 = graph.leakyReLU(with: linear2.out, alpha: 0.03, name: "relu2")

        let linear3 = graph.makeTrainableLinear(input: relu2, out: 1, name: "linear3")
        variables.append(linear3.weightsVariable)
        variables.append(linear3.biasVariable)
        
        self.out = linear3.out
        self.loss = graph.MSELoss(out, labels: label, name: "loss")
        
        let gradients = graph.gradients(of: loss, with: variables, name: nil)
        self.assignOps = graph.adamOptimiseOperations(gradients: gradients, lr: 0.0001)
    }
    
    
    
    func train(data: [CGPoint], iterations: Int) {
        let data = data.map { Float32($0.y) }
        let inputDataArray = MPSNDArray(
            device: mtlDevice,
            descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: input.shape!)
        )
        let labelDataArray = MPSNDArray(
            device: mtlDevice,
            descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: label.shape!)
        )
        var inputBuffer = [Float32](repeating: 0, count: 40)
        var labelBuffer: Float32 = 0

        var lossBuffer: Float32 = 0
        var outBuffer: Float32 = 0
        
        for _ in 0..<iterations {
            let randomChunkStartPos = Int.random(in: 0..<data.count-41)
            for i in 0..<41 {
                if i < 40 {
                    inputBuffer[i] = data[randomChunkStartPos + i]
                } else {
                    labelBuffer = data[randomChunkStartPos + i]
                }
            }
            inputDataArray.writeBytes(&inputBuffer, strideBytes: nil)
            labelDataArray.writeBytes(&labelBuffer, strideBytes: nil)
            
            let inputData = MPSGraphTensorData(inputDataArray)
            let labelData = MPSGraphTensorData(labelDataArray)
            
            autoreleasepool {
                let results = graph.run(
                    with: commandQueue,
                    feeds: [
                        input: inputData,
                        label: labelData
                    ],
                    targetTensors: [loss, out],
                    targetOperations: assignOps
                )
                
                results[loss]?.mpsndarray()
                    .readBytes(&lossBuffer, strideBytes: nil)
                results[out]?.mpsndarray()
                    .readBytes(&outBuffer, strideBytes: nil)
                print(lossBuffer, outBuffer)
            }
        }
    }
    
    func predict(startSequence: [CGPoint], predictNext itemsCountToPredict: Int) -> [CGPoint] {
        var resultSeq = startSequence
        let xDist = startSequence[1].x - startSequence[0].x
        
        for _ in 0..<itemsCountToPredict {
            let batch = Array(resultSeq.suffix(40))
            let inputData = batch
                .map { Float32($0.y) }
                .asMPSGraphTensorData(on: mtlDevice, shape: input.shape!)!
            
            let results = graph.run(
                with: commandQueue,
                feeds: [input: inputData],
                targetTensors: [out],
                targetOperations: []
            )
            
            var resultBuffer: Float32 = 0
            results[out]?.mpsndarray()
                .readBytes(&resultBuffer, strideBytes: nil)
            
            let x = batch.last!.x + xDist
            let y = CGFloat(resultBuffer)
            resultSeq.append(.init(x: x, y: y))
        }
        
        return resultSeq
    }
}
