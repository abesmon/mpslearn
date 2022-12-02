//
//  SimpleInteractor.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 16.11.2022.
//

import Foundation
import MetalPerformanceShadersGraph

class SimpleInteractor {
    func simpleCalc() {
        DispatchQueue.main.async {
            let graph = MPSGraph()
            let shape: [NSNumber] = [3, 5]
            let zeroTwos = graph.constant(0.2, shape: shape, dataType: .float32)
            
            let data = ([1,2,3,4] as [Int32])
                .withContiguousStorageIfAvailable { p in
                    Data(buffer: p)
                }!
            
            let zeroFives = graph.constant(data, shape: shape, dataType: .int32)
            let castedZeroFives = graph.cast(zeroFives, to: .float32, name: "castedZeroFives")
            
            let addTensor = graph.addition(zeroTwos,
                                           castedZeroFives,
                                           name: nil)
            
            
            
            let results = graph.run(
                feeds: [:],
                targetTensors: [addTensor],
                targetOperations: nil
            )
            
            
            let length = addTensor.shape?.featuresCount() ?? 0
            let metalDevice = MTLCreateSystemDefaultDevice()!
            let commandQueue = metalDevice.makeCommandQueue()!
            let arr = results[addTensor]?.mpsndarray()
                .toContigousArray(commandQueue: commandQueue, length: length)
            print(arr!)
        }
    }
    
    func testGelu() {
        let metalDevice = MTLCreateSystemDefaultDevice()!
        
        let vector = ([1.0, 2, 3, 4, 5, 6] as [Float32]).asVector(on: metalDevice, dataType: .float32)!
        
        let graph = MPSGraph()
        
        let inputTensor = graph.placeholder(shape: nil, dataType: .float32, name: nil)
        
        let gelu = graph.gelu(input: inputTensor)
        
        let realInput = MPSGraphTensorData(vector)
        
        let results = graph.run(
            feeds: [
                inputTensor: realInput
            ], targetTensors: [gelu],
            targetOperations: nil
        )
        
        let outputValues: [Float32]? = results[gelu]?.exportData()
        print(outputValues ?? [])
    }
    
    func testResduction() {
        let inputData: [Float32] = [1, 2, 3]
        let weightsData: [Float32] = [
            4, 1,
            5, 2,
            6, 3
        ]
        
        let graph = MPSGraph()
        let input = graph.constant(inputData.asData()!, shape: [NSNumber(value: inputData.count)], dataType: .float32)
        let weights = graph.constant(weightsData.asData()!, shape: [3, 2], dataType: .float32)
        
        let bias = graph.constant(0.1, dataType: .float32)
        
        let linear = graph.linear(input: input, weights: weights, bias: nil, name: "bababoy")
        let linearWBias = graph.linear(input: input, weights: weights, bias: bias, name: "bababoy-biased")
        
        let graphRun = graph.run(feeds: [:], targetTensors: [linear, linearWBias], targetOperations: nil)
        var linearResult: [Float32] = .init(repeating: 0, count: weights.shape![1].intValue)
        var linearWBiasResult: [Float32] = .init(repeating: 0, count: weights.shape![1].intValue)
        graphRun[linear]?.mpsndarray()
            .readBytes(&linearResult, strideBytes: nil)
        graphRun[linearWBias]?.mpsndarray()
            .readBytes(&linearWBiasResult, strideBytes: nil)
        
        print(linearResult)
        print(linearWBiasResult)
    }

    func testLogStuff() {
        let graph = MPSGraph()
        let log0 = graph.logarithmBase2(with: graph.constant(0, dataType: .float32), name: nil)
        let log1 = graph.logarithmBase2(with: graph.constant(1, dataType: .float32), name: nil)
        let log0maxed = graph.maximum(graph.constant(-100, shape: log0.shape!, dataType: log0.dataType), log0, name: nil)

        let results = graph.run(feeds: [:], targetTensors: [log0, log1, log0maxed], targetOperations: nil)

        var log0res: Float32 = 0
        var log1res: Float32 = 0
        var log0maxRes: Float32 = 0

        results[log0]?.mpsndarray()
            .readBytes(&log0res, strideBytes: nil)
        results[log1]?.mpsndarray()
            .readBytes(&log1res, strideBytes: nil)
        results[log0maxed]?.mpsndarray()
            .readBytes(&log0maxRes, strideBytes: nil)
        print(log0res, log1res, log0maxRes)
    }

    func testRandoms() {
        let graph = MPSGraph()
        let rand = graph.randomUniformTensor(withShape: [1, 2], name: nil)

        var randoms: [[Float32]] = []
        for _ in 0..<10 {
            let result = graph.run(feeds: [:], targetTensors: [rand], targetOperations: nil)
            randoms.append(result[rand]!.exportData())
        }
        print(randoms)
    }
}
