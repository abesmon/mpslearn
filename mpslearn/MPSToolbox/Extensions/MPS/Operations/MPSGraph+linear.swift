//
//  MPSGraph+linear.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 17.11.2022.
//

import Foundation
import MetalPerformanceShadersGraph

extension MPSGraph {
    func linear(input: MPSGraphTensor, weights: MPSGraphTensor, bias: MPSGraphTensor?, name: String?) -> MPSGraphTensor {
        assert(weights.shape!.count == 2)
        
        var sourceTensor = input
        if input.shape!.count > 1 {
            let inputFeaturesCount = input.shape!.featuresCount() as NSNumber
            sourceTensor = reshape(input, shape: [1, inputFeaturesCount], name: name.map { $0 + ".reshapedInput" })
        }
        
        let fcTensor = matrixMultiplication(primary: sourceTensor, secondary: weights, name: name.map { $0 + ".fc" })
        
        if let bias = bias {
            return addition(fcTensor, bias, name: name.map { $0 + ".biased" })
        } else {
            return fcTensor
        }
    }
    
    typealias TrainableOut = (out: MPSGraphTensor, weightsVariable: MPSGraphTensor, biasVariable: MPSGraphTensor)
    func makeTrainableLinear(input: MPSGraphTensor,
                             out: NSNumber,
                             name: String?)  -> TrainableOut {
        let k = 1 / Float32(out.intValue)

        let inputFeaturesCount = input.shape!.featuresCount() as NSNumber
        let weightsShape = [inputFeaturesCount, out]
        let weightsFeturesCount = weightsShape.featuresCount().intValue
        
        let weightsData = (0..<weightsFeturesCount).map { _ in Float32.random(in: -k.squareRoot()...k.squareRoot()) }
        let biasData = (0..<out.intValue).map { _ in Float32.random(in: -k.squareRoot()...k.squareRoot()) }
        
        
        let weights = variable(
            with: weightsData.asData()!,
            shape: weightsShape,
            dataType: .float32, name: name.map { $0 + ".weightsVar" }
        )
        let bias = variable(
            with: biasData.asData()!,
            shape: [out],
            dataType: .float32,
            name: name.map { $0 + ".biasVar" }
        )
        
        let linear = linear(input: input, weights: weights, bias: bias, name: name)
        return (linear, weights, bias)
    }
}
