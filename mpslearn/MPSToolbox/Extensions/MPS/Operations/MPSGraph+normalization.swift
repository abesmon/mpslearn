//
//  MPSGraph+normalization.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 28.11.2022.
//

import MetalPerformanceShadersGraph

extension MPSGraph {
    func trainableBatchNorm(input: MPSGraphTensor,
                            outChannels: NSNumber,
                            vars: VariablesPair? = nil,
                            eps: Float = 1e-5, name: String?) -> TrainableRes {
        let trainableGamma: MPSGraphTensor
        if let varGamma = vars?.gamma {
            trainableGamma = varGamma
        } else {
            let trainableGammaSeq = Array(repeating: Float32(1), count: input.featuresCount())
            let trainableGammaData = Data(bytes: trainableGammaSeq, count: trainableGammaSeq.count * 4)
            trainableGamma = self.variable(with: trainableGammaData,
                                               shape: input.shape!,
                                               dataType: .float32,
                                               name: name.map { $0 + ".gammaVar" })
        }

        let trainableBeta: MPSGraphTensor
        if let varBeta = vars?.beta {
            trainableBeta = varBeta
        } else {
            let trainableBetaSeq = Array(repeating: Float32(0), count: input.featuresCount())
            let trainableBetaData = Data(bytes: trainableBetaSeq, count: trainableBetaSeq.count * 4)
            trainableBeta = self.variable(with: trainableBetaData, shape: input.shape!, dataType: .float32, name: name.map { $0 + ".betaVar" })
        }

        let mean = self.mean(of: input, axes: [0, 1, 2], name: name.map { $0 + ".mean" })
        let variance = self.variance(of: input, axes: [0, 1, 2], name: name.map { $0 + ".variance" })
        let norm = self.normalize(input,
                       mean: mean,
                       variance: variance,
                       gamma: trainableGamma,
                       beta: trainableBeta,
                       epsilon: eps,
                                  name: name.map { $0 + ".batchNorm" })

        return (norm, [trainableGamma, trainableBeta])
    }
}
