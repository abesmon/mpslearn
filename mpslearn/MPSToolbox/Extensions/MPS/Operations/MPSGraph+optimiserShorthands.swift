//
//  MPSGraph+optimiserShorthands.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 18.11.2022.
//

import Foundation
import MetalPerformanceShadersGraph

extension MPSGraph {
    func adamOptimiseOperations(gradients: [MPSGraphTensor: MPSGraphTensor],
                                lr: Double = 1e-4,
                                beta1: Double = 0.9, beta2: Double = 0.999,
                                epsilon: Double = 1e-8) -> [MPSGraphOperation] {
        let learningRate = constant(lr, dataType: .float32)
        let beta1 = constant(beta1, dataType: .float32)
        let beta2 = constant(beta2, dataType: .float32)
        let epsilon = constant(epsilon, dataType: .float32)
        
        var assignOps: [MPSGraphOperation] = []
        for (variable, gradient) in gradients {
            let variableFeaturesCount = variable.shape!.featuresCount() as NSNumber
            let momentumVar = self.variable(
                with: [Float32](repeating: 1, count: variableFeaturesCount.intValue).asData()!,
                shape: variable.shape!,
                dataType: .float32, name: nil
            )
            let velocityVar = self.variable(
                with: [Float32](repeating: 1, count: variableFeaturesCount.intValue).asData()!,
                shape: variable.shape!,
                dataType: .float32, name: nil
            )
            
            let adamUpdate = adam(
                currentLearningRate: learningRate,
                beta1: beta1,
                beta2: beta2,
                epsilon: epsilon,
                values: variable,
                momentum: momentumVar,
                velocity: velocityVar,
                maximumVelocity: nil,
                gradient: gradient,
                name: nil
            )
            let assign = self.assign(variable, tensor: adamUpdate[0], name: nil)
            assignOps.append(assign)
            let momentumAssign = self.assign(momentumVar, tensor: adamUpdate[1], name: nil)
            assignOps.append(momentumAssign)
            let velocityAssign = self.assign(velocityVar, tensor: adamUpdate[2], name: nil)
            assignOps.append(velocityAssign)
        }
        return assignOps
    }
    
    func adamDynamicOptimiseOperations(gradients: [MPSGraphTensor: MPSGraphTensor],
                                       lr: Double = 1e-4,
                                       beta1: Double = 0.9, beta2: Double = 0.999,
                                       beta1p: Double = 0.9, beta2p: Double = 0.999,
                                       epsilon: Double = 1e-8) -> [MPSGraphOperation] {
        let learningRate = constant(lr, dataType: .float32)
        let beta1с = constant(beta1, dataType: .float32)
        let beta2с = constant(beta2, dataType: .float32)
        let epsilon = constant(epsilon, dataType: .float32)
        
        let beta1Power = constant(beta1p, dataType: .float32)
        let beta2Power = constant(beta2p, dataType: .float32)
        
        var assignOps: [MPSGraphOperation] = []
        for (variable, gradient) in gradients {
            let momentumVar = self.variable(
                with: [Float32](repeating: 1, count: variable.featuresCount()).asData()!,
                shape: variable.shape!,
                dataType: .float32, name: nil
            )
            let velocityVar = self.variable(
                with: [Float32](repeating: 1, count: variable.featuresCount()).asData()!,
                shape: variable.shape!,
                dataType: .float32, name: nil
            )

            let adamUpdate = self.adam(
                learningRate: learningRate,
                beta1: beta1с,
                beta2: beta2с,
                epsilon: epsilon,
                beta1Power: beta1Power,
                beta2Power: beta2Power,
                values: variable,
                momentum: momentumVar,
                velocity: velocityVar,
                maximumVelocity: nil,
                gradient: gradient,
                name: nil
            )
            let assign = self.assign(variable, tensor: adamUpdate[0], name: nil)
            assignOps.append(assign)
            let momentumAssign = self.assign(momentumVar, tensor: adamUpdate[1], name: nil)
            assignOps.append(momentumAssign)
            let velocityAssign = self.assign(velocityVar, tensor: adamUpdate[2], name: nil)
            assignOps.append(velocityAssign)
        }
        return assignOps
    }
    
    func stochasticGradientDescentOptimiseOperations(gradients: [MPSGraphTensor : MPSGraphTensor],
                                                     lr: Double = 0.0001) -> [MPSGraphOperation] {
        var assignOps: [MPSGraphOperation] = []
        for (variable, gradient) in gradients {
            let learningRate = constant(lr, dataType: .float32)
            let update = stochasticGradientDescent(learningRate: learningRate, values: variable, gradient: gradient, name: nil)
            let assign = assign(variable, tensor: update, name: nil)
            assignOps.append(assign)
        }
        return assignOps
    }
}
