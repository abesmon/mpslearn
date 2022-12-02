//
//  MPSGraph+gelu.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 16.11.2022.
//

import Foundation
import MetalPerformanceShadersGraph

extension MPSGraph {
    func gelu(input: MPSGraphTensor) -> MPSGraphTensor {        
        let ones = constant(1, shape: [1], dataType: .float32)
        let half = constant(0.5, shape: [1], dataType: .float32)
        
        let sqrt = squareRoot(with: half, name: nil)
        
        let multiply = multiplication(sqrt, input, name: nil)
        let multiply2 = multiplication(half, input, name: nil)
        
        let erf = erf(with: multiply, name: nil)
        
        let add = addition(erf, ones, name: nil)
        
        return multiplication(multiply2, add, name: nil)
    }
}
