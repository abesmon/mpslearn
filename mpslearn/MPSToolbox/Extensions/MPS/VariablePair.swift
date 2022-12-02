//
//  VariablePair.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 28.11.2022.
//

import MetalPerformanceShadersGraph

struct VariablesPair: ExpressibleByArrayLiteral {
    let a: MPSGraphTensor?
    let b: MPSGraphTensor?

    init(arrayLiteral elements: MPSGraphTensor?...) {
        a = !elements.isEmpty ? elements[0] : nil
        b = elements.count > 1 ? elements[1] : nil
    }
}

extension VariablesPair {
    var weights: MPSGraphTensor? { return self.a }
    var biases: MPSGraphTensor? { return self.b }

    var gamma: MPSGraphTensor? { return self.a }
    var beta: MPSGraphTensor? { return self.b }
}


