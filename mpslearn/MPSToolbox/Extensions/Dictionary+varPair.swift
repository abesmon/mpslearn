//
//  Dictionary+varPair.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 28.11.2022.
//

import MetalPerformanceShadersGraph

extension Dictionary where Key == String, Value == VariablesPair {
    var trainableVariables: [MPSGraphTensor] {
        return values.compactMap { $0.a } + values.compactMap { $0.b }
    }
}
