//
//  MPSGraph+loss.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 17.11.2022.
//

import Foundation
import MetalPerformanceShadersGraph

enum LossReduction {
    case none
    case mean([NSNumber])
    case sum([NSNumber])
}

extension MPSGraph {
    func MSELoss(_ input: MPSGraphTensor, labels: MPSGraphTensor, reduction: LossReduction = .mean([0]), name: String?) -> MPSGraphTensor {
        let sub = subtraction(input, labels, name: name.map { $0 + ".sub" })
        let sq = power(sub, constant(2, dataType: .float32), name: name.map { $0 + ".pow" })
        switch reduction {
        case .none: return sq
        case .mean(let axis): return mean(of: sq, axes: axis, name: name.map { $0 + ".mean" })
        case .sum(let axis): return reductionSum(with: sq, axes: axis, name:  name.map { $0 + ".sum" })
        }
    }
}

extension MPSGraph {
    // y * log(x) + (1 - y) * log(1 - x)
    // y * a + b * log(c)
    // y * a + b * d

    // y is target; x is input
    func BCELoss(input x: MPSGraphTensor, target y: MPSGraphTensor, reduction: LossReduction = .mean([0])) -> MPSGraphTensor {
        let eps = constant(1e-7, dataType: .float32)
        let a1 = logarithmBase2(with: maximum(x, eps, name: nil), name: nil)
//        let a1 = maximum(a, constant(-100, shape: a.shape!, dataType: a.dataType), name: nil)

        let b = subtraction(constant(1, dataType: y.dataType), y, name: nil)
        let c = subtraction(constant(1, dataType: x.dataType), x, name: nil)
        let d1 = logarithmBase2(with: maximum(c, eps, name: nil) , name: nil)
//        let d1 = maximum(d, constant(-100, shape: d.shape!, dataType: d.dataType), name: nil)
        let e = multiplication(y, a1, name: nil)
        let f = multiplication(b, d1, name: nil)
        let sum = addition(e, f, name: nil)
        let mult = multiplication(constant(-1, dataType: sum.dataType), sum, name: nil)
        switch reduction {
        case .none: return mult
        case .mean(let axis): return mean(of: mult, axes: axis, name: nil)
        case .sum(let axis): return reductionSum(with: mult, axes: axis, name: nil)
        }
    }
}
