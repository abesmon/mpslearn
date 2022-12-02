//
//  BalanceTransformer.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 27.11.2022.
//

import Foundation

struct BalanceTransformer: Transformer {
    let mult: Float32
    let shift: Float32

    func transform(_ input: [Float32]) -> [Float32] {
        input.map { ($0 + shift) * mult }
    }
}
