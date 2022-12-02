//
//  Transformer.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 27.11.2022.
//

import Foundation

protocol Transformer {
    func transform(_ input: [Float32]) -> [Float32]
}

extension Sequence where Element == Transformer {
    func transform(_ input: [Float32]) -> [Float32] {
        self.reduce(input) { partialResult, transformer in
            return transformer.transform(partialResult)
        }
    }
}
