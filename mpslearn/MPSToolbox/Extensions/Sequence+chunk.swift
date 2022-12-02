//
//  Sequence+chunk.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 17.11.2022.
//

import Foundation

extension Sequence where Self: RandomAccessCollection, Self.Index == Int {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
    
    func chuncked(into size: Int, dropTailIfSmall: Bool) -> [[Element]] {
        let prechunked = chunked(into: size)
        guard prechunked.last?.count == prechunked.first?.count else {
            return prechunked.dropLast(1)
        }
        return prechunked
    }
}
