//
//  Sequence+touples.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 23.11.2022.
//

import Foundation

typealias Pair<T> = (T, T)
typealias Quad<T> = (T, T, T, T)

extension Sequence where Self: RandomAccessCollection, Self.Index == Int {
    var pair: Pair<Element>? {
        count == 2 ? (self[0], self[1]) : nil
    }

    var quad: Quad<Element>? {
        count == 4 ? (self[0], self[1], self[2], self[3]) : nil
    }
}
