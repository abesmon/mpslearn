//
//  Sequence+adding.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 24.11.2022.
//

import Foundation

extension Sequence where Self: RandomAccessCollection, Self.Index == Int {
    func adding(_ element: Element, afterEvery n: Int) -> [Element] {
        guard n > 0 else { fatalError("afterEvery value must be greater than 0") }
        let newcount = count + count / n
        return (0 ..< newcount).map { (i: Int) -> Element in
            (i + 1) % (n + 1) == 0 ? element : self[i - i / (n + 1)]
        }
    }
}
