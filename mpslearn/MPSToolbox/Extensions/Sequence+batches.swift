//
//  Sequence+batches.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 27.11.2022.
//

import Foundation

extension Sequence where Self: RandomAccessCollection, Index == Int {
    func item(index: Int, fromBatchSized batchSize: Int) -> Self.SubSequence {
        let elementsInItem = (count / batchSize)
        return self[index * elementsInItem..<(index + 1) * elementsInItem]
    }
}
