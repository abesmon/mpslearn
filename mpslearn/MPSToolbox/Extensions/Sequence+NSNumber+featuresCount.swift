//
//  Sequence+NSNumber+featuresCount.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 18.11.2022.
//

import Foundation

extension Sequence where Element == NSNumber {
    func featuresCount() -> NSNumber { NSNumber(value: featuresCount() as Int) }
    func featuresCount() -> Int { reduce(1, { $0 * $1.intValue }) }
}
