//
//  Sequence+NSNumbers.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 23.11.2022.
//

import Foundation

extension Sequence where Element: BinaryInteger {
    var nsnumbers: [NSNumber] { map { NSNumber(value: Int($0)) } }
}
