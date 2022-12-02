//
//  Data+array.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 30.11.2022.
//

import Foundation

extension Data {
    func asArray<T>(ofType: T.Type = T.self, length: Int) -> [T] {
        return [T](unsafeUninitializedCapacity: length) { buffer, initializedCount in
            initializedCount = self.copyBytes(to: buffer) / MemoryLayout<T>.stride
        }
    }
}
