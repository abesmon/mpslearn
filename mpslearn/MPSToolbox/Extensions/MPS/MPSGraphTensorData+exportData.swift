//
//  MPSGraphTensorData+exportData.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 25.11.2022.
//

import Foundation
import MetalPerformanceShadersGraph

extension MPSGraphTensorData {
    func exportData_f32() -> [Float32] { exportData() }

    func exportData<T: ExpressibleByIntegerLiteral>() -> [T] {
        var exportBuffer = [T](repeating: 0, count: shape.featuresCount().intValue)
        mpsndarray().readBytes(&exportBuffer, strideBytes: nil)
        return exportBuffer
    }

    func exportRawData<T: ExpressibleByIntegerLiteral>(itemType: T.Type = T.self) -> Data {
        return Data(bytes: exportData() as [T], count: shape.featuresCount().intValue * MemoryLayout<T>.stride)
    }
}
