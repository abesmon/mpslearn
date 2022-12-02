//
//  MPSGraphTensorData+init.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 27.11.2022.
//

import Foundation
import MetalPerformanceShadersGraph

extension MPSGraphTensorData {
    static func withRandomValues(in range: ClosedRange<Float32>, device: MTLDevice, shape: [NSNumber]) -> MPSGraphTensorData {
        let buffer = (0..<shape.featuresCount()).map { _ in Float32.random(in: range) }
        return withValues(of: buffer, device: device, shape: shape)
    }

    static func withValues(of buffer: [Float32], device: MTLDevice, shape: [NSNumber]) -> MPSGraphTensorData {
        return buffer.asMPSGraphTensorData(on: device, shape: shape)!
    }
}
