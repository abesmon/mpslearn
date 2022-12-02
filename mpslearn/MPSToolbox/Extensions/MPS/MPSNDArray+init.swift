//
//  MPSNDArray+init.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 26.11.2022.
//

import Foundation
import MetalPerformanceShadersGraph

extension MPSNDArray {
    static func withRandomValues(in range: ClosedRange<Float32>, device: MTLDevice, shape: [NSNumber]) -> Self {
        var randomValuesBuffer = (0..<shape.featuresCount()).map { _ in Float32.random(in: range) }
        return withValues(of: &randomValuesBuffer, device: device, shape: shape)
    }

    static func withValues(of buffer: [Float32], device: MTLDevice, shape: [NSNumber]) -> Self {
        var mutableBuffer = buffer
        return withValues(of: &mutableBuffer, device: device, shape: shape)
    }

    static func withValues(of mutableBuffer: inout [Float32], device: MTLDevice, shape: [NSNumber]) -> Self {
        let array = Self(device: device, descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: shape))
        array.writeBytes(&mutableBuffer, strideBytes: nil)
        return array
    }
}
