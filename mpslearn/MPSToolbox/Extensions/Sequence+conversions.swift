//
//  Sequence+conversions.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 13.11.2022.
//

import Foundation
import MetalPerformanceShadersGraph
import MetalPerformanceShaders
import Metal


// MARK: - as buffer
extension Sequence {
    func asBuffer(on device: MTLDevice) -> MTLBuffer? {
        withContiguousStorageIfAvailable { p -> MTLBuffer? in
            guard let baseAddress = p.baseAddress else {
                return nil
            }
            return device.makeBuffer(
                bytes: baseAddress,
                length: p.count * MemoryLayout<Element>.size,
                options: .storageModeShared
            )
        } ?? nil
    }
}

// MARK: - as vector
extension Sequence where Self: RandomAccessCollection {
    func asVector(on device: MTLDevice, dataType: MPSDataType) -> MPSVector? {
        guard let buffer = asBuffer(on: device) else { return nil }
        return MPSVector(
            buffer: buffer,
            descriptor: MPSVectorDescriptor(
                length: count,
                dataType: dataType
            )
        )
    }
}

// MARK: - asData
extension Sequence where Self: RandomAccessCollection {
    func asData() -> Data? {
        withContiguousStorageIfAvailable { buffer in Data(buffer: buffer) }
    }
}

// MARK: - asMPSGraphTensorData
extension Sequence where Self: RandomAccessCollection {
    func asMPSGraphTensorData(on device: MTLDevice, shape: [NSNumber], dataType: MPSDataType) -> MPSGraphTensorData? {
        asMPSGraphTensorData(on: MPSGraphDevice(mtlDevice: device), shape: shape, dataType: dataType)
    }
    
    func asMPSGraphTensorData(on device: MPSGraphDevice, shape: [NSNumber], dataType: MPSDataType) -> MPSGraphTensorData? {
        guard let data = asData() else { return nil }
        return MPSGraphTensorData(device: device, data: data, shape: shape, dataType: dataType)
    }
}
