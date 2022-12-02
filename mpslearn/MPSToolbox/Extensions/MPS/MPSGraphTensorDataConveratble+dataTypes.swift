//
//  MPSGraphTensorDataConveratble+dataTypes.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 29.11.2022.
//

import MetalPerformanceShadersGraph

extension Array: MPSGraphTensorDataConveratble where Element == Float32 {
    func asMPSGraphTensorData(on device: MTLDevice, shape: [NSNumber]) -> MPSGraphTensorData? {
        asMPSGraphTensorData(on: device, shape: shape, dataType: .float32)
    }
}

#if !targetEnvironment(simulator)
//extension Array where Element == Float16 {
//    func asMPSGraphTensorData(on device: MTLDevice, shape: [NSNumber]) -> MPSGraphTensorData? {
//        asMPSGraphTensorData(on: device, shape: shape, dataType: .float16)
//    }
//}
#endif

extension Array where Element == Bool {
    func asMPSGraphTensorData(on device: MTLDevice, shape: [NSNumber]) -> MPSGraphTensorData? {
        asMPSGraphTensorData(on: device, shape: shape, dataType: .bool)
    }
}

extension Array where Element == Int32 {
    func asMPSGraphTensorData(on device: MTLDevice, shape: [NSNumber]) -> MPSGraphTensorData? {
        asMPSGraphTensorData(on: device, shape: shape, dataType: .int32)
    }
}

extension Array where Element == Int16 {
    func asMPSGraphTensorData(on device: MTLDevice, shape: [NSNumber]) -> MPSGraphTensorData? {
        asMPSGraphTensorData(on: device, shape: shape, dataType: .int16)
    }
}
