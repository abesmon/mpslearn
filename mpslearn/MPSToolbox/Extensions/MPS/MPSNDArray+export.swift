//
//  MPSNDArray+export.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 13.11.2022.
//

import MetalPerformanceShadersGraph

extension MPSNDArray {
    func toContigousArray<T>(commandQueue: MTLCommandQueue, destType: MPSDataType, length: Int) -> ContiguousArray<T>? {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return nil }
        let buffer = commandQueue.device.makeBuffer(length: length * MemoryLayout<T>.stride)!
        exportData(with: commandBuffer,
                   to: buffer,
                   destinationDataType: destType, offset: 0, rowStrides: nil)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let pointer = buffer.contents().assumingMemoryBound(to: T.self)
        let dataBufferPointer = UnsafeBufferPointer<T>(start: pointer, count: length)
        return ContiguousArray<T>(dataBufferPointer)
    }
    
    func toContigousArray(commandQueue: MTLCommandQueue, length: Int) -> ContiguousArray<Float32>? {
        toContigousArray(commandQueue: commandQueue, destType: .float32, length: length)
    }
}
