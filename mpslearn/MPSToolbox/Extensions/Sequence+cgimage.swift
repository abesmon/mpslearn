//
//  Sequence+cgimage.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 25.11.2022.
//

import Foundation
import CoreGraphics

extension Sequence where Self: RandomAccessCollection, Index == Int, Element == Float32 {
    func makeCGImage(width: Int, height: Int, balance: (shift: Float32, mult: Float32)? = nil) -> CGImage? {
        var values = self
            .adding(0, afterEvery: 3)
            .map {
                var val = $0
                if let balance = balance {
                    val = (val + balance.shift) * balance.mult
                }
                return UInt8(truncatingIfNeeded: Int(val * 255))
            }

        let bitmapInfo = CGImageAlphaInfo.noneSkipLast
        guard let ctx = (values.withContiguousMutableStorageIfAvailable { pointer in
            return CGContext(
                data: pointer.baseAddress,
                width: width, height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4 * 1, // width * C * 1 bytes per channel
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: bitmapInfo.rawValue
            )
        }) else { return nil }

        return ctx?.makeImage()
    }
}

extension Array where Element == Float32 {
    func makeCGImage(width: Int, height: Int, transformers: [Transformer]) -> CGImage? {
        return transformers
            .transform(self)
            .makeCGImage(width: width, height: height)
    }
}

#if canImport(AppKit)
import AppKit
extension NSImage {
    func makeRGBBuffer(balance: (shift: Float32, mult: Float32)? = nil) -> [Float32]? {
        guard let data = cgImage(forProposedRect: nil, context: nil, hints: nil)?.dataProvider?.data else { return nil }

        let pixelCount = Int(size.width * size.height)

        guard let pointer = CFDataGetBytePtr(data) else { return nil }
        var buffer = [Float32]()
        var b = 0
        for i in (0..<pixelCount * 4) {
            if b == 3 { b = 0; continue }
            var val = Float32(pointer[i]) / 255
            if let balance = balance {
                val = (val + balance.shift) * balance.mult
            }
            buffer.append(val)
            b += 1
        }
        return buffer
    }
}
#endif
