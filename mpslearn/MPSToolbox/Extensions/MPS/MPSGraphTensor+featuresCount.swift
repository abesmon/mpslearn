//
//  MPSGraphTensor+featuresCount.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 24.11.2022.
//

import Foundation
import MetalPerformanceShadersGraph

extension MPSGraphTensor {
    func featuresCount() -> NSNumber { shape!.featuresCount() }
    func featuresCount() -> Int { shape!.featuresCount() }
}
