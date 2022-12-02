//
//  MPSGraphTensor+quadShape.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 23.11.2022.
//

import Foundation
import MetalPerformanceShadersGraph

extension MPSGraphTensor {
    var quadShape: Quad<Int>? {
        shape?.map(\.intValue).quad
    }
}
