//
//  MPSGraphTensorDataConveratble.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 29.11.2022.
//

import MetalPerformanceShadersGraph

protocol MPSGraphTensorDataConveratble {
    func asMPSGraphTensorData(on device: MTLDevice, shape: [NSNumber]) -> MPSGraphTensorData?
}
