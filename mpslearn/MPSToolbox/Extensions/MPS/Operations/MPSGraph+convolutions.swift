//
//  MPSGraph+convolutions.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 24.11.2022.
//

import Foundation
import MetalPerformanceShadersGraph

enum WeightsInitializationRule<T: FloatingPoint> {
    case normal
    case dist(mean: T, stdev: T)

    func range(forNormal k: T) -> ClosedRange<T> {
        switch self {
        case .normal: return -k.squareRoot()...k.squareRoot()
        case .dist(let mean, let stdev): return mean-stdev...mean+stdev
        }
    }
}

// MARK: ConvTranspose2d
extension MPSGraph {
    func addConvolutionTranspose2D(input: MPSGraphTensor,
                                   outChannels: NSNumber,
                                   weightsInitRule: WeightsInitializationRule<Float32> = .normal,
                                   biasInitRule: WeightsInitializationRule<Float32> = .normal,
                                   kernelSize: NSNumber,
                                   stride: Int = 1,
                                   padding: Int = 0,
                                   outputPadding: [NSNumber]?,
                                   groups: Int = 1,
                                   bias: Bool = true,
                                   dilation: Int = 1,
                                   name: String?) -> TrainableRes {
        let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: stride,
            strideInY: stride,
            dilationRateInX: dilation,
            dilationRateInY: dilation,
            groups: groups,
            paddingLeft: padding,
            paddingRight: padding,
            paddingTop: padding,
            paddingBottom: padding,
            paddingStyle: .explicit,
            dataLayout: .NHWC,
            weightsLayout: .HWIO
        )!

        let weightsShape = [kernelSize, kernelSize, outChannels, input.shape![3]]
        let k = Float32(descriptor.groups) / Float32(weightsShape[0].intValue * weightsShape[1].intValue * weightsShape[2].intValue)

        let weightsSeq = (0..<weightsShape.featuresCount()).map { _ in Float32.random(in: weightsInitRule.range(forNormal: k)) }
        let weightsData = Data(bytes: weightsSeq, count: weightsSeq.count * MemoryLayout<Float32>.stride)
        let weightsVar = variable(with: weightsData, shape: weightsShape, dataType: .float32, name: name.map { $0 + ".weightsVar" })

        let inputShape = input.quadShape!
        let weightsShapeQuad = weightsVar.quadShape!
        let strides = [descriptor.strideInY, descriptor.strideInX].pair!
        let pads = [descriptor.paddingTop, descriptor.paddingLeft, descriptor.paddingBottom, descriptor.paddingRight].quad!
        let dilations = [descriptor.dilationRateInY, descriptor.dilationRateInX].pair!
        let outputPadding = (outputPadding ?? [0, 0]).map(\.intValue).pair!

        let batches = inputShape.0
        let channels = weightsShapeQuad.2 * descriptor.groups
        let height = strides.0 * (inputShape.1 - 1) + outputPadding.0 + ((weightsShapeQuad.0 - 1) * dilations.0 + 1) - pads.0 - pads.2
        let width = strides.1 * (inputShape.2 - 1) + outputPadding.1 + ((weightsShapeQuad.1 - 1) * dilations.1 + 1) - pads.1 - pads.3
        //NHWC
        let outputShape = [batches, height, width, channels].nsnumbers

        let convolution = convolutionTranspose2D(
            input,
            weights: weightsVar,
            outputShape: outputShape,
            descriptor: descriptor,
            name: name.map { $0 + ".convolution" }
        )

        if bias {
            let biasSeq = (0..<convolution.featuresCount()).map { _ in Float32.random(in: biasInitRule.range(forNormal: k)) }
            let biasData = Data(bytes: biasSeq, count: biasSeq.count * MemoryLayout<Float32>.stride)
            let biasVar = variable(with: biasData, shape: outputShape, dataType: .float32, name: name.map { $0 + ".biasVar" })
            let biased = addition(convolution, biasVar, name: name.map { $0 + ".biasedConvolution" })
            return (biased, [weightsVar, biasVar])
        } else {
            return (convolution, [weightsVar, nil])
        }
    }
}

// MARK: Conv2d
extension MPSGraph{
    func addConvolution2D(input: MPSGraphTensor, outChannels: NSNumber,
                          vars: VariablesPair? = nil,
                          weightsInitRule: WeightsInitializationRule<Float32> = .normal,
                          biasInitRule: WeightsInitializationRule<Float32> = .normal,
                          kernelSize: NSNumber, stride: Int = 1,
                          padding: Int = 0, dilation: Int = 1,
                          groups: Int = 1, bias: Bool = true, name: String?) -> TrainableRes {
        let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: stride,
            strideInY: stride,
            dilationRateInX: dilation,
            dilationRateInY: dilation,
            groups: groups,
            paddingLeft: padding,
            paddingRight: padding,
            paddingTop: padding,
            paddingBottom: padding,
            paddingStyle: .explicit,
            dataLayout: .NHWC,
            weightsLayout: .HWIO
        )!

        let weightsShape = [kernelSize, kernelSize, input.shape![3], outChannels]
        let k = Float32(descriptor.groups) / Float32(weightsShape[0].intValue * weightsShape[1].intValue * weightsShape[2].intValue)

        let weightsVar: MPSGraphTensor
        if let weights = vars?.weights {
            weightsVar = weights
        } else {
            let weightsSeq = (0..<weightsShape.featuresCount()).map { _ in Float32.random(in: weightsInitRule.range(forNormal: k) ) }
            let weightsData = Data(bytes: weightsSeq, count: weightsSeq.count * MemoryLayout<Float32>.stride)
            weightsVar = variable(with: weightsData, shape: weightsShape, dataType: .float32, name: name.map { $0 + ".weightsVar" })
        }

        let convolution = convolution2D(
            input,
            weights: weightsVar,
            descriptor: descriptor,
            name: name.map { $0 + ".convolution" }
        )

        if bias {
            let biasVar: MPSGraphTensor
            if let biases = vars?.biases {
                biasVar = biases
            } else {
                let biasSeq = (0..<convolution.featuresCount()).map { _ in Float32.random(in: biasInitRule.range(forNormal: k)) }
                let biasData = Data(bytes: biasSeq, count: biasSeq.count * MemoryLayout<Float32>.stride)
                biasVar = variable(with: biasData, shape: convolution.shape!, dataType: .float32, name: name.map { $0 + ".biasVar" })
            }
            let biased = addition(convolution, biasVar, name: name.map { $0 + ".biasedConvolution" })
            return (biased, [weightsVar, biasVar])
        } else {
            return (convolution, [weightsVar, nil])
        }
    }
}
