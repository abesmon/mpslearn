//
//  Nets+shorthands.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 27.11.2022.
//

import Foundation
import MetalPerformanceShadersGraph

extension MPSGraph {
    func makeDisc(
        input: MPSGraphTensor,
        vars: [String: VariablesPair]? = nil
    ) -> (
        out: MPSGraphTensor,
        vars: [String: VariablesPair]
    ) {
        let l1 = self.addConvolution2D(input: input, outChannels: 64,
                                       vars: vars?["l1"],
                                       weightsInitRule: .dist(mean: 0, stdev: 0.02),
                                       biasInitRule: .dist(mean: 0, stdev: 0.02),
                                       kernelSize: 4, stride: 2, padding: 1,
                                       bias: false, name: "disc.l1")
        let l1relu = self.leakyReLU(with: l1.out, alpha: 0.2, name: "disc.l1relu")

        let l2 = self.addConvolution2D(input: l1relu, outChannels: 128,
                                       vars: vars?["l2"],
                                       weightsInitRule: .dist(mean: 0, stdev: 0.02),
                                       biasInitRule: .dist(mean: 0, stdev: 0.02),
                                       kernelSize: 4, stride: 2, padding: 1,
                                       bias: false, name: "disc.l2")
        let l2relu = self.leakyReLU(with: l2.out, alpha: 0.2, name: "disc.l2relu")

        let l3 = self.addConvolution2D(input: l2relu, outChannels: 256,
                                       vars: vars?["l3"],
                                       weightsInitRule: .dist(mean: 0, stdev: 0.02),
                                       biasInitRule: .dist(mean: 0, stdev: 0.02),
                                       kernelSize: 4, stride: 2, padding: 1,
                                       bias: false, name: "disc.l3")
        let l3relu = self.leakyReLU(with: l3.out, alpha: 0.2, name:"disc.l3relu")

        let l4 = self.addConvolution2D(input: l3relu, outChannels: 512,
                                       vars: vars?["l4"],
                                       weightsInitRule: .dist(mean: 0, stdev: 0.02),
                                       biasInitRule: .dist(mean: 0, stdev: 0.02),
                                       kernelSize: 4, stride: 2, padding: 1,
                                       bias: false, name: "disc.l4")
        let bn4 = self.trainableBatchNorm(input: l4.out, outChannels: 512, vars: vars?["bn4"], name: "disc.bn4")
        let l4relu = self.leakyReLU(with: bn4.out, alpha: 0.2, name: "disc.l4relu")

        let l5 = self.addConvolution2D(input: l4relu, outChannels: 1,
                                       vars: vars?["l5"],
                                       weightsInitRule: .dist(mean: 0, stdev: 0.02),
                                       biasInitRule: .dist(mean: 0, stdev: 0.02),
                                       kernelSize: 4, stride: 1, padding: 0,
                                       bias: false, name: "disc.l5")
        let out = self.sigmoid(with: l5.out, name: "disc.sigm")
        return (
            out,
            [
                "l1": l1.vars,
                "l2": l2.vars,
                "l3": l3.vars,
                "l4": l4.vars,
                "bn4": bn4.vars,
                "l5": l5.vars,
            ]
        )
    }

    func makeGen(
        input: MPSGraphTensor
    ) -> (
        out: MPSGraphTensor,
        vars: [String: VariablesPair]
    ) {
        let l1 = self.addConvolutionTranspose2D(
            input: input, outChannels: 1024,
            weightsInitRule: .dist(mean: 0, stdev: 0.02),
            biasInitRule: .dist(mean: 0, stdev: 0.02),
            kernelSize: 4, outputPadding: nil,
            bias: true, name: "gen.l1"
        )
        let bn1 = self.trainableBatchNorm(input: l1.out, outChannels: 1024, name: "gen.bn1")
        let l1relu = self.reLU(with: bn1.out, name: "gen.l1relu")

        let l2 = self.addConvolutionTranspose2D(
            input: l1relu, outChannels: 512,
            weightsInitRule: .dist(mean: 0, stdev: 0.02),
            biasInitRule: .dist(mean: 0, stdev: 0.02),
            kernelSize: 4, stride: 2,
            padding: 1, outputPadding: nil,
            bias: true, name: "gen.l2"
        )
        let bn2 = self.trainableBatchNorm(input: l2.out, outChannels: 512, name: "gen.bn2")
        let l2relu = self.reLU(with: bn2.out, name: "gen.l2relu")

        let l3 = self.addConvolutionTranspose2D(
            input: l2relu, outChannels: 256,
            weightsInitRule: .dist(mean: 0, stdev: 0.02),
            biasInitRule: .dist(mean: 0, stdev: 0.02),
            kernelSize: 4, stride: 2,
            padding: 1, outputPadding: nil,
            bias: true, name: "gen.l3"
        )
        let bn3 = self.trainableBatchNorm(input: l3.out, outChannels: 256, name: "gen.bn3")
        let l3relu = self.reLU(with: bn3.out, name: "gen.l3relu")

        let l4 = self.addConvolutionTranspose2D(
            input: l3relu, outChannels: 128,
            weightsInitRule: .dist(mean: 0, stdev: 0.02),
            biasInitRule: .dist(mean: 0, stdev: 0.02),
            kernelSize: 4, stride: 2,
            padding: 1, outputPadding: nil,
            bias: true, name: "gen.l4"
        )
        let bn4 = self.trainableBatchNorm(input: l4.out, outChannels: 128, name: "gen.bn4")
        let l4relu = self.reLU(with: bn4.out, name: "gen.l4relu")

        let l5 = self.addConvolutionTranspose2D(
            input: l4relu, outChannels: 3,
            weightsInitRule: .dist(mean: 0, stdev: 0.02),
            biasInitRule: .dist(mean: 0, stdev: 0.02),
            kernelSize: 4, stride: 2,
            padding: 1, outputPadding: nil,
            bias: true, name: "gen.l5"
        )

        let out = self.tanh(with: l5.out, name: "gen.tanh")

        return (
            out,
            [
                "l1": l1.vars,
                "bn1": bn1.vars,
                "l2": l2.vars,
                "bn2": bn2.vars,
                "l3": l3.vars,
                "bn3": bn3.vars,
                "l4": l4.vars,
                "bn4": bn4.vars,
                "l5": l5.vars,
            ]
        )
    }
}
