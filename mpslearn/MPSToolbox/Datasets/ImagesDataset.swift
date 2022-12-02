//
//  ImagesDataset.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 25.11.2022.
//

import AppKit

class ImagesDataset: Dataset {
    private let imageNames = ["squirtle", "charmeleon"]

    private var counter: Int = 0
    private var shuffleMode = true

    override func next() -> [Float32]? {
        if counter == imageNames.count {
            counter = 0
            return nil
        }
        let index = shuffleMode ? imageNames.indices.randomElement()! : counter
        let imageName = imageNames[index % imageNames.count]
        counter += 1
        return NSImage(named: imageName)?.makeRGBBuffer(balance: (-0.5, 2)) ?? []
    }
}

