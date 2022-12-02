//
//  FolderDataset.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 27.11.2022.
//

import AppKit

class FolderDataset: Dataset {
    private let folder: URL
    private let batchSize: Int

    private let filePaths: [URL]

    private var counter: Int = 0
    private var shuffleMode = true

    init(folder: URL, batchSize: Int = 1) {
        self.folder = folder
        self.batchSize = batchSize

        var filePaths = [URL]()
        for subapth in FileManager.default.subpaths(atPath: folder.path)! {
            let newPath = folder.appending(path: subapth)
            guard !newPath.isDirectory else { continue }
            filePaths.append(newPath)
        }
        let maxFiles = Int((Double(filePaths.count) / Double(batchSize)).rounded(.towardZero)) * batchSize
        self.filePaths = Array(filePaths[0..<maxFiles])
    }

    override func next() -> [Float32]? {
        if counter == filePaths.count {
            counter = 0
            return nil
        }

        var buffer = [Float32]()

        for _ in 0..<batchSize {
            let index = shuffleMode ? filePaths.indices.randomElement()! : counter
            let imagePath = filePaths[index % filePaths.count]
            buffer += NSImage(contentsOfFile: imagePath.path)?.makeRGBBuffer(balance: (-0.5, 2)) ?? []
            counter += 1
        }
        return buffer
    }
}

extension URL {
    var isDirectory: Bool {
       (try? resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
    }
}
