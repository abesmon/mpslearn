//
//  FileManager+checkpoints.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 30.11.2022.
//

import Foundation

extension FileManager {
    func urlForCheckpintsDirectory() throws -> URL {
        let docsUrl = try self.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        return docsUrl.appending(path: "checkpoints", directoryHint: .isDirectory)
    }
}
