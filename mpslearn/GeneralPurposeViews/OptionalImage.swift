//
//  OptionalImage.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 30.11.2022.
//

import SwiftUI

struct OptionalImage: View {
    let cgImage: CGImage?
    let width: CGFloat
    let height: CGFloat

    var body: some View {
        Group {
            if let image = cgImage {
                Image(decorative: image, scale: 1)
            } else {
                Rectangle()
                    .fill(.gray)
                    .frame(width: width, height: height)
            }
        }
        .cornerRadius(5)
    }
}
