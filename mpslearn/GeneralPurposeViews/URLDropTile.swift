//
//  URLDropTile.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 30.11.2022.
//
import SwiftUI

struct URLDropTile: View {
    @State private var dropTarg = false

    let color: Color
    let text: String
    let iconSystemName: String
    let callback: (URL) -> Void

    var body: some View {
        Rectangle()
            .fill(color)
            .frame(width: 64, height: 64)
            .overlay(content: {
                VStack {
                    Text(text)
                        .font(.caption2)

                    Image(systemName: iconSystemName)
                }
                .padding(2)
            })
            .opacity(dropTarg ? 0.5 : 1)
            .cornerRadius(5)
            .onDrop(of: [.fileURL], isTargeted: $dropTarg) { providers in
                var loaded = false
                for provider in providers {
                    if provider.canLoadObject(ofClass: URL.self) {
                        _ = provider.loadObject(ofClass: URL.self) { url, error in
                            guard let url = url else { return }
                            callback(url)
                        }
                        loaded = loaded || true
                    }
                }
                return loaded
            }
    }
}
