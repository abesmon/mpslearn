//
//  SimpleView.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 20.11.2022.
//

import SwiftUI

struct SimpleView: View {
    private let interactor: SimpleInteractor?
    
    init(interactor: SimpleInteractor?) {
        self.interactor = interactor
    }
    
    var body: some View {
        HStack {
            VStack {
                Image(systemName: "globe")
                    .imageScale(.large)
                    .foregroundColor(.accentColor)
                Text("Hello, world!")
            }
            
            Button("simple calc") {
                interactor?.simpleCalc()
            }
            
            Button("Gelu test") {
                interactor?.testGelu()
            }
            
            Button("test reduction sum") {
                interactor?.testResduction()
            }

            Button("test logs") {
                interactor?.testLogStuff()
            }

            Button("test randoms") {
                interactor?.testRandoms()
            }
        }
    }
}

