//
//  ContentView.swift
//  mpslearn
//
//  Created by Алексей Лысенко on 13.11.2022.
//

import SwiftUI

struct ContentView: View {
    @State private var graphLearn = AnyView(EmptyView())
    @State private var simple  = AnyView(EmptyView())
    @State private var gan = AnyView(EmptyView())
    
    var body: some View {
        TabView {
            graphLearn
                .padding()
                .tabItem {
                    Label("Graph Learn", systemImage: "chart.xyaxis.line")
                }
            
            simple
                .padding()
                .tabItem {
                    Label("Simple", systemImage: "function")
                }

            gan
                .padding()
                .tabItem {
                    Label("GAN", systemImage: "photo")
                }
        }
        .onAppear {
            setupGraphLearn()
            setupSimple()
            setupGAN()
        }
    }
    
    private func setupGraphLearn() {
        let model = GraphLearnModel.sinInit()
        let interactor = GraphLearnInteractor(model: model)
        graphLearn = AnyView(GraphLearnView(interactor: interactor, model: model))
    }
    
    private func setupSimple() {
        let simpleInteractor = SimpleInteractor()
        simple = AnyView(SimpleView(interactor: simpleInteractor))
    }

    private func setupGAN() {
        let ganInteractor = GANLearnInteractor()
        gan = AnyView(GANLearnView(
            model: GANLearnModel(),
            interactor: ganInteractor
        ))
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
