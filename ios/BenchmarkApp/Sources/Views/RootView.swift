import SwiftUI

struct RootView: View {
    @EnvironmentObject private var session: AppSession

    var body: some View {
        TabView {
            NavigationStack {
                RunView()
            }
            .tabItem { Label("Run", systemImage: "play.circle") }

            NavigationStack {
                CameraVLMView()
            }
            .tabItem { Label("Camera", systemImage: "camera.viewfinder") }

            NavigationStack {
                HistoryView()
            }
            .tabItem { Label("History", systemImage: "list.bullet.rectangle") }

            NavigationStack {
                AboutView()
            }
            .tabItem { Label("About", systemImage: "info.circle") }
        }
    }
}
