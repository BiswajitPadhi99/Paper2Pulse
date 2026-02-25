import SwiftUI

@main
struct MMECGApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

// MARK: - Benchmark Access (for development only)
// To access BenchmarkView during development, temporarily replace
// ContentView() above with BenchmarkView()
