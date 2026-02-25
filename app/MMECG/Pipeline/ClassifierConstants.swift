// Auto-generated classifier constants
// Add this to your Constants.swift or create ClassifierConstants.swift

import Foundation

struct ClassifierConstants {
    // Class labels (order matches model output)
    static let labelNames = ["CD", "HYP", "MI", "NORM", "STTC"]
    
    // Classification thresholds (learned during training)
    // Order: CD, HYP, MI, NORM, STTC
    static let thresholds: [Float] = [0.51, 0.5, 0.51, 0.54, 0.54]
    
    // Normalization statistics (per-lead from training data)
    static let signalMean: [Float] = [-0.00127763, -0.00134505, -0.00059723, 0.00062197, -0.00052196, -4.611e-05, -0.00147435, -0.00298645, -0.00458671, -0.00230535, -0.00120006, -0.00126306]
    
    static let signalStd: [Float] = [0.16235043, 0.16215008, 0.16528705, 0.13600624, 0.13458662, 0.13528119, 0.22273302, 0.32375798, 0.31501389, 0.29835835, 0.28519872, 0.23471278]
    
    // Model input shape
    static let expectedSamples = 5000
    static let expectedLeads = 12
    static let samplingRate = 500
    
    // Lead names in order
    static let leadNames = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    // Full class names for display
    static let classFullNames: [String: String] = [
        "CD": "Conduction Disturbance",
        "HYP": "Hypertrophy", 
        "MI": "Myocardial Infarction",
        "NORM": "Normal",
        "STTC": "ST/T Change"
    ]
}
