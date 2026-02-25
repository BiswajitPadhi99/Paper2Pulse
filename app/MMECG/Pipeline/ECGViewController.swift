import UIKit

// MARK: - Example View Controller
// Shows how to use the ECG pipeline in an iOS app

class ECGViewController: UIViewController {
    
    // MARK: - UI Elements
    
    private let imageView: UIImageView = {
        let iv = UIImageView()
        iv.contentMode = .scaleAspectFit
        iv.backgroundColor = .systemGray6
        iv.translatesAutoresizingMaskIntoConstraints = false
        return iv
    }()
    
    private let processButton: UIButton = {
        let btn = UIButton(type: .system)
        btn.setTitle("Process ECG", for: .normal)
        btn.titleLabel?.font = .boldSystemFont(ofSize: 18)
        btn.translatesAutoresizingMaskIntoConstraints = false
        return btn
    }()
    
    private let progressLabel: UILabel = {
        let label = UILabel()
        label.textAlignment = .center
        label.font = .systemFont(ofSize: 14)
        label.textColor = .secondaryLabel
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    private let resultsTextView: UITextView = {
        let tv = UITextView()
        tv.isEditable = false
        tv.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        tv.backgroundColor = .systemGray6
        tv.layer.cornerRadius = 8
        tv.translatesAutoresizingMaskIntoConstraints = false
        return tv
    }()
    
    private let activityIndicator: UIActivityIndicatorView = {
        let ai = UIActivityIndicatorView(style: .large)
        ai.hidesWhenStopped = true
        ai.translatesAutoresizingMaskIntoConstraints = false
        return ai
    }()
    
    // MARK: - Pipeline
    
    private var pipeline: ECGPipeline?
    private var selectedImage: UIImage?
    
    // MARK: - Lifecycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupPipeline()
    }
    
    private func setupUI() {
        title = "ECG Digitization"
        view.backgroundColor = .systemBackground
        
        view.addSubview(imageView)
        view.addSubview(processButton)
        view.addSubview(progressLabel)
        view.addSubview(resultsTextView)
        view.addSubview(activityIndicator)
        
        NSLayoutConstraint.activate([
            imageView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            imageView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            imageView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
            imageView.heightAnchor.constraint(equalTo: view.heightAnchor, multiplier: 0.3),
            
            processButton.topAnchor.constraint(equalTo: imageView.bottomAnchor, constant: 16),
            processButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            
            progressLabel.topAnchor.constraint(equalTo: processButton.bottomAnchor, constant: 8),
            progressLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            progressLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
            
            resultsTextView.topAnchor.constraint(equalTo: progressLabel.bottomAnchor, constant: 16),
            resultsTextView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            resultsTextView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
            resultsTextView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -16),
            
            activityIndicator.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            activityIndicator.centerYAnchor.constraint(equalTo: view.centerYAnchor)
        ])
        
        processButton.addTarget(self, action: #selector(processButtonTapped), for: .touchUpInside)
        
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(selectImage))
        imageView.isUserInteractionEnabled = true
        imageView.addGestureRecognizer(tapGesture)
        
        progressLabel.text = "Tap image to select ECG"
    }
    
    private func setupPipeline() {
        do {
            pipeline = try ECGPipeline()
            progressLabel.text = "Pipeline ready. Tap image to select ECG"
        } catch {
            progressLabel.text = "Error loading models: \(error.localizedDescription)"
            processButton.isEnabled = false
        }
    }
    
    // MARK: - Actions
    
    @objc private func selectImage() {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .photoLibrary
        present(picker, animated: true)
    }
    
    @objc private func processButtonTapped() {
        guard let image = selectedImage, let pipeline = pipeline else {
            progressLabel.text = "Please select an image first"
            return
        }
        
        processButton.isEnabled = false
        activityIndicator.startAnimating()
        resultsTextView.text = ""
        
        pipeline.processAsync(
            image: image,
            targetSignalLength: 5000,
            progressHandler: { [weak self] progress, message in
                self?.progressLabel.text = message
            },
            completion: { [weak self] result in
                self?.handleResult(result)
            }
        )
    }
    
    private func handleResult(_ result: ECGPipelineResult) {
        activityIndicator.stopAnimating()
        processButton.isEnabled = true
        
        if result.success {
            // Show rectified image
            if let rectifiedCG = result.rectifiedImage {
                imageView.image = UIImage(cgImage: rectifiedCG)
            }
            
            // Show signal statistics
            var text = "✅ ECG Signals Extracted!\n\n"
            text += "Lead       Min(mV)   Max(mV)   Duration\n"
            text += "─────────────────────────────────────────\n"
            
            for leadName in ECGConstants.allLeadNames {
                if let signal = result.signals[leadName] {
                    text += String(format: "%-10s %6.2f    %6.2f    %.1fs\n",
                                   (leadName as NSString).utf8String!,
                                   signal.minValue,
                                   signal.maxValue,
                                   signal.duration)
                }
            }
            
            if let rhythm = result.signals["II-rhythm"] {
                text += String(format: "\n%-10s %6.2f    %6.2f    %.1fs\n",
                               ("II-rhythm" as NSString).utf8String!,
                               rhythm.minValue,
                               rhythm.maxValue,
                               rhythm.duration)
            }
            
            resultsTextView.text = text
            
        } else {
            progressLabel.text = "Processing failed"
            resultsTextView.text = "❌ Error: \(result.errorMessage ?? "Unknown error")"
        }
    }
}

// MARK: - Image Picker Delegate

extension ECGViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    func imagePickerController(
        _ picker: UIImagePickerController,
        didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
    ) {
        picker.dismiss(animated: true)
        
        if let image = info[.originalImage] as? UIImage {
            selectedImage = image
            imageView.image = image
            progressLabel.text = "Image selected. Tap 'Process ECG' to start"
        }
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true)
    }
}

// MARK: - Usage Example

/*
 
 // In your AppDelegate or SceneDelegate:
 
 let viewController = ECGViewController()
 let navigationController = UINavigationController(rootViewController: viewController)
 window?.rootViewController = navigationController
 
 
 // Or programmatic usage without UI:
 
 do {
     let pipeline = try ECGPipeline()
     let image = UIImage(named: "ecg_sample")!
     
     let result = pipeline.process(image: image)
     
     if result.success {
         for (leadName, signal) in result.signals {
             print("\(leadName): \(signal.samples.count) samples, range [\(signal.minValue), \(signal.maxValue)] mV")
         }
     }
 } catch {
     print("Pipeline error: \(error)")
 }
 
 */
