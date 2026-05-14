# Dual-Engine AI Document Classifier

A robust C# console application designed for high-accuracy document classification. This system utilizes a **dual-engine approach**, cross-validating results between **Azure AI Document Intelligence (Cloud)** and a **local ONNX Runtime model**. It is specifically optimized for financial documents like bank statements and invoices.

---

## 🛠 Features
- **Hybrid AI Validation**: Combines the power of Cloud-based AI with the speed of Local Edge AI.
- **Recursive Directory Scanning**: Automatically discovers all `.jpg` and `.png` files in the source folder and all its subfolders.
- **Excel-Ready Reporting**: Generates a comprehensive `;` delimited CSV report containing predictions and confidence scores.
- **Workflow Automation**: Successfully processed files are automatically moved to a `Done` directory to prevent duplicate processing.
- **Advanced Pre-processing**: Implements image normalization, auto-orientation, and resizing (300x300) for consistent local inference.

---

## 📂 Project Structure
```text
├── Program.cs           # Main application logic & AI orchestration
├── AI_Classifier.csproj # Project configuration & NuGet dependencies
├── labels.txt           # REQUIRED: Plain text list of category labels
├── model.onnx           # REQUIRED: Pre-trained ONNX model for local inference
└── README.md            # Documentation and setup guide
```
---

## ⚙️ Configuration & Setup
To get this project running, you need to configure your credentials and local assets:

- **Azure AI Setup**: Open `Program.cs` and replace the placeholders with your actual Azure resource details:
  ```csharp
  const string azureEndpoint = "https://YOUR_RESOURCE_[NAME.cognitiveservices.azure.com/](https://NAME.cognitiveservices.azure.com/)";
  const string azureKey = "YOUR_SECRET_API_KEY";
  const string azureModelId = "YOUR_CUSTOM_CLASSIFICATION_MODEL_ID";```

- **Local ONNX Model Setup**: Ensure the `model.onnx` file is present in the application's execution directory (e.g., `bin/Debug/net8.0/`).

- **Labels Configuration**: The `labels.txt` file must contain your class names, one per line, in the exact order the model was trained on.

- **Required NuGet Packages**:
  - `Azure.AI.FormRecognizer`
  - `Microsoft.ML.OnnxRuntime`
  - `SixLabors.ImageSharp`
  - `System.Drawing.Common`

---

## 🚀 How to Run
- **1. Clone the Repository**:
  ```bash
  git clone [https://github.com/YOUR_USERNAME/AI-Document-Classifier.git](https://github.com/YOUR_USERNAME/AI-Document-Classifier.git)
  cd AI-Document-Classifier ```

- **2. Restore and Build**:
  ```bash
dotnet restore
dotnet build ```

- **3. Execute the Application**:
```bash
dotnet run```

- **4. Interactive Usage**: A folder browser dialog will open. Select the root folder containing your images. The application will process all images and output the results to `results_summary.csv`.

---

## 📊 Output Format (CSV)
The report is saved as `results_summary.csv` with the following schema:

| Column | Description |
| :--- | :--- |
| **File Name** | Name of the processed image file. |
| **Azure Class** | Category predicted by the Azure Cloud model. |
| **Azure Confidence** | Probability score from Azure (Formatted as %). |
| **ONNX Class** | Category predicted by the local ONNX model. |
| **ONNX Confidence** | Probability score from the local model (Formatted as %). |
| **Date Processed** | Timestamp of the analysis. |

---

## 🔒 Security Note
- **DO NOT** commit your actual Azure API keys or endpoints to a public GitHub repository.
- **Placeholders**: This project uses placeholder constants for safety.
- **Best Practice**: For production environments, it is highly recommended to use Environment Variables or Azure Key Vault.

---

## 📄 License
This project is licensed under the **MIT License**. Feel free to use, modify, and distribute as needed.
