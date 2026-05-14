using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using System.Collections.Generic;
using Azure;
using Azure.AI.FormRecognizer.DocumentAnalysis;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace AI_Classifier_Project
{
    class Program
    {
        // --- AZURE Configuration---
        const string azureEndpoint = "PASTE_YOUR_AZURE_ENDPOINT";
        const string azureKey = "PASTE_YOUR_KEY_HERE";
        const string azureModelId = "PAST_YOUR_MODEL_ID"; 

        [STAThread]
        static async Task Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            Console.WriteLine("=== AI Dual Classifier System ===\n");

            // 1. Choose a folder
            string sourcePath = ""; //Initialize an empty string to store the selected folder path

            // Create a new thread to run the Folder Selection UI. 
            // We use a separate thread because FolderBrowserDialog requires an STA (Single Threaded Apartment) state.
            Thread t = new Thread(() => {
                // Instantiate the standard Windows folder browser dialog
                using (FolderBrowserDialog fbd = new FolderBrowserDialog())
                {
                    fbd.Description = "Please select the folder containing the images"; // Set the prompt message that appears at the top of the window
                    if (fbd.ShowDialog() == DialogResult.OK) // Show the dialog and check if the user actually clicked "OK"
                    {
                        sourcePath = fbd.SelectedPath; // Store the path of the folder chosen by the user
                    }
                }
            });
            t.SetApartmentState(ApartmentState.STA); // Set the thread's apartment state to STA, which is mandatory for Windows Forms components to work correctly
            t.Start(); // Start the thread to display the window
            t.Join(); // Wait for the thread to finish

            if (string.IsNullOrEmpty(sourcePath)) return; // Check if the user canceled the folder selection; if so, exit the program

            // 2. Preparation Phase
            string donePath = Path.Combine(sourcePath, "Done"); // Define the path for the "Done" folder where processed images will be moved
            if (!Directory.Exists(donePath)) Directory.CreateDirectory(donePath); // Create the "Done" directory if it doesn't already exist

            // Locate the labels file and the ONNX model in the application's base directory
            string labelsPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "labels.txt");
            string onnxPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "model.onnx");

            string[] labels = File.ReadAllLines(labelsPath); // Load all classification labels from the text file into an array
            
            // Initialize the Azure Document Intelligence client using the provided endpoint and key
            var azureClient = new DocumentAnalysisClient(new Uri(azureEndpoint), new AzureKeyCredential(azureKey));
            // Initialize the ONNX Runtime session to load the local AI model for inference
            using var onnxSession = new InferenceSession(onnxPath);

            // Prepare a StringBuilder to store CSV results
            var csvResults = new StringBuilder();
            // Add the CSV header using semicolons (;) as delimiters for compatibility with European Excel settings
            csvResults.AppendLine("File Name;Azure Class;Azure Confidence;ONNX Class;ONNX Confidence");

            // 3. Processing Phase

            // Get all files from the source folder and subfolders and filter them to include only .jpg and .png images
            // StringComparison.OrdinalIgnoreCase ensures it finds both .JPG and .jpg
            var files = Directory.GetFiles(sourcePath, "*.*", SearchOption.AllDirectories)
            .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) || 
                        f.EndsWith(".png", StringComparison.OrdinalIgnoreCase)).ToList();

            // Loop through each image file found in the list
            foreach (var filePath in files)
            {
                string fileName = Path.GetFileName(filePath); // Extract only the file name (e.g., "invoice.jpg") from the full path
                Console.Write($"Επεξεργασία: {fileName}... "); // Print a message to the console to show which file is currently being processed

                try
                {
                    // 1. Call Azure AI (Cloud)
                    // Sends the image to Azure Document Intelligence for custom classification
                    var azureData = await GetAzureResult(azureClient, azureModelId, filePath);

                    // 2. Call ONNX Model (Local)
                    // Processes the same image using the local ONNX model and the provided labels
                    var onnxData = GetOnnxResult(onnxSession, filePath, labels);

                    // 3. Prepare CSV Data Row
                    // Formats the results into a single string separated by semicolons (;)
                    // :P2 formats the confidence scores as percentages (e.g., 95.50%)
                    string row = $"{fileName};{azureData.Label};{azureData.Score:P2};{onnxData.Label};{onnxData.Score:P2}";
                    csvResults.AppendLine(row);

                    // 4. Move Processed File
                    // Moves the image to the "Done" folder to keep the source folder clean
                    // The 'true' parameter allows overwriting if a file with the same name exists in "Done"
                    File.Move(filePath, Path.Combine(donePath, fileName), true);

                    Console.WriteLine("OK"); // Indicate successful processing for the current file
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"ΣΦΑΛΜΑ: {ex.Message}"); // If anything goes wrong (network error, file locked, etc.), catch the error and show it
                }
            }

            // 4. Save CSV
            File.WriteAllText(Path.Combine(sourcePath, "results_summary.csv"), csvResults.ToString(), Encoding.UTF8);
            Console.WriteLine("\nΤΕΛΟΣ! Το αρχείο CSV δημιουργήθηκε.");
        }

        // Method to get classification results from Azure Cloud Service
        static async Task<(string Label, float Score)> GetAzureResult(DocumentAnalysisClient client, string modelId, string path)
        {
            using var stream = File.OpenRead(path);// Open the image file as a stream for uploading
            var operation = await client.ClassifyDocumentAsync(WaitUntil.Completed, modelId, stream); // Start the asynchronous classification process and wait until it completes
            var doc = operation.Value.Documents.FirstOrDefault(); // Retrieve the first document result from the operation
            // Return the detected Label (DocumentType) and the Confidence score
            // If no document is found, return "Unknown" and 0 score
            return (doc?.DocumentType ?? "Unknown", (float)(doc?.Confidence ?? 0));
        }

        // Method to get classification results from the local ONNX model
        static (string Label, float Score) GetOnnxResult(InferenceSession session, string path, string[] labels)
        {
            // Load the image using SixLabors.ImageSharp library in Rgb24 format
            using var image = SixLabors.ImageSharp.Image.Load<Rgb24>(path);
            
            // Pre-processing: Auto-orient and resize the image to 300x300 pixels to match model input requirements
            image.Mutate(x => x.AutoOrient().Resize(new ResizeOptions { 
                Size = new SixLabors.ImageSharp.Size(300, 300), 
                Mode = ResizeMode.Stretch 
            }));

            // Create a 4D Tensor (Batch size: 1, Channels: 3, Height: 300, Width: 300)
            var inputTensor = new DenseTensor<float>(new[] { 1, 3, 300, 300 });

            // Extract RGB values from each pixel and normalize them into the tensor
            image.ProcessPixelRows(accessor => {
                for (int y = 0; y < accessor.Height; y++) {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++) {
                        inputTensor[0, 0, y, x] = row[x].R;
                        inputTensor[0, 1, y, x] = row[x].G;
                        inputTensor[0, 2, y, x] = row[x].B;
                    }
                }
            });

            // Prepare the inputs for the ONNX session
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(session.InputMetadata.Keys.First(), inputTensor) };
            using var results = session.Run(inputs); // Run the local inference (the actual AI "thinking" process)
            var output = results.First().AsEnumerable<float>().ToArray(); // Convert the output probabilities into a float array
            float maxScore = output.Max(); // Find the highest probability score in the output array
            return (labels[Array.IndexOf(output, maxScore)], maxScore); // Map the highest score index to its corresponding label from the labels.txt file
        }
    }
}