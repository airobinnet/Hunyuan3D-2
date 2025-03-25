using UnityEngine;
using UnityEditor;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Security;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using UnityEngine.Networking;
using GLTFast;
using UnityEditor.SceneManagement;

namespace AIRobin.ModelGenerator
{
    public class ModelGeneratorWindow : EditorWindow
    {
        // Server settings
        private string serverUrl = "http://localhost:8081";
        
        // Model selection
        private string selectedModel = "Hunyuan3D-2mini-Turbo";
        private string[] modelOptions = new string[] {
            "Hunyuan3D-2mini-Turbo",
            "Hunyuan3D-2mini-Fast",
            "Hunyuan3D-2mini",
            "Hunyuan3D-2-Turbo",
            "Hunyuan3D-2-Fast",
            "Hunyuan3D-2"
        };
        
        // Mapping of friendly names to actual model paths and subfolders
        private Dictionary<string, (string, string)> modelPathMapping = new Dictionary<string, (string, string)>() {
            { "Hunyuan3D-2mini-Turbo", ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo") },
            { "Hunyuan3D-2mini-Fast", ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-fast") },
            { "Hunyuan3D-2mini", ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini") },
            { "Hunyuan3D-2-Turbo", ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0-turbo") },
            { "Hunyuan3D-2-Fast", ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0-fast") },
            { "Hunyuan3D-2", ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0") }
        };
        
        // Multiview model selection
        private string selectedMVModel = "Hunyuan3D-2mv-Turbo";
        private string[] mvModelOptions = new string[] {
            "Hunyuan3D-2mv-Turbo",
            "Hunyuan3D-2mv-Fast",
            "Hunyuan3D-2mv"
        };
        
        // Mapping of multiview model friendly names to paths and subfolders
        private Dictionary<string, (string, string)> mvModelPathMapping = new Dictionary<string, (string, string)>() {
            { "Hunyuan3D-2mv-Turbo", ("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv-turbo") },
            { "Hunyuan3D-2mv-Fast", ("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv-fast") },
            { "Hunyuan3D-2mv", ("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv") }
        };
        
        // Input options
        private bool useImage = true;
        private Texture2D inputImage;
        private string inputText = "";
        
        // Multiview support
        private bool useMultiview = false;
        private Dictionary<string, Texture2D> multiviewImages = new Dictionary<string, Texture2D>() {
            { "front", null },
            { "left", null },
            { "back", null }
        };
        private Vector2 multiviewScrollPosition = Vector2.zero;
        
        // Generation parameters
        private int seed = 1234;
        private float guidanceScale = 5.0f;
        private int octreeResolution = 128;
        private int numInferenceSteps = 5;
        private bool generateTexture = false;
        private int faceCount = 5000;
        
        // Multiview-specific parameters
        private int numChunks = 20000;
        
        // Status
        private string status = "Ready";
        private string detailedStatus = "";
        private string generatedModelPath = "";
        private GameObject generatedModel;
        private string currentUid = "";
        private bool isGenerating = false;
        private bool cancelRequested = false;
        private EditorCoroutine activeCoroutine;
        
        // Progress tracking
        private float currentProgress = 0f;
        private string currentPhase = "";
        private int pollCount = 0;
        private int maxPollCount = 9999;
        private DateTime generationStartTime;
        
        // GUI Styles
        private GUIStyle headerStyle;
        private GUIStyle subHeaderStyle;
        private GUIStyle statusStyle;
        private GUIStyle errorStyle;
        private GUIStyle successStyle;
        private bool stylesInitialized = false;
        
        // Categories foldout states
        private bool showServerSettings = true;
        private bool showInputOptions = true;
        private bool showGenerationParams = true;

        [MenuItem("Tools/3D Model Generator")]
        public static void ShowWindow()
        {
            var window = GetWindow<ModelGeneratorWindow>("3D Model Generator");
            window.minSize = new Vector2(400, 650);
            window.Show();
        }

        private void OnEnable()
        {
            // Always bypass certificate validation for simplicity
            ServicePointManager.ServerCertificateValidationCallback = CertificateValidationCallback;
        }
        
        private void InitializeStyles()
        {
            if (stylesInitialized) return;
            
            headerStyle = new GUIStyle(EditorStyles.boldLabel);
            headerStyle.fontSize = 14;
            headerStyle.margin = new RectOffset(0, 0, 10, 5);
            
            subHeaderStyle = new GUIStyle(EditorStyles.boldLabel);
            subHeaderStyle.fontSize = 12;
            subHeaderStyle.margin = new RectOffset(0, 0, 5, 3);
            
            statusStyle = new GUIStyle(EditorStyles.label);
            statusStyle.wordWrap = true;
            
            errorStyle = new GUIStyle(EditorStyles.label);
            errorStyle.normal.textColor = Color.red;
            errorStyle.wordWrap = true;
            
            successStyle = new GUIStyle(EditorStyles.label);
            successStyle.normal.textColor = Color.green;
            successStyle.wordWrap = true;
            
            stylesInitialized = true;
        }

        private bool CertificateValidationCallback(object sender, X509Certificate certificate, X509Chain chain, SslPolicyErrors sslPolicyErrors)
        {
            // Always return true to bypass certificate validation
            return true;
        }

        private void OnGUI()
        {
            InitializeStyles();
            
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("3D Model Generator", headerStyle);
            EditorGUILayout.Space();

            // Server settings
            showServerSettings = EditorGUILayout.Foldout(showServerSettings, "Server Settings", true);
            if (showServerSettings)
            {
                EditorGUI.indentLevel++;
                EditorGUILayout.BeginHorizontal();
                serverUrl = EditorGUILayout.TextField("Server URL", serverUrl);
                
                GUI.enabled = !isGenerating;
                if (GUILayout.Button("Test Connection", GUILayout.Width(120)))
                {
                    TestServerConnection();
                }
                GUI.enabled = true;
                
                EditorGUILayout.EndHorizontal();
                
                // Add model selection dropdown
                EditorGUILayout.Space(5);
                EditorGUILayout.LabelField("Model Selection", subHeaderStyle);
                
                // For multiview mode, show multiview models
                if (useMultiview)
                {
                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.PrefixLabel("Multiview Model");
                    int selectedMVModelIndex = Array.IndexOf(mvModelOptions, selectedMVModel);
                    int newMVModelIndex = EditorGUILayout.Popup(selectedMVModelIndex, mvModelOptions);
                    if (newMVModelIndex != selectedMVModelIndex)
                    {
                        selectedMVModel = mvModelOptions[newMVModelIndex];
                    }
                    EditorGUILayout.EndHorizontal();
                    
                    // Show a help box with model description
                    EditorGUILayout.HelpBox($"Selected multiview model: {selectedMVModel}\nSpecialized for multi-view 3D reconstruction.", MessageType.Info);
                }
                else
                {
                    // Show regular model options for single view mode
                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.PrefixLabel("Shape Model");
                    int selectedModelIndex = Array.IndexOf(modelOptions, selectedModel);
                    int newModelIndex = EditorGUILayout.Popup(selectedModelIndex, modelOptions);
                    if (newModelIndex != selectedModelIndex)
                    {
                        selectedModel = modelOptions[newModelIndex];
                    }
                    EditorGUILayout.EndHorizontal();
                    
                    // Show a help box with model description
                    string modelSize = selectedModel.Contains("mini") ? "smaller (0.6B)" : "larger (1.1B)";
                    string modelSpeed = selectedModel.Contains("Turbo") ? "fastest" : 
                                        (selectedModel.Contains("Fast") ? "faster" : "standard speed");
                    
                    EditorGUILayout.HelpBox($"Selected model: {selectedModel}\nSize: {modelSize}, Speed: {modelSpeed}", MessageType.Info);
                }
                
                EditorGUI.indentLevel--;
                EditorGUILayout.Space();
            }

            // Input options
            showInputOptions = EditorGUILayout.Foldout(showInputOptions, "Input Options", true);
            if (showInputOptions)
            {
                EditorGUI.indentLevel++;
                
                // Model type selection (Single view, Multiview, Text)
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.PrefixLabel("Generation Mode");
                
                bool oldUseImage = useImage;
                bool oldUseMultiview = useMultiview;
                
                // Create a 3-button toggle group
                useMultiview = GUILayout.Toggle(!useImage && useMultiview, "Multiview", EditorStyles.miniButtonLeft, GUILayout.Height(20));
                useImage = GUILayout.Toggle(useImage && !useMultiview, "Single Image", EditorStyles.miniButtonMid, GUILayout.Height(20));
                bool useText = GUILayout.Toggle(!useImage && !useMultiview, "Text", EditorStyles.miniButtonRight, GUILayout.Height(20));
                
                // Logic to ensure only one option is selected
                if (useImage && oldUseMultiview != useMultiview)
                {
                    useMultiview = false;
                }
                else if (useMultiview && oldUseImage != useImage)
                {
                    useImage = false;
                }
                else if (useText)
                {
                    useImage = false;
                    useMultiview = false;
                }
                
                EditorGUILayout.EndHorizontal();
                
                EditorGUILayout.Space();
                
                if (useMultiview)
                {
                    EditorGUILayout.LabelField("Multiview Images", subHeaderStyle);
                    EditorGUILayout.HelpBox("Upload images from different viewpoints for better 3D reconstruction.", MessageType.Info);
                    
                    // Scroll view for multiple image fields
                    multiviewScrollPosition = EditorGUILayout.BeginScrollView(multiviewScrollPosition, GUILayout.Height(300));
                    
                    // Display image fields for each view
                    foreach (var view in new string[] { "front", "left", "back" })
                    {
                        EditorGUILayout.Space(5);
                        
                        // Header for the view
                        EditorGUILayout.LabelField(char.ToUpper(view[0]) + view.Substring(1) + " View", EditorStyles.boldLabel);
                        
                        // Image field
                        EditorGUILayout.BeginHorizontal();
                        EditorGUILayout.PrefixLabel("Select Image");
                        multiviewImages[view] = (Texture2D)EditorGUILayout.ObjectField(
                            multiviewImages[view], 
                            typeof(Texture2D), 
                            false,
                            GUILayout.Width(150)
                        );
                        EditorGUILayout.EndHorizontal();
                        
                        // Preview for this view
                        if (multiviewImages[view] != null)
                        {
                            // Create a preview area with fixed height
                            Rect previewAreaRect = EditorGUILayout.GetControlRect(GUILayout.Height(120));
                            
                            // Calculate aspect ratio
                            float aspectRatio = (float)multiviewImages[view].width / multiviewImages[view].height;
                            
                            // Calculate preview dimensions while maintaining aspect ratio
                            float previewHeight = previewAreaRect.height - 10; // margins
                            float previewWidth = previewHeight * aspectRatio;
                            
                            // If too wide, adjust width and recalculate height
                            if (previewWidth > previewAreaRect.width - 20)
                            {
                                previewWidth = previewAreaRect.width - 20;
                                previewHeight = previewWidth / aspectRatio;
                            }
                            
                            // Center the preview
                            float xOffset = (previewAreaRect.width - previewWidth) / 2;
                            float yOffset = (previewAreaRect.height - previewHeight) / 2;
                            
                            // Draw a preview background for better visibility
                            EditorGUI.DrawRect(
                                new Rect(previewAreaRect.x + xOffset - 2, previewAreaRect.y + yOffset - 2, 
                                        previewWidth + 4, previewHeight + 4), 
                                new Color(0.2f, 0.2f, 0.2f, 1f));
                            
                            // Draw the texture with proper aspect ratio
                            Rect previewRect = new Rect(
                                previewAreaRect.x + xOffset,
                                previewAreaRect.y + yOffset,
                                previewWidth,
                                previewHeight
                            );
                            
                            EditorGUI.DrawPreviewTexture(previewRect, multiviewImages[view], null, ScaleMode.ScaleToFit);
                        }
                        
                        // Separator
                        EditorGUILayout.Space(5);
                        EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
                    }
                    
                    EditorGUILayout.EndScrollView();
                    
                    // Show warning if no images are selected
                    int imageCount = 0;
                    foreach (var img in multiviewImages.Values)
                    {
                        if (img != null) imageCount++;
                    }
                    
                    if (imageCount == 0)
                    {
                        EditorGUILayout.HelpBox("Please upload at least one image. The front view is recommended.", MessageType.Warning);
                    }
                }
                else if (useImage)
                {
                    EditorGUILayout.LabelField("Input Image", subHeaderStyle);
                    
                    // Image field with proper label
                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.PrefixLabel("Select Image");
                    inputImage = (Texture2D)EditorGUILayout.ObjectField(
                        inputImage, 
                        typeof(Texture2D), 
                        false,
                        GUILayout.Width(150)
                    );
                    EditorGUILayout.EndHorizontal();
                    
                    // Preview area
                    if (inputImage != null)
                    {
                        EditorGUILayout.Space(5);
                        
                        // Create a preview area with fixed height
                        Rect previewAreaRect = EditorGUILayout.GetControlRect(GUILayout.Height(150));
                        
                        // Calculate aspect ratio
                        float aspectRatio = (float)inputImage.width / inputImage.height;
                        
                        // Calculate preview dimensions while maintaining aspect ratio
                        float previewHeight = previewAreaRect.height - 10; // margins
                        float previewWidth = previewHeight * aspectRatio;
                        
                        // If too wide, adjust width and recalculate height
                        if (previewWidth > previewAreaRect.width - 20)
                        {
                            previewWidth = previewAreaRect.width - 20;
                            previewHeight = previewWidth / aspectRatio;
                        }
                        
                        // Center the preview
                        float xOffset = (previewAreaRect.width - previewWidth) / 2;
                        float yOffset = (previewAreaRect.height - previewHeight) / 2;
                        
                        // Draw a preview background for better visibility
                        EditorGUI.DrawRect(
                            new Rect(previewAreaRect.x + xOffset - 2, previewAreaRect.y + yOffset - 2, 
                                    previewWidth + 4, previewHeight + 4), 
                            new Color(0.2f, 0.2f, 0.2f, 1f));
                        
                        // Draw the texture with proper aspect ratio
                        Rect previewRect = new Rect(
                            previewAreaRect.x + xOffset,
                            previewAreaRect.y + yOffset,
                            previewWidth,
                            previewHeight
                        );
                        
                        EditorGUI.DrawPreviewTexture(previewRect, inputImage, null, ScaleMode.ScaleToFit);
                        
                        // Show image info
                        EditorGUILayout.HelpBox($"Image size: {inputImage.width}x{inputImage.height}px. Large images will be automatically resized while preserving aspect ratio.", MessageType.Info);
                    }
                    else
                    {
                        // Show instructions when no image selected
                        EditorGUILayout.HelpBox("Please select an image to use as reference for the 3D model generation.", MessageType.Info);
                    }
                }
                else
                {
                    EditorGUILayout.LabelField("Text Prompt", subHeaderStyle);
                    inputText = EditorGUILayout.TextArea(inputText, GUILayout.Height(80));
                    
                    EditorGUILayout.HelpBox("Enter a detailed description of the 3D model you want to generate.", MessageType.Info);
                }
                
                EditorGUI.indentLevel--;
                EditorGUILayout.Space();
            }

            // Generation parameters
            showGenerationParams = EditorGUILayout.Foldout(showGenerationParams, "Generation Parameters", true);
            if (showGenerationParams)
            {
                EditorGUI.indentLevel++;
                
                // Basic parameters
                EditorGUILayout.BeginHorizontal();
                
                EditorGUILayout.BeginVertical();
                seed = EditorGUILayout.IntField("Seed", seed);
                EditorGUILayout.EndVertical();
                
                EditorGUILayout.BeginVertical();
                if (GUILayout.Button("Random Seed", GUILayout.Height(18)))
                {
                    seed = UnityEngine.Random.Range(0, 99999);
                    Repaint();
                }
                EditorGUILayout.EndVertical();
                
                EditorGUILayout.EndHorizontal();
                
                EditorGUILayout.Space(5);
                
                // Sliders with more informative labels
                EditorGUILayout.LabelField("Guidance Scale (how closely to follow input)");
                guidanceScale = EditorGUILayout.Slider(guidanceScale, 1.0f, 10.0f);
                
                EditorGUILayout.Space(5);
                
                // Resolution parameter - adjust range based on mode
                if (useMultiview)
                {
                    EditorGUILayout.LabelField("Resolution (higher = more detail but slower)");
                    octreeResolution = EditorGUILayout.IntSlider(octreeResolution, 128, 512);
                    
                    EditorGUILayout.Space(5);
                    
                    EditorGUILayout.LabelField("Number of Chunks (higher = more detail but slower)");
                    numChunks = EditorGUILayout.IntSlider(numChunks, 10000, 50000);
                    
                    EditorGUILayout.Space(5);
                    
                    EditorGUILayout.LabelField("Inference Steps (higher = better quality but slower)");
                    // For multiview, default is 50 as in the example
                    numInferenceSteps = EditorGUILayout.IntSlider(numInferenceSteps, 20, 100);
                    
                    EditorGUILayout.HelpBox("Multiview generation takes longer but produces more accurate models.", MessageType.Info);
                }
                else
                {
                    EditorGUILayout.LabelField("Resolution (higher = more detail but slower)");
                    octreeResolution = EditorGUILayout.IntSlider(octreeResolution, 64, 256);
                    
                    EditorGUILayout.Space(5);
                    
                    EditorGUILayout.LabelField("Inference Steps (higher = better quality but slower)");
                    // For single view with FlashVDM, 5 steps is recommended
                    numInferenceSteps = EditorGUILayout.IntSlider(numInferenceSteps, 1, 10);
                }
                
                EditorGUILayout.Space(10);
                
                // Texture options
                EditorGUILayout.BeginHorizontal();
                generateTexture = EditorGUILayout.Toggle("Generate Texture", generateTexture);
                
                // Show a warning about texture generation time
                if (generateTexture)
                {
                    EditorGUILayout.HelpBox("Texture generation takes additional time", MessageType.Info);
                }
                EditorGUILayout.EndHorizontal();
                
                if (generateTexture)
                {
                    EditorGUILayout.LabelField("Face Count (lower = faster texturing)");
                    faceCount = EditorGUILayout.IntSlider(faceCount, 5000, 40000);
                }
                
                EditorGUI.indentLevel--;
                EditorGUILayout.Space();
            }

            // Generation controls
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Generation", headerStyle);
            
            if (isGenerating)
            {
                // Show elapsed time
                TimeSpan elapsed = DateTime.Now - generationStartTime;
                string elapsedTime = string.Format("{0:00}:{1:00}", elapsed.Minutes, elapsed.Seconds);
                
                // Show phase info
                EditorGUILayout.LabelField($"Phase: {currentPhase}");
                EditorGUILayout.LabelField($"Elapsed time: {elapsedTime}");
                
                // Progress bar
                EditorGUI.ProgressBar(EditorGUILayout.GetControlRect(GUILayout.Height(24)), 
                                     currentProgress, 
                                     $"{(currentProgress * 100):F0}%");
                
                // Additional status info
                if (!string.IsNullOrEmpty(detailedStatus))
                {
                    EditorGUILayout.LabelField(detailedStatus, statusStyle);
                }
                
                EditorGUILayout.Space();
                
                // Cancel button with red background
                GUI.backgroundColor = new Color(0.9f, 0.3f, 0.3f);
                if (GUILayout.Button("Cancel Generation", GUILayout.Height(30)))
                {
                    CancelGeneration();
                }
                GUI.backgroundColor = Color.white;
            }
            else
            {
                // Generate button - only enable if we have valid input
                bool hasValidInput = HasValidInput();
                GUI.enabled = !isGenerating && hasValidInput;
                
                GUI.backgroundColor = new Color(0.3f, 0.8f, 0.4f);
                if (GUILayout.Button("Generate 3D Model", GUILayout.Height(30)))
                {
                    GenerateModel();
                }
                GUI.backgroundColor = Color.white;
                
                GUI.enabled = true;
                
                // Show message if input is missing
                if (!hasValidInput)
                {
                    string missingItem = useMultiview ? "multiview images" : (useImage ? "an image" : "a text prompt");
                    EditorGUILayout.HelpBox($"Please provide {missingItem} before generating.", MessageType.Warning);
                }
            }

            // Status display
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Status", headerStyle);
            
            // Display status with appropriate style
            GUIStyle currentStyle = statusStyle;
            if (status.Contains("Error"))
            {
                currentStyle = errorStyle;
            } 
            else if (status.Contains("completed") || status.Contains("imported"))
            {
                currentStyle = successStyle;
            }
            
            EditorGUILayout.LabelField(status, currentStyle);
            
            if (isGenerating)
            {
                // Force repaint to update progress
                Repaint();
            }

            // Generated model section
            if (!string.IsNullOrEmpty(generatedModelPath) && !isGenerating)
            {
                EditorGUILayout.Space(10);
                EditorGUILayout.LabelField("Generated Model", headerStyle);
                
                EditorGUILayout.LabelField("Path:", EditorStyles.boldLabel);
                EditorGUILayout.SelectableLabel(generatedModelPath, EditorStyles.textField, GUILayout.Height(20));
                
                EditorGUILayout.Space();
                
                GUI.backgroundColor = new Color(0.3f, 0.6f, 0.9f);
                if (GUILayout.Button("Create in Scene", GUILayout.Height(30)))
                {
                    CreateModelInScene();
                }
                GUI.backgroundColor = Color.white;
                
                EditorGUILayout.Space(5);
                
                if (GUILayout.Button("Reveal in Project", GUILayout.Height(24)))
                {
                    // Ping the asset in the Project window
                    UnityEngine.Object obj = AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(generatedModelPath);
                    if (obj != null)
                    {
                        EditorGUIUtility.PingObject(obj);
                    }
                }
            }
        }

        private void CancelGeneration()
        {
            if (!isGenerating) return;
            
            cancelRequested = true;
            status = "Cancelling generation...";
            Debug.Log("User requested to cancel the generation process.");
        }

        private async void GenerateModel()
        {
            if (isGenerating) return;
            
            generationStartTime = DateTime.Now;
            isGenerating = true;
            cancelRequested = false;
            currentProgress = 0f;
            currentPhase = "Preparing data";
            status = "Starting model generation...";
            detailedStatus = "Preparing request data...";
            
            // Create the request parameters in JSON manually since JsonUtility doesn't handle dictionaries well
            StringBuilder jsonBuilder = new StringBuilder();
            jsonBuilder.Append("{");
            jsonBuilder.Append($"\"seed\": {seed},");
            jsonBuilder.Append($"\"guidance_scale\": {guidanceScale.ToString("F1").Replace(',', '.')},");
            jsonBuilder.Append($"\"octree_resolution\": {octreeResolution},");
            jsonBuilder.Append($"\"num_inference_steps\": {numInferenceSteps},");
            jsonBuilder.Append($"\"texture\": {generateTexture.ToString().ToLower()}");
            
            if (generateTexture)
            {
                jsonBuilder.Append($",\"face_count\": {faceCount}");
            }
            
            // Add model information
            if (useMultiview)
            {
                // Add multiview model parameters
                var mvModelInfo = mvModelPathMapping[selectedMVModel];
                jsonBuilder.Append($",\"model_path\": \"{mvModelInfo.Item1}\"");
                jsonBuilder.Append($",\"subfolder\": \"{mvModelInfo.Item2}\"");
                jsonBuilder.Append($",\"multiview\": true");
                jsonBuilder.Append($",\"num_chunks\": {numChunks}");
            }
            else
            {
                // Add regular model parameters
                var modelInfo = modelPathMapping[selectedModel];
                jsonBuilder.Append($",\"model_path\": \"{modelInfo.Item1}\"");
                jsonBuilder.Append($",\"subfolder\": \"{modelInfo.Item2}\"");
            }

            if (useMultiview)
            {
                // Check if we have at least one image
                bool hasAtLeastOneImage = false;
                foreach (var img in multiviewImages.Values)
                {
                    if (img != null)
                    {
                        hasAtLeastOneImage = true;
                        break;
                    }
                }
                
                if (!hasAtLeastOneImage)
                {
                    status = "Error: No multiview images selected";
                    isGenerating = false;
                    return;
                }

                currentPhase = "Processing multiview images";
                detailedStatus = "Converting images to base64...";
                
                // Create a JSON object for the multiview images
                jsonBuilder.Append(",\"images\": {");
                
                bool firstImage = true;
                foreach (var view in multiviewImages.Keys)
                {
                    if (multiviewImages[view] != null)
                    {
                        if (!firstImage)
                        {
                            jsonBuilder.Append(",");
                        }
                        
                        string base64Image = TextureToBase64(multiviewImages[view]);
                        jsonBuilder.Append($"\"{view}\": \"{base64Image}\"");
                        firstImage = false;
                    }
                }
                
                jsonBuilder.Append("}");
                
                detailedStatus = "Images processed successfully";
            }
            else if (useImage)
            {
                if (inputImage == null)
                {
                    status = "Error: No input image selected";
                    isGenerating = false;
                    return;
                }

                currentPhase = "Processing image";
                detailedStatus = "Converting image to base64...";
                
                string base64Image = TextureToBase64(inputImage);
                jsonBuilder.Append($",\"image\": \"{base64Image}\"");
                
                detailedStatus = "Image processed successfully";
            }
            else
            {
                if (string.IsNullOrEmpty(inputText))
                {
                    status = "Error: Text prompt is empty";
                    isGenerating = false;
                    return;
                }

                currentPhase = "Processing text prompt";
                jsonBuilder.Append($",\"text\": \"{EscapeJsonString(inputText)}\"");
            }
            
            jsonBuilder.Append("}");
            
            currentProgress = 0.1f;

            try
            {
                // Use async generation route
                currentPhase = "Sending generation request";
                await SendGenerationRequest(jsonBuilder.ToString());
            }
            catch (Exception e)
            {
                status = "Error: " + e.Message;
                detailedStatus = e.ToString();
                isGenerating = false;
                currentProgress = 0;
            }
        }
        
        private string EscapeJsonString(string text)
        {
            return text.Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");
        }

        private string TextureToBase64(Texture2D texture)
        {
            // Preserve aspect ratio while ensuring max dimensions don't exceed 1024 pixels
            texture = ResizeTextureWithAspectRatio(texture, 1024);
            
            // Create a readable copy of the texture
            RenderTexture tmp = RenderTexture.GetTemporary(
                texture.width,
                texture.height,
                0,
                RenderTextureFormat.Default,
                RenderTextureReadWrite.Linear
            );
            
            Graphics.Blit(texture, tmp);
            RenderTexture previous = RenderTexture.active;
            RenderTexture.active = tmp;
            Texture2D readableTexture = new Texture2D(texture.width, texture.height);
            readableTexture.ReadPixels(new Rect(0, 0, tmp.width, tmp.height), 0, 0);
            readableTexture.Apply();
            RenderTexture.active = previous;
            RenderTexture.ReleaseTemporary(tmp);

            byte[] bytes = readableTexture.EncodeToPNG();
            if (readableTexture != texture)
            {
                DestroyImmediate(readableTexture);
            }
            
            return Convert.ToBase64String(bytes);
        }
        
        private Texture2D ResizeTextureWithAspectRatio(Texture2D source, int maxDimension)
        {
            int width = source.width;
            int height = source.height;
            float aspectRatio = (float)width / height;
            
            // Only resize if dimensions exceed maxDimension
            if (width <= maxDimension && height <= maxDimension)
                return source;
                
            // Calculate new dimensions while maintaining aspect ratio
            if (width > height)
            {
                width = maxDimension;
                height = Mathf.RoundToInt(width / aspectRatio);
            }
            else
            {
                height = maxDimension;
                width = Mathf.RoundToInt(height * aspectRatio);
            }
            
            // Create temporary RT for resizing
            RenderTexture rt = RenderTexture.GetTemporary(width, height, 0, RenderTextureFormat.ARGB32);
            rt.filterMode = FilterMode.Bilinear;
            
            // Blit the original texture to the render texture
            RenderTexture.active = rt;
            Graphics.Blit(source, rt);
            
            // Create new texture with the target size
            Texture2D resized = new Texture2D(width, height, TextureFormat.RGBA32, false);
            resized.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            resized.Apply();
            
            // Cleanup
            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(rt);
            
            Debug.Log($"Resized image from {source.width}x{source.height} to {width}x{height}");
            return resized;
        }

        private async Task SendGenerationRequest(string jsonData)
        {
            status = "Sending generation request to server...";
            detailedStatus = "Connecting to AI server...";
            byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);

            try
            {
                currentProgress = 0.15f;
                
                // Primary method: UnityWebRequest
                using (UnityWebRequest request = new UnityWebRequest(serverUrl + "/send", "POST"))
                {
                    request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                    request.downloadHandler = new DownloadHandlerBuffer();
                    request.SetRequestHeader("Content-Type", "application/json");
                    
                    // Send the request
                    request.SendWebRequest();
                    
                    detailedStatus = "Sending data to server...";

                    while (!request.isDone)
                    {
                        await Task.Delay(100);
                    }

                    currentProgress = 0.2f;

                    if (request.result == UnityWebRequest.Result.Success)
                    {
                        string responseText = request.downloadHandler.text;
                        GenerationResponse response = JsonUtility.FromJson<GenerationResponse>(responseText);

                        currentUid = response.uid;
                        status = "Generation started on server";
                        detailedStatus = "Server is now generating your 3D model. This may take several minutes.";
                        currentPhase = "3D Model Generation";
                        
                        // Poll for status
                        activeCoroutine = EditorCoroutine.Start(PollGenerationStatus());
                        return;
                    }
                    
                    status = "Primary request failed. Trying fallback...";
                    detailedStatus = $"Error: {request.error}";
                }

                // Fallback using WebClient
                using (WebClient webClient = new WebClient())
                {
                    webClient.Headers.Add("Content-Type", "application/json");
                    string response = await webClient.UploadStringTaskAsync(serverUrl + "/send", jsonData);
                    
                    GenerationResponse genResponse = JsonUtility.FromJson<GenerationResponse>(response);
                    currentUid = genResponse.uid;
                    status = "Generation started on server (fallback method)";
                    detailedStatus = "Server is now generating your 3D model. This may take several minutes.";
                    currentPhase = "3D Model Generation";
                    
                    // Poll for status
                    activeCoroutine = EditorCoroutine.Start(PollGenerationStatus());
                    return;
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"All request methods failed: {e.Message}");
                throw;
            }
        }

        private IEnumerator PollGenerationStatus()
        {
            bool completed = false;
            pollCount = 0;
            int failedAttempts = 0;
            
            // Polling settings
            float initialPollIntervalSeconds = 5;
            float pollIntervalSeconds = initialPollIntervalSeconds;
            int frameCounter = 0;
            int framesPerPoll = (int)(initialPollIntervalSeconds * 60);
            
            // Keep track of consecutive processing responses to adjust polling interval
            int consecutiveProcessingResponses = 0;

            Debug.Log($"Started polling for model generation status at {DateTime.Now}. UID: {currentUid}");
            
            // Continue polling indefinitely while server reports processing
            while (!completed && !cancelRequested)
            {
                // Use frame counting instead of time comparison
                frameCounter++;
                if (frameCounter < framesPerPoll)
                {
                    yield return null; // Wait for next frame
                    continue;
                }
                
                // Reset frame counter
                frameCounter = 0;
                
                // Track poll attempts
                pollCount++;
                
                // Only show detailed status during development builds
                #if UNITY_EDITOR && DEBUG
                //detailedStatus = $"Polling server for model status... (Attempt {pollCount})";
                Debug.Log($"Poll {pollCount} at {DateTime.Now} - UID: {currentUid}");
                #else
                //detailedStatus = "Checking model generation progress...";
                #endif
                
                bool success = false;
                StatusResponse statusResponse = null;
                
                // Primary method: UnityWebRequest
                using (UnityWebRequest request = UnityWebRequest.Get(serverUrl + "/status/" + currentUid))
                {
                    request.SendWebRequest();
                    
                    // Manual waiting for the request to complete
                    int requestFrameCounter = 0;
                    int maxRequestFrames = 120;
                    
                    while (!request.isDone && requestFrameCounter < maxRequestFrames)
                    {
                        // Check for cancellation
                        if (cancelRequested)
                        {
                            Debug.Log("Polling cancelled by user");
                            break;
                        }
                        
                        requestFrameCounter++;
                        yield return null; // Wait a frame
                    }

                    if (cancelRequested) break;

                    if (request.isDone && request.result == UnityWebRequest.Result.Success)
                    {
                        string responseText = request.downloadHandler.text;
                        #if UNITY_EDITOR && DEBUG
                        Debug.Log($"Poll {pollCount} - Response: {responseText}");
                        #endif
                        
                        try {
                            statusResponse = JsonUtility.FromJson<StatusResponse>(responseText);
                            success = true;
                            
                            // Reset failed attempts on success
                            failedAttempts = 0;
                        }
                        catch (Exception e) {
                            Debug.LogError($"Failed to parse response: {e.Message}");
                            Debug.LogError($"Response text: {responseText}");
                        }
                    }
                    else
                    {
                        Debug.LogWarning($"Primary polling method failed: {request.error}");
                    }
                }

                if (cancelRequested) break;

                // Fallback to WebClient if UnityWebRequest failed
                if (!success && failedAttempts < 3)
                {
                    try
                    {
                        using (WebClient webClient = new WebClient())
                        {
                            string response = webClient.DownloadString(serverUrl + "/status/" + currentUid);
                            #if UNITY_EDITOR && DEBUG
                            Debug.Log($"Fallback Poll {pollCount} - Response: {response}");
                            #endif
                            
                            try {
                                statusResponse = JsonUtility.FromJson<StatusResponse>(response);
                                success = true;
                                
                                #if UNITY_EDITOR && DEBUG
                                Debug.Log("Fallback polling succeeded");
                                #endif
                                
                                // Reset failed attempts on success
                                failedAttempts = 0;
                            }
                            catch (Exception e) {
                                Debug.LogError($"Failed to parse fallback response: {e.Message}");
                                Debug.LogError($"Response text: {response}");
                            }
                        }
                    }
                    catch (Exception e)
                    {
                        Debug.LogError($"Fallback polling failed: {e.Message}");
                        failedAttempts++;
                        
                        // If we've failed multiple times, increase the polling interval
                        if (failedAttempts >= 3)
                        {
                            pollIntervalSeconds = Mathf.Min(pollIntervalSeconds * 2, 60); // Double interval up to 60
                            framesPerPoll = (int)(pollIntervalSeconds * 60);
                            Debug.Log($"Increased polling interval to {pollIntervalSeconds} seconds after {failedAttempts} failed attempts");
                        }
                    }
                }

                if (success && statusResponse != null)
                {
                    #if UNITY_EDITOR && DEBUG
                    Debug.Log($"Poll {pollCount} - Status: {statusResponse.status}, Phase: {statusResponse.phase}, Progress: {statusResponse.progress}");
                    #endif
                    
                    if (statusResponse.status == "completed")
                    {
                        currentProgress = 1.0f;
                        currentPhase = "Processing Final Model";
                        status = "Model generation completed!";
                        detailedStatus = "Processing model data received from server...";
                        
                        try {
                            // Check if the model_base64 field is present and not empty
                            if (string.IsNullOrEmpty(statusResponse.model_base64))
                            {
                                Debug.LogError("Model completed but base64 data is empty!");
                                status = "Error: Received empty model data";
                                isGenerating = false;
                                yield break;
                            }
                            
                            // Save the model to disk
                            string tempPath = Path.Combine(Application.temporaryCachePath, currentUid + ".glb");
                            File.WriteAllBytes(tempPath, Convert.FromBase64String(statusResponse.model_base64));
                            Debug.Log($"Model saved to temporary path: {tempPath}");
                            
                            // Create asset folder if it doesn't exist
                            string assetPath = Path.Combine("Assets", "GeneratedModels");
                            if (!Directory.Exists(assetPath))
                            {
                                Directory.CreateDirectory(assetPath);
                                Debug.Log($"Created directory: {assetPath}");
                            }
                            
                            // Copy to Assets folder
                            generatedModelPath = Path.Combine(assetPath, currentUid + ".glb");
                            File.Copy(tempPath, generatedModelPath, true);
                            Debug.Log($"Model copied to Assets: {generatedModelPath}");
                            
                            detailedStatus = "Importing model into Unity project...";
                            AssetDatabase.Refresh();
                            completed = true;
                            detailedStatus = "Model ready! Use 'Create in Scene' to add it to your scene.";
                            
                            // Calculate time taken
                            TimeSpan elapsed = DateTime.Now - generationStartTime;
                            string elapsedTime = string.Format("{0:00}:{1:00}", elapsed.Minutes, elapsed.Seconds);
                            status = $"Model generation completed in {elapsedTime}!";
                        }
                        catch (Exception e) {
                            Debug.LogError($"Error processing completed model: {e.Message}");
                            status = "Error processing model: " + e.Message;
                            detailedStatus = e.ToString();
                            isGenerating = false;
                            yield break;
                        }
                    }
                    else if (statusResponse.status == "processing")
                    {
                        // Server is still processing, update the phase and progress
                        consecutiveProcessingResponses++;
                        
                        // Use server-provided phase information if available
                        if (!string.IsNullOrEmpty(statusResponse.phase) && statusResponse.phase != "unknown")
                        {
                            string phase = statusResponse.phase;
                            // Map server phase names to user-friendly display names
                            switch (phase)
                            {
                                case "initializing":
                                    currentPhase = "Initializing";
                                    break;
                                case "queued":
                                    currentPhase = "Queued for processing";
                                    break;
                                case "processing_image":
                                    currentPhase = "Processing input image";
                                    break;
                                case "processing_text":
                                    currentPhase = "Processing text prompt";
                                    break;
                                case "background_removal":
                                    currentPhase = "Removing image background";
                                    break;
                                case "shape_generation":
                                    currentPhase = "Generating 3D shape";
                                    break;
                                case "shape_complete":
                                    currentPhase = "Shape generation complete";
                                    break;
                                case "mesh_cleaning":
                                    currentPhase = "Preparing mesh for texturing";
                                    break;
                                case "removing_floaters":
                                    currentPhase = "Cleaning mesh: Removing floaters";
                                    break;
                                case "fixing_mesh":
                                    currentPhase = "Cleaning mesh: Fixing geometry";
                                    break;
                                case "reducing_faces":
                                    currentPhase = "Optimizing mesh for texturing";
                                    break;
                                case "texture_generation":
                                    currentPhase = "Generating textures";
                                    break;
                                case "texture_complete":
                                    currentPhase = "Texturing complete";
                                    break;
                                case "exporting":
                                    currentPhase = "Exporting final model";
                                    break;
                                default:
                                    if (phase == "unknown" && generateTexture && currentProgress >= 0.4f)
                                    {
                                        // If we know texturing was requested and we're at least past shape generation
                                        currentPhase = "Applying textures";
                                    }
                                    else if (phase == "unknown" && currentProgress < 0.4f)
                                    {
                                        currentPhase = "Generating 3D model";
                                    }
                                    else
                                    {
                                        // Format other phase names nicely
                                        currentPhase = phase.Replace("_", " ");
                                        currentPhase = char.ToUpper(currentPhase[0]) + currentPhase.Substring(1);
                                    }
                                    break;
                            }
                        }
                        else if (currentPhase == "unknown" || string.IsNullOrEmpty(currentPhase))
                        {
                            // Fallback phases based on progress if server doesn't provide phase info
                            if (currentProgress < 0.3f)
                                currentPhase = "Generating 3D shape";
                            else if (currentProgress < 0.7f && generateTexture)
                                currentPhase = "Applying textures";
                            else
                                currentPhase = "Processing 3D model";
                        }
                        
                        // Check if the server provided any progress information
                        if (statusResponse.progress > 0)
                        {
                            // If server provides actual progress, use it
                            currentProgress = Mathf.Clamp01(statusResponse.progress);
                            
                            // Calculate remaining time estimate if we have consistent progress updates
                            if (consecutiveProcessingResponses > 2 && currentProgress > 0.1f)
                            {
                                TimeSpan elapsed = DateTime.Now - generationStartTime;
                                float estimatedTotalSeconds = (float)elapsed.TotalSeconds / currentProgress;
                                float remainingSeconds = estimatedTotalSeconds - (float)elapsed.TotalSeconds;
                                
                                if (remainingSeconds > 0)
                                {
                                    int minutes = (int)(remainingSeconds / 60);
                                    int seconds = (int)(remainingSeconds % 60);
                                    detailedStatus = $"Estimated time remaining: {minutes}m {seconds}s";
                                }
                            }
                        }
                        else
                        {
                            // Otherwise use our best guess based on poll count
                            // Update progress - follow a curve that approaches 1.0 slowly
                            float progressEstimate = 0.2f + 0.6f * (1.0f - (1.0f / (1.0f + 0.03f * pollCount)));
                            currentProgress = Mathf.Clamp01(progressEstimate);
                        }
                        
                        status = "Model generation in progress...";
                        
                        // Check if there's a message from the server
                        if (!string.IsNullOrEmpty(statusResponse.message))
                        {
                            detailedStatus = statusResponse.message;
                        }
                        
                        // Adjust polling interval based on how long we've been processing
                        if (consecutiveProcessingResponses > 10)
                        {
                            // Gradually increase polling interval for long-running operations
                            pollIntervalSeconds = Mathf.Min(initialPollIntervalSeconds * (1 + consecutiveProcessingResponses/20), 15);
                            framesPerPoll = (int)(pollIntervalSeconds * 60);
                        }
                        else
                        {
                            // Reset to initial interval
                            pollIntervalSeconds = initialPollIntervalSeconds;
                            framesPerPoll = (int)(pollIntervalSeconds * 60);
                        }
                    }
                    else if (statusResponse.status == "error")
                    {
                        status = "Server error: " + statusResponse.message;
                        detailedStatus = "The server encountered an error while processing your request.";
                        isGenerating = false;
                        yield break;
                    }
                }
                else
                {
                    // Increment failed attempts if we couldn't get a valid response
                    failedAttempts++;
                    
                    // If we've failed too many times in a row, gradually increase the polling interval
                    if (failedAttempts > 3)
                    {
                        pollIntervalSeconds = Mathf.Min(pollIntervalSeconds * 1.5f, 60); // Increase interval up to 60 seconds max
                        framesPerPoll = (int)(pollIntervalSeconds * 60);
                        Debug.Log($"Increased polling interval to {pollIntervalSeconds} seconds after {failedAttempts} failed attempts");
                    }
                    
                    // If we've failed too many times consecutively, give up
                    if (failedAttempts > 10)
                    {
                        status = "Error: Failed to communicate with server after multiple attempts";
                        detailedStatus = "Please check if the server is still running and accessible.";
                        isGenerating = false;
                        yield break;
                    }
                    
                    // Update status with error info but hide technical details from user
                    detailedStatus = "Communication with server interrupted. Retrying...";
                }
            }

            if (cancelRequested)
            {
                status = "Generation cancelled by user.";
                detailedStatus = "";
                Debug.Log("Model generation cancelled by user after " + pollCount + " attempts");
                cancelRequested = false;
            }
            else if (!completed)
            {
                status = "Generation process appears stuck.";
                detailedStatus = "The server is taking unusually long to respond. You may need to restart the server.";
                Debug.LogError($"Model generation may be stuck after {pollCount} polling attempts");
            }

            isGenerating = false;
        }

        private void CreateModelInScene()
        {
            if (string.IsNullOrEmpty(generatedModelPath)) return;

            try
            {
                // Log important information for debugging
                string fullPath = Path.GetFullPath(generatedModelPath);
                bool fileExists = File.Exists(fullPath);
                Debug.Log($"Importing model from: {fullPath}");
                Debug.Log($"File exists: {fileExists}");
                
                if (!fileExists)
                {
                    status = "Error: Generated model file not found";
                    return;
                }
                
                // Get the relative path for Unity's asset system
                string assetPath = generatedModelPath;
                
                // Make sure Unity knows about the file
                AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceUpdate);
                
                // Get the imported asset
                UnityEngine.Object modelAsset = AssetDatabase.LoadAssetAtPath(assetPath, typeof(GameObject));
                
                if (modelAsset == null)
                {
                    status = "Error: Failed to load model asset";
                    Debug.LogError("Failed to load model asset at path: " + assetPath);
                    return;
                }
                
                // Instantiate the model in the scene
                GameObject modelInstance = (GameObject)PrefabUtility.InstantiatePrefab(modelAsset);
                modelInstance.name = "GeneratedModel_" + Path.GetFileNameWithoutExtension(assetPath);
                
                // Process the loaded model
                ProcessLoadedModel(modelInstance);
                
                // Select the model in the editor
                Selection.activeGameObject = modelInstance;
                SceneView.FrameLastActiveSceneView();
                
                status = "Model successfully imported into scene!";
            }
            catch (Exception e)
            {
                Debug.LogError($"Error creating model in scene: {e.Message}\n{e.StackTrace}");
                status = "Error: " + e.Message;
            }
        }
        
        private void ProcessLoadedModel(GameObject modelContainer)
        {
            try
            {
                // Find all child renderers
                Renderer[] renderers = modelContainer.GetComponentsInChildren<Renderer>(true);
                
                if (renderers.Length == 0)
                {
                    Debug.LogWarning("No renderers found in imported model");
                    status = "Warning: Model imported but no renderers found";
                    return;
                }
                
                Debug.Log($"Found {renderers.Length} renderers in imported model");
                
                // Calculate bounds to normalize scale if needed
                Bounds bounds = CalculateBounds(modelContainer);
                float maxSize = Mathf.Max(bounds.size.x, bounds.size.y, bounds.size.z);
                Debug.Log($"Model bounds: {bounds.size}, max size: {maxSize}");
                
                // Center the model at the origin
                Vector3 center = bounds.center;
                center.y = bounds.min.y; // Place bottom at origin
                modelContainer.transform.position = -center;
                
                // If the model is very large or small, normalize its scale
                if (maxSize > 10f || maxSize < 0.1f)
                {
                    float scaleFactor = 1f / maxSize;
                    modelContainer.transform.localScale = Vector3.one * scaleFactor;
                    Debug.Log($"Normalized model scale. Original size: {maxSize}, Scale factor: {scaleFactor}");
                }
                
                // Select the model in the editor
                Selection.activeGameObject = modelContainer;
                SceneView.FrameLastActiveSceneView();
                
                status = "Model imported and ready!";
            }
            catch (Exception e)
            {
                Debug.LogError($"Error processing loaded model: {e.Message}");
                status = "Error processing model: " + e.Message;
            }
        }
        
        private Bounds CalculateBounds(GameObject gameObject)
        {
            Bounds bounds = new Bounds(gameObject.transform.position, Vector3.zero);
            Renderer[] renderers = gameObject.GetComponentsInChildren<Renderer>();
            
            if (renderers.Length > 0)
            {
                // Initialize with the first renderer's bounds
                bounds = renderers[0].bounds;
                
                // Expand to include all other renderers
                for (int i = 1; i < renderers.Length; i++)
                {
                    bounds.Encapsulate(renderers[i].bounds);
                }
            }
            
            return bounds;
        }

        private async void TestServerConnection()
        {
            status = "Testing server connection...";
            
            try
            {
                // Try to connect to the server
                using (UnityWebRequest request = UnityWebRequest.Get(serverUrl))
                {
                    request.SendWebRequest();
                    
                    while (!request.isDone)
                    {
                        await Task.Delay(100);
                    }
                    
                    if (request.result == UnityWebRequest.Result.Success)
                    {
                        status = "✓ Server connection successful!";
                        Debug.Log("Server connection test successful");
                    }
                    else
                    {
                        status = $"Connection error: {request.error}. Trying fallback...";
                        
                        // Try fallback method
                        try
                        {
                            using (WebClient webClient = new WebClient())
                            {
                                webClient.DownloadString(serverUrl);
                                status = "✓ Server connection successful (fallback)!";
                                Debug.Log("Server connection test successful via fallback");
                            }
                        }
                        catch (Exception e)
                        {
                            status = $"Server connection failed: {e.Message}";
                            Debug.LogError($"All connection methods failed: {e.Message}");
                        }
                    }
                }
            }
            catch (Exception e)
            {
                status = $"Error testing connection: {e.Message}";
                Debug.LogError($"Connection test error: {e.Message}");
            }
        }

        private bool HasValidInput()
        {
            if (useMultiview)
            {
                foreach (var img in multiviewImages.Values)
                {
                    if (img != null) return true;
                }
                return false;
            }
            else if (useImage)
            {
                return inputImage != null;
            }
            else
            {
                return !string.IsNullOrEmpty(inputText);
            }
        }
    }

    // Helper class for editor coroutines
    public class EditorCoroutine
    {
        public static EditorCoroutine Start(IEnumerator routine)
        {
            EditorCoroutine coroutine = new EditorCoroutine(routine);
            coroutine.Start();
            return coroutine;
        }

        readonly IEnumerator routine;
        EditorCoroutine(IEnumerator routine)
        {
            this.routine = routine;
        }

        void Start()
        {
            EditorApplication.update += Update;
        }

        private void Update()
        {
            if (!routine.MoveNext())
            {
                EditorApplication.update -= Update;
            }
        }

        public void Stop()
        {
            EditorApplication.update -= Update;
        }
    }

    [Serializable]
    public class GenerationResponse
    {
        public string uid;
    }

    [Serializable]
    public class StatusResponse
    {
        public string status;
        public string model_base64;
        public string message;
        public float progress;
        public string phase;
    }
} 