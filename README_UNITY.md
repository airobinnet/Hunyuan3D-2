# Hunyuan3D Unity Server and Plugin

This guide will help you set up the Hunyuan3D Unity server and plugin for 3D model generation in Unity.

## Server Setup

### Prerequisites

- Python 3.8+ 
- CUDA-compatible GPU (recommended for optimal performance)
- Required Python dependencies (see [Requirements](#requirements))

### Installation

1. Clone the Hunyuan3D-2 repository or unzip the package:
   ```
   git clone https://github.com/Tencent/Hunyuan3D.git
   cd Hunyuan3D-2
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. **(Important for Performance)** Build the C++ mesh processor for accelerated texture generation:
   ```
   cd hy3dgen/texgen/differentiable_renderer
   python setup.py install
   ```
   
   This step is **crucial** for reasonable texture generation performance. Without the C++ implementation, the Python fallback is much slower.

### Running the Server

Start the server with the following command:

```
python unity_server.py --host 0.0.0.0 --port 8081
```

Options:
- `--host`: Specify the host address (default: 0.0.0.0)
- `--port`: Specify the port (default: 8081)
- `--model_path`: Model path (default: tencent/Hunyuan3D-2mini)
- `--tex_model_path`: Texture model path (default: tencent/Hunyuan3D-2)
- `--subfolder`: Model subfolder (default: hunyuan3d-dit-v2-mini-turbo)
- `--device`: Device to use (default: cuda)
- `--low_vram_mode`: Enable low VRAM mode for devices with limited memory
- `--disable_flashvdm`: Disable FlashVDM acceleration (not recommended)
- `--verbose`: Enable verbose logging
- `--mv_model_path`: Path to multiview model (default: tencent/Hunyuan3D-2mv)
- `--mv_subfolder`: Multiview model subfolder (default: hunyuan3d-dit-v2-mv)

### Model Variants

The server supports multiple model variants:

1. **Shape Models**:
   - **Hunyuan3D-2mini-Turbo**: Smallest model size with fastest generation
   - **Hunyuan3D-2mini-Fast**: Small model with faster generation
   - **Hunyuan3D-2mini**: Small model with standard generation speed
   - **Hunyuan3D-2-Turbo**: Larger model with faster generation
   - **Hunyuan3D-2-Fast**: Larger model with faster generation
   - **Hunyuan3D-2**: Largest model with standard generation speed

2. **Multiview Models**:
   - **Hunyuan3D-2mv-Turbo**: Fastest multiview model
   - **Hunyuan3D-2mv-Fast**: Faster multiview model
   - **Hunyuan3D-2mv**: Standard multiview model

### Memory Optimization

The server uses dynamic model loading to optimize memory usage:
- Models are loaded only when needed
- Single-view, multiview, and texture models are loaded/unloaded as required
- For maximum memory efficiency, use the `--low_vram_mode` flag

### Performance Optimization

For optimal texture generation performance, enable the C++ mesh processor implementation:

1. The server installation steps above include building the C++ implementation.
2. If you encounter slow texture generation, verify that the C++ implementation is being used by checking the server logs:
   ```
   Successfully imported C++ mesh_processor implementation
   ```

3. If you're still experiencing slow texture generation with the error "meshVerticeInpaint_smooth takes extremely long time", ensure the correct implementation is used by modifying `mesh_render.py`:
   ```python
   # from .mesh_processor import meshVerticeInpaint
   from mesh_processor import meshVerticeInpaint
   ```

## Unity Plugin Setup

### Installation

1. Drag the `UnityPlugin/3DModelGenerator.cs` to your Project's Asset folder

### Using the Plugin

1. In Unity, navigate to `Tools` → `3D Model Generator` to open the plugin window
2. Configure the server connection:
   - Enter the server URL (default: http://localhost:8081)
   - Click "Test Connection" to verify connectivity
   
3. Input options (select one):
   - **Single Image**: Upload a single reference image
   - **Multiview**: Upload multiple images from different viewpoints (front, left, back)
   - **Text**: Enter a detailed description of the 3D model you want to generate

4. Model Selection:
   - For single-image or text mode, choose from shape models (mini-Turbo to full)
   - For multiview mode, choose from multiview models (Turbo to standard)

5. Generation parameters:
   - **Seed**: Controls randomness (set a specific number or click "Random Seed")
   - **Guidance Scale**: How closely to follow the input (1.0-10.0)
   - **Resolution**: Higher values produce more detailed models but slower generation
   - **Inference Steps**: Higher values improve quality but increase generation time
     - For FlashVDM with single view: 1-10 steps (5 recommended)
     - For multiview: 20-100 steps
   - **Generate Texture**: Enable to create textured models (slower)
   - **Face Count**: For textured models, lower face count (5000-10000) improves texturing speed
   - **Number of Chunks**: (Multiview only) Higher values produce more detailed models

6. Click "Generate" to start the model generation process

7. Progress Tracking:
   - The plugin shows real-time progress, current phase, and estimated time
   - Generation phases are displayed (background removal, shape generation, texturing, etc.)
   - You can cancel generation at any time

8. After Completion:
   - Click "Create in Scene" to add the model to your current scene
   - Click "Reveal in Project" to locate the model in your Assets folder
   - Models are automatically scaled and positioned appropriately

### Generation Modes

1. **Image-to-3D (Single Image)**:
   - Upload a single reference image
   - The system automatically removes the background
   - Best for simple objects with clear silhouettes

2. **Multiview-to-3D**:
   - Upload images from different angles (front, left, back views)
   - Produces more geometrically accurate models
   - Useful for complex shapes or when accuracy is important

3. **Text-to-3D**:
   - Simply describe the 3D model you want
   - The system generates an image from your text, then creates a 3D model
   - Best for creative or conceptual models

## Troubleshooting

### Server Issues

- **Out of Memory Errors**: Use `--low_vram_mode` to reduce VRAM usage
- **Slow Texture Generation**: Ensure the C++ mesh processor is properly built and used
- **Connection Errors**: Verify the server is running and accessible from Unity
- **Generation Stuck**: Check server logs for errors; you may need to restart the server

### Unity Plugin Issues

- **Import Errors**: Ensure you're using a compatible Unity version (2020.3+)
- **Connection Failures**: Check firewall settings and server URL configuration
- **Generation Failures**: Check server logs for detailed error information
- **Slow Progress**: Texture generation is CPU-intensive and may take several minutes

## Tips for Best Results

1. Use clear, well-lit images against simple backgrounds
2. For multiview mode, ensure consistent lighting across all images
3. Use lower face count (5000-10000) for faster texture generation
4. For text-to-3D, provide detailed, specific descriptions
5. Start with "mini-Turbo" models for faster iteration, then use larger models for final quality

## License

Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT. 