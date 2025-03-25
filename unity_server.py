# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""
A model worker executes the model.
"""
import argparse
import asyncio
import base64
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import traceback
import uuid
from io import BytesIO
import time
import gc
import psutil
import multiprocessing
import signal
import functools
import json
from PIL import Image

import torch
import trimesh
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer, \
    MeshSimplifier
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline

LOGDIR = '.'

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None

# Global dictionary to store generation status for each request
generation_status = {}

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        # Skip empty messages
        if not buf.strip():
            return
        
        # Filter out HTTP access logs and base64 image data
        # 1. Skip all HTTP access logs regardless of format
        if ('INFO:     ' in buf and '- "' in buf and ' HTTP/' in buf and ('"' in buf and ' 200' in buf)):
            return
        
        # 2. Skip any line containing base64 image data (these are typically very long)
        if 'model_base64' in buf or ';base64,' in buf or (len(buf) > 100 and any(c in buf for c in '+/=')):
            return
            
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("controller", f"{SAVE_DIR}/controller.log")


def load_image_from_base64(image_b64):
    """
    Load and validate an image from a base64 string.
    Standardizes format and resizes if needed.
    """
    try:
        # Decode base64 string to binary
        image_data = base64.b64decode(image_b64)
        
        # Open the image using PIL
        image = Image.open(BytesIO(image_data))
        
        # Standardize to RGB format if needed
        if image.mode != 'RGB' and image.mode != 'RGBA':
            image = image.convert('RGB')
            
        # Check for very large images and resize if necessary
        # This helps prevent excessive memory usage
        max_dimension = 1024  # Reasonable size for 3D generation
        width, height = image.size
        if width > max_dimension or height > max_dimension:
            logger.info(f"Resizing large image from {width}x{height}")
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"Resized image to {new_width}x{new_height}")
            
        return image
    except base64.binascii.Error:
        raise ValueError("Invalid base64 image data provided")
    except IOError:
        raise ValueError("Could not open image - invalid image data")
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")


def load_multiview_images_from_base64(images_dict):
    """
    Load and validate multiple images from base64 strings.
    Processes a dictionary of images for multiview generation.
    """
    result = {}
    for view, image_b64 in images_dict.items():
        try:
            image = load_image_from_base64(image_b64)
            result[view] = image
        except ValueError as e:
            raise ValueError(f"Error processing {view} image: {str(e)}")
    
    # Ensure we have at least one valid image
    if not result:
        raise ValueError("No valid images provided for multiview generation")
        
    return result


class ModelWorker:
    def __init__(self,
                 model_path='tencent/Hunyuan3D-2mini',
                 tex_model_path='tencent/Hunyuan3D-2',
                 subfolder='hunyuan3d-dit-v2-mini-turbo',
                 device='cuda',
                 enable_tex=False,
                 low_vram_mode=False,
                 enable_flashvdm=True,
                 enable_multiview=False,
                 mv_model_path='tencent/Hunyuan3D-2mv',
                 mv_subfolder='hunyuan3d-dit-v2-mv'):
        self.model_path = model_path
        self.worker_id = worker_id
        self.device = device
        self.request_count = 0
        self.max_requests_before_reset = 10  # Reset model after this many requests
        
        # Store model configuration for dynamic loading
        self.enable_multiview = enable_multiview
        self.mv_model_path = mv_model_path
        self.mv_subfolder = mv_subfolder
        self.tex_model_path = tex_model_path
        self.enable_tex = enable_tex
        self.low_vram_mode = low_vram_mode
        self.enable_flashvdm = enable_flashvdm
        self.subfolder = subfolder
        
        # Track loaded models
        self.single_view_loaded = False
        self.multiview_loaded = False
        self.texture_loaded = False
        
        # Set environment variables to optimize performance
        os.environ['TEXGEN_USE_CPP'] = '1'
        os.environ['TEXGEN_MERGE_METHOD'] = 'fast'
        os.environ['TEXGEN_TEXTURE_SIZE'] = '1024'
        
        # Allow texture generation to be interrupted
        os.environ['TEXGEN_INTERRUPTIBLE'] = '1'
        
        logger.info(f"Starting worker {worker_id} with dynamic model loading...")
        logger.info(f"Configuration prepared for single view model: {model_path}/{subfolder}")
        
        if enable_multiview:
            logger.info(f"Multiview support enabled. Model will be loaded on demand: {mv_model_path}/{mv_subfolder}")
            
        # Install signal handler to handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self._signal_handler)
        
        try:
            # Try to import mesh_processor directly (C++ version)
            import mesh_processor
            logger.info("Successfully imported C++ mesh_processor implementation")
        except ImportError:
            logger.warning("C++ mesh_processor not found - texture generation may be slow")
        
        # Initialize common utilities that are needed regardless of model type
        self.rembg = BackgroundRemover()
        
        # Initialize mesh processing utilities - these are needed for both single view and multiview
        self.floater_remover = FloaterRemover()
        self.degenerate_face_remover = DegenerateFaceRemover()
        self.face_reducer = FaceReducer()
        
        # Initialize text-to-image pipeline
        self.pipeline_t2i = HunyuanDiTPipeline(
            'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
            device=device
        )
        
        # Initialize the default single view model
        self._init_single_view_model()
        
    def _init_single_view_model(self, model_path=None, subfolder=None):
        """Initialize the standard single view model"""
        # Unload multiview model if it's loaded to free up VRAM
        if self.multiview_loaded:
            self._unload_multiview_model()
            
        # Unload texture model if it's loaded to free up VRAM
        if self.texture_loaded:
            self._unload_texture_model()
            
        if self.single_view_loaded:
            logger.info("Single view model already loaded")
            return
            
        logger.info(f"Loading single view model: {model_path}/{subfolder}")
        
        # Force garbage collection before loading
        gc.collect()
        torch.cuda.empty_cache()
        
        load_start = time.time()
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path or self.model_path,
            subfolder=subfolder or self.subfolder,
            use_safetensors=True,
            device=self.device,
            low_vram_mode=self.low_vram_mode,
        )
        logger.info(f"Single view model loading time: {time.time() - load_start:.2f} seconds")
        
        # Enable FlashVDM for fast shape generation
        if self.enable_flashvdm:
            logger.info("Enabling FlashVDM for single view model...")
            flashvdm_start = time.time()
            self.pipeline.enable_flashvdm(mc_algo='mc')
            logger.info(f"FlashVDM initialization time: {time.time() - flashvdm_start:.2f} seconds")
        
        self.single_view_loaded = True
        self._log_memory_usage("After single view model initialization")
    
    def _unload_single_view_model(self):
        """Unload the single view model to free memory"""
        if not self.single_view_loaded:
            return
            
        logger.info("Unloading single view model to free VRAM")
        del self.pipeline
        self.single_view_loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        self._log_memory_usage("After single view model unloading")
        
    def _init_multiview_model(self, model_path=None, subfolder=None):
        """Initialize the multiview model on demand"""
        # Unload single view model if it's loaded to free up VRAM
        if self.single_view_loaded:
            self._unload_single_view_model()
            
        # Unload texture model if it's loaded to free up VRAM
        if self.texture_loaded:
            self._unload_texture_model()
            
        if self.multiview_loaded:
            logger.info("Multiview model already loaded")
            return
            
        logger.info(f"Loading multiview model on demand: {model_path}/{subfolder}")
        
        # Force garbage collection before loading a new model
        gc.collect()
        torch.cuda.empty_cache()
        
        mv_start = time.time()
        self.pipeline_mv = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path or self.mv_model_path,
            subfolder=subfolder or self.mv_subfolder,
            use_safetensors=True,
            device=self.device,
            low_vram_mode=self.low_vram_mode,
        )
        logger.info(f"Multiview model loading time: {time.time() - mv_start:.2f} seconds")
        
        # Enable FlashVDM for multiview model if configured
        if self.enable_flashvdm:
            logger.info("Enabling FlashVDM for multiview model...")
            mv_flashvdm_start = time.time()
            self.pipeline_mv.enable_flashvdm(mc_algo='mc')
            logger.info(f"Multiview FlashVDM initialization time: {time.time() - mv_flashvdm_start:.2f} seconds")
        
        self.multiview_loaded = True
        self._log_memory_usage("After multiview model initialization")
    
    def _unload_multiview_model(self):
        """Unload the multiview model to free memory"""
        if not self.multiview_loaded:
            return
            
        logger.info("Unloading multiview model to free VRAM")
        del self.pipeline_mv
        self.multiview_loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        self._log_memory_usage("After multiview model unloading")
        
    def _init_texture_model(self):
        """Initialize the texture generation model on demand"""
        # Unload shape models to free up VRAM
        if self.single_view_loaded:
            self._unload_single_view_model()
            
        if self.multiview_loaded:
            self._unload_multiview_model()
            
        if self.texture_loaded:
            logger.info("Texture model already loaded")
            return
            
        logger.info(f"Loading texture generation model on demand: {self.tex_model_path}")
        
        # Force garbage collection before loading a new model
        gc.collect()
        torch.cuda.empty_cache()
        
        tex_start = time.time()
        self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(self.tex_model_path)
        
        if self.low_vram_mode:
            logger.info("Enabling CPU offload for texture generation model")
            self.pipeline_tex.enable_model_cpu_offload()
            
        logger.info(f"Texture model loading time: {time.time() - tex_start:.2f} seconds")
        self.texture_loaded = True
        self._log_memory_usage("After texture model initialization")
    
    def _unload_texture_model(self):
        """Unload the texture model to free memory"""
        if not self.texture_loaded:
            return
            
        logger.info("Unloading texture model to free VRAM")
        del self.pipeline_tex
        self.texture_loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        self._log_memory_usage("After texture model unloading")

    def _init_models(self, model_path, subfolder, device, low_vram_mode, enable_flashvdm, enable_tex, tex_model_path):
        """Initialize or reinitialize models to prevent performance degradation"""
        # Force garbage collection before loading models
        gc.collect()
        torch.cuda.empty_cache()
        
        # Update model paths
        self.model_path = model_path
        self.subfolder = subfolder
        
        # Install signal handler to handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Reset model loaded flags
        self.single_view_loaded = False
        self.multiview_loaded = False
        self.texture_loaded = False
        
        # Initialize shared components
        try:
            # Try to import mesh_processor directly (C++ version)
            import mesh_processor
            logger.info("Successfully imported C++ mesh_processor implementation")
        except ImportError:
            logger.warning("C++ mesh_processor not found - texture generation may be slow")
            
        # Reinitialize the background remover
        self.rembg = BackgroundRemover()
        
        # Reinitialize mesh processing utilities
        self.floater_remover = FloaterRemover()
        self.degenerate_face_remover = DegenerateFaceRemover()
        self.face_reducer = FaceReducer()
        
        # Initialize the single view pipeline
        self._init_single_view_model()
            
        # Reinitialize the text-to-image pipeline
        self.pipeline_t2i = HunyuanDiTPipeline(
            'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
            device=device
        )

    def _log_memory_usage(self, label):
        """Log current memory usage for debugging"""
        if not torch.cuda.is_available():
            return
            
        # Log GPU memory
        gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
        logger.info(f"{label} - GPU memory allocated: {gpu_allocated:.2f} GB")
        logger.info(f"{label} - GPU memory reserved: {gpu_reserved:.2f} GB")
        
        # Log system memory
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        logger.info(f"{label} - System memory used: {memory_mb:.2f} MB")

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }
        
    def maybe_reset_model(self):
        """Reset the model if we've processed too many requests to prevent performance degradation"""
        self.request_count += 1
        if self.request_count >= self.max_requests_before_reset:
            logger.info(f"Resetting models after {self.request_count} requests to maintain performance")
            enable_tex = hasattr(self, 'pipeline_tex')
            tex_model_path = getattr(args, 'tex_model_path', 'tencent/Hunyuan3D-2')
            
            # Use current model paths instead of args to preserve any runtime changes
            current_model_path = self.model_path
            current_subfolder = self.subfolder
            
            self._init_models(
                current_model_path, 
                current_subfolder,
                self.device,
                getattr(args, 'low_vram_mode', False),
                not getattr(args, 'disable_flashvdm', False),
                enable_tex,
                tex_model_path
            )
            self.request_count = 0
            logger.info("Model reset complete")

    def _signal_handler(self, sig, frame):
        """Handle signals like Ctrl+C"""
        logger.info("Received interrupt signal, cleaning up and exiting...")
        # Clean up GPU resources
        torch.cuda.empty_cache()
        gc.collect()
        sys.exit(0)

    def update_status(self, uid, status, message=None, progress=0.0, phase=None):
        """Update the status of a generation request"""
        global generation_status
        
        # Convert UUID to string to ensure consistent key type
        uid_str = str(uid)
        
        if uid_str not in generation_status:
            generation_status[uid_str] = {"status": "processing", "message": "", "progress": 0.0, "phase": "starting"}
            logger.info(f"Created new status entry for UID: {uid_str}")
            
        if status:
            generation_status[uid_str]["status"] = status
        if message:
            generation_status[uid_str]["message"] = message
        if phase:
            generation_status[uid_str]["phase"] = phase
            
        # Always update progress
        generation_status[uid_str]["progress"] = progress
        
        # Log the current status for debugging
        # logger.info(f"Updated status for {uid_str}: phase={generation_status[uid_str]['phase']}, progress={generation_status[uid_str]['progress']:.2f}")

    @torch.inference_mode()
    def generate(self, uid, params):
        import time
        total_start_time = time.time()
        
        # Clear CUDA cache and force garbage collection at start
        torch.cuda.empty_cache()
        gc.collect()
        
        # Make sure models are unloaded if a new generation is starting with different params
        # This ensures we don't have stale data from previous generations
        prev_gen_was_multiview = self.multiview_loaded
        current_gen_is_multiview = params.get('multiview', False)
        
        # If we're switching modes, make sure to unload the previous model
        if prev_gen_was_multiview and not current_gen_is_multiview and self.multiview_loaded:
            self._unload_multiview_model()
        elif not prev_gen_was_multiview and current_gen_is_multiview and self.single_view_loaded:
            self._unload_single_view_model()
            
        # Always unload texture model if it's loaded
        if self.texture_loaded:
            self._unload_texture_model()
        
        # Convert UID to string to ensure consistency
        uid_str = str(uid)
        
        # Local image variables to ensure cleanup
        local_image = None
        local_multiview_images = {}
        
        try:
            # Set default parameters if not provided
            if 'num_inference_steps' not in params:
                params['num_inference_steps'] = 5  # Default to 5 steps for FlashVDM
            if 'guidance_scale' not in params:
                params['guidance_scale'] = 5.0
            
            # Handle model_path and subfolder if provided by the client
            model_path = params.get('model_path', self.model_path)
            subfolder = params.get('subfolder', self.subfolder)
            
            logger.info(f"Request parameters: Steps={params.get('num_inference_steps')}, Guidance={params.get('guidance_scale')}")
            logger.info(f"Model path: {model_path}, Subfolder: {subfolder}")
            
            uid = uuid.uuid4()
            
            # Filter out base64 data from log output
            filtered_params = {k: ('BASE64_DATA' if k in ['image', 'images', 'mesh'] or (isinstance(v, str) and len(v) > 100 and any(c in v for c in '+/=')) else v) 
                              for k, v in params.items()}
            logger.info(f"Request received for UID {uid_str} with params: {filtered_params}")
            self.update_status(uid_str, "processing", "Starting model generation...", 0.05, "initializing")
            
            # Check if this is a multiview request
            is_multiview = params.get('multiview', False)
            
            # Check if texture generation is requested - but don't load it yet!
            needs_texture = params.get('texture', False)
            
            load_start = time.time()
            
            # Process input based on generation type (single view, multiview, or text-to-image)
            if is_multiview:
                if 'images' in params:
                    try:
                        images_dict = params["images"]
                        # Process multiview images
                        self.update_status(uid_str, None, "Processing multiview images...", 0.1, "processing_images")
                        local_multiview_images = load_multiview_images_from_base64(images_dict)
                        logger.info(f"Loaded {len(local_multiview_images)} multiview images")
                        
                        # Apply background removal to each image if needed
                        self.update_status(uid_str, None, "Removing background from images...", 0.15, "background_removal")
                        for view, img in local_multiview_images.items():
                            if img.mode != 'RGBA':  # Only remove background if not already RGBA
                                local_multiview_images[view] = self.rembg(img)
                        
                        params['image'] = local_multiview_images  # Set the images for pipeline
                    except ValueError as e:
                        logger.error(f"Multiview image loading error: {str(e)}")
                        self.update_status(uid_str, "error", f"Failed to process multiview images: {str(e)}")
                        raise ValueError(f"Failed to process multiview images: {str(e)}")
                else:
                    self.update_status(uid_str, "error", "No multiview images provided")
                    raise ValueError("No multiview images provided for multiview generation")
            elif 'image' in params:
                try:
                    image_b64 = params["image"]
                    local_image = load_image_from_base64(image_b64)
                    logger.info(f"Loaded image, mode: {local_image.mode}")
                    self.update_status(uid_str, None, "Processing input image...", 0.1, "processing_image")
                except ValueError as e:
                    logger.error(f"Image loading error: {str(e)}")
                    self.update_status(uid_str, "error", f"Failed to process image: {str(e)}")
                    raise ValueError(f"Failed to process image: {str(e)}")
            else:
                if 'text' in params:
                    self.update_status(uid_str, None, "Processing text prompt...", 0.1, "processing_text")
                    text = params["text"]
                    
                    # First generate an image from text using the text-to-image pipeline
                    self.update_status(uid_str, None, "Generating image from text prompt...", 0.12, "text_to_image")
                    t2i_start = time.time()
                    try:
                        local_image = self.pipeline_t2i(text)
                        logger.info(f"Text-to-image generation time: {time.time() - t2i_start:.2f} seconds")
                        
                        # Ensure the image is properly converted to RGB if needed
                        if local_image.mode != 'RGB' and local_image.mode != 'RGBA':
                            local_image = local_image.convert('RGB')
                            
                        logger.info(f"Generated image size: {local_image.size} mode: {local_image.mode}")
                        
                        # Instead of adding to params, we'll treat this like a regular image input from now on
                        # to reuse the existing image processing code
                        
                        # We no longer need the text parameter
                        if 'text' in params:
                            del params['text']
                    except Exception as e:
                        error_msg = f"Failed to generate image from text: {str(e)}"
                        logger.error(error_msg)
                        self.update_status(uid_str, "error", error_msg)
                        raise ValueError(error_msg)
                else:
                    self.update_status(uid_str, "error", "No input image or text provided")
                    raise ValueError("No input image or text provided")
            logger.info(f"Image loading time: {time.time() - load_start:.2f} seconds")

            # Apply background removal for single-view image
            # For text-to-image, we need to remove background from local_image (not from params)
            if not is_multiview and local_image is not None and 'images' not in params:
                self.update_status(uid_str, None, "Removing background from image...", 0.15, "background_removal")
                rembg_start = time.time()
                local_image = self.rembg(local_image)
                logger.info(f"Background removal time: {time.time() - rembg_start:.2f} seconds")
                
                # Now put the processed image into params
                params['image'] = local_image

            # Save a reference to the image for texture generation
            texture_ref_image = None
            if is_multiview and 'front' in local_multiview_images:
                texture_ref_image = local_multiview_images['front']
            elif is_multiview and local_multiview_images:
                texture_ref_image = list(local_multiview_images.values())[0]
            elif 'image' in params:
                texture_ref_image = params['image']

            if 'mesh' in params:
                self.update_status(uid_str, None, "Processing existing mesh...", 0.2, "processing_mesh")
                mesh = trimesh.load(BytesIO(base64.b64decode(params["mesh"])), file_type='glb')
            else:
                # Load the appropriate shape generation model
                if is_multiview:
                    if not self.multiview_loaded:
                        self.update_status(uid_str, None, "Loading multiview model...", 0.15, "loading_model")
                        try:
                            # Use the provided model path and subfolder if specified
                            mv_model_path = params.get('model_path', self.mv_model_path)
                            mv_subfolder = params.get('subfolder', self.mv_subfolder)
                            logger.info(f"Loading multiview model from {mv_model_path}/{mv_subfolder}")
                            self._init_multiview_model(model_path=mv_model_path, subfolder=mv_subfolder)
                        except Exception as e:
                            error_msg = f"Failed to load multiview model: {str(e)}"
                            logger.error(error_msg)
                            self.update_status(uid_str, "error", error_msg)
                            raise ValueError(error_msg)
                else:
                    if not self.single_view_loaded:
                        self.update_status(uid_str, None, "Loading single view model...", 0.15, "loading_model")
                        try:
                            # Use the provided model path and subfolder if specified
                            sv_model_path = params.get('model_path', self.model_path)
                            sv_subfolder = params.get('subfolder', self.subfolder)
                            logger.info(f"Loading single view model from {sv_model_path}/{sv_subfolder}")
                            self._init_single_view_model(model_path=sv_model_path, subfolder=sv_subfolder)
                        except Exception as e:
                            error_msg = f"Failed to load single view model: {str(e)}"
                            logger.error(error_msg)
                            self.update_status(uid_str, "error", error_msg)
                            raise ValueError(error_msg)
                            
                # Now generate the shape
                seed = params.get("seed", 1234)
                # Create a new generator for each request
                generator = torch.Generator(self.device).manual_seed(seed)
                params['generator'] = generator
                params['octree_resolution'] = params.get("octree_resolution", 128)
                # Use 5 inference steps for FlashVDM as recommended
                params['num_inference_steps'] = params.get("num_inference_steps", 5)
                params['guidance_scale'] = params.get('guidance_scale', 5.0)
                
                # Print GPU memory usage before inference
                self._log_memory_usage("Before shape generation")
                
                # Clear CUDA cache before generation
                gc.collect()
                torch.cuda.empty_cache()
                
                # Choose appropriate pipeline based on generation type
                if is_multiview:
                    self.update_status(uid_str, None, f"Generating 3D shape from multiple views with {params.get('num_inference_steps', 5)} steps...", 0.2, "shape_generation")
                    inference_start = time.time()
                    logger.info(f"Starting multiview shape generation with {params.get('num_inference_steps', 5)} steps...")
                    # Add multiview-specific parameters if needed
                    params['num_chunks'] = params.get('num_chunks', 20000)  # Default for multiview from example
                    params['output_type'] = 'trimesh'
                    mesh = self.pipeline_mv(**params)[0]
                else:
                    self.update_status(uid_str, None, f"Generating 3D shape with {params.get('num_inference_steps', 5)} steps...", 0.2, "shape_generation")
                    inference_start = time.time()
                    logger.info(f"Starting shape generation with {params.get('num_inference_steps', 5)} steps...")
                    mesh = self.pipeline(**params)[0]
                    
                inference_time = time.time() - inference_start
                logger.info(f"Shape generation time: {inference_time:.2f} seconds")
                self.update_status(uid_str, None, "Shape generation complete!", 0.4, "shape_complete")
                
                # Clean up GPU memory
                del generator
                torch.cuda.empty_cache()
                gc.collect()
                
                # Remove the generator from params to free memory
                if 'generator' in params:
                    del params['generator']
                
                # IMPORTANT: Unload shape generation model before texturing to free up VRAM
                if is_multiview:
                    self._unload_multiview_model()
                else:
                    self._unload_single_view_model()

            # Clean up the image references from params to ensure no memory leaks
            if 'image' in params:
                del params['image']
            if 'images' in params:
                del params['images']

            if params.get('texture', False):
                # Now that shape model is unloaded, we can load the texture model
                self.update_status(uid_str, None, "Loading texture model...", 0.45, "loading_texture_model")
                try:
                    self._init_texture_model()
                except Exception as e:
                    error_msg = f"Failed to load texture model: {str(e)}"
                    logger.error(error_msg)
                    self.update_status(uid_str, "error", error_msg)
                    raise ValueError(error_msg)
                    
                texture_start = time.time()
                self.update_status(uid_str, None, "Starting mesh cleaning and texturing...", 0.45, "mesh_cleaning")
                logger.info("Starting mesh cleaning and texturing...")
                
                # First, clean the mesh - keep face count very low for better performance
                mesh_clean_start = time.time()
                self.update_status(uid_str, None, "Removing floaters...", 0.5, "removing_floaters")
                logger.info("Removing floaters...")
                mesh = self.floater_remover(mesh)
                
                self.update_status(uid_str, None, "Removing degenerate faces...", 0.55, "fixing_mesh")
                logger.info("Removing degenerate faces...")
                mesh = self.degenerate_face_remover(mesh)
                
                self.update_status(uid_str, None, "Reducing face count for texturing...", 0.6, "reducing_faces")
                logger.info("Reducing face count...")
                # Use a VERY low face count for faster texture generation
                max_facenum = params.get('face_count', 7500)  # Even lower than gradio app's 10,000 for speed
                mesh = self.face_reducer(mesh, max_facenum=max_facenum)
                logger.info(f"Mesh cleaning time: {time.time() - mesh_clean_start:.2f} seconds")
                
                # Apply texturing directly - no threads, no multiprocessing
                tex_start = time.time()
                self.update_status(uid_str, None, "Generating textures...", 0.7, "texture_generation")
                logger.info("Applying texture...")
                
                # Force GC to clear memory before texturing
                gc.collect()
                torch.cuda.empty_cache()
                
                try:
                    # Debug logging to track progress
                    logger.info("Starting texture pipeline...")
                    
                    # Use the saved reference image
                    if texture_ref_image is None:
                        raise ValueError("No valid reference image available for texturing")
                        
                    # Apply texturing with the reference image
                    mesh = self.pipeline_tex(mesh, texture_ref_image)
                    
                    logger.info("Texture generation complete")
                    self.update_status(uid_str, None, "Texture generation complete!", 0.9, "texture_complete")
                except KeyboardInterrupt:
                    logger.warning("Texture generation interrupted by user")
                    self.update_status(uid_str, "error", "Texture generation was interrupted by user")
                    raise ValueError("Texture generation was interrupted by user")
                except Exception as e:
                    logger.error(f"Error during texture generation: {str(e)}")
                    self.update_status(uid_str, "error", f"Texture generation failed: {str(e)}")
                    raise ValueError(f"Texture generation failed: {str(e)}")
                    
                logger.info(f"Texturing time: {time.time() - tex_start:.2f} seconds")
                logger.info(f"Total texturing time: {time.time() - texture_start:.2f} seconds")
                
                # Unload texture model to free up memory
                self._unload_texture_model()
                
                # Clean up memory after texturing
                torch.cuda.empty_cache()
                gc.collect()

            self.update_status(uid_str, None, "Exporting final model...", 0.95, "exporting")
            export_start = time.time()
            type = params.get('type', 'glb')
            with tempfile.NamedTemporaryFile(suffix=f'.{type}', delete=False) as temp_file:
                mesh.export(temp_file.name)
                mesh = trimesh.load(temp_file.name)
                save_path = os.path.join(SAVE_DIR, f'{str(uid_str)}.{type}')
                mesh.export(save_path)
            logger.info(f"Mesh export time: {time.time() - export_start:.2f} seconds")

            # Final cleanup
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"Total processing time: {time.time() - total_start_time:.2f} seconds")
            
            # Update status to completed
            self.update_status(uid_str, "completed", "Model generation complete!", 1.0, "complete")
            
            # Check if we need to reset the model
            self.maybe_reset_model()
            
            return save_path, uid_str
            
        finally:
            # Ensure cleanup happens even if something went wrong
            # Clear all local references to images
            local_image = None
            local_multiview_images.clear()
            texture_ref_image = None
            
            # Clean up CUDA memory
            torch.cuda.empty_cache()
            gc.collect()


app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 你可以指定允许的来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)


@app.get("/")
async def test_connection():
    """Simple endpoint to test if the server is running."""
    return JSONResponse({
        "status": "success",
        "message": "Unity server is connected and running!",
        "version": "Hunyuan3D-2"
    }, status_code=200)


@app.post("/generate")
async def generate(request: Request):
    logger.info("Worker generating...")
    start_time = time.time()
    params = await request.json()
    
    # Set default parameters if not provided
    if 'num_inference_steps' not in params:
        params['num_inference_steps'] = 5  # Default to 5 steps for FlashVDM
    if 'guidance_scale' not in params:
        params['guidance_scale'] = 5.0
    
    logger.info(f"Request parameters: Steps={params.get('num_inference_steps')}, Guidance={params.get('guidance_scale')}")
    
    uid = uuid.uuid4()
    try:
        # Clear CUDA cache before starting a new generation
        torch.cuda.empty_cache()
        
        logger.info("Starting model generation...")
        file_path, uid = worker.generate(uid, params)
        
        # Clear CUDA cache after generation completes
        torch.cuda.empty_cache()
        
        logger.info(f"Total request processing time: {time.time() - start_time:.2f} seconds")
        return FileResponse(file_path)
    except ValueError as e:
        traceback.print_exc()
        print("Caught ValueError:", e)
        torch.cuda.empty_cache()  # Try to free memory on error
        ret = {
            "text": f"Error: {str(e)}",
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except torch.cuda.CudaError as e:
        print("Caught torch.cuda.CudaError:", e)
        torch.cuda.empty_cache()  # Try to free up memory
        ret = {
            "text": f"GPU memory error: {str(e)}. Try using --low_vram_mode flag.",
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except Exception as e:
        print("Caught Unknown Error", e)
        traceback.print_exc()
        torch.cuda.empty_cache()  # Try to free memory on any error
        ret = {
            "text": f"Error: {str(e)}",
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)


@app.post("/send")
async def generate(request: Request):
    logger.info("Worker send...")
    params = await request.json()
    uid = str(uuid.uuid4())
    
    # Initialize status in global dictionary using string UID
    worker.update_status(uid, "processing", "Request queued, waiting to start...", 0.0, "queued")
    logger.info(f"Created generation task with UID: {uid}")
    
    threading.Thread(target=worker.generate, args=(uid, params,)).start()
    ret = {"uid": uid}
    return JSONResponse(ret, status_code=200)


@app.get("/status/{uid}")
async def status(uid: str):
    save_file_path = os.path.join(SAVE_DIR, f'{uid}.glb')
    
    global generation_status
    
    # # Debug log to see what's in the status dictionary
    # logger.info(f"Status check for UID: {uid}")
    # logger.info(f"Current generation_status keys: {list(generation_status.keys())}")
    
    # Check if we have status information for this UID
    if uid in generation_status:
        status_info = generation_status[uid]
        # logger.info(f"Found status for {uid}: {status_info}")
        
        # If model is completed and file exists, include the model data
        if status_info["status"] == "completed" and os.path.exists(save_file_path):
            try:
                with open(save_file_path, 'rb') as f:
                    base64_str = base64.b64encode(f.read()).decode()
                response = {
                    'status': 'completed',
                    'model_base64': base64_str,
                    'message': status_info.get('message', 'Generation complete'),
                    'progress': 1.0,
                    'phase': status_info.get('phase', 'complete')
                }
                # Once sent, we can clean up this UID from our status dictionary
                # Only clean up completed models to avoid issues with polling
                del generation_status[uid]
                return JSONResponse(response, status_code=200)
            except Exception as e:
                logger.error(f"Error reading result file: {str(e)}")
                response = {'status': 'error', 'message': str(e)}
                return JSONResponse(response, status_code=500)
        
        # Return current status for in-progress or error states
        response = {
            'status': status_info["status"],
            'message': status_info.get('message', ''),
            'progress': status_info.get('progress', 0.0),
            'phase': status_info.get('phase', 'processing')
        }
        return JSONResponse(response, status_code=200)
    
    # If no status info but file exists, it's completed
    if os.path.exists(save_file_path):
        try:
            with open(save_file_path, 'rb') as f:
                base64_str = base64.b64encode(f.read()).decode()
            response = {
                'status': 'completed', 
                'model_base64': base64_str,
                'progress': 1.0,
                'phase': 'complete'
            }
            return JSONResponse(response, status_code=200)
        except Exception as e:
            logger.error(f"Error reading result file: {str(e)}")
            response = {'status': 'error', 'message': str(e)}
            return JSONResponse(response, status_code=500)
    
    # Default response if no status info and no file
    logger.warning(f"No status info found for UID: {uid}")
    response = {'status': 'processing', 'progress': 0.0, 'phase': 'unknown'}
    return JSONResponse(response, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mini')
    parser.add_argument("--tex_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument("--texgen_model_path", type=str, help="Alias for --tex_model_path", dest="tex_model_path")
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-mini-turbo', 
                        help="Model subfolder to use, e.g., 'hunyuan3d-dit-v2-mini-turbo'")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument('--low_vram_mode', action='store_true', help="Enable low VRAM mode for devices with limited memory")
    parser.add_argument('--disable_flashvdm', action='store_true', help="Disable FlashVDM acceleration (not recommended)")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")
    
    # Multiview configuration (these are now used for configuration, not immediate loading)
    parser.add_argument('--mv_model_path', type=str, default='tencent/Hunyuan3D-2mv', help="Path to multiview model")
    parser.add_argument('--mv_subfolder', type=str, default='hunyuan3d-dit-v2-mv', help="Multiview model subfolder")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = build_logger("controller", f"{SAVE_DIR}/controller.log")
    logger.setLevel(log_level)
    
    logger.info(f"args: {args}")
    logger.info("Server starting in dynamic model loading mode. Models will be loaded on demand.")

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    # Register signal handler to handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.info("Received SIGINT, shutting down gracefully...")
        torch.cuda.empty_cache()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    worker = ModelWorker(
        model_path=args.model_path, 
        device=args.device, 
        enable_tex=False,  # We'll load texturing model on demand
        tex_model_path=args.tex_model_path, 
        subfolder=args.subfolder,
        low_vram_mode=args.low_vram_mode,
        enable_flashvdm=not args.disable_flashvdm,
        enable_multiview=True,  # Enable multiview support by default (model will load on demand)
        mv_model_path=args.mv_model_path,
        mv_subfolder=args.mv_subfolder
    )
    
    # Configure uvicorn with logging that shows startup but hides access logs
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn"]["level"] = "INFO"
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", log_config=log_config)
