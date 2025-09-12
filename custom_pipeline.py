import os
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Union, Optional

from paddlex.inference.models.text_detection import TextDetPredictor
from paddlex.inference.models.text_recognition import TextRecPredictor
from paddlex.inference.models.image_classification import ClasPredictor 
from paddlex.inference.common.batch_sampler import ImageBatchSampler
from paddlex.inference.common.reader import ReadImage

from paddlex.inference.pipelines.components import (
    CropByPolys,
    SortPolyBoxes,
    SortQuadBoxes,
    convert_points_to_boxes,
    rotate_image,
)
from paddlex.inference.pipelines.ocr.result import OCRResult
from paddlex.utils import logging

logger = logging._logger

CHECKPOINTS_DIR = "./checkpoints"
_models = {}

def initialize_manual_pipeline(config=None):
    """Initialize individual OCR components manually based on original pipeline logic"""
    print("=== Initializing Manual OCR Components ===")
    
    global _models
    
    # Default configuration for general text (based on original pipeline)
    if config is None:
        config = {
            "text_type": "general",
            "use_textline_orientation": True,
            "batch_size": 1,
            "SubModules": {
                "TextDetection": {
                    "model_dir": f"{CHECKPOINTS_DIR}/PP-OCRv5_mobile_det",
                    "limit_side_len": 960,
                    "limit_type": "max",
                    "max_side_limit": 4000,
                    "thresh": 0.3,
                    "box_thresh": 0.6,
                    "unclip_ratio": 2.0
                },
                "TextRecognition": {
                    "model_dir": f"{CHECKPOINTS_DIR}/en_PP-OCRv5_mobile_rec",
                    "score_thresh": 0.0,
                    "return_word_box": False
                },
                "TextLineOrientation": {
                    "model_dir": f"{CHECKPOINTS_DIR}/PP-LCNet_x1_0_textline_ori"
                }
            }
        }
    
    # Store configuration
    _models['config'] = config
    
    # Initialize batch sampler and image reader (same as original)
    print("1. Initializing Batch Sampler and Image Reader...")
    _models['batch_sampler'] = ImageBatchSampler(batch_size=config.get("batch_size", 1))
    _models['img_reader'] = ReadImage(format="BGR")
    
    # Initialize text detection model (same parameters as original)
    print("2. Initializing Text Detection Model...")
    text_det_config = config["SubModules"]["TextDetection"]
    
    _models['text_det_model'] = TextDetPredictor(
        model_dir=text_det_config["model_dir"],
        limit_side_len=text_det_config.get("limit_side_len", 960),
        limit_type=text_det_config.get("limit_type", "max"),
        max_side_limit=text_det_config.get("max_side_limit", 4000),
        thresh=text_det_config.get("thresh", 0.3),
        box_thresh=text_det_config.get("box_thresh", 0.6),
        unclip_ratio=text_det_config.get("unclip_ratio", 2.0)
    )
    print(f"   ✓ Text detection model loaded from: {text_det_config['model_dir']}")
    
    # Initialize text recognition model
    print("3. Initializing Text Recognition Model...")
    text_rec_config = config["SubModules"]["TextRecognition"]
    
    _models['text_rec_model'] = TextRecPredictor(
        model_dir=text_rec_config["model_dir"]
    )
    # Store score threshold separately (used during processing)
    _models['text_rec_score_thresh'] = text_rec_config.get("score_thresh", 0.0)
    print(f"   ✓ Text recognition model loaded from: {text_rec_config['model_dir']}")
    
    # Initialize text orientation model (optional)
    print("4. Initializing Text Orientation Model...")
    if config.get("use_textline_orientation", True):
        try:
            textline_orientation_config = config["SubModules"]["TextLineOrientation"]
            _models['textline_orientation_model'] = ClasPredictor(
                model_dir=textline_orientation_config["model_dir"]
            )
            _models['use_textline_orientation'] = True
            print(f"   ✓ Text orientation model loaded from: {textline_orientation_config['model_dir']}")
        except Exception as e:
            print(f"   ⚠ Text orientation model not available: {e}")
            _models['textline_orientation_model'] = None
            _models['use_textline_orientation'] = False
    else:
        _models['textline_orientation_model'] = None
        _models['use_textline_orientation'] = False
    
    # Initialize components based on text type (same logic as original)
    text_type = config.get("text_type", "general")
    print(f"5. Initializing Components for text_type: {text_type}...")
    
    if text_type == "general":
        _models['sort_boxes'] = SortQuadBoxes()
        _models['crop_by_polys'] = CropByPolys(det_box_type="quad")
    elif text_type == "seal":
        _models['sort_boxes'] = SortPolyBoxes()
        _models['crop_by_polys'] = CropByPolys(det_box_type="poly")
    else:
        raise ValueError(f"Unsupported text type {text_type}")
    
    return True

def stage1_load_and_batch_images(input_path: str):
    """
    STAGE 1: Load and batch images (same as original pipeline lines 339-340)
    Input: image file path
    Output: batched image data
    """
    print(f"\n=== STAGE 1: Load and Batch Images ===")
    print(f"  → Input image path: {input_path}")
    
    batch_data_generator = _models['batch_sampler']([input_path])
    batch_data = next(batch_data_generator)  # Get first batch
    print(f"  → Batch created with {len(batch_data.instances)} items")
    
    image_arrays = _models['img_reader'](batch_data.instances)
    print(f"  → Image loaded with shape: {image_arrays[0].shape} (H×W×C)")
    
    return {
        'batch_data': batch_data,
        'image_arrays': image_arrays,
        'input_paths': batch_data.input_paths
    }

def stage2_text_detection(data: Dict):
    """
    STAGE 2: Text Detection (same as original pipeline lines 357-363) 
    Input: image arrays
    Output: detection results with sorted boxes
    """
    print(f"\n=== STAGE 2: Text Detection ===")
    
    image_arrays = data['image_arrays']
    print("  → Running text detection model...")
    
    # Run text detection (same as original line 358)
    det_results = list(_models['text_det_model'](image_arrays))
    
    # Extract detection polygons (same as original line 361)
    dt_polys_list = [item["dt_polys"] for item in det_results]
    print(f"  → Detected text regions: {[len(polys) for polys in dt_polys_list]}")
    
    # Sort boxes (same as original line 363)
    dt_polys_list = [_models['sort_boxes'](item) for item in dt_polys_list]
    print("  → Boxes sorted using original sorting logic")
    
    data.update({
        'det_results': det_results,
        'dt_polys_list': dt_polys_list
    })
    
    return data

def stage3_crop_text_regions(data: Dict):
    """
    STAGE 3: Crop text regions (same as original pipeline lines 393-402)
    Input: images and detection polygons
    Output: cropped text region images
    """
    print(f"\n=== STAGE 3: Crop Text Regions ===")
    
    image_arrays = data['image_arrays']
    dt_polys_list = data['dt_polys_list']
    
    # Find indices with detections (same as original lines 389-390)
    indices = list(range(len(image_arrays)))
    indices = [idx for idx in indices if len(dt_polys_list[idx]) > 0]
    print(f"  → Images with text detections: {len(indices)}")
    
    if not indices:
        print("  → No text regions to crop")
        data.update({
            'indices': [],
            'all_subs_of_imgs': [],
            'chunk_indices': [0]
        })
        return data
    
    # Crop all sub-images (same as original lines 393-402)
    all_subs_of_imgs = []
    chunk_indices = [0]
    
    for idx in indices:
        print(f"    → Cropping image {idx} with {len(dt_polys_list[idx])} text regions...")
        all_subs_of_img = list(
            _models['crop_by_polys'](image_arrays[idx], dt_polys_list[idx])
        )
        all_subs_of_imgs.extend(all_subs_of_img)
        chunk_indices.append(chunk_indices[-1] + len(all_subs_of_img))
    
    print(f"  → Total cropped regions: {len(all_subs_of_imgs)}")
    
    data.update({
        'indices': indices,
        'all_subs_of_imgs': all_subs_of_imgs,
        'chunk_indices': chunk_indices
    })
    
    return data

def stage4_text_orientation(data: Dict):
    """
    STAGE 4: Text orientation correction (same as original pipeline lines 405-419)
    Input: cropped text images
    Output: orientation-corrected images
    """
    print(f"\n=== STAGE 4: Text Orientation Correction ===")
    
    all_subs_of_imgs = data['all_subs_of_imgs']
    
    if not _models['use_textline_orientation'] or not all_subs_of_imgs:
        print("  → Skipping text orientation correction")
        angles = [-1] * len(all_subs_of_imgs)
        data['angles'] = angles
        return data
    
    print("  → Running text orientation model...")
    
    # Get orientation angles (same as original lines 406-411)
    angles = [
        int(textline_angle_info["class_ids"][0])
        for textline_angle_info in _models['textline_orientation_model'](all_subs_of_imgs)
    ]
    
    # Rotate images based on angles (same as original line 412)
    all_subs_of_imgs = rotate_images_by_angles(all_subs_of_imgs, angles)
    
    print(f"  → Corrected orientation for {len(all_subs_of_imgs)} regions")
    print(f"  → Angle distribution: {dict(zip(*np.unique(angles, return_counts=True)))}")
    
    data.update({
        'all_subs_of_imgs': all_subs_of_imgs,
        'angles': angles
    })
    
    return data

def rotate_images_by_angles(image_array_list: List[np.ndarray], rotate_angle_list: List[int]) -> List[np.ndarray]:
    """
    Rotate images by angles (same as original pipeline lines 141-175)
    0 corresponds to 0 degrees, 1 corresponds to 180 degrees
    """
    assert len(image_array_list) == len(rotate_angle_list), \
        f"Length mismatch: {len(image_array_list)} vs {len(rotate_angle_list)}"
    
    rotated_images = []
    for image_array, rotate_indicator in zip(image_array_list, rotate_angle_list):
        if rotate_indicator == 1:
            # Rotate by 180 degrees
            rotate_angle = 180
            rotated_image = rotate_image(image_array, rotate_angle)
            rotated_images.append(rotated_image)
        else:
            # No rotation needed
            rotated_images.append(image_array)
    
    return rotated_images

def stage5_text_recognition(data: Dict):
    """
    STAGE 5: Text Recognition (same as original pipeline lines 422-467)
    Input: orientation-corrected cropped images
    Output: recognized texts and scores
    """
    print(f"\n=== STAGE 5: Text Recognition ===")
    
    indices = data['indices']
    all_subs_of_imgs = data['all_subs_of_imgs']
    chunk_indices = data['chunk_indices']
    dt_polys_list = data['dt_polys_list']
    batch_data = data['batch_data']
    config = _models['config']
    
    # Initialize results structure (same as original lines 365-387)
    results = []
    for input_path, page_index, dt_polys in zip(
        batch_data.input_paths,
        batch_data.page_indexes,
        dt_polys_list
    ):
        result = {
            "input_path": input_path,
            "page_index": page_index,
            "dt_polys": dt_polys,
            "text_type": config.get("text_type", "general"),
            "rec_texts": [],
            "rec_scores": [],
            "rec_polys": [],
        }
        results.append(result)
    
    if not indices:
        print("  → No text regions to recognize")
        data['results'] = results
        return data
    
    print(f"  → Recognizing text in {len(all_subs_of_imgs)} cropped regions...")
    
    # Process each image with detections (same as original lines 422-467)
    text_rec_score_thresh = _models['text_rec_score_thresh']
    
    for i, idx in enumerate(indices):
        # Get sub-images for this main image
        all_subs_of_img = all_subs_of_imgs[chunk_indices[i]:chunk_indices[i + 1]]
        res = results[idx]
        dt_polys = dt_polys_list[idx]
        
        # Sort sub-images by aspect ratio (same as original lines 428-440)
        sub_img_info_list = [
            {
                "sub_img_id": img_id,
                "sub_img_ratio": sub_img.shape[1] / float(sub_img.shape[0]),
            }
            for img_id, sub_img in enumerate(all_subs_of_img)
        ]
        sorted_subs_info = sorted(sub_img_info_list, key=lambda x: x["sub_img_ratio"])
        sorted_subs_of_img = [all_subs_of_img[x["sub_img_id"]] for x in sorted_subs_info]
        
        print(f"    → Processing {len(sorted_subs_of_img)} regions for image {idx}...")
        
        # Run text recognition (same as original lines 441-447)
        for j, rec_res in enumerate(_models['text_rec_model'](sorted_subs_of_img)):
            sub_img_id = sorted_subs_info[j]["sub_img_id"]
            sub_img_info_list[sub_img_id]["rec_res"] = rec_res
        
        # Collect results above threshold (same as original lines 451-467)
        vis_fonts = []
        for sno in range(len(sub_img_info_list)):
            rec_res = sub_img_info_list[sno]["rec_res"]
            if rec_res["rec_score"] >= text_rec_score_thresh:
                res["rec_texts"].append(rec_res["rec_text"])
                res["rec_scores"].append(rec_res["rec_score"])
                res["rec_polys"].append(dt_polys[sno])
                # Collect vis_fonts (required for proper visualization)
                vis_fonts.append(rec_res.get("vis_font", None))
        
        # Store vis_fonts in result
        res["vis_fonts"] = vis_fonts
    
    # Convert polygons to boxes for general text type (same as original lines 469-471)
    for res in results:
        if config.get("text_type") == "general":
            rec_boxes = convert_points_to_boxes(res["rec_polys"])
            res["rec_boxes"] = rec_boxes
        else:
            res["rec_boxes"] = np.array([])
    
    print(f"  → Recognition complete. Results: {[len(r['rec_texts']) for r in results]}")
    
    data['results'] = results
    return data

def stage6_format_results(data: Dict):
    """
    STAGE 6: Format final results using original OCRResult class
    Input: recognition results
    Output: OCRResult objects with proper visualization support
    """
    print(f"\n=== STAGE 6: Format Results ===")
    
    results = data['results']
    batch_data = data['batch_data']
    image_arrays = data['image_arrays']
    config = _models['config']
    
    ocr_results = []
    
    for i, result in enumerate(results):
        # Create doc preprocessor result (required by OCRResult)
        doc_preprocessor_res = {
            "output_img": image_arrays[i]  # Original processed image
        }
        
        # Create model settings (required by OCRResult)
        model_settings = {
            "use_doc_preprocessor": False,  # We didn't use doc preprocessor in this example
            "use_textline_orientation": config.get("use_textline_orientation", True)
        }
        
        # Create text detection parameters (required by OCRResult)
        text_det_params = {
            "limit_side_len": config["SubModules"]["TextDetection"].get("limit_side_len", 960),
            "thresh": config["SubModules"]["TextDetection"].get("thresh", 0.3),
            "box_thresh": config["SubModules"]["TextDetection"].get("box_thresh", 0.6),
            "unclip_ratio": config["SubModules"]["TextDetection"].get("unclip_ratio", 2.0)
        }
        
        # Get vis_fonts from recognition results (if available)
        vis_fonts = []
        if 'vis_fonts' in result:
            vis_fonts = result['vis_fonts']
        else:
            # Create default vis_fonts list
            vis_fonts = [None] * len(result['rec_texts'])
        
        # Create OCRResult with all required fields (same format as original pipeline)
        ocr_result_data = {
            "input_path": result['input_path'],
            "page_index": batch_data.page_indexes[i],
            "doc_preprocessor_res": doc_preprocessor_res,
            "dt_polys": result['dt_polys'],
            "model_settings": model_settings,
            "text_det_params": text_det_params,
            "text_type": config.get("text_type", "general"),
            "text_rec_score_thresh": _models['text_rec_score_thresh'],
            "return_word_box": False,  # We didn't implement word boxes in this example
            "rec_texts": result['rec_texts'],
            "rec_scores": result['rec_scores'],
            "rec_polys": result['rec_polys'],
            "rec_boxes": result.get('rec_boxes', []),
            "vis_fonts": vis_fonts
        }
        
        # Add textline orientation angles if available
        if 'angles' in data and data['angles']:
            # Map angles back to results for each image
            chunk_indices = data.get('chunk_indices', [0, len(data.get('angles', []))])
            indices = data.get('indices', [])
            if i in indices:
                idx_in_indices = indices.index(i)
                if idx_in_indices < len(chunk_indices) - 1:
                    start_idx = chunk_indices[idx_in_indices]
                    end_idx = chunk_indices[idx_in_indices + 1]
                    ocr_result_data["textline_orientation_angles"] = data['angles'][start_idx:end_idx]
        
        # Create OCRResult object using original class
        ocr_result = OCRResult(ocr_result_data)
        ocr_results.append(ocr_result)
        
        print(f"  → Image: {os.path.basename(result['input_path'])}")
        print(f"    Total detected regions: {len(result['dt_polys'])}")
        print(f"    Successfully recognized: {len(result['rec_texts'])}")
        if result['rec_texts']:
            avg_score = np.mean(result['rec_scores'])
            print(f"    Average confidence: {avg_score:.3f}")
    
    return ocr_results

def run_manual_ocr_pipeline(image_path: str, config=None):
    """
    Complete manual OCR pipeline replicating original PaddleX pipeline logic
    Each stage corresponds exactly to the original pipeline implementation
    """
    print(f"\n{'='*70}")
    print(f"MANUAL OCR PIPELINE: {os.path.basename(image_path)}")
    print(f"{'='*70}")
    
    # STAGE 1: Load and batch images
    data = stage1_load_and_batch_images(image_path)
    
    # STAGE 2: Text detection
    data = stage2_text_detection(data)
    
    # STAGE 3: Crop text regions
    data = stage3_crop_text_regions(data)
    
    # STAGE 4: Text orientation correction
    data = stage4_text_orientation(data)
    
    # STAGE 5: Text recognition
    data = stage5_text_recognition(data)
    
    # STAGE 6: Format results
    results = stage6_format_results(data)
    
    print(f"\n{'='*70}")
    print("MANUAL PIPELINE COMPLETE!")
    print(f"{'='*70}")
    
    return results

def save_results(results, output_dir="output/manual"):
    """Save OCR results using original PaddleX visualization"""
    print(f"\n=== Saving Results (Original PaddleX Format) ===")
    
    output_paths = []
    for result in results:
        # Use original OCRResult.save_to_img method for proper visualization
        result.save_to_img(output_dir)
        
        # The original save method creates files with specific naming
        input_filename = os.path.basename(result['input_path'])
        # Find the actual output file created by save_to_img
        expected_files = [f for f in os.listdir(output_dir) if input_filename in f]
        
        if expected_files:
            output_path = os.path.join(output_dir, expected_files[-1])  # Get most recent
            output_paths.append(output_path)
            print(f"  → Saved with original PaddleX visualization: {output_path}")
        else:
            print(f"  → Warning: Could not find output file for {input_filename}")
    
    return output_paths   
    

def main():
    """
    Main function demonstrating manual OCR pipeline
    
    This implementation exactly replicates the original PaddleX OCR pipeline
    but makes each processing stage explicit and transparent:
    
    1. stage1_load_and_batch_images() → Batch creation and image loading
    2. stage2_text_detection() → Text detection with box sorting  
    3. stage3_crop_text_regions() → Text region cropping
    4. stage4_text_orientation() → Text orientation correction
    5. stage5_text_recognition() → Text recognition with confidence filtering
    6. stage6_format_results() → Result formatting
    
    Each stage uses the exact same logic, components, and parameters as the
    original pipeline, just organized into separate functions for clarity.
    """
    
    os.makedirs("output/manual", exist_ok=True)    

    config = {
        "text_type": "general",
        "use_textline_orientation": True,
        "batch_size": 1,
        "SubModules": {
            "TextDetection": {
                "model_dir": f"{CHECKPOINTS_DIR}/PP-OCRv5_mobile_det",
                "limit_side_len": 960,
                "thresh": 0.3,
                "box_thresh": 0.6,
                "unclip_ratio": 2.0
            },
            "TextRecognition": {
                "model_dir": f"{CHECKPOINTS_DIR}/en_PP-OCRv5_mobile_rec",
                "score_thresh": 0.0
            },
            "TextLineOrientation": {
                "model_dir": f"{CHECKPOINTS_DIR}/PP-LCNet_x1_0_textline_ori"
            }
        }
    }
    
    initialize_manual_pipeline(config)    
    test_image = "../data/some_test_data/en/PXL_20250819_103421347.jpg"
    results = run_manual_ocr_pipeline(test_image, config)    
    output_paths = save_results(results)
    
    print(f"\n Manual Pipeline Test Results:")
    print(f"  • Input: {os.path.basename(test_image)}")
    print(f"  • Detected texts: {len(results[0]['rec_texts'])}")
    print(f"  • Sample texts: {results[0]['rec_texts'][:3]}")
    print(f"  • Output saved to: {output_paths}")
    

if __name__ == "__main__":
    main()