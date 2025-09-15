import os
from typing import List, Dict

import cv2
import numpy as np

from paddlex.inference.models.text_detection import TextDetPredictor
from paddlex.inference.models.text_recognition import TextRecPredictor
from paddlex.inference.models.image_classification import ClasPredictor 

from paddlex.inference.pipelines.components import (
    CropByPolys,
    SortPolyBoxes,
    SortQuadBoxes,
    convert_points_to_boxes,
    rotate_image,
)
# optional
from paddlex.inference.pipelines.ocr.result import OCRResult


def stage1_load_and_batch_images(input_path: str):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    image_arrays = [img]
    input_paths = [input_path]
    return input_paths, image_arrays

def stage2_text_detection(
    image_arrays: List[np.ndarray],
    text_det_model,
    sort_boxes_alg,
):
    det_results = list(text_det_model(image_arrays))
    
    # Extract detection polygons (same as original line 361)
    dt_polys_list = [item["dt_polys"] for item in det_results]
    print(f"  → Detected text regions: {[len(polys) for polys in dt_polys_list]}")
    
    # Sort boxes (same as original line 363)
    dt_polys_list = [sort_boxes_alg(item) for item in dt_polys_list]
    
    return det_results, dt_polys_list

def stage3_crop_text_regions(
    image_arrays: List[np.ndarray], 
    dt_polys_list: List[np.ndarray],
    crop_by_polys_alg,
):    
    # Find indices with detections 
    indices = list(range(len(image_arrays)))
    indices = [idx for idx in indices if len(dt_polys_list[idx]) > 0]
    print(f"  → Images with text detections: {len(indices)}")
    
    if not indices:
        indices =  []
        all_subs_of_imgs =  []
        chunk_indices =  [0]
        return indices, all_subs_of_imgs, chunk_indices
    
    # Crop all sub-images 
    all_subs_of_imgs = []
    chunk_indices = [0]
    
    for idx in indices:
        print(f"    → Cropping image {idx} with {len(dt_polys_list[idx])} text regions...")
        all_subs_of_img = list(
            crop_by_polys_alg(image_arrays[idx], dt_polys_list[idx])
        )
        all_subs_of_imgs.extend(all_subs_of_img)
        chunk_indices.append(chunk_indices[-1] + len(all_subs_of_img))
    
    print(f"  → Total cropped regions: {len(all_subs_of_imgs)}")
    return indices, all_subs_of_imgs, chunk_indices

def stage4_text_orientation(
    all_subs_of_imgs: List[np.ndarray],
    textline_orientation_model,
):
    if textline_orientation_model is None or not all_subs_of_imgs:
        print("  → Skipping text orientation correction")
        angles = [-1] * len(all_subs_of_imgs)
        return all_subs_of_imgs, angles
        
    # Get orientation angles (same as original lines 406-411)
    angles = [
        int(textline_angle_info["class_ids"][0])
        for textline_angle_info in textline_orientation_model(all_subs_of_imgs)
    ]
    
    # Rotate images based on angles (same as original line 412)
    all_subs_of_imgs = rotate_images_by_angles(all_subs_of_imgs, angles)
    
    print(f"  → Corrected orientation for {len(all_subs_of_imgs)} regions")
    print(f"  → Angle distribution: {dict(zip(*np.unique(angles, return_counts=True)))}")
    
    return all_subs_of_imgs, angles

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

def stage5_text_recognition(
    input_paths: List[str],
    indices: List[int],
    all_subs_of_imgs: List[np.ndarray],
    chunk_indices: List[int],
    text_dt_polys_list: List[np.ndarray],
    text_rec_model,
    config,
    text_rec_score_thresh: float
):
    
    # Initialize results structure (same as original lines 365-387)
    results = []
    for page_index, (input_path, dt_polys) in enumerate(zip(input_paths, text_dt_polys_list)):
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
        return results
        
    # Process each image with detections (same as original lines 422-467)
    
    for i, idx in enumerate(indices):
        # Get sub-images for this main image
        all_subs_of_img = all_subs_of_imgs[chunk_indices[i]:chunk_indices[i + 1]]
        res = results[idx]
        dt_polys = text_dt_polys_list[idx]
        
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
        for j, rec_res in enumerate(text_rec_model(sorted_subs_of_img)):
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
    
    return results

def stage6_format_results(
    image_arrays: List[np.ndarray], 
    indices: List[int],
    chunk_indices: List[int],
    angles: List[int],
    results: List[Dict],
    config: Dict,
    text_rec_score_thresh: float = 0.0,
):    
    ocr_results = []
    
    for i, result in enumerate(results):
        # Create doc preprocessor result (required by OCRResult)
        doc_preprocessor_res = {
            "output_img": image_arrays[i]  # Original processed image
        }
        
        use_textline_orientation = config.get("use_textline_orientation", True)
        # Create model settings (required by OCRResult)
        model_settings = {
            "use_doc_preprocessor": False,  # We didn't use doc preprocessor in this example
            "use_textline_orientation": use_textline_orientation
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
            "page_index": i, 
            "doc_preprocessor_res": doc_preprocessor_res,
            "dt_polys": result['dt_polys'],
            "model_settings": model_settings,
            "text_det_params": text_det_params,
            "text_type": config.get("text_type", "general"),
            "text_rec_score_thresh": text_rec_score_thresh,
            "return_word_box": False,  # We didn't implement word boxes in this example
            "rec_texts": result['rec_texts'],
            "rec_scores": result['rec_scores'],
            "rec_polys": result['rec_polys'],
            "rec_boxes": result.get('rec_boxes', []),
            "vis_fonts": vis_fonts
        }
        
        # Add textline orientation angles if available
        if len(angles) != 0 and use_textline_orientation:
            # Map angles back to results for each image
            if i in indices:
                idx_in_indices = indices.index(i)
                if idx_in_indices < len(chunk_indices) - 1:
                    start_idx = chunk_indices[idx_in_indices]
                    end_idx = chunk_indices[idx_in_indices + 1]
                    ocr_result_data["textline_orientation_angles"] = angles[start_idx:end_idx]
        
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

def run_manual_ocr_pipeline(
    img_path: str,
    text_det_model,
    text_rec_model,
    textline_orientation_model,
    text_rec_score_thresh,
    sort_boxes_alg,
    crop_by_polys_alg,
    config: Dict,
):

    input_paths, image_arrays = stage1_load_and_batch_images(
        input_path=img_path
    )
    text_det_results, text_dt_polys_list = stage2_text_detection(
        image_arrays=image_arrays,
        text_det_model=text_det_model,
        sort_boxes_alg=sort_boxes_alg,
    )
    indices, all_subs_of_imgs, chunk_indices = stage3_crop_text_regions(
        image_arrays=image_arrays,
        dt_polys_list=text_dt_polys_list,
        crop_by_polys_alg=crop_by_polys_alg,
    )
    all_subs_of_imgs, angles = stage4_text_orientation(
        all_subs_of_imgs=all_subs_of_imgs,
        textline_orientation_model=textline_orientation_model,
    )
    results = stage5_text_recognition(
        input_paths=input_paths,
        indices=indices,
        all_subs_of_imgs=all_subs_of_imgs,
        chunk_indices=chunk_indices,
        text_dt_polys_list=text_dt_polys_list,
        text_rec_model=text_rec_model,
        config=config,
        text_rec_score_thresh=text_rec_score_thresh,
    )
    results = stage6_format_results(
        image_arrays=image_arrays,
        indices=indices,
        chunk_indices=chunk_indices,
        angles=angles,
        results=results,
        config=config,
        text_rec_score_thresh=text_rec_score_thresh,
    )
    return results

def save_results(results, output_dir):
    
    output_paths = []
    for result in results:
        # Use original OCRResult.save_to_img method for proper visualization
        result.save_to_img(output_dir)
        input_filename = os.path.basename(result['input_path'])
        expected_files = [f for f in os.listdir(output_dir) if input_filename in f]
        
        if expected_files:
            output_path = os.path.join(output_dir, expected_files[-1])  # Get most recent
            output_paths.append(output_path)
            print(f"  → Saved with original PaddleX visualization: {output_path}")
        else:
            print(f"  → Warning: Could not find output file for {input_filename}")
    
    return output_paths   
    

def main(
    img_path: str, 
):    

    CHECKPOINTS_DIR = "./checkpoints"
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
    
    # Initialize text detection model (same parameters as original)
    text_det_config = config["SubModules"]["TextDetection"]
    text_det_model = TextDetPredictor(
        model_dir=text_det_config["model_dir"],
        limit_side_len=text_det_config.get("limit_side_len", 960),
        limit_type=text_det_config.get("limit_type", "max"),
        max_side_limit=text_det_config.get("max_side_limit", 4000),
        thresh=text_det_config.get("thresh", 0.3),
        box_thresh=text_det_config.get("box_thresh", 0.6),
        unclip_ratio=text_det_config.get("unclip_ratio", 2.0)
    )    

    # Initialize text recognition model
    text_rec_config = config["SubModules"]["TextRecognition"]
    text_rec_model = TextRecPredictor(
        model_dir=text_rec_config["model_dir"]
    )
    # Store score threshold separately (used during processing)
    text_rec_score_thresh = text_rec_config.get("score_thresh", 0.0)
    
    # Initialize text orientation model (optional)
    if config.get("use_textline_orientation", True):
        textline_orientation_config = config["SubModules"]["TextLineOrientation"]
        textline_orientation_model = ClasPredictor(
            model_dir=textline_orientation_config["model_dir"]
        )
    else:
        textline_orientation_model = None
    
    # Initialize components based on text type (same logic as original)
    text_type = config.get("text_type", "general")
    if text_type == "general":
        sort_boxes_alg = SortQuadBoxes()
        crop_by_polys_alg = CropByPolys(det_box_type="quad")
    elif text_type == "seal":
        sort_boxes_alg = SortPolyBoxes()
        crop_by_polys_alg = CropByPolys(det_box_type="poly")
    else:
        raise ValueError(f"Unsupported text type {text_type}")
    
    results = run_manual_ocr_pipeline(
        img_path=img_path,
        text_det_model=text_det_model,
        text_rec_model=text_rec_model,
        textline_orientation_model=textline_orientation_model,
        text_rec_score_thresh=text_rec_score_thresh,
        sort_boxes_alg=sort_boxes_alg,
        crop_by_polys_alg=crop_by_polys_alg,
        config=config
    )    
    return results
    
    
if __name__ == "__main__":
    img_path = "../data/some_test_data/en/PXL_20250819_103421347.jpg"
    output_dir = "./output/manual"
    os.makedirs(output_dir, exist_ok=True)    

    results = main(img_path=img_path)

    output_paths = save_results(results, output_dir=output_dir)

    print(f"\n Manual Pipeline Test Results:")
    print(f"  • Input: {os.path.basename(img_path)}")
    print(f"  • Detected texts: {len(results[0]['rec_texts'])}")
    print(f"  • Sample texts: {results[0]['rec_texts'][:3]}")
    print(f"  • Output saved to: {output_paths}")