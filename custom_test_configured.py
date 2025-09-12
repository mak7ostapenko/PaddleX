"""
PaddleX OCR Test with Custom Model Configuration
Shows different ways to configure pipelines and models
"""

from paddlex import create_pipeline
import os

def test_default_pipeline():
    """Test with default OCR pipeline (server models)"""
    print("=== Method 1: Default Pipeline ===")
    pipeline = create_pipeline(pipeline="OCR")
    
    output = pipeline.predict(
        "../data/some_test_data/en/PXL_20250819_103421347.jpg",
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
    )
    
    for res in output:
        print(f"✓ Detected {len(res['rec_texts'])} text regions")
        res.save_to_img("./output/default/")
    print()

def test_custom_config_pipeline():
    """Test with custom pipeline configuration"""
    print("=== Method 2: Custom Configuration ===")
    
    # Custom configuration with mobile models for faster inference
    custom_config = {
        "pipeline_name": "OCR",
        "text_type": "general",
        "use_doc_preprocessor": True,
        "use_textline_orientation": True,
        
        "SubPipelines": {
            "DocPreprocessor": {
                "pipeline_name": "doc_preprocessor",
                "use_doc_orientation_classify": True,
                "use_doc_unwarping": True,
                "SubModules": {
                    "DocOrientationClassify": {
                        "module_name": "doc_text_orientation",
                        "model_name": "PP-LCNet_x1_0_doc_ori",
                        "model_dir": "checkpoints/PP-LCNet_x1_0_doc_ori"
                    },
                    "DocUnwarping": {
                        "module_name": "image_unwarping", 
                        "model_name": "UVDoc",
                        "model_dir": "checkpoints/UVDoc"
                    }
                }
            }
        },
        
        "SubModules": {
            "TextDetection": {
                "module_name": "text_detection",
                "model_name": "PP-OCRv5_mobile_det",  # Mobile version for speed
                "model_dir": "checkpoints/PP-OCRv5_mobile_det",
                "limit_side_len": 64,
                "limit_type": "min", 
                "max_side_limit": 4000,
                "thresh": 0.3,
                "box_thresh": 0.6,
                "unclip_ratio": 1.5
            },
            "TextLineOrientation": {
                "module_name": "textline_orientation",
                "model_name": "PP-LCNet_x1_0_textline_ori",
                "model_dir": "checkpoints/PP-LCNet_x1_0_textline_ori",
                "batch_size": 6
            },
            "TextRecognition": {
                "module_name": "text_recognition",
                "model_name": "en_PP-OCRv5_mobile_rec",  # English mobile model
                "model_dir": "checkpoints/en_PP-OCRv5_mobile_rec",
                "batch_size": 6,
                "score_thresh": 0.0
            }
        }
    }
    
    pipeline = create_pipeline(config=custom_config)
    
    output = pipeline.predict(
        "../data/some_test_data/en/PXL_20250819_103421347.jpg",
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
    )
    
    for res in output:
        print(f"✓ Mobile models detected {len(res['rec_texts'])} text regions")
        res.save_to_img("./output/mobile/")
    print()

def test_layout_parsing_pipeline():
    """Test layout parsing with custom model paths"""
    print("=== Method 3: Layout Parsing Pipeline ===")
    
    try:
        # Custom layout parsing config
        layout_config = {
            "pipeline_name": "layout_parsing",
            "use_doc_preprocessor": True,
            "use_seal_recognition": False,  # Disable for faster testing
            "use_table_recognition": False,
            "use_formula_recognition": False,
            
            "SubModules": {
                "LayoutDetection": {
                    "module_name": "layout_detection",
                    "model_name": "RT-DETR-H_layout_17cls",
                    "model_dir": "checkpoints/RT-DETR-H_layout_17cls"
                }
            },
            
            "SubPipelines": {
                "DocPreprocessor": {
                    "pipeline_name": "doc_preprocessor",
                    "use_doc_orientation_classify": True,
                    "use_doc_unwarping": True,
                    "SubModules": {
                        "DocOrientationClassify": {
                            "module_name": "doc_text_orientation",
                            "model_name": "PP-LCNet_x1_0_doc_ori",
                            "model_dir": "checkpoints/PP-LCNet_x1_0_doc_ori"
                        },
                        "DocUnwarping": {
                            "module_name": "image_unwarping",
                            "model_name": "UVDoc",
                            "model_dir": "checkpoints/UVDoc"
                        }
                    }
                }
            }
        }
        
        pipeline = create_pipeline(config=layout_config)
        print("✓ Layout parsing pipeline created successfully with local models!")
        
    except Exception as e:
        print(f"✗ Layout parsing error: {str(e)[:100]}...")
    print()

def test_model_comparison():
    """Compare different model configurations"""
    print("=== Method 4: Model Performance Comparison ===")
    
    test_image = "../data/some_test_data/en/PXL_20250819_103421347.jpg"
    
    # Server models configuration (accuracy)
    server_config = {
        "pipeline_name": "OCR",
        "text_type": "general",
        "use_doc_preprocessor": False,
        "use_textline_orientation": False,
        "SubModules": {
            "TextDetection": {
                "module_name": "text_detection",
                "model_name": "PP-OCRv5_server_det",
                "model_dir": "checkpoints/PP-OCRv5_server_det",
                "limit_side_len": 960,
                "thresh": 0.3,
                "box_thresh": 0.6
            },
            "TextRecognition": {
                "module_name": "text_recognition", 
                "model_name": "PP-OCRv5_server_rec",
                "model_dir": "checkpoints/PP-OCRv5_server_rec",
                "batch_size": 6
            }
        }
    }
    
    # Mobile models configuration (speed)
    mobile_config = {
        "pipeline_name": "OCR",
        "text_type": "general",
        "use_doc_preprocessor": False,
        "use_textline_orientation": False,
        "SubModules": {
            "TextDetection": {
                "module_name": "text_detection",
                "model_name": "PP-OCRv5_mobile_det",
                "model_dir": "checkpoints/PP-OCRv5_mobile_det",
                "limit_side_len": 960,
                "thresh": 0.3,
                "box_thresh": 0.6
            },
            "TextRecognition": {
                "module_name": "text_recognition",
                "model_name": "en_PP-OCRv5_mobile_rec", 
                "model_dir": "checkpoints/en_PP-OCRv5_mobile_rec",
                "batch_size": 6
            }
        }
    }
    
    import time
    
    # Test server models
    print("Testing server models (high accuracy)...")
    start_time = time.time()
    server_pipeline = create_pipeline(config=server_config)
    server_output = server_pipeline.predict(test_image)
    server_time = time.time() - start_time
    
    # Test mobile models  
    print("Testing mobile models (high speed)...")
    start_time = time.time()
    mobile_pipeline = create_pipeline(config=mobile_config)
    mobile_output = mobile_pipeline.predict(test_image)
    mobile_time = time.time() - start_time
    
    # Compare results
    for server_res, mobile_res in zip(server_output, mobile_output):
        server_texts = len(server_res['rec_texts'])
        mobile_texts = len(mobile_res['rec_texts'])
        
        print(f"Server models: {server_texts} texts in {server_time:.1f}s")
        print(f"Mobile models: {mobile_texts} texts in {mobile_time:.1f}s")
        print(f"Speed improvement: {server_time/mobile_time:.1f}x faster")

def show_available_models():
    """Show all available models in checkpoints"""
    print("=== Available Models in Checkpoints ===")
    
    checkpoints_dir = "checkpoints"
    models = {}
    
    for model_dir in os.listdir(checkpoints_dir):
        model_path = os.path.join(checkpoints_dir, model_dir)
        if os.path.isdir(model_path):
            onnx_path = os.path.join(model_path, "inference.onnx")
            paddle_path = os.path.join(model_path, "inference.pdiparams")
            
            formats = []
            if os.path.exists(onnx_path):
                formats.append(f"ONNX ({os.path.getsize(onnx_path)/(1024*1024):.1f}MB)")
            if os.path.exists(paddle_path):
                formats.append(f"Paddle ({os.path.getsize(paddle_path)/(1024*1024):.1f}MB)")
            
            models[model_dir] = formats
    
    # Categorize models
    categories = {
        "🔍 Text Detection": ["det", "detection"],
        "🔤 Text Recognition": ["rec", "recognition"],
        "📄 Document Processing": ["doc", "UVDoc"],
        "📐 Layout Analysis": ["layout", "RT-DETR"],
        "🧭 Orientation": ["ori", "orientation"]
    }
    
    for category, keywords in categories.items():
        print(f"\n{category}:")
        for model, formats in models.items():
            if any(keyword in model.lower() for keyword in keywords):
                print(f"  • {model}: {', '.join(formats)}")

def main():
    """Main function demonstrating all configuration methods"""
    os.makedirs("output/default", exist_ok=True)
    os.makedirs("output/mobile", exist_ok=True)
    
    show_available_models()
    
    test_default_pipeline()
    test_custom_config_pipeline() 
    test_layout_parsing_pipeline()
    test_model_comparison()
    
    print("✅ All configuration methods demonstrated!")
    print("📍 Check output/ directory for results")

if __name__ == "__main__":
    main()