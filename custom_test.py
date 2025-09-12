"""
PaddleX OCR Test - Configuration Examples
Shows how to configure pipelines and models in custom_test.py
"""

from paddlex import create_pipeline
import os

def test_default_ocr():
    """Method 1: Default OCR Pipeline (Server Models)"""
    print("=== Method 1: Default OCR Pipeline ===")
    pipeline = create_pipeline(pipeline="OCR")
    
    output = pipeline.predict(
        "../data/some_test_data/en/PXL_20250819_103421347.jpg",
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
    )
    
    for res in output:
        print(f"✓ Server models detected {len(res['rec_texts'])} text regions")
        res.save_to_img("./output/server/")

def test_mobile_ocr():
    """Method 2: Custom Configuration for Mobile Models"""
    print("\n=== Method 2: Mobile Models Configuration ===")
    
    # Custom configuration using mobile models (faster inference)
    mobile_config = {
        "pipeline_name": "OCR",
        "text_type": "general",
        "use_doc_preprocessor": False,  # Simplified for speed
        "use_textline_orientation": False,
        
        "SubModules": {
            "TextDetection": {
                "module_name": "text_detection",
                "model_name": "PP-OCRv5_mobile_det",
                "model_dir": "checkpoints/PP-OCRv5_mobile_det",
                "limit_side_len": 480,  # Smaller for speed
                "thresh": 0.3,
                "box_thresh": 0.6
            },
            "TextRecognition": {
                "module_name": "text_recognition",
                "model_name": "en_PP-OCRv5_mobile_rec",  # English-specific
                "model_dir": "checkpoints/en_PP-OCRv5_mobile_rec",
                "batch_size": 8  # Larger batch for mobile
            }
        }
    }
    
    pipeline = create_pipeline(config=mobile_config)
    
    output = pipeline.predict(
        "../data/some_test_data/en/PXL_20250819_103421347.jpg"
    )
    
    for res in output:
        print(f"✓ Mobile models detected {len(res['rec_texts'])} text regions")
        res.save_to_img("./output/mobile/")

def test_multilingual_ocr():
    """Method 3: Multilingual Configuration"""
    print("\n=== Method 3: Multilingual OCR Configuration ===")
    
    multilingual_config = {
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
                "model_name": "PP-OCRv5_server_det",
                "model_dir": "checkpoints/PP-OCRv5_server_det"
            },
            "TextLineOrientation": {
                "module_name": "textline_orientation", 
                "model_name": "PP-LCNet_x1_0_textline_ori",
                "model_dir": "checkpoints/PP-LCNet_x1_0_textline_ori"
            },
            "TextRecognition": {
                "module_name": "text_recognition",
                "model_name": "eslav_PP-OCRv5_mobile_rec",  # Eastern Slavic languages
                "model_dir": "checkpoints/eslav_PP-OCRv5_mobile_rec"
            }
        }
    }
    
    pipeline = create_pipeline(config=multilingual_config)
    print("✓ Multilingual OCR pipeline configured with Eastern Slavic support")

def show_configuration_options():
    """Show all available configuration options"""
    print("\n=== Configuration Guide ===")
    
    print("\n📋 Pipeline Types:")
    print("  • OCR - General text recognition")
    print("  • layout_parsing - Document structure analysis")  
    print("  • table_recognition - Table extraction")
    print("  • seal_recognition - Seal/stamp recognition")
    
    print("\n🔧 Key Configuration Parameters:")
    print("  • model_name - Which model to use")
    print("  • model_dir - Local path to model (null = auto-download)")
    print("  • batch_size - Processing batch size")
    print("  • limit_side_len - Image resize limit")
    print("  • thresh/box_thresh - Detection thresholds")
    
    print("\n📁 Available Models:")
    checkpoints_dir = "checkpoints/"
    models = {
        "Speed (Mobile)": ["mobile_det", "mobile_rec"],
        "Accuracy (Server)": ["server_det", "server_rec"], 
        "Languages": ["en_", "eslav_"],
        "Document": ["doc_ori", "UVDoc", "textline_ori"],
        "Layout": ["DocLayout", "RT-DETR"]
    }
    
    for category, keywords in models.items():
        print(f"\n  {category}:")
        for model_dir in os.listdir(checkpoints_dir):
            if any(kw in model_dir for kw in keywords):
                onnx_exists = os.path.exists(os.path.join(checkpoints_dir, model_dir, "inference.onnx"))
                status = "✓ ONNX" if onnx_exists else "○ Paddle"
                print(f"    {status} {model_dir}")

def main():
    """Main function demonstrating different configuration approaches"""
    os.makedirs("output/server", exist_ok=True)
    os.makedirs("output/mobile", exist_ok=True)
    
    show_configuration_options()
    
    test_default_ocr()
    test_mobile_ocr() 
    test_multilingual_ocr()
    
    print("\n✅ All configuration methods demonstrated!")
    print("📍 Models location: checkpoints/")
    print("📍 Check output/ directory for results")
    print("\n💡 Tip: Copy any configuration above into your custom_test.py to use specific models!")

if __name__ == "__main__":
    main()