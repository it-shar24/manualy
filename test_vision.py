import sys
from pathlib import Path
from src.ingestion.vision_annotator import LocalVisionAnnotator

if __name__ == "__main__":
    annotator = LocalVisionAnnotator()
    
    # Path to the first extracted figure from Page 3
    sample_img = "data/extracted_images/SmartBrew_SB3000_Manual/page_3_fig_1.png"
    
    if not Path(sample_img).exists():
        print(f"File not found: {sample_img}")
        sys.exit(1)

    print(f"\n[Manualy Vision Engine] Annotating diagram: {sample_img}")
    description = annotator.describe_diagram(
        image_path=sample_img,
        page_number=3,
        doc_name="SmartBrew_SB3000_Manual.pdf"
    )

    print("\n" + "="*50)
    print("AI-GENERATED DIAGRAM KNOWLEDGE ANNOTATION:")
    print("="*50)
    print(description)
    print("="*50)