from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from pathlib import Path
from io import BytesIO # Used to save the Pillow Image object as binary data
import sys # Added to read command-line arguments

# Get PDF filename from the command-line argument
if len(sys.argv) < 2:
    print("Error: No PDF file path provided.")
    print("Usage: python Base.py <path_to_pdf_file>")
    sys.exit(1)
    
pdf_filename = sys.argv[1] # e.g., "pdf/2501.17887v1.pdf"

# creating output dir with same name as the PDF stem
# e.g., "pdf/2501.17887v1.pdf" -> "2501.17887v1"
output_dir_name = Path(pdf_filename).stem 

# 1. Setup Converter and Process PDF
print(f"Initializing Marker converter for: {pdf_filename}")
converter = PdfConverter(
    artifact_dict=create_model_dict(),
)
rendered = converter(pdf_filename)

# 2. Extract Text, Metadata, and Images
print("Extracting text and images...")
text, _, images = text_from_rendered(rendered)

# 3. Create an output directory for the MD file and images
# This will create a directory in the same folder where the script is run
output_dir = Path(output_dir_name)
output_dir.mkdir(exist_ok=True)
print(f"Created output directory: {output_dir}")

# 4. Save the Markdown text file
md_filename = output_dir / f"{output_dir_name}.md"

try:
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Successfully saved Markdown text to {md_filename}")
except Exception as e:
    print(f"An error occurred while writing the MD file: {e}")

# 5. Save the image files
print(f"\nSaving {len(images)} images...")
for filename, image_object in images.items():
    image_path = output_dir / filename
    
    byte_io = BytesIO()
    
    try:
        # Determine image format from suffix (e.g., .jpeg, .png)
        img_format = image_path.suffix.lstrip('.').upper()
        if img_format == 'JPEG':
            img_format = 'JPEG'
        elif img_format == 'JPG':
            img_format = 'JPEG'
        elif img_format == 'PNG':
            img_format = 'PNG'
        else:
            print(f"Warning: Unknown image format '{img_format}' for {filename}. Defaulting to PNG.")
            img_format = 'PNG'
            image_path = image_path.with_suffix('.png')

        image_object.save(byte_io, format=img_format)
        image_data = byte_io.getvalue()
        
        with open(image_path, "wb") as f:
            f.write(image_data)
        
    except Exception as e:
        print(f"An error occurred while writing image {filename} (format: {img_format}): {e}")

print(f"Image saving complete for {output_dir_name}.")