from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
import os

# Get list of image files in the folder
image_folder = "./"
image_files = [f for f in os.listdir(image_folder) if (f.endswith('.png') or f.endswith('.jpg')) and f.startswith('TracePrediction')]

# Sort and take the first 50 images
image_files = sorted(image_files)[:50]

# Create a PDF
pdf_path = "PDF_of_Traces.pdf"

# Set custom page size to better fit the images, or use landscape(letter)
custom_page_size = (1500, 1000)  # Width x Height in points (1 inch = 72 points)
c = canvas.Canvas(pdf_path, pagesize=custom_page_size)

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    img = Image.open(image_path)
    
    # Convert image size to fit the PDF
    img_width, img_height = img.size
    aspect_ratio = img_height / img_width
    new_width = 1400  # Adjust to fit your needs
    new_height = int(new_width * aspect_ratio)
    
    # Calculate position to center image on page
    x_position = (custom_page_size[0] - new_width) / 2
    y_position = (custom_page_size[1] - new_height) / 2
    
    # Add image to PDF
    c.drawImage(image_path, x_position, y_position, width=new_width, height=new_height)
    c.showPage()  # Move to next page

c.save()
