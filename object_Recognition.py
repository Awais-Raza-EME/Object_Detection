# import torch
# from PIL import Image
# import matplotlib.pyplot as plt

# # Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)  # Nano model
# model.to('cpu')
# # Load an image
# image_path = 'awais_Image.jpg'  # Replace with your image path
# image = Image.open(image_path)

# # Perform inference
# results = model(image)

# # Display results
# results.show()  # Show in a popup window
# results.print()  # Print detected objects and confidence scores

# # Save the annotated image
# results.save()  # Saves the image with bounding boxes in 'runs/detect/'
# Ensure ultralytics is installed
# from ultralytics import YOLO

# # Load YOLO model
# model = YOLO('yolov5n.pt')  # Download 'yolov5n.pt' if not already available

# # Run inference on an image
# results = model('awais_Image.jpg')  # Replace with an actual image path

# # Show results
# results.show()






from ultralytics import YOLO

# Load the pretrained YOLOv5 model
model = YOLO("yolov5n.pt")  # Adjust model path as needed

# Perform inference
results = model("E:\PYTHON Projects\Object_Recognition\myself.jpg")  # Replace with actual image or video source
# Access the first result in the list (if it's a list of results)
result = results[0]

# Now you can call .show() on the result
result.show()


