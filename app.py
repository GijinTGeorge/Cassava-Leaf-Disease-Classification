import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
from flask import Flask, request, render_template, url_for

# Define the model
class CassavaResNet50(nn.Module):
    def __init__(self, num_classes=5):
        super(CassavaResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# Flask app setup
app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CassavaResNet50(num_classes=5)
checkpoint = torch.load('best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Preprocessing transforms
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Disease class mapping
DISEASE_CLASSES = {
    0: 'Cassava Bacterial Blight (CBB)',
    1: 'Cassava Brown Streak Disease (CBSD)',
    2: 'Cassava Green Mottle (CGM)',
    3: 'Cassava Mosaic Disease (CMD)',
    4: 'Healthy'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save the uploaded image to display it later
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Preprocess the image
            img = Image.open(file_path)
            img_tensor = val_test_transform(img).unsqueeze(0).to(device)

            # Make the prediction
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                disease_class = DISEASE_CLASSES[predicted.item()]

            # Pass image path and prediction result to the template
            return render_template('app1.html', disease=disease_class, uploaded_image=url_for('static', filename=f'uploads/{file.filename}'))

    return render_template('app1.html')

if __name__ == '__main__':
    app.run(debug=True)
