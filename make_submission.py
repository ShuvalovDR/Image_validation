import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from torchvision.models import resnet50
import torch.nn as nn

MODEL_WEIGHTS = "baseline.pth"
TEST_IMAGES_DIR = "./data/test/"
SUBMISSION_PATH = "./data/submission.csv"

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()

    img_size = 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_image_names = os.listdir(TEST_IMAGES_DIR)
    all_preds = []

    for image_name in all_image_names:
        img_path = os.path.join(TEST_IMAGES_DIR, image_name)
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            pred = torch.sigmoid(output).item() >= 0.5
            all_preds.append(int(pred))

    with open(SUBMISSION_PATH, "w") as f:
        f.write("image_name\tlabel_id\n")
        for name, cl_id in zip(all_image_names, all_preds):
            f.write(f"{name}\t{cl_id}\n")