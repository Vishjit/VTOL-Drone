import torch
import clip
from PIL import Image
from ultralytics import YOLO

# Load models
yolo = YOLO("yolov8n.pt")
clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

def find_person(image_path, prompt):
    image = Image.open(image_path)
    results = yolo(image_path, classes=[0])  # class 0 = person

    text = clip.tokenize([prompt]).to("cuda")
    text_features = clip_model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    matches = []
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        crop = image.crop((x1, y1, x2, y2))
        
        img_input = preprocess(crop).unsqueeze(0).to("cuda")
        img_features = clip_model.encode_image(img_input)
        img_features /= img_features.norm(dim=-1, keepdim=True)

        similarity = (img_features @ text_features.T).item()
        matches.append({"box": (x1,y1,x2,y2), "score": similarity})

    return sorted(matches, key=lambda x: -x["score"])


if __name__ == "__main__":
    image_path = "input.png"
    prompt = "Homosexual man"
    matches = find_person(image_path, prompt)
    for match in matches:
        print(f"Box: {match['box']}, Similarity: {match['score']:.4f}")