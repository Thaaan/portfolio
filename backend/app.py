import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
import io
import base64
from PIL import Image
import threading
import time

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Slightly simplified CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

model = Net()
model_trained = threading.Event()

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_model():
    print("Starting model training...")
    socketio.emit('training_log', {'data': "Starting model training..."})
    start_time = time.time()

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataset_size = len(full_dataset)
    reduced_size = int(0.7 * dataset_size)  # Use 70% of the data
    _, reduced_dataset = random_split(full_dataset, [dataset_size - reduced_size, reduced_size])

    train_size = int(0.8 * reduced_size)
    val_size = reduced_size - train_size
    train_dataset, val_dataset = random_split(reduced_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                log_message = f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}'
                print(log_message)
                socketio.emit('training_log', {'data': log_message})
                running_loss = 0.0

        # Validate the model
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        log_message = f'Epoch {epoch + 1} validation loss: {val_loss:.4f}, accuracy: {accuracy:.2f}%'
        print(log_message)
        socketio.emit('training_log', {'data': log_message})

        scheduler.step(val_loss)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            socketio.emit('training_log', {'data': "Early stopping triggered"})
            break

    end_time = time.time()
    training_time = end_time - start_time
    print(f'Finished Training. Total time: {training_time:.2f} seconds')
    socketio.emit('training_log', {'data': f'Finished Training. Total time: {training_time:.2f} seconds'})
    model.eval()  # Set the model to evaluation mode
    model_trained.set()

@app.route('/train', methods=['POST'])
def train():
    if not model_trained.is_set():
        try:
            # Start training in a separate thread
            training_thread = threading.Thread(target=train_model)
            training_thread.start()
            return jsonify({"message": "Training started"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"message": "Model already trained"}), 200

@app.route('/use_pretrained', methods=['POST'])
def use_pretrained():
    global model
    try:
        # Load a pre-trained model (you'll need to have this saved somewhere)
        model.load_state_dict(torch.load('pretrained_mnist_model.pth'))
        model.eval()
        model_trained.set()
        return jsonify({"message": "Pretrained model loaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    # Wait for the model to be trained
    model_trained.wait()

    # Get the image data from the request
    image_data = request.json['image']

    # Decode the base64 image
    image_data = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_data)).convert('L')

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(dim=1).item()

    return jsonify({"prediction": prediction})

@socketio.on('connect')
def handle_connect():
    print('Client connected')

if __name__ == '__main__':
    print("Starting Flask server...")
    socketio.run(app, debug=True, use_reloader=False)