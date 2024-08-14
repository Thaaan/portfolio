from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
from flask_mail import Mail, Message
import io, os, base64, time, threading, queue
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

# Disable multiprocessing on Heroku
if os.environ.get('DYNO'):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

app = Flask(__name__, static_folder="../build", static_url_path="/")
CORS(app)

app.config['MAIL_SERVER'] = 'live.smtp.mailtrap.io'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'api'
app.config['MAIL_PASSWORD'] = os.getenv('MAILTRAP_API_KEY')
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

# Serve React app
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/email', methods=['POST'])
def send_email():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        if not all(key in data for key in ('name', 'email', 'message')):
            return jsonify({'error': 'Missing required fields'}), 400

        msg = Message('New Contact Form Submission',
                      sender='portfolio@demomailtrap.com',
                      recipients=['ethalo10@gmail.com'])

        msg.html = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <h2 style="color: #4a4a4a;">New Contact Form Submission</h2>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Name:</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #ddd;">{data['name']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Email:</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #ddd;">
                            <a href="mailto:{data['email']}">{data['email']}</a>
                        </td>
                    </tr>
                </table>
                <h3 style="color: #4a4a4a; margin-top: 20px;">Message:</h3>
                <p style="background-color: #f9f9f9; padding: 15px; border-radius: 5px;">
                    {data['message']}
                </p>
            </body>
        </html>
        """

        msg.body = f"""
        New Contact Form Submission

        Name: {data['name']}
        Email: {data['email']}

        Message:
        {data['message']}
        """

        mail.send(msg)
        return jsonify({'message': 'Email sent successfully'}), 200
    except Exception as e:
        app.logger.error(f"Error sending email: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Improved CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
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

# Queue for storing training updates
training_updates = queue.Queue()

def train_model():
    print("Starting model training...")
    training_updates.put("Starting model training...")
    start_time = time.time()

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    num_epochs = 10
    best_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                log_message = f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}'
                print(log_message)
                training_updates.put(log_message)
                running_loss = 0.0

        # Validate the model
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
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
        training_updates.put(log_message)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
            training_updates.put(f"New best model saved with accuracy: {best_accuracy:.2f}%")

        scheduler.step(val_loss)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            training_updates.put("Early stopping triggered")
            break

    end_time = time.time()
    training_time = end_time - start_time
    final_message = f'Finished Training. Total time: {training_time:.2f} seconds. Best accuracy: {best_accuracy:.2f}%'
    print(final_message)
    training_updates.put(final_message)

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()  # Set the model to evaluation mode
    model_trained.set()
    training_updates.put(None)  # Signal end of updates

@app.route('/train', methods=['POST'])
def train():
    if not model_trained.is_set():
        threading.Thread(target=train_model).start()
        return jsonify({"message": "Training started"}), 200
    else:
        return jsonify({"message": "Model already trained"}), 200

@app.route('/train_updates')
def train_updates():
    def generate():
        while True:
            update = training_updates.get()
            if update is None:
                break
            yield f"data: {update}\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_trained.is_set():
        return jsonify({"error": "Model is not trained yet"}), 400

    try:
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
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            prediction = output.argmax(dim=1).item()

        return jsonify({"prediction": prediction})
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "An error occurred during prediction"}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)