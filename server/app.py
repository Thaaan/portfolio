import os
import io
import base64
import time
import threading
import uuid
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
from flask_mail import Mail, Message
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from threading import Timer
import redis
import json

# Disable multiprocessing on Heroku
if os.environ.get('DYNO'):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

app = Flask(__name__, static_folder="../build", static_url_path="/")
CORS(app, supports_credentials=True)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

app.config['MAIL_SERVER'] = 'live.smtp.mailtrap.io'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'api'
app.config['MAIL_PASSWORD'] = os.getenv('MAILTRAP_API_KEY')
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

redis_url = os.getenv('REDIS_URL')
redis_client = redis.from_url(redis_url)

INACTIVE_THRESHOLD = 600

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(4608, 64)
        self.fc2 = nn.Linear(64, 10)

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

# Timestamp handling functions
def get_current_timestamp():
    return time.time()

def timestamp_to_string(timestamp):
    return str(timestamp)

def string_to_timestamp(timestamp_string):
    return float(timestamp_string)

# Model management functions
def save_model_to_redis(user_id, model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    try:
        redis_client.set(f'model_{user_id}', buffer.getvalue())
        print(f"Model for user {user_id} saved to Redis successfully")
    except redis.RedisError as e:
        print(f"Error saving model to Redis: {e}")
        raise

def load_model_from_redis(user_id):
    try:
        model_data = redis_client.get(f'model_{user_id}')
        if model_data is None:
            raise ValueError(f"No model found for user {user_id}")
        buffer = io.BytesIO(model_data)
        model = Net().to(device)
        model.load_state_dict(torch.load(buffer, map_location=device))
        print(f"Model for user {user_id} loaded from Redis successfully")
        return model
    except (redis.RedisError, ValueError) as e:
        print(f"Error loading model from Redis: {e}")
        raise

# User management
def update_user_activity(user_id):
    try:
        current_time = get_current_timestamp()
        redis_client.hset('user_last_activity', user_id, timestamp_to_string(current_time))
    except redis.RedisError as e:
        print(f"Error updating user activity in Redis: {e}")

def get_user_last_activity(user_id):
    try:
        last_activity = redis_client.hget('user_last_activity', user_id)
        if last_activity:
            return string_to_timestamp(last_activity.decode('utf-8'))
        return None
    except redis.RedisError as e:
        print(f"Error getting user activity from Redis: {e}")
        return None

def get_or_create_user_id():
    user_id = request.cookies.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())
        response = jsonify({"message": "New user ID created"})
        response.set_cookie('user_id', user_id, max_age=30*24*60*60)  # Cookie lasts 30 days
        return user_id, response
    return user_id, None

# Queue management
def enqueue_update(user_id, update):
    redis_client.rpush(f'updates_{user_id}', json.dumps(update))

def dequeue_update(user_id):
    update = redis_client.blpop(f'updates_{user_id}', timeout=20)
    if update:
        return json.loads(update[1])
    return None

# Routes
@app.route('/email', methods=['POST'])
def send_email():
    try:
        data = request.json
        if not data or not all(key in data for key in ('name', 'email', 'message')):
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

        mail.send(msg)
        return jsonify({'message': 'Email sent successfully'}), 200
    except Exception as e:
        app.logger.error(f"Error sending email: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    user_id, _ = get_or_create_user_id()
    update_user_activity(user_id)
    return jsonify({"message": "Heartbeat received"}), 200

@app.route('/train', methods=['POST'])
def train():
    user_id, response = get_or_create_user_id()
    update_user_activity(user_id)

    try:
        # Check if model already exists in Redis
        if redis_client.exists(f'model_{user_id}'):
            result = {"message": "Model already trained"}
        else:
            # Model doesn't exist, start training
            threading.Thread(target=train_model, args=(user_id,)).start()
            result = {"message": "Training started"}

    except redis.RedisError as e:
        app.logger.error(f"Redis error: {str(e)}")
        result = {"error": "An error occurred while checking model status"}

    if response:
        response.data = jsonify(result).data
        return response
    else:
        return jsonify(result)

@app.route('/train_updates')
def train_updates():
    user_id, _ = get_or_create_user_id()
    update_user_activity(user_id)

    def generate():
        while True:
            update = dequeue_update(user_id)
            if update is None:
                yield ": keep-alive\n\n"
            else:
                yield f"data: {update}\n\n"
            update_user_activity(user_id)

    return Response(generate(), mimetype='text/event-stream')

@app.route('/predict', methods=['POST'])
def predict():
    user_id, _ = get_or_create_user_id()
    update_user_activity(user_id)

    try:
        # Load model from Redis
        model = load_model_from_redis(user_id)

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

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Helper functions
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
        elif self.best_loss - val_loss >= self.min_delta:  # Changed > to >=
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_model(user_id):
    model = Net().to(device)

    print("Starting model training...")
    enqueue_update(user_id, "Starting model training...")
    start_time = time.time()

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataset_size = len(full_dataset)
    reduced_size = int(0.8 * dataset_size)  # Use 80% of the data
    _, reduced_dataset = random_split(full_dataset, [dataset_size - reduced_size, reduced_size])

    train_size = int(0.8 * reduced_size)
    val_size = reduced_size - train_size
    train_dataset, val_dataset = random_split(reduced_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    num_epochs = 5
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
                enqueue_update(user_id, log_message)
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
        enqueue_update(user_id, log_message)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model_to_redis(user_id, model)
            print(f"New best model saved to Redis with accuracy: {best_accuracy:.2f}%")
            enqueue_update(user_id, f"New best model saved to Redis with accuracy: {best_accuracy:.2f}%")

        scheduler.step(val_loss)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            enqueue_update(user_id, "Early stopping triggered")
            break

    end_time = time.time()
    training_time = end_time - start_time
    final_message = f'Finished Training. Total time: {training_time:.2f} seconds. Best accuracy: {best_accuracy:.2f}%'
    print(final_message)
    enqueue_update(user_id, final_message)

    # Load the best model
    model = load_model_from_redis(user_id)
    model.eval()  # Set the model to evaluation mode
    update_user_activity(user_id)
    enqueue_update(user_id, None)  # Signal end of updates

def cleanup_inactive_models():
    current_time = get_current_timestamp()
    print(f"Starting cleanup at {current_time}")
    try:
        all_user_activities = redis_client.hgetall('user_last_activity')
        print(f"Found {len(all_user_activities)} user activities")

        inactive_users = []
        for user_id, last_activity in all_user_activities.items():
            last_activity_time = string_to_timestamp(last_activity.decode('utf-8'))
            time_difference = current_time - last_activity_time
            print(f"User {user_id}: Last activity at {last_activity_time}, Time difference: {time_difference}")
            if time_difference > INACTIVE_THRESHOLD:
                inactive_users.append(user_id)

        print(f"Identified {len(inactive_users)} inactive users")

        for user_id in inactive_users:
            try:
                model_key = f'model_{user_id}'
                if redis_client.exists(model_key):
                    redis_client.delete(model_key)
                    print(f"Deleted model for user {user_id}")
                else:
                    print(f"No model found for user {user_id}")

                redis_client.hdel('user_last_activity', user_id)
                print(f"Deleted activity for user {user_id}")
            except redis.RedisError as e:
                print(f"Error deleting data for user {user_id} from Redis: {e}")

        print(f"Cleanup completed. Processed {len(inactive_users)} inactive users")

    except redis.RedisError as e:
        print(f"Error during cleanup: {e}")

    # Schedule the next cleanup
    Timer(INACTIVE_THRESHOLD, cleanup_inactive_models).start()

cleanup_inactive_models()

def clear_redis():
    try:
        for key in redis_client.scan_iter("model_*"):
            redis_client.delete(key)
        redis_client.delete('user_last_activity')
        print("Application-specific Redis data cleared successfully")
    except redis.RedisError as e:
        print(f"Error clearing Redis data: {e}")

#clear redis on deployment
clear_redis()

if __name__ == '__main__':
    print("Starting Flask server...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)