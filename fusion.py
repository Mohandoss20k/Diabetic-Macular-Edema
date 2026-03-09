import os
import sqlite3
import hashlib
import secrets
import traceback
from functools import wraps
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, g, jsonify
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json

# Initialize Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
DATABASE = 'database.db'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== IMAGE TRANSFORMS ====================
# Define transform at module level so it's accessible everywhere
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==================== MODEL DEFINITIONS ====================

# OCT Model Classes (5-class)
OCT_CLASSES = ['Mild', 'Moderate', 'Normal', 'Proliferate', 'Severe']

# Fundus Model Classes (7-class)
FUNDUS_CLASSES = [
    'No DR signs',
    'Mild NPDR',
    'Moderate NPDR',
    'Severe NPDR',
    'Very Severe NPDR',
    'PDR',
    'Advanced PDR'
]

# Define the model architectures
class LightweightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.feat_dim = 960

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.feat_dim, self.feat_dim//8, 1),
            nn.ReLU(),
            nn.Conv2d(self.feat_dim//8, self.feat_dim, 1),
            nn.Sigmoid()
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x * self.channel_attn(x)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        x = x * self.spatial_attn(spatial)

        x = F.adaptive_avg_pool2d(x, 1)
        return x.view(x.size(0), -1)

class LightweightViT(nn.Module):
    def __init__(self, img_size=128, patch=16, embed_dim=256, depth=6, heads=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch, stride=patch)
        self.num_patches = (img_size//patch)**2
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim)*0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim*4,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.norm(x)

class OCTFusionModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.cnn = LightweightCNN()
        self.vit = LightweightViT()

        self.classifier = nn.Sequential(
            nn.Linear(960+256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        return self.classifier(combined)

class FundusModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(960, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==================== LOAD MODELS ====================

print("\n" + "="*50)
print("LOADING MODELS...")
print("="*50)

# Use the extracted fundus model
FUNDUS_MODEL_PATH = 'fundus_model_extracted.pth'
OCT_MODEL_PATH = 'best_model_5class_fixed.pth'

print(f"OCT model path: {OCT_MODEL_PATH}")
print(f"Fundus model path: {FUNDUS_MODEL_PATH}")
print(f"Files exist: OCT={os.path.exists(OCT_MODEL_PATH)}, Fundus={os.path.exists(FUNDUS_MODEL_PATH)}")

oct_model = None
fundus_model = None

# Load OCT Model
print("\n" + "-"*30)
print("Loading OCT Model...")
try:
    # Initialize OCT model
    oct_model = OCTFusionModel(num_classes=5)
    
    # Load the checkpoint
    checkpoint = torch.load(OCT_MODEL_PATH, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            oct_model.load_state_dict(checkpoint['model_state_dict'])
            print("  ✓ Loaded from checkpoint with 'model_state_dict' key")
        else:
            oct_model.load_state_dict(checkpoint)
            print("  ✓ Loaded direct state dict")
    
    oct_model = oct_model.to(device)
    oct_model.eval()
    print("  ✓ OCT Model loaded successfully!")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 128, 128).to(device)
    with torch.no_grad():
        output = oct_model(dummy_input)
        print(f"  ✓ Test inference successful. Output shape: {output.shape}")
            
except Exception as e:
    print(f"  ✗ Error loading OCT model: {str(e)}")
    traceback.print_exc()
    oct_model = None

# Load Fundus Model
print("\n" + "-"*30)
print("Loading Fundus Model...")
try:
    # Initialize Fundus model
    fundus_model = FundusModel(num_classes=7)
    
    # Load the checkpoint
    checkpoint = torch.load(FUNDUS_MODEL_PATH, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            fundus_model.load_state_dict(checkpoint['model_state_dict'])
            print("  ✓ Loaded from checkpoint with 'model_state_dict' key")
        else:
            fundus_model.load_state_dict(checkpoint)
            print("  ✓ Loaded direct state dict")
    
    fundus_model = fundus_model.to(device)
    fundus_model.eval()
    print("  ✓ Fundus Model loaded successfully!")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 128, 128).to(device)
    with torch.no_grad():
        output = fundus_model(dummy_input)
        print(f"  ✓ Test inference successful. Output shape: {output.shape}")
            
except Exception as e:
    print(f"  ✗ Error loading Fundus model: {str(e)}")
    traceback.print_exc()
    fundus_model = None

print("\n" + "="*50)
print("MODEL LOADING SUMMARY:")
print(f"OCT Model: {'✓ Loaded' if oct_model is not None else '✗ Failed'}")
print(f"Fundus Model: {'✓ Loaded' if fundus_model is not None else '✗ Failed'}")
print("="*50 + "\n")

# ==================== DATABASE FUNCTIONS ====================

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                image_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        db.commit()

# Initialize database
init_db()

# ==================== AUTHENTICATION HELPERS ====================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ==================== IMAGE PROCESSING ====================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path, model_type='oct'):
    """Process image and get prediction from appropriate model"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if model_type == 'oct':
                if oct_model is None:
                    return {'success': False, 'error': 'OCT model not loaded'}
                outputs = oct_model(image_tensor)
                classes = OCT_CLASSES
            else:
                if fundus_model is None:
                    return {'success': False, 'error': 'Fundus model not loaded'}
                outputs = fundus_model(image_tensor)
                classes = FUNDUS_CLASSES
            
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            prediction = classes[predicted.item()]
            confidence_score = confidence.item() * 100
            all_probs = probabilities[0].cpu().numpy()
            
            # Get top-3 predictions
            top3_idx = np.argsort(all_probs)[-3:][::-1]
            top3_predictions = [
                {
                    'class': classes[idx],
                    'probability': float(all_probs[idx] * 100)
                }
                for idx in top3_idx
            ]
            
        return {
            'success': True,
            'prediction': prediction,
            'confidence': confidence_score,
            'top3': top3_predictions,
            'probabilities': all_probs.tolist()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def fuse_predictions(oct_result, fundus_result):
    """Fuse predictions from both models using weighted ensemble"""
    
    # Define severity weights (higher for more severe conditions)
    severity_map = {
        # OCT Classes severity (0-4 scale)
        'Normal': 0,
        'Mild': 1,
        'Moderate': 2,
        'Severe': 3,
        'Proliferate': 4,
        
        # Fundus Classes severity (0-6 scale)
        'No DR signs': 0,
        'Mild NPDR': 1,
        'Moderate NPDR': 2,
        'Severe NPDR': 3,
        'Very Severe NPDR': 4,
        'PDR': 5,
        'Advanced PDR': 6
    }
    
    # Get severity scores
    oct_severity = severity_map.get(oct_result['prediction'], 0)
    fundus_severity = severity_map.get(fundus_result['prediction'], 0)
    
    # Normalize fundus severity to 0-4 scale for comparison
    fundus_severity_norm = (fundus_severity / 6) * 4
    
    # Weighted fusion
    oct_weight = 0.5
    fundus_weight = 0.5
    
    fused_severity = (oct_severity * oct_weight + fundus_severity_norm * fundus_weight)
    
    # Determine final classification
    if fused_severity < 0.5:
        final_prediction = "No DR / Normal"
        risk_level = "Low"
        recommendations = [
            "Regular eye check-ups recommended",
            "Maintain healthy lifestyle",
            "Control blood sugar levels if diabetic"
        ]
    elif fused_severity < 1.5:
        final_prediction = "Mild DR"
        risk_level = "Mild"
        recommendations = [
            "Follow up in 6-12 months",
            "Monitor blood sugar levels",
            "Maintain healthy diet and exercise"
        ]
    elif fused_severity < 2.5:
        final_prediction = "Moderate DR"
        risk_level = "Moderate"
        recommendations = [
            "Follow up in 3-6 months",
            "Consult with ophthalmologist",
            "Strict blood sugar control",
            "Consider retinal imaging"
        ]
    elif fused_severity < 3.5:
        final_prediction = "Severe DR"
        risk_level = "High"
        recommendations = [
            "Immediate ophthalmologist consultation",
            "Follow up in 1-3 months",
            "Possible laser treatment",
            "Strict metabolic control"
        ]
    else:
        final_prediction = "Proliferative DR"
        risk_level = "Critical"
        recommendations = [
            "Urgent ophthalmologist referral",
            "Immediate treatment required",
            "Possible vitrectomy",
            "Strict blood pressure control"
        ]
    
    return {
        'final_prediction': final_prediction,
        'risk_level': risk_level,
        'recommendations': recommendations,
        'oct_prediction': oct_result['prediction'],
        'oct_confidence': oct_result['confidence'],
        'fundus_prediction': fundus_result['prediction'],
        'fundus_confidence': fundus_result['confidence'],
        'fused_severity': fused_severity
    }

# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))
        
        db = get_db()
        cursor = db.cursor()
        
        # Check if user exists
        cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        if cursor.fetchone():
            flash('Username or email already exists', 'error')
            return redirect(url_for('register'))
        
        # Insert new user
        hashed_password = hash_password(password)
        cursor.execute(
            'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
            (username, email, hashed_password)
        )
        db.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        if user and user['password'] == hash_password(password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()
    cursor = db.cursor()
    
    # Get user's prediction statistics
    cursor.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN image_type = 'oct' THEN 1 ELSE 0 END) as oct_count,
            SUM(CASE WHEN image_type = 'fundus' THEN 1 ELSE 0 END) as fundus_count
        FROM predictions 
        WHERE user_id = ?
    ''', (session['user_id'],))
    
    stats = cursor.fetchone()
    
    # Get recent predictions
    cursor.execute('''
        SELECT * FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 5
    ''', (session['user_id'],))
    
    recent_predictions = cursor.fetchall()
    
    return render_template('dashboard.html', 
                         stats=stats, 
                         recent_predictions=recent_predictions,
                         models_loaded=(oct_model is not None and fundus_model is not None))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if oct_model is None or fundus_model is None:
        missing = []
        if oct_model is None:
            missing.append("OCT")
        if fundus_model is None:
            missing.append("Fundus")
        flash(f'Models are loading. Please wait a moment and try again. Missing: {", ".join(missing)}', 'warning')
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        if 'oct_image' not in request.files or 'fundus_image' not in request.files:
            flash('Both OCT and Fundus images are required', 'error')
            return redirect(request.url)
        
        oct_file = request.files['oct_image']
        fundus_file = request.files['fundus_image']
        
        if oct_file.filename == '' or fundus_file.filename == '':
            flash('No files selected', 'error')
            return redirect(request.url)
        
        if oct_file and allowed_file(oct_file.filename) and fundus_file and allowed_file(fundus_file.filename):
            # Generate unique filenames
            timestamp = datetime.now().timestamp()
            oct_filename = secure_filename(f"oct_{session['user_id']}_{timestamp}_{oct_file.filename}")
            fundus_filename = secure_filename(f"fundus_{session['user_id']}_{timestamp}_{fundus_file.filename}")
            
            oct_path = os.path.join(app.config['UPLOAD_FOLDER'], oct_filename)
            fundus_path = os.path.join(app.config['UPLOAD_FOLDER'], fundus_filename)
            
            oct_file.save(oct_path)
            fundus_file.save(fundus_path)
            
            # Process images
            oct_result = process_image(oct_path, 'oct')
            fundus_result = process_image(fundus_path, 'fundus')
            
            if not oct_result['success'] or not fundus_result['success']:
                error_msg = oct_result.get('error', '') or fundus_result.get('error', '')
                flash(f'Error processing images: {error_msg}', 'error')
                return redirect(request.url)
            
            # Fuse predictions
            fusion_result = fuse_predictions(oct_result, fundus_result)
            
            # Save to database
            db = get_db()
            cursor = db.cursor()
            
            # Save OCT prediction
            cursor.execute('''
                INSERT INTO predictions (user_id, image_type, filename, prediction, confidence, probabilities)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                session['user_id'],
                'oct',
                oct_filename,
                oct_result['prediction'],
                oct_result['confidence'],
                json.dumps(oct_result['probabilities'])
            ))
            
            # Save Fundus prediction
            cursor.execute('''
                INSERT INTO predictions (user_id, image_type, filename, prediction, confidence, probabilities)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                session['user_id'],
                'fundus',
                fundus_filename,
                fundus_result['prediction'],
                fundus_result['confidence'],
                json.dumps(fundus_result['probabilities'])
            ))
            
            db.commit()
            
            return render_template('result.html',
                                 oct_result=oct_result,
                                 fundus_result=fundus_result,
                                 fusion_result=fusion_result,
                                 oct_classes=OCT_CLASSES,
                                 fundus_classes=FUNDUS_CLASSES,
                                 oct_image=oct_filename,
                                 fundus_image=fundus_filename)
    
    return render_template('predict.html', models_loaded=(oct_model is not None and fundus_model is not None))

@app.route('/history')
@login_required
def history():
    db = get_db()
    cursor = db.cursor()
    
    # Get all predictions for user
    cursor.execute('''
        SELECT * FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', (session['user_id'],))
    
    predictions = cursor.fetchall()
    
    return render_template('history.html', predictions=predictions)

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """API endpoint for predictions"""
    if oct_model is None or fundus_model is None:
        return jsonify({'error': 'Models not loaded'}), 500
        
    if 'oct_image' not in request.files or 'fundus_image' not in request.files:
        return jsonify({'error': 'Both images required'}), 400
    
    oct_file = request.files['oct_image']
    fundus_file = request.files['fundus_image']
    
    if oct_file.filename == '' or fundus_file.filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    # Save temporarily
    oct_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(oct_file.filename))
    fundus_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(fundus_file.filename))
    
    oct_file.save(oct_path)
    fundus_file.save(fundus_path)
    
    # Process
    oct_result = process_image(oct_path, 'oct')
    fundus_result = process_image(fundus_path, 'fundus')
    
    # Clean up
    os.remove(oct_path)
    os.remove(fundus_path)
    
    if not oct_result['success'] or not fundus_result['success']:
        return jsonify({'error': 'Processing failed'}), 500
    
    fusion_result = fuse_predictions(oct_result, fundus_result)
    
    return jsonify({
        'oct_prediction': oct_result,
        'fundus_prediction': fundus_result,
        'fusion_result': fusion_result
    })

@app.route('/api/prediction/<int:prediction_id>')
@login_required
def get_prediction_details(prediction_id):
    """API endpoint to get prediction details"""
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute('''
        SELECT * FROM predictions 
        WHERE id = ? AND user_id = ?
    ''', (prediction_id, session['user_id']))
    
    prediction = cursor.fetchone()
    
    if prediction:
        return jsonify({
            'id': prediction['id'],
            'image_type': prediction['image_type'],
            'filename': prediction['filename'],
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'probabilities': json.loads(prediction['probabilities']),
            'created_at': prediction['created_at']
        })
    
    return jsonify({'error': 'Prediction not found'}), 404

@app.route('/check_models')
def check_models():
    """Route to check model status"""
    status = {
        'oct_model': oct_model is not None,
        'fundus_model': fundus_model is not None,
        'device': str(device),
        'oct_classes': OCT_CLASSES,
        'fundus_classes': FUNDUS_CLASSES,
        'current_directory': os.getcwd(),
        'files': [f for f in os.listdir() if f.endswith('.pth')]
    }
    
    # Test inference if models are loaded
    if oct_model is not None:
        try:
            dummy = torch.randn(1, 3, 128, 128).to(device)
            with torch.no_grad():
                out = oct_model(dummy)
            status['oct_test'] = f"Success: {out.shape}"
        except Exception as e:
            status['oct_test'] = f"Failed: {str(e)}"
    
    if fundus_model is not None:
        try:
            dummy = torch.randn(1, 3, 128, 128).to(device)
            with torch.no_grad():
                out = fundus_model(dummy)
            status['fundus_test'] = f"Success: {out.shape}"
        except Exception as e:
            status['fundus_test'] = f"Failed: {str(e)}"
    
    return jsonify(status)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("DR Fusion Application Starting...")
    print("="*50)
    print(f"OCT Model Loaded: {oct_model is not None}")
    print(f"Fundus Model Loaded: {fundus_model is not None}")
    print(f"Device: {device}")
    print(f"OCT Classes: {OCT_CLASSES}")
    print(f"Fundus Classes: {FUNDUS_CLASSES}")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)