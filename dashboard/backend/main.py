"""
FastAPI Backend for Federated Learning Dashboard
=================================================
Real-time monitoring and control of federated learning experiments.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import asyncio
import base64
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from src.models import create_model
from src.explainability.gradcam import GradCAMExplainer

# Initialize FastAPI app
app = FastAPI(
    title="Hospital Federated Learning Dashboard",
    description="Real-time monitoring for privacy-preserving medical AI training",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")

manager = ConnectionManager()

# Global state
class ExperimentState:
    def __init__(self):
        self.is_running = False
        self.current_round = 0
        self.total_rounds = 50
        self.start_time: Optional[datetime] = None
        self.metrics_history: List[Dict] = []
        self.hospital_metrics: Dict[int, List[Dict]] = {}
        self.privacy_budget_used = 0.0
        self.config: Dict = {}
    
    def to_dict(self) -> Dict:
        return {
            "is_running": self.is_running,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "progress": self.current_round / self.total_rounds if self.total_rounds > 0 else 0,
            "privacy_budget_used": self.privacy_budget_used,
            "config": self.config
        }

state = ExperimentState()


# Pydantic models
class ExperimentConfig(BaseModel):
    num_hospitals: int = 5
    num_rounds: int = 50
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001
    epsilon: float = 8.0
    delta: float = 1e-5
    aggregation_strategy: str = "adaptive"
    partition_type: str = "non_iid"
    dirichlet_alpha: float = 0.5


class RoundMetrics(BaseModel):
    round_num: int
    global_loss: float
    global_accuracy: float
    auc_roc: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    epsilon_spent: float = 0.0
    num_clients: int = 0
    hospital_metrics: Optional[Dict[str, Dict]] = None


class HospitalStatus(BaseModel):
    hospital_id: int
    name: str
    status: str  # 'training', 'idle', 'uploading', 'error'
    samples: int
    current_loss: float
    current_accuracy: float
    last_update: Optional[str] = None


# API Routes

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Hospital Federated Learning Dashboard API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/api/status")
async def get_status():
    """Get current experiment status."""
    return {
        "status": "success",
        "data": state.to_dict()
    }


@app.get("/api/metrics/history")
async def get_metrics_history():
    """Get complete metrics history."""
    return {
        "status": "success",
        "data": state.metrics_history
    }


@app.get("/api/metrics/latest")
async def get_latest_metrics():
    """Get latest round metrics."""
    if not state.metrics_history:
        return {"status": "success", "data": None}
    return {
        "status": "success",
        "data": state.metrics_history[-1]
    }


@app.get("/api/hospitals")
async def get_hospitals():
    """Get all hospital statuses."""
    hospitals = []
    hospital_names = [
        "Metro General Hospital",
        "City Medical Center",
        "University Hospital",
        "Regional Health Center",
        "Community Medical"
    ]
    
    for i in range(state.config.get("num_hospitals", 5)):
        h_metrics = state.hospital_metrics.get(i, [])
        latest = h_metrics[-1] if h_metrics else {}
        
        hospitals.append({
            "hospital_id": i,
            "name": hospital_names[i] if i < len(hospital_names) else f"Hospital {i+1}",
            "status": "training" if state.is_running else "idle",
            "samples": latest.get("num_samples", 0),
            "current_loss": latest.get("train_loss", 0),
            "current_accuracy": latest.get("train_accuracy", 0),
            "last_update": latest.get("timestamp", None)
        })
    
    return {"status": "success", "data": hospitals}


@app.get("/api/hospitals/{hospital_id}")
async def get_hospital(hospital_id: int):
    """Get specific hospital details."""
    if hospital_id not in state.hospital_metrics:
        return {"status": "success", "data": {"history": []}}
    
    return {
        "status": "success",
        "data": {
            "hospital_id": hospital_id,
            "history": state.hospital_metrics[hospital_id]
        }
    }


@app.get("/api/privacy")
async def get_privacy_status():
    """Get privacy budget consumption."""
    target_epsilon = state.config.get("epsilon", 8.0)
    
    return {
        "status": "success",
        "data": {
            "target_epsilon": target_epsilon,
            "epsilon_spent": state.privacy_budget_used,
            "epsilon_remaining": max(0, target_epsilon - state.privacy_budget_used),
            "budget_percentage": (state.privacy_budget_used / target_epsilon * 100) if target_epsilon > 0 else 0,
            "delta": state.config.get("delta", 1e-5)
        }
    }


@app.post("/api/experiment/start")
async def start_experiment(config: ExperimentConfig, background_tasks: BackgroundTasks):
    """Start a new federated learning experiment."""
    if state.is_running:
        raise HTTPException(status_code=400, detail="Experiment already running")
    
    state.config = config.dict()
    state.is_running = True
    state.current_round = 0
    state.total_rounds = config.num_rounds
    state.start_time = datetime.now()
    state.metrics_history = []
    state.hospital_metrics = {i: [] for i in range(config.num_hospitals)}
    state.privacy_budget_used = 0.0
    
    # Broadcast start
    await manager.broadcast({
        "type": "experiment_started",
        "data": state.to_dict()
    })
    
    return {"status": "success", "message": "Experiment started"}


@app.post("/api/experiment/stop")
async def stop_experiment():
    """Stop the current experiment."""
    if not state.is_running:
        raise HTTPException(status_code=400, detail="No experiment running")
    
    state.is_running = False
    
    await manager.broadcast({
        "type": "experiment_stopped",
        "data": state.to_dict()
    })
    
    return {"status": "success", "message": "Experiment stopped"}


@app.post("/api/metrics/update")
async def update_metrics(metrics: RoundMetrics):
    """Update metrics (called by training process)."""
    metrics_dict = metrics.dict()
    metrics_dict["timestamp"] = datetime.now().isoformat()
    
    state.current_round = metrics.round_num
    state.metrics_history.append(metrics_dict)
    state.privacy_budget_used = metrics.epsilon_spent
    
    # Update hospital metrics
    if metrics.hospital_metrics:
        for h_id, h_metrics in metrics.hospital_metrics.items():
            hospital_id = int(h_id)
            if hospital_id not in state.hospital_metrics:
                state.hospital_metrics[hospital_id] = []
            state.hospital_metrics[hospital_id].append(h_metrics)
    
    # Broadcast update
    await manager.broadcast({
        "type": "metrics_update",
        "data": metrics_dict
    })
    
    return {"status": "success"}


@app.get("/api/config")
async def get_config():
    """Get current experiment configuration."""
    return {"status": "success", "data": state.config}


@app.get("/api/explanations")
async def get_explanations():
    """Get available Grad-CAM explanations."""
    explanations_dir = Path("./explanations")
    if not explanations_dir.exists():
        return {"status": "success", "data": []}
    
    explanations = []
    for file in explanations_dir.glob("*.png"):
        explanations.append({
            "filename": file.name,
            "path": f"/api/explanations/{file.name}",
            "created": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
        })
    
    return {"status": "success", "data": explanations}


@app.get("/api/explanations/{filename}")
async def get_explanation_image(filename: str):
    """Get a specific explanation image."""
    file_path = Path("./explanations") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await manager.connect(websocket)
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "initial_state",
            "data": {
                "status": state.to_dict(),
                "metrics_history": state.metrics_history,
                "hospital_metrics": state.hospital_metrics
            }
        })
        
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    logger.info("Starting up... Attempting to load inference model...")
    try:
        model = InferenceModel.get_instance()
        if model._model is not None:
            logger.info("✅ Inference model loaded successfully on startup")
        else:
            logger.warning("⚠️ Inference model not loaded on startup. Use /api/inference/reload-model to load it.")
    except Exception as e:
        logger.error(f"Error loading model on startup: {e}")


# Model inference
class InferenceModel:
    """Singleton model loader for inference."""
    _instance = None
    _model = None
    _device = None
    _explainer = None
    _transform = None
    
    def __init__(self):
        if InferenceModel._instance is not None:
            raise Exception("InferenceModel is a singleton")
        InferenceModel._instance = self
        self._load_model()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _load_model(self):
        """Load the trained model."""
        # Get project root (3 levels up from dashboard/backend/main.py)
        project_root = Path(__file__).parent.parent.parent
        
        # Try multiple possible paths
        possible_paths = [
            project_root / "checkpoints" / "best_model.pt",  # Primary
            project_root / "checkpoints" / "final_model.pt",  # Fallback 1
            project_root / "checkpoints" / "best_model_dp.pt",  # Fallback 2 (DP model)
            Path("./checkpoints/best_model.pt"),  # Relative from current dir
            Path("../checkpoints/best_model.pt"),  # Relative from parent
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            abs_path = path.resolve() if path.is_absolute() or str(path).startswith('.') else path
            if abs_path.exists():
                checkpoint_path = abs_path
                logger.info(f"Found checkpoint at: {checkpoint_path}")
                break
        
        if checkpoint_path is None:
            logger.error(f"No model checkpoint found. Searched:")
            for p in possible_paths:
                abs_p = p.resolve() if p.is_absolute() or str(p).startswith('.') else p
                logger.error(f"  - {abs_p} (exists: {abs_p.exists()})")
            return
        
        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading model from {checkpoint_path} on {self._device}")
            
            # Create model
            self._model = create_model(
                model_name="efficientnet_b0",
                num_classes=2,
                pretrained=False,
                dropout=0.3
            )
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self._device)
            if "model_state_dict" in checkpoint:
                self._model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self._model.load_state_dict(checkpoint)
            
            self._model.to(self._device)
            self._model.eval()
            
            # Create explainer (optional - model can work without it)
            try:
                self._explainer = GradCAMExplainer(self._model, self._device)
                logger.info("Grad-CAM explainer initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Grad-CAM explainer: {e}. Model will work without explanations.")
                self._explainer = None
            
            # Image preprocessing
            self._transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("Model loaded successfully")
            logger.info(f"Model device: {self._device}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self._model.parameters()):,}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._model = None
    
    def predict(self, image: Image.Image, generate_explanation: bool = True) -> Dict[str, Any]:
        """Predict on an image."""
        if self._model is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
        
        try:
            # Preprocess image
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Transform
            input_tensor = self._transform(image).unsqueeze(0).to(self._device)
            
            # Predict
            with torch.no_grad():
                output = self._model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                pred_class = output.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()
            
            result = {
                "prediction": "Pneumonia" if pred_class == 1 else "Normal",
                "confidence": float(confidence),
                "class_probabilities": {
                    "Normal": float(probs[0, 0].item()),
                    "Pneumonia": float(probs[0, 1].item())
                },
                "prediction_class": int(pred_class)
            }
            
            # Generate Grad-CAM explanation if requested
            if generate_explanation and self._explainer is not None:
                try:
                    # Convert PIL to numpy for explainer
                    img_array = np.array(image.resize((224, 224)))
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)
                    
                    heatmap = self._explainer.explain(input_tensor, pred_class)
                    visualization = self._explainer.visualize(img_array, heatmap)
                    
                    # Convert visualization to base64
                    vis_img = Image.fromarray((visualization * 255).astype(np.uint8))
                    buffer = BytesIO()
                    vis_img.save(buffer, format="PNG")
                    vis_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    result["explanation"] = {
                        "heatmap_available": True,
                        "visualization_base64": f"data:image/png;base64,{vis_base64}",
                        "important_regions": self._explainer._find_important_region(heatmap)
                    }
                except Exception as e:
                    logger.warning(f"Could not generate explanation: {e}")
                    result["explanation"] = {"heatmap_available": False, "error": str(e)}
            
            return result
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def reload_model(self):
        """Reload the model from checkpoint."""
        self._load_model()


@app.post("/api/inference/predict")
async def predict_image(
    file: UploadFile = File(...),
    hospital_id: Optional[int] = None,
    generate_explanation: bool = True
):
    """
    Predict pneumonia from uploaded X-ray image.
    
    Args:
        file: Uploaded image file
        hospital_id: Optional hospital ID for tracking
        generate_explanation: Whether to generate Grad-CAM explanation
    
    Returns:
        Prediction results with confidence scores and optional explanation
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        
        # Get model instance
        model = InferenceModel.get_instance()
        
        # Predict
        result = model.predict(image, generate_explanation=generate_explanation)
        result["hospital_id"] = hospital_id
        result["timestamp"] = datetime.now().isoformat()
        result["image_size"] = image.size
        
        logger.info(f"Inference request from hospital {hospital_id}: {result['prediction']} ({result['confidence']:.2%})")
        
        return {"status": "success", "data": result}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/inference/predict-base64")
async def predict_image_base64(
    request: dict,
    hospital_id: Optional[int] = None,
    generate_explanation: bool = True
):
    """
    Predict from base64 encoded image.
    
    Args:
        request: {"image": "base64_string"}
        hospital_id: Optional hospital ID
        generate_explanation: Whether to generate explanation
    """
    try:
        if "image" not in request:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
        
        # Decode base64
        image_data = base64.b64decode(request["image"])
        image = Image.open(BytesIO(image_data))
        
        # Get model instance
        model = InferenceModel.get_instance()
        
        # Predict
        result = model.predict(image, generate_explanation=generate_explanation)
        result["hospital_id"] = hospital_id
        result["timestamp"] = datetime.now().isoformat()
        
        return {"status": "success", "data": result}
    
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/inference/reload-model")
async def reload_model():
    """Reload the inference model from checkpoint."""
    try:
        # Reset singleton to force reload
        InferenceModel._instance = None
        InferenceModel._model = None
        InferenceModel._device = None
        InferenceModel._explainer = None
        InferenceModel._transform = None
        
        # Create new instance (will trigger _load_model)
        model = InferenceModel.get_instance()
        
        if model._model is None:
            return {"status": "error", "message": "Failed to load model. Check server logs."}
        
        return {
            "status": "success", 
            "message": "Model reloaded",
            "device": str(model._device),
            "parameters": sum(p.numel() for p in model._model.parameters())
        }
    except Exception as e:
        logger.error(f"Reload error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


@app.get("/api/inference/model-status")
async def get_model_status():
    """Check if inference model is loaded."""
    try:
        model = InferenceModel.get_instance()
        is_loaded = model._model is not None
        
        # Get project root
        project_root = Path(__file__).parent.parent.parent
        
        # Check multiple possible paths
        possible_paths = [
            project_root / "checkpoints" / "best_model.pt",
            project_root / "checkpoints" / "final_model.pt",
            project_root / "checkpoints" / "best_model_dp.pt",
        ]
        
        found_path = None
        for path in possible_paths:
            if path.exists():
                found_path = str(path)
                break
        
        return {
            "status": "success",
            "data": {
                "model_loaded": is_loaded,
                "checkpoint_exists": found_path is not None,
                "checkpoint_path": found_path,
                "device": str(model._device) if model._device else None,
                "error": None if is_loaded else "Model not loaded. Try reloading or check server logs.",
                "can_reload": found_path is not None
            }
        }
    except Exception as e:
        logger.error(f"Model status check error: {e}")
        return {
            "status": "error",
            "data": {
                "model_loaded": False,
                "error": str(e)
            }
        }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()

