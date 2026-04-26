import React, { useState, useRef, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Activity, Shield, Brain, Building2, Zap, TrendingUp,
  Lock, Play, Pause, RotateCcw, CheckCircle, AlertCircle,
  Upload, Database, Cpu, Network, Server, Image as ImageIcon,
  FileImage, X, Loader
} from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer
} from 'recharts';

// Styles
const styles = `
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
  
  :root {
    --bg-primary: #0a0a1a;
    --bg-secondary: #12122a;
    --bg-tertiary: #1a1a3a;
    --accent-cyan: #00e5ff;
    --accent-purple: #b24dff;
    --accent-green: #00ff9d;
    --accent-orange: #ff9500;
    --accent-red: #ff4d6a;
    --text-primary: #ffffff;
    --text-secondary: #8892b0;
    --border-color: rgba(255, 255, 255, 0.08);
  }

  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: 'Space Grotesk', -apple-system, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    overflow-x: hidden;
  }

  .dashboard {
    min-height: 100vh;
    background: var(--bg-primary);
    background-image: 
      radial-gradient(ellipse at 0% 0%, rgba(0, 229, 255, 0.08) 0%, transparent 50%),
      radial-gradient(ellipse at 100% 100%, rgba(178, 77, 255, 0.08) 0%, transparent 50%);
  }

  .header {
    padding: 20px 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-color);
    backdrop-filter: blur(20px);
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(10, 10, 26, 0.9);
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 14px;
  }

  .logo-icon {
    width: 50px;
    height: 50px;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 8px 32px rgba(0, 229, 255, 0.2);
  }

  .logo-text h1 {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
  }

  .logo-text span {
    font-size: 0.7rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 3px;
    font-weight: 500;
  }

  .status-badge {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 20px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 500;
  }

  .status-running {
    background: rgba(0, 255, 157, 0.15);
    color: var(--accent-green);
    border: 1px solid rgba(0, 255, 157, 0.4);
    animation: pulse-glow 2s infinite;
  }

  @keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0, 255, 157, 0.4); }
    50% { box-shadow: 0 0 20px 5px rgba(0, 255, 157, 0.2); }
  }

  .status-idle {
    background: rgba(136, 146, 176, 0.15);
    color: var(--text-secondary);
    border: 1px solid rgba(136, 146, 176, 0.3);
  }

  .main-content {
    padding: 28px 40px;
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 20px;
  }

  .card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    padding: 24px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
  }

  .card:hover {
    border-color: rgba(255, 255, 255, 0.15);
    transform: translateY(-2px);
  }

  .card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-purple), transparent);
    opacity: 0.6;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }

  .card-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text-secondary);
  }

  .metric-card {
    grid-column: span 3;
  }

  .metric-value {
    font-size: 2.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
  }

  .metric-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 6px;
  }

  .fl-visualization {
    grid-column: span 12;
    min-height: 420px;
  }

  .fl-diagram {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 30px 60px;
    position: relative;
  }

  .hospital-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    z-index: 2;
  }

  .hospital-icon-wrapper {
    width: 72px;
    height: 72px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    transition: all 0.4s ease;
  }

  .hospital-icon-wrapper.training {
    animation: pulse-training 1.2s infinite ease-in-out;
    box-shadow: 0 0 30px 10px rgba(0, 229, 255, 0.3);
  }

  .hospital-icon-wrapper.uploading {
    animation: pulse-upload 0.8s infinite;
  }

  @keyframes pulse-training {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.12); }
  }

  @keyframes pulse-upload {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
  }

  .central-server {
    width: 100px;
    height: 100px;
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-cyan));
    border-radius: 24px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    z-index: 3;
    box-shadow: 0 12px 40px rgba(178, 77, 255, 0.3);
  }

  .central-server.aggregating {
    animation: aggregate-pulse 0.6s infinite;
    box-shadow: 0 12px 50px rgba(178, 77, 255, 0.5);
  }

  @keyframes aggregate-pulse {
    0%, 100% { transform: translateX(-50%) scale(1) rotate(0deg); }
    50% { transform: translateX(-50%) scale(1.15) rotate(5deg); }
  }

  .step-indicator {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-top: 24px;
    flex-wrap: wrap;
  }

  .step {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border-radius: 10px;
    font-size: 0.7rem;
    font-weight: 500;
    transition: all 0.3s ease;
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    border: 1px solid transparent;
  }

  .step.active {
    background: rgba(0, 229, 255, 0.2);
    color: var(--accent-cyan);
    border-color: var(--accent-cyan);
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(0, 229, 255, 0.3);
  }

  .step.completed {
    background: rgba(0, 255, 157, 0.15);
    color: var(--accent-green);
    border-color: rgba(0, 255, 157, 0.3);
  }

  .data-comparison {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 24px;
  }

  .comparison-box {
    padding: 24px;
    border-radius: 16px;
    text-align: center;
    transition: all 0.3s ease;
  }

  .comparison-box.bad {
    background: rgba(255, 77, 106, 0.1);
    border: 1px solid rgba(255, 77, 106, 0.3);
  }

  .comparison-box.good {
    background: rgba(0, 255, 157, 0.1);
    border: 1px solid rgba(0, 255, 157, 0.3);
  }

  .comparison-box.good.active {
    box-shadow: 0 0 30px rgba(0, 255, 157, 0.2);
    transform: scale(1.02);
  }

  .hospital-item {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 16px;
    background: var(--bg-tertiary);
    border-radius: 14px;
    margin-bottom: 10px;
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
  }

  .hospital-item:hover {
    border-color: rgba(0, 229, 255, 0.3);
  }

  .hospital-item.active {
    border-color: var(--accent-green);
    box-shadow: 0 0 25px rgba(0, 255, 157, 0.15);
    background: rgba(0, 255, 157, 0.05);
  }

  .hospital-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1.1rem;
    font-family: 'JetBrains Mono', monospace;
  }

  .hospital-info {
    flex: 1;
  }

  .hospital-name {
    font-weight: 500;
    margin-bottom: 3px;
    font-size: 0.9rem;
  }

  .hospital-stats {
    display: flex;
    gap: 14px;
    font-size: 0.7rem;
    color: var(--text-secondary);
  }

  .control-btn {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 28px;
    border-radius: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    border: none;
    font-size: 0.9rem;
    font-family: 'Space Grotesk', sans-serif;
  }

  .control-btn-primary {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
    color: white;
    box-shadow: 0 8px 30px rgba(0, 229, 255, 0.25);
  }

  .control-btn-primary:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0, 229, 255, 0.35);
  }

  .control-btn-secondary {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    border: 1px solid var(--border-color);
  }

  .control-btn-secondary:hover {
    border-color: var(--accent-cyan);
    color: var(--accent-cyan);
  }

  .timeline-item {
    display: flex;
    gap: 14px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border-color);
    animation: fadeIn 0.3s ease;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .timeline-icon {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .packet-fly {
    position: absolute;
    font-size: 0.6rem;
    padding: 4px 8px;
    background: var(--accent-green);
    color: #000;
    border-radius: 6px;
    font-weight: 600;
    white-space: nowrap;
    z-index: 10;
  }

  /* Connection lines */
  .connection-line {
    position: absolute;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan) 50%, transparent);
    opacity: 0.3;
    z-index: 1;
  }

  .scrollable {
    max-height: 340px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--accent-cyan) var(--bg-tertiary);
  }

  .scrollable::-webkit-scrollbar {
    width: 6px;
  }

  .scrollable::-webkit-scrollbar-track {
    background: var(--bg-tertiary);
    border-radius: 3px;
  }

  .scrollable::-webkit-scrollbar-thumb {
    background: var(--accent-cyan);
    border-radius: 3px;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .spin {
    animation: spin 1s linear infinite;
  }
`;

// FL Step descriptions for professor demo
const FL_STEPS = [
  { id: 1, name: "Download Model", icon: "📥" },
  { id: 2, name: "Local Training", icon: "🏥" },
  { id: 3, name: "Compute Updates", icon: "📊" },
  { id: 4, name: "Add DP Noise", icon: "🔒" },
  { id: 5, name: "Send Updates", icon: "📤" },
  { id: 6, name: "Aggregate", icon: "⚖️" },
  { id: 7, name: "Distribute", icon: "✅" }
];

// Custom tooltip for charts
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{
        background: '#1a1a3a',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: '10px',
        padding: '12px 16px',
        boxShadow: '0 8px 32px rgba(0,0,0,0.3)'
      }}>
        <p style={{ color: '#8892b0', marginBottom: '8px', fontSize: '0.8rem' }}>Round {label}</p>
        {payload.map((entry, index) => (
          <p key={index} style={{ color: entry.color, fontSize: '0.85rem', fontWeight: '500' }}>
            {entry.name}: {typeof entry.value === 'number' ? (entry.value * 100).toFixed(1) + '%' : entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

// Delay helper
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Hospital Inference Panel Component
function HospitalInferencePanel() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [hospitalId, setHospitalId] = useState(0);
  
  const hospitals = [
    { id: 0, name: 'Metro General Hospital' },
    { id: 1, name: 'City Medical Center' },
    { id: 2, name: 'University Hospital' },
    { id: 3, name: 'Regional Health Center' },
    { id: 4, name: 'Community Medical' },
  ];

  useEffect(() => {
    // Check model status
    fetch('http://localhost:8000/api/inference/model-status')
      .then(res => res.json())
      .then(data => setModelStatus(data.data))
      .catch(err => console.error('Model status check failed:', err));
  }, []);

  const handleImageSelect = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }

    setSelectedImage(file);
    setError(null);
    setPrediction(null);

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(file);
  };

  const handlePredict = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);
      formData.append('hospital_id', hospitalId);
      formData.append('generate_explanation', 'true');

      const response = await fetch('http://localhost:8000/api/inference/predict', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const result = await response.json();
      setPrediction(result.data);
    } catch (err) {
      setError(err.message);
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedImage(null);
    setPreview(null);
    setPrediction(null);
    setError(null);
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
      {/* Left: Upload Section */}
      <div>
        <div style={{ marginBottom: '16px' }}>
          <label style={{ 
            display: 'block', 
            marginBottom: '8px', 
            fontSize: '0.85rem', 
            color: 'var(--text-secondary)',
            fontWeight: '500'
          }}>
            Select Hospital:
          </label>
          <select
            value={hospitalId}
            onChange={(e) => setHospitalId(parseInt(e.target.value))}
            style={{
              width: '100%',
              padding: '10px',
              background: 'var(--bg-tertiary)',
              border: '1px solid var(--border-color)',
              borderRadius: '10px',
              color: 'var(--text-primary)',
              fontSize: '0.9rem'
            }}
          >
            {hospitals.map((h, idx) => (
              <option key={idx} value={idx}>Hospital {idx + 1}: {h.name}</option>
            ))}
          </select>
        </div>

        <div style={{
          border: '2px dashed var(--border-color)',
          borderRadius: '16px',
          padding: '40px',
          textAlign: 'center',
          background: 'var(--bg-tertiary)',
          transition: 'all 0.3s ease',
          cursor: 'pointer',
          position: 'relative'
        }}
        onDragOver={(e) => { e.preventDefault(); e.currentTarget.style.borderColor = 'var(--accent-cyan)'; }}
        onDragLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border-color)'; }}
        onDrop={(e) => {
          e.preventDefault();
          e.currentTarget.style.borderColor = 'var(--border-color)';
          const file = e.dataTransfer.files[0];
          if (file && file.type.startsWith('image/')) {
            setSelectedImage(file);
            const reader = new FileReader();
            reader.onloadend = () => setPreview(reader.result);
            reader.readAsDataURL(file);
          }
        }}
        >
          {preview ? (
            <div style={{ position: 'relative' }}>
              <img 
                src={preview} 
                alt="Preview" 
                style={{ 
                  maxWidth: '100%', 
                  maxHeight: '300px', 
                  borderRadius: '12px',
                  border: '1px solid var(--border-color)'
                }} 
              />
              <button
                onClick={handleClear}
                style={{
                  position: 'absolute',
                  top: '8px',
                  right: '8px',
                  background: 'rgba(255, 77, 106, 0.9)',
                  border: 'none',
                  borderRadius: '50%',
                  width: '32px',
                  height: '32px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  color: 'white'
                }}
              >
                <X size={16} />
              </button>
            </div>
          ) : (
            <>
              <ImageIcon size={48} color="var(--accent-cyan)" style={{ margin: '0 auto 16px' }} />
              <p style={{ color: 'var(--text-secondary)', marginBottom: '12px' }}>
                Drag & drop X-ray image here, or click to browse
              </p>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                style={{ display: 'none' }}
                id="image-upload"
              />
              <label
                htmlFor="image-upload"
                style={{
                  display: 'inline-block',
                  padding: '10px 20px',
                  background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-purple))',
                  borderRadius: '10px',
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                  fontWeight: '500'
                }}
              >
                Select Image
              </label>
            </>
          )}
        </div>

        {selectedImage && (
          <div style={{ marginTop: '16px', display: 'flex', gap: '12px' }}>
            <button
              onClick={handlePredict}
              disabled={loading || !modelStatus?.model_loaded}
              style={{
                flex: 1,
                padding: '12px 24px',
                background: loading || !modelStatus?.model_loaded 
                  ? 'var(--bg-tertiary)' 
                  : 'linear-gradient(135deg, var(--accent-green), var(--accent-cyan))',
                border: 'none',
                borderRadius: '12px',
                color: 'white',
                fontWeight: '600',
                cursor: loading || !modelStatus?.model_loaded ? 'not-allowed' : 'pointer',
                opacity: loading || !modelStatus?.model_loaded ? 0.5 : 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px'
              }}
            >
              {loading ? (
                <>
                  <Loader size={18} className="spin" />
                  Analyzing...
                </>
              ) : !modelStatus?.model_loaded ? (
                'Model Not Loaded'
              ) : (
                <>
                  <Brain size={18} />
                  Diagnose X-Ray
                </>
              )}
            </button>
            <button
              onClick={handleClear}
              style={{
                padding: '12px 20px',
                background: 'var(--bg-tertiary)',
                border: '1px solid var(--border-color)',
                borderRadius: '12px',
                color: 'var(--text-secondary)',
                cursor: 'pointer'
              }}
            >
              Clear
            </button>
          </div>
        )}

        {error && (
          <div style={{
            marginTop: '16px',
            padding: '12px',
            background: 'rgba(255, 77, 106, 0.1)',
            border: '1px solid rgba(255, 77, 106, 0.3)',
            borderRadius: '10px',
            color: 'var(--accent-red)',
            fontSize: '0.85rem'
          }}>
            <AlertCircle size={16} style={{ display: 'inline', marginRight: '8px' }} />
            {error}
          </div>
        )}

        {modelStatus && (
          <div style={{
            marginTop: '16px',
            padding: '12px',
            background: modelStatus.model_loaded 
              ? 'rgba(0, 255, 157, 0.1)' 
              : 'rgba(255, 149, 0, 0.1)',
            border: `1px solid ${modelStatus.model_loaded ? 'rgba(0, 255, 157, 0.3)' : 'rgba(255, 149, 0, 0.3)'}`,
            borderRadius: '10px',
            fontSize: '0.75rem',
            color: 'var(--text-secondary)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <div>
                <strong style={{ color: modelStatus.model_loaded ? 'var(--accent-green)' : 'var(--accent-orange)' }}>
                  Model Status:
                </strong> {modelStatus.model_loaded ? '✅ Loaded and Ready' : '⚠️ Not Loaded'}
              </div>
              {!modelStatus.model_loaded && modelStatus.can_reload && (
                <button
                  onClick={async () => {
                    setLoading(true);
                    try {
                      const response = await fetch('http://localhost:8000/api/inference/reload-model', {
                        method: 'POST'
                      });
                      const result = await response.json();
                      if (result.status === 'success') {
                        // Refresh model status
                        const statusRes = await fetch('http://localhost:8000/api/inference/model-status');
                        const statusData = await statusRes.json();
                        setModelStatus(statusData.data);
                        setError(null);
                      } else {
                        setError(result.message || 'Failed to reload model');
                      }
                    } catch (err) {
                      setError('Failed to reload model: ' + err.message);
                    } finally {
                      setLoading(false);
                    }
                  }}
                  disabled={loading}
                  style={{
                    padding: '6px 12px',
                    background: 'var(--accent-cyan)',
                    border: 'none',
                    borderRadius: '6px',
                    color: 'white',
                    fontSize: '0.7rem',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    opacity: loading ? 0.5 : 1
                  }}
                >
                  {loading ? 'Loading...' : '🔄 Reload Model'}
                </button>
              )}
            </div>
            {modelStatus.checkpoint_path && (
              <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Checkpoint: {modelStatus.checkpoint_path.split('/').pop()}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Right: Results Section */}
      <div>
        {prediction ? (
          <div>
            <h3 style={{ fontSize: '1.1rem', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <CheckCircle size={20} color="var(--accent-green)" />
              Diagnosis Results
            </h3>

            <div style={{
              padding: '20px',
              background: prediction.prediction === 'Pneumonia' 
                ? 'rgba(255, 77, 106, 0.1)' 
                : 'rgba(0, 255, 157, 0.1)',
              border: `2px solid ${prediction.prediction === 'Pneumonia' ? 'var(--accent-red)' : 'var(--accent-green)'}`,
              borderRadius: '16px',
              marginBottom: '20px'
            }}>
              <div style={{ fontSize: '1.5rem', fontWeight: '700', marginBottom: '8px' }}>
                {prediction.prediction === 'Pneumonia' ? '⚠️ Pneumonia Detected' : '✅ Normal'}
              </div>
              <div style={{ fontSize: '2rem', fontWeight: '700', fontFamily: 'JetBrains Mono', marginBottom: '12px' }}>
                {(prediction.confidence * 100).toFixed(1)}% Confidence
              </div>
              
              <div style={{ marginTop: '16px' }}>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                  Class Probabilities:
                </div>
                <div style={{ display: 'flex', gap: '12px' }}>
                  <div style={{ flex: 1, padding: '10px', background: 'var(--bg-tertiary)', borderRadius: '8px' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>Normal</div>
                    <div style={{ fontSize: '1.2rem', fontWeight: '600' }}>
                      {(prediction.class_probabilities.Normal * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div style={{ flex: 1, padding: '10px', background: 'var(--bg-tertiary)', borderRadius: '8px' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>Pneumonia</div>
                    <div style={{ fontSize: '1.2rem', fontWeight: '600' }}>
                      {(prediction.class_probabilities.Pneumonia * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {prediction.explanation && prediction.explanation.heatmap_available && (
              <div>
                <h4 style={{ fontSize: '0.9rem', marginBottom: '12px', color: 'var(--text-secondary)' }}>
                  Grad-CAM Explanation (Important Regions Highlighted):
                </h4>
                <img 
                  src={prediction.explanation.visualization_base64} 
                  alt="Grad-CAM Explanation"
                  style={{
                    width: '100%',
                    borderRadius: '12px',
                    border: '1px solid var(--border-color)'
                  }}
                />
              </div>
            )}
          </div>
        ) : (
          <div style={{
            padding: '60px 20px',
            textAlign: 'center',
            color: 'var(--text-secondary)'
          }}>
            <FileImage size={48} style={{ margin: '0 auto 16px', opacity: 0.3 }} />
            <p>Upload an X-ray image to get diagnosis</p>
          </div>
        )}
      </div>
    </div>
  );
}

// Main App Component
function App() {
  const [isRunning, setIsRunning] = useState(false);
  const [currentRound, setCurrentRound] = useState(0);
  const [totalRounds] = useState(10);
  const [currentStep, setCurrentStep] = useState(0);
  const [metricsHistory, setMetricsHistory] = useState([]);
  const [activityLog, setActivityLog] = useState([]);
  const [hospitals, setHospitals] = useState([
    { id: 0, name: 'Metro General Hospital', status: 'idle', samples: 354, accuracy: 0, loss: 0, color: '#00e5ff', dataSize: '847 MB', updateSize: '17.6 MB' },
    { id: 1, name: 'City Medical Center', status: 'idle', samples: 593, accuracy: 0, loss: 0, color: '#b24dff', dataSize: '1.2 GB', updateSize: '17.6 MB' },
    { id: 2, name: 'University Hospital', status: 'idle', samples: 2027, accuracy: 0, loss: 0, color: '#00ff9d', dataSize: '4.1 GB', updateSize: '17.6 MB' },
    { id: 3, name: 'Regional Health Center', status: 'idle', samples: 671, accuracy: 0, loss: 0, color: '#ff9500', dataSize: '1.4 GB', updateSize: '17.6 MB' },
    { id: 4, name: 'Community Medical', status: 'idle', samples: 523, accuracy: 0, loss: 0, color: '#ff4d6a', dataSize: '1.1 GB', updateSize: '17.6 MB' },
  ]);
  const [privacyBudget, setPrivacyBudget] = useState({ used: 0, total: 8.0 });
  const stopRef = useRef(false);

  const addLog = useCallback((message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setActivityLog(prev => [...prev.slice(-15), { timestamp, message, type, id: Date.now() }]);
  }, []);

  // Simulate FL training step by step
  const runDemo = useCallback(async () => {
    stopRef.current = false;
    setIsRunning(true);
    setMetricsHistory([]);
    setActivityLog([]);
    setPrivacyBudget({ used: 0, total: 8.0 });
    setHospitals(prev => prev.map(h => ({ ...h, status: 'idle', accuracy: 0, loss: 0 })));
    
    addLog('🚀 Starting Federated Learning Demo...', 'success');
    await delay(800);
    addLog(`📋 Config: 5 hospitals, ${totalRounds} rounds, ε=8.0`, 'info');
    await delay(500);
    addLog('⚠️ KEY: Patient data NEVER leaves hospitals!', 'warning');
    await delay(1000);

    for (let round = 1; round <= totalRounds; round++) {
      if (stopRef.current) break;
      
      setCurrentRound(round);
      addLog(`\n━━━ ROUND ${round}/${totalRounds} ━━━`, 'info');
      await delay(500);

      // Step 1: Download model
      setCurrentStep(1);
      addLog('📥 Central server → Sending global model...', 'info');
      setHospitals(prev => prev.map(h => ({ ...h, status: 'downloading' })));
      await delay(1200);
      if (stopRef.current) break;

      // Step 2: Local training
      setCurrentStep(2);
      addLog('🏥 LOCAL TRAINING (data stays at hospital!)', 'success');
      
      for (let i = 0; i < 5; i++) {
        if (stopRef.current) break;
        setHospitals(prev => prev.map((h, idx) => 
          idx === i ? { ...h, status: 'training' } : h
        ));
        addLog(`  → H${i + 1}: Training on ${hospitals[i].samples} X-rays...`, 'info');
        await delay(600);
        
        const newAcc = Math.min(0.99, 0.82 + (round * 0.015) + Math.random() * 0.05);
        const newLoss = Math.max(0.05, 0.45 - (round * 0.03) + Math.random() * 0.04);
        
        setHospitals(prev => prev.map((h, idx) => 
          idx === i ? { ...h, accuracy: newAcc, loss: newLoss, status: 'trained' } : h
        ));
      }
      if (stopRef.current) break;

      // Step 3: Compute updates
      setCurrentStep(3);
      addLog('📊 Computing weight gradients...', 'info');
      await delay(800);
      if (stopRef.current) break;

      // Step 4: Add DP noise
      setCurrentStep(4);
      const newEpsilon = Math.min(8.0, privacyBudget.used + 0.8);
      addLog(`🔒 Adding DP noise (ε now = ${newEpsilon.toFixed(1)})`, 'warning');
      setPrivacyBudget(prev => ({ ...prev, used: newEpsilon }));
      await delay(1000);
      if (stopRef.current) break;

      // Step 5: Send updates
      setCurrentStep(5);
      addLog('📤 Sending ONLY model updates:', 'success');
      for (let i = 0; i < 5; i++) {
        if (stopRef.current) break;
        addLog(`  → H${i + 1}: 17.6 MB (NOT ${hospitals[i].dataSize}!)`, 'success');
        setHospitals(prev => prev.map((h, idx) => 
          idx === i ? { ...h, status: 'uploading' } : h
        ));
        await delay(400);
      }
      if (stopRef.current) break;

      // Step 6: Aggregate
      setCurrentStep(6);
      addLog('⚖️ Aggregating with adaptive weights...', 'info');
      setHospitals(prev => prev.map(h => ({ ...h, status: 'aggregating' })));
      await delay(1200);
      if (stopRef.current) break;

      // Step 7: Distribute
      setCurrentStep(7);
      const globalAcc = Math.min(0.99, 0.85 + (round * 0.012) + Math.random() * 0.02);
      const globalLoss = Math.max(0.03, 0.35 - (round * 0.025) + Math.random() * 0.02);
      const globalAuc = Math.min(0.99, 0.88 + (round * 0.01) + Math.random() * 0.015);
      
      addLog(`✅ Round ${round} done! Accuracy: ${(globalAcc * 100).toFixed(1)}%`, 'success');
      
      setMetricsHistory(prev => [...prev, {
        round,
        accuracy: globalAcc,
        loss: globalLoss,
        auc: globalAuc
      }]);

      setHospitals(prev => prev.map(h => ({ ...h, status: 'idle' })));
      setCurrentStep(0);
      await delay(800);
    }
    
    if (!stopRef.current) {
      addLog('🎉 TRAINING COMPLETE!', 'success');
      addLog('✅ Model improved while preserving privacy!', 'success');
    }
    setIsRunning(false);
    setCurrentStep(0);
  }, [addLog, hospitals, privacyBudget.used, totalRounds]);

  const handleStop = () => {
    stopRef.current = true;
    setIsRunning(false);
    addLog('⏹️ Demo stopped', 'warning');
  };

  const handleReset = () => {
    stopRef.current = true;
    setIsRunning(false);
    setCurrentRound(0);
    setCurrentStep(0);
    setMetricsHistory([]);
    setActivityLog([]);
    setPrivacyBudget({ used: 0, total: 8.0 });
    setHospitals(prev => prev.map(h => ({ ...h, status: 'idle', accuracy: 0, loss: 0 })));
  };

  const latestMetrics = metricsHistory[metricsHistory.length - 1] || {};
  const progress = (currentRound / totalRounds) * 100;

  return (
    <>
      <style>{styles}</style>
      <div className="dashboard">
        {/* Header */}
        <header className="header">
          <div className="logo">
            <div className="logo-icon">
              <Brain size={28} color="white" />
            </div>
            <div className="logo-text">
              <h1>Hospital FL</h1>
              <span>Privacy-Preserving AI</span>
            </div>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div className={`status-badge ${isRunning ? 'status-running' : 'status-idle'}`}>
              {isRunning ? (
                <>
                  <Activity size={16} />
                  Training Round {currentRound}/{totalRounds}
                </>
              ) : (
                <>
                  <Pause size={16} />
                  Ready for Demo
                </>
              )}
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="main-content">
          {/* Key Message Banner */}
          <motion.div
            className="card"
            style={{ gridColumn: 'span 12', background: 'linear-gradient(135deg, rgba(0, 229, 255, 0.08), rgba(178, 77, 255, 0.08))' }}
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
              <div style={{ 
                width: '64px', 
                height: '64px', 
                borderRadius: '16px', 
                background: 'linear-gradient(135deg, #00e5ff, #b24dff)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 8px 32px rgba(0, 229, 255, 0.2)'
              }}>
                <Shield size={32} color="white" />
              </div>
              <div>
                <h2 style={{ fontSize: '1.2rem', marginBottom: '6px', fontWeight: '600' }}>
                  🔐 Key Privacy Guarantee
                </h2>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: '1.6' }}>
                  <strong style={{ color: 'var(--accent-green)' }}>Patient X-ray data NEVER leaves the hospital.</strong> Only mathematical model updates 
                  (17.6 MB each) are shared — not the actual images (GBs of sensitive data).
                </p>
              </div>
            </div>
          </motion.div>

          {/* Metric Cards */}
          <motion.div className="card metric-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <div className="card-title"><TrendingUp size={18} color="var(--accent-cyan)" />Global Accuracy</div>
            <div className="metric-value">{((latestMetrics.accuracy || 0) * 100).toFixed(1)}%</div>
            <div className="metric-label">Across all hospitals</div>
          </motion.div>

          <motion.div className="card metric-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <div className="card-title"><Activity size={18} color="var(--accent-purple)" />AUC-ROC Score</div>
            <div className="metric-value">{((latestMetrics.auc || 0) * 100).toFixed(1)}%</div>
            <div className="metric-label">Classification performance</div>
          </motion.div>

          <motion.div className="card metric-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
            <div className="card-title"><Shield size={18} color="var(--accent-green)" />Privacy Budget</div>
            <div className="metric-value">ε = {privacyBudget.used.toFixed(1)}</div>
            <div className="metric-label">of {privacyBudget.total} total</div>
          </motion.div>

          <motion.div className="card metric-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
            <div className="card-title"><Zap size={18} color="var(--accent-orange)" />Progress</div>
            <div className="metric-value">{progress.toFixed(0)}%</div>
            <div className="metric-label">Round {currentRound} of {totalRounds}</div>
          </motion.div>

          {/* FL Visualization */}
          <motion.div
            className="card fl-visualization"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <div className="card-header">
              <div className="card-title">
                <Network size={18} color="var(--accent-cyan)" />
                Federated Learning Process Visualization
              </div>
              <div style={{ 
                fontSize: '0.75rem', 
                color: currentStep > 0 ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                fontWeight: currentStep > 0 ? '600' : '400'
              }}>
                {currentStep > 0 ? `${FL_STEPS[currentStep - 1]?.icon} ${FL_STEPS[currentStep - 1]?.name}` : 'Click Start Demo'}
              </div>
            </div>

            {/* Step Progress */}
            <div className="step-indicator">
              {FL_STEPS.map((step) => (
                <div 
                  key={step.id}
                  className={`step ${currentStep === step.id ? 'active' : ''} ${currentStep > step.id ? 'completed' : ''}`}
                >
                  {currentStep > step.id ? <CheckCircle size={12} /> : <span>{step.icon}</span>}
                  <span>{step.name}</span>
                </div>
              ))}
            </div>

            {/* Visual Diagram */}
            <div className="fl-diagram">
              {/* Hospitals */}
              {hospitals.map((hospital, idx) => (
                <motion.div 
                  key={hospital.id}
                  className="hospital-node"
                  animate={{
                    scale: hospital.status === 'training' ? 1.15 : 1,
                    y: hospital.status === 'uploading' ? -12 : 0
                  }}
                  transition={{ duration: 0.3 }}
                  style={{ 
                    marginLeft: idx === 0 ? 0 : 'auto',
                    marginRight: idx === 4 ? 0 : 'auto'
                  }}
                >
                  <div 
                    className={`hospital-icon-wrapper ${hospital.status === 'training' ? 'training' : ''} ${hospital.status === 'uploading' ? 'uploading' : ''}`}
                    style={{ 
                      background: `${hospital.color}20`,
                      border: `2px solid ${hospital.status !== 'idle' ? hospital.color : 'transparent'}`,
                      boxShadow: hospital.status === 'training' ? `0 0 30px ${hospital.color}40` : 'none'
                    }}
                  >
                    <Building2 size={32} color={hospital.color} />
                    {hospital.status === 'uploading' && (
                      <motion.div
                        className="packet-fly"
                        initial={{ opacity: 0, y: 0 }}
                        animate={{ opacity: [0, 1, 1, 0], y: [-5, -15, -25, -35] }}
                        transition={{ duration: 1.2, repeat: Infinity }}
                      >
                        📤 17.6 MB
                      </motion.div>
                    )}
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ 
                      fontSize: '0.8rem', 
                      fontWeight: '600',
                      color: hospital.status !== 'idle' ? hospital.color : 'inherit'
                    }}>
                      H{hospital.id + 1}
                    </div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>
                      {hospital.samples} samples
                    </div>
                    {hospital.accuracy > 0 && (
                      <div style={{ 
                        fontSize: '0.65rem', 
                        color: 'var(--accent-green)',
                        fontWeight: '600',
                        marginTop: '2px'
                      }}>
                        {(hospital.accuracy * 100).toFixed(1)}%
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}

              {/* Central Server */}
              <motion.div 
                className={`central-server ${currentStep === 6 ? 'aggregating' : ''}`}
              >
                <Server size={36} color="white" />
                <div style={{ fontSize: '0.6rem', color: 'white', marginTop: '4px', fontWeight: '600' }}>CENTRAL</div>
              </motion.div>
            </div>

            {/* Data Size Comparison */}
            <div className="data-comparison">
              <div className="comparison-box bad">
                <Database size={28} color="var(--accent-red)" style={{ margin: '0 auto 12px' }} />
                <div style={{ fontSize: '1.6rem', fontWeight: '700', color: 'var(--accent-red)' }}>~8.6 GB</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '6px' }}>
                  ❌ Traditional: Send ALL patient X-rays
                </div>
                <div style={{ fontSize: '0.65rem', color: 'var(--accent-red)', marginTop: '8px', fontWeight: '500' }}>
                  Privacy Risk: Patient images exposed!
                </div>
              </div>
              <div className={`comparison-box good ${currentStep === 5 ? 'active' : ''}`}>
                <Upload size={28} color="var(--accent-green)" style={{ margin: '0 auto 12px' }} />
                <div style={{ fontSize: '1.6rem', fontWeight: '700', color: 'var(--accent-green)' }}>~88 MB</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '6px' }}>
                  ✅ Federated: Send ONLY model weights
                </div>
                <div style={{ fontSize: '0.65rem', color: 'var(--accent-green)', marginTop: '8px', fontWeight: '500' }}>
                  Privacy Protected: No patient data shared!
                </div>
              </div>
            </div>
          </motion.div>

          {/* Activity Log */}
          <motion.div
            className="card"
            style={{ gridColumn: 'span 5' }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            <div className="card-header">
              <div className="card-title">
                <Activity size={18} color="var(--accent-cyan)" />
                Live Activity Log
              </div>
            </div>
            <div className="scrollable">
              {activityLog.length === 0 ? (
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', textAlign: 'center', padding: '30px 20px' }}>
                  👆 Click "Start Demo" to watch<br/>federated learning in action!
                </p>
              ) : (
                activityLog.map((log) => (
                  <div key={log.id} className="timeline-item">
                    <div className="timeline-icon" style={{
                      background: log.type === 'success' ? 'rgba(0, 255, 157, 0.15)' :
                                  log.type === 'warning' ? 'rgba(255, 149, 0, 0.15)' :
                                  'rgba(0, 229, 255, 0.15)'
                    }}>
                      {log.type === 'success' ? <CheckCircle size={14} color="var(--accent-green)" /> :
                       log.type === 'warning' ? <AlertCircle size={14} color="var(--accent-orange)" /> :
                       <Activity size={14} color="var(--accent-cyan)" />}
                    </div>
                    <div style={{ minWidth: 0 }}>
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>{log.timestamp}</div>
                      <div style={{ fontSize: '0.8rem', wordBreak: 'break-word' }}>{log.message}</div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </motion.div>

          {/* Hospital Status */}
          <motion.div
            className="card"
            style={{ gridColumn: 'span 4' }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
          >
            <div className="card-header">
              <div className="card-title">
                <Building2 size={18} color="var(--accent-cyan)" />
                Hospital Status
              </div>
            </div>
            <div className="scrollable">
              {hospitals.map((hospital) => (
                <div 
                  key={hospital.id}
                  className={`hospital-item ${hospital.status !== 'idle' ? 'active' : ''}`}
                >
                  <div 
                    className="hospital-icon"
                    style={{ background: `${hospital.color}20`, color: hospital.color }}
                  >
                    H{hospital.id + 1}
                  </div>
                  <div className="hospital-info">
                    <div className="hospital-name">{hospital.name}</div>
                    <div className="hospital-stats">
                      <span>{hospital.samples} samples</span>
                      <span>•</span>
                      <span style={{ 
                        color: hospital.status === 'training' ? 'var(--accent-green)' : 
                               hospital.status === 'uploading' ? 'var(--accent-cyan)' :
                               'var(--text-secondary)',
                        fontWeight: hospital.status !== 'idle' ? '600' : '400'
                      }}>
                        {hospital.status === 'idle' ? 'Ready' :
                         hospital.status === 'training' ? '🔄 Training...' :
                         hospital.status === 'uploading' ? '📤 Sending...' :
                         hospital.status === 'aggregating' ? '⚙️ Aggregating' :
                         hospital.status === 'downloading' ? '📥 Receiving' :
                         '✅ Done'}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Privacy Explanation */}
          <motion.div
            className="card"
            style={{ gridColumn: 'span 3' }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
          >
            <div className="card-header">
              <div className="card-title">
                <Lock size={18} color="var(--accent-green)" />
                Differential Privacy
              </div>
            </div>
            
            <div style={{ textAlign: 'center', padding: '15px 0' }}>
              <div style={{ 
                fontSize: '2.2rem', 
                fontWeight: '700',
                fontFamily: 'JetBrains Mono',
                color: privacyBudget.used < privacyBudget.total * 0.7 ? 'var(--accent-green)' : 'var(--accent-orange)'
              }}>
                ε = {privacyBudget.used.toFixed(2)}
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Privacy Budget Used
              </div>
            </div>

            <div style={{ 
              height: '10px', 
              background: 'var(--bg-tertiary)', 
              borderRadius: '5px', 
              margin: '15px 0',
              overflow: 'hidden'
            }}>
              <motion.div 
                style={{
                  height: '100%',
                  background: 'linear-gradient(90deg, var(--accent-green), var(--accent-cyan))',
                  borderRadius: '5px'
                }}
                animate={{ width: `${(privacyBudget.used / privacyBudget.total) * 100}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>

            <div style={{ 
              padding: '12px',
              background: 'var(--bg-tertiary)',
              borderRadius: '10px',
              fontSize: '0.7rem',
              color: 'var(--text-secondary)',
              lineHeight: '1.6'
            }}>
              <strong style={{ color: 'var(--accent-green)' }}>How it works:</strong><br/>
              Gaussian noise added to gradients makes it mathematically impossible 
              to extract individual patient data from the model.
            </div>
          </motion.div>

          {/* Training Chart */}
          <motion.div
            className="card"
            style={{ gridColumn: 'span 8', minHeight: '280px' }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9 }}
          >
            <div className="card-header">
              <div className="card-title">
                <TrendingUp size={18} color="var(--accent-cyan)" />
                Training Progress
              </div>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={metricsHistory}>
                <defs>
                  <linearGradient id="colorAcc" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00e5ff" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#00e5ff" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorAuc" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#b24dff" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#b24dff" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="round" stroke="#8892b0" fontSize={11} />
                <YAxis stroke="#8892b0" domain={[0.7, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} fontSize={11} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="accuracy" stroke="#00e5ff" strokeWidth={2} fill="url(#colorAcc)" name="Accuracy" />
                <Area type="monotone" dataKey="auc" stroke="#b24dff" strokeWidth={2} fill="url(#colorAuc)" name="AUC-ROC" />
              </AreaChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Adaptive Aggregation */}
          <motion.div
            className="card"
            style={{ gridColumn: 'span 4' }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.0 }}
          >
            <div className="card-header">
              <div className="card-title">
                <Cpu size={18} color="var(--accent-purple)" />
                Adaptive Aggregation
              </div>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
              {[
                { label: 'Sample Weight', value: '30%', color: 'var(--accent-cyan)' },
                { label: 'Stability', value: '25%', color: 'var(--accent-purple)' },
                { label: 'Loss Improve', value: '25%', color: 'var(--accent-green)' },
                { label: 'Quality', value: '20%', color: 'var(--accent-orange)' }
              ].map((item, i) => (
                <div key={i} style={{
                  padding: '12px',
                  background: 'var(--bg-tertiary)',
                  borderRadius: '10px',
                  borderLeft: `3px solid ${item.color}`
                }}>
                  <div style={{ fontSize: '1.2rem', fontWeight: '700', color: item.color }}>
                    {item.value}
                  </div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>
                    {item.label}
                  </div>
                </div>
              ))}
            </div>

            <p style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '14px', lineHeight: '1.5' }}>
              Unlike FedAvg, weights are dynamically adjusted based on client contribution quality.
            </p>
          </motion.div>

          {/* Control Panel */}
          <motion.div
            style={{ gridColumn: 'span 12', display: 'flex', gap: '16px', justifyContent: 'center', padding: '15px 0' }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.1 }}
          >
            {!isRunning ? (
              <button className="control-btn control-btn-primary" onClick={runDemo}>
                <Play size={20} />
                Start Demo (Watch FL in Action!)
              </button>
            ) : (
              <button className="control-btn control-btn-secondary" onClick={handleStop} style={{ borderColor: 'var(--accent-red)', color: 'var(--accent-red)' }}>
                <Pause size={20} />
                Stop Demo
              </button>
            )}
            <button className="control-btn control-btn-secondary" onClick={handleReset}>
              <RotateCcw size={20} />
              Reset
            </button>
          </motion.div>

          {/* Hospital Inference - Image Upload & Prediction */}
          <motion.div
            className="card"
            style={{ gridColumn: 'span 12', minHeight: '500px' }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2 }}
          >
            <div className="card-header">
              <div className="card-title">
                <FileImage size={18} color="var(--accent-cyan)" />
                Hospital Inference - Upload X-Ray for Diagnosis
              </div>
            </div>
            
            <HospitalInferencePanel />
          </motion.div>

          {/* Professor Summary */}
          <motion.div
            className="card"
            style={{ gridColumn: 'span 12', background: 'linear-gradient(135deg, rgba(0, 255, 157, 0.05), rgba(0, 229, 255, 0.05))' }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.3 }}
          >
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px', fontSize: '0.85rem' }}>
              <div>
                <h4 style={{ color: 'var(--accent-cyan)', marginBottom: '12px', fontSize: '0.9rem' }}>🏥 What Hospitals Send</h4>
                <ul style={{ color: 'var(--text-secondary)', lineHeight: '2', paddingLeft: '20px' }}>
                  <li>Model weight updates (~17.6 MB)</li>
                  <li>Training metrics (loss, accuracy)</li>
                  <li>Number of samples used</li>
                  <li><strong style={{ color: 'var(--accent-red)' }}>❌ NOT patient data!</strong></li>
                </ul>
              </div>
              <div>
                <h4 style={{ color: 'var(--accent-purple)', marginBottom: '12px', fontSize: '0.9rem' }}>🔐 Privacy Mechanisms</h4>
                <ul style={{ color: 'var(--text-secondary)', lineHeight: '2', paddingLeft: '20px' }}>
                  <li>Data never leaves hospitals</li>
                  <li>Differential Privacy (ε=8, δ=10⁻⁵)</li>
                  <li>Gradient clipping (max norm=1.0)</li>
                  <li>Gaussian noise injection</li>
                </ul>
              </div>
              <div>
                <h4 style={{ color: 'var(--accent-green)', marginBottom: '12px', fontSize: '0.9rem' }}>⚖️ Key Innovations</h4>
                <ul style={{ color: 'var(--text-secondary)', lineHeight: '2', paddingLeft: '20px' }}>
                  <li>Adaptive aggregation weights</li>
                  <li>Non-IID data handling</li>
                  <li>Grad-CAM explainability</li>
                  <li>Real-time monitoring</li>
                </ul>
              </div>
            </div>
          </motion.div>
        </main>
      </div>
    </>
  );
}

export default App;
