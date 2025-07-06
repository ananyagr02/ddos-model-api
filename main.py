import os
import json
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, RootModel # <-- IMPORTANT: Import RootModel
from typing import List, Dict

# ===================================================================================
# 1. DEFINE MODEL ARCHITECTURES (Unchanged)
# ===================================================================================

device = torch.device('cpu')

class CNNBranch(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__();self.conv1=nn.Conv1d(in_features,64,3,padding=1);self.conv2=nn.Conv1d(64,128,3,padding=1);self.pool=nn.AdaptiveAvgPool1d(1);self.fc=nn.Linear(128,num_classes)
    def forward(self,x):
        x=x.permute(0,2,1);x=F.relu(self.conv1(x));x=F.relu(self.conv2(x));feat=self.pool(x).squeeze(-1);return self.fc(feat),feat
class TransformerBranch(nn.Module):
    def __init__(self,in_features,num_classes,d_model=64,nhead=4):
        super().__init__();self.project=nn.Linear(in_features,d_model);encoder_layer=nn.TransformerEncoderLayer(d_model,nhead,batch_first=True,dropout=0.1);self.encoder=nn.TransformerEncoder(encoder_layer,num_layers=2);self.fc=nn.Linear(d_model,num_classes)
    def forward(self,x):
        x=self.project(x);x=self.encoder(x);feat=x.mean(dim=1);return self.fc(feat),feat
class GatingNet(nn.Module):
    def __init__(self,input_dim,num_classes):
        super().__init__();self.fc=nn.Sequential(nn.Linear(input_dim,64),nn.ReLU(),nn.Linear(64,num_classes),nn.Sigmoid())
    def forward(self,cnn_feat,trans_feat):return self.fc(torch.cat([cnn_feat,trans_feat],dim=1))
class HybridGatedNet(nn.Module):
    def __init__(self,in_features,num_classes,d_model=64,nhead=4):
        super().__init__();self.cnn=CNNBranch(in_features,num_classes);self.trans=TransformerBranch(in_features,num_classes,d_model,nhead);self.gate=GatingNet(128+d_model,num_classes)
    def forward(self,x_seq):
        log_cnn,f_cnn=self.cnn(x_seq);log_trans,f_trans=self.trans(x_seq);gate=self.gate(f_cnn.detach(),f_trans.detach());return gate*log_cnn+(1-gate)*log_trans,gate
class TemporalAttention(nn.Module):
    def __init__(self,input_dim):super().__init__();self.attn=nn.Linear(input_dim,1)
    def forward(self,x):return(x*torch.softmax(self.attn(x),dim=1)).sum(dim=1)
class TemporalCNNExpert(nn.Module):
    def __init__(self,input_dim,num_classes,dropout_prob=0.5):
        super().__init__();self.cnn=nn.Sequential(nn.Conv1d(input_dim,64,3,padding=1),nn.ReLU(),nn.Dropout(dropout_prob),nn.Conv1d(64,128,3,padding=1),nn.ReLU(),nn.Dropout(dropout_prob));self.attn=TemporalAttention(128);self.fc=nn.Linear(128,num_classes)
    def forward(self,x):
        x=self.cnn(x.permute(0,2,1)).permute(0,2,1);pooled=self.attn(x);return self.fc(pooled),pooled
class CrossModalAttentionFusion(nn.Module):
    def __init__(self,tab_dim,cnn_dim,fusion_dim=64,heads=2):
        super().__init__();self.tab_proj=nn.Linear(tab_dim,fusion_dim);self.cnn_proj=nn.Linear(cnn_dim,fusion_dim);self.attn=nn.MultiheadAttention(fusion_dim,heads,batch_first=True,dropout=0.1);self.fc=nn.Sequential(nn.Linear(fusion_dim*2,fusion_dim),nn.ReLU(),nn.Dropout(0.1))
    def forward(self,tab,cnn):
        tab,cnn=self.tab_proj(tab),self.cnn_proj(cnn);x=torch.stack([tab,cnn],dim=1);attn_out,_=self.attn(x,x,x);return self.fc(torch.cat([attn_out[:,0],attn_out[:,1]],dim=1))
class HierarchicalGatingClassifier(nn.Module):
    def __init__(self,lgb_dim,cnn_dim,fusion_dim,num_classes,dropout_prob=0.3):
        super().__init__();self.fc=nn.Sequential(nn.Linear(lgb_dim+cnn_dim+fusion_dim,128),nn.ReLU(),nn.Dropout(dropout_prob),nn.Linear(128,64),nn.ReLU(),nn.Dropout(dropout_prob),nn.Linear(64,num_classes))
    def forward(self,lgb,cnn,fused):return self.fc(torch.cat([lgb,cnn,fused],dim=1))

# ===================================================================================
# 2. LOAD ARTIFACTS (Unchanged)
# ===================================================================================
ARTIFACTS_PATH = "artifacts"
try:
    print("Loading deployment artifacts...")
    scaler = joblib.load(os.path.join(ARTIFACTS_PATH, "scaler.pkl"))
    class_names = joblib.load(os.path.join(ARTIFACTS_PATH, "class_names.pkl"))
    feature_names = joblib.load(os.path.join(ARTIFACTS_PATH, "feature_names.pkl"))
    with open(os.path.join(ARTIFACTS_PATH, "deployment_config.json"), 'r') as f:
        config = json.load(f)
    TIME_STEPS = config['time_steps']
    NUM_CLASSES = len(class_names)
    hybrid_model = HybridGatedNet(len(feature_names), NUM_CLASSES).to(device)
    hybrid_model.load_state_dict(torch.load(os.path.join(ARTIFACTS_PATH, "hybrid_model.pth"), map_location=device))
    hybrid_model.eval()
    hgae_lgbm = joblib.load(os.path.join(ARTIFACTS_PATH, "hgae_lgbm.txt"))
    hgae_temporal = TemporalCNNExpert(len(feature_names), NUM_CLASSES).to(device)
    hgae_temporal.load_state_dict(torch.load(os.path.join(ARTIFACTS_PATH, "hgae_temporal.pth"), map_location=device))
    hgae_temporal.eval()
    hgae_fusion = CrossModalAttentionFusion(NUM_CLASSES, 128).to(device)
    hgae_fusion.load_state_dict(torch.load(os.path.join(ARTIFACTS_PATH, "hgae_fusion.pth"), map_location=device))
    hgae_fusion.eval()
    hgae_gating = HierarchicalGatingClassifier(NUM_CLASSES, NUM_CLASSES, 64, NUM_CLASSES).to(device)
    hgae_gating.load_state_dict(torch.load(os.path.join(ARTIFACTS_PATH, "hgae_gating.pth"), map_location=device))
    hgae_gating.eval()
    print("✅ Artifacts loaded successfully.")
except Exception as e:
    print(f"🔥 Error loading artifacts: {e}")
    scaler, class_names, feature_names, hybrid_model, hgae_lgbm, hgae_temporal, hgae_fusion, hgae_gating = [None]*8

# ===================================================================================
# 3. PREDICTION LOGIC (Unchanged)
# ===================================================================================
def get_probs_hybrid(model, X_seq_tensor):
    with torch.no_grad():
        logits, _ = model(X_seq_tensor.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs

def get_probs_hgae(lgbm, temporal, fusion, gating, X_tab, X_seq):
    lgb_probs = lgbm.predict_proba(X_tab)
    with torch.no_grad():
        X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
        logits, feats = temporal(X_seq_tensor)
        cnn_probs = F.softmax(logits, dim=1).cpu().numpy()
        cnn_feats = feats.cpu().numpy()
    with torch.no_grad():
        lgbp_t = torch.tensor(lgb_probs, dtype=torch.float32).to(device)
        cnp_t = torch.tensor(cnn_probs, dtype=torch.float32).to(device)
        cnf_t = torch.tensor(cnn_feats, dtype=torch.float32).to(device)
        fused = fusion(lgbp_t, cnf_t)
        logits_final = gating(lgbp_t, cnp_t, fused)
        final_probs = F.softmax(logits_final, dim=1).cpu().numpy()
    return final_probs

def predict_pipeline(input_df: pd.DataFrame):
    input_df_reordered = input_df[feature_names]
    X_scaled = scaler.transform(input_df_reordered)
    if len(X_scaled) < TIME_STEPS:
        raise ValueError(f"Input data must have at least {TIME_STEPS} rows.")
    X_seq = X_scaled[-TIME_STEPS:].reshape(1, TIME_STEPS, -1)
    X_tab = X_scaled[-1].reshape(1, -1)
    X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)
    hybrid_probs = get_probs_hybrid(hybrid_model, X_seq_tensor)
    hgae_probs = get_probs_hgae(hgae_lgbm, hgae_temporal, hgae_fusion, hgae_gating, X_tab, X_seq)
    weights = np.ones_like(hybrid_probs) * 0.5
    if hybrid_probs.shape[1] > 2:
        weights[:, 1] = 0.7
        weights[:, 2] = 0.7
    ensemble_probs = (1 - weights) * hybrid_probs + weights * hgae_probs
    ensemble_probs[:, [1, 2]] *= 1.2
    capped = np.clip(ensemble_probs, 1e-8, 0.97)
    final_probs = capped / np.sum(capped, axis=1, keepdims=True)
    prediction_index = np.argmax(final_probs[0])
    confidence = final_probs[0][prediction_index]
    predicted_class = class_names[prediction_index]
    return {"predicted_class": predicted_class, "confidence": float(confidence)}

# ===================================================================================
# 4. API ENDPOINTS (NEW, CORRECTED VERSION FOR Pydantic v2)
# ===================================================================================
app = FastAPI(title="DDoS Detection API")

# Define the input data model for a list of dictionaries using the new Pydantic v2 style
class InputData(RootModel[List[Dict[str, float]]]):
    root: List[Dict[str, float]]

@app.get("/", summary="Health check endpoint")
def read_root():
    return {"status": "DDoS Detection API is running"}

@app.post("/predict", summary="Predict DDoS attack class from input data")
def predict(data: InputData):
    """
    Receives a JSON array of objects, where each object is a row of data.
    - **data**: Must contain at least `time_steps` (10) rows.
    
    Returns the predicted class and confidence score.
    """
    if not all([scaler, class_names, feature_names]):
        raise HTTPException(status_code=500, detail="Models or artifacts not loaded. API is not operational.")
    
    try:
        # Pydantic v2 automatically unpacks the root model
        input_df = pd.DataFrame(data.root)

        # Check for missing columns
        missing_cols = set(feature_names) - set(input_df.columns)
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing input columns: {list(missing_cols)}")
            
        result = predict_pipeline(input_df)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Catch-all for other errors
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")# import os
# import json
# import joblib
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Dict

# # ===================================================================================
# # 1. DEFINE MODEL ARCHITECTURES
# # This MUST match the architectures from your training script exactly.
# # ===================================================================================

# device = torch.device('cpu') # Set to CPU for deployment

# # --- Hybrid Model (CNN + Transformer) ---
# class CNNBranch(nn.Module):
#     def __init__(self, in_features, num_classes):
#         super().__init__();self.conv1=nn.Conv1d(in_features,64,3,padding=1);self.conv2=nn.Conv1d(64,128,3,padding=1);self.pool=nn.AdaptiveAvgPool1d(1);self.fc=nn.Linear(128,num_classes)
#     def forward(self,x):
#         x=x.permute(0,2,1);x=F.relu(self.conv1(x));x=F.relu(self.conv2(x));feat=self.pool(x).squeeze(-1);return self.fc(feat),feat
# class TransformerBranch(nn.Module):
#     def __init__(self,in_features,num_classes,d_model=64,nhead=4):
#         super().__init__();self.project=nn.Linear(in_features,d_model);encoder_layer=nn.TransformerEncoderLayer(d_model,nhead,batch_first=True,dropout=0.1);self.encoder=nn.TransformerEncoder(encoder_layer,num_layers=2);self.fc=nn.Linear(d_model,num_classes)
#     def forward(self,x):
#         x=self.project(x);x=self.encoder(x);feat=x.mean(dim=1);return self.fc(feat),feat
# class GatingNet(nn.Module):
#     def __init__(self,input_dim,num_classes):
#         super().__init__();self.fc=nn.Sequential(nn.Linear(input_dim,64),nn.ReLU(),nn.Linear(64,num_classes),nn.Sigmoid())
#     def forward(self,cnn_feat,trans_feat):return self.fc(torch.cat([cnn_feat,trans_feat],dim=1))
# class HybridGatedNet(nn.Module):
#     def __init__(self,in_features,num_classes,d_model=64,nhead=4):
#         super().__init__();self.cnn=CNNBranch(in_features,num_classes);self.trans=TransformerBranch(in_features,num_classes,d_model,nhead);self.gate=GatingNet(128+d_model,num_classes)
#     def forward(self,x_seq):
#         log_cnn,f_cnn=self.cnn(x_seq);log_trans,f_trans=self.trans(x_seq);gate=self.gate(f_cnn.detach(),f_trans.detach());return gate*log_cnn+(1-gate)*log_trans,gate

# # --- HGAE Model (LGBM + Temporal CNN) ---
# class TemporalAttention(nn.Module):
#     def __init__(self,input_dim):super().__init__();self.attn=nn.Linear(input_dim,1)
#     def forward(self,x):return(x*torch.softmax(self.attn(x),dim=1)).sum(dim=1)
# class TemporalCNNExpert(nn.Module):
#     def __init__(self,input_dim,num_classes,dropout_prob=0.5):
#         super().__init__();self.cnn=nn.Sequential(nn.Conv1d(input_dim,64,3,padding=1),nn.ReLU(),nn.Dropout(dropout_prob),nn.Conv1d(64,128,3,padding=1),nn.ReLU(),nn.Dropout(dropout_prob));self.attn=TemporalAttention(128);self.fc=nn.Linear(128,num_classes)
#     def forward(self,x):
#         x=self.cnn(x.permute(0,2,1)).permute(0,2,1);pooled=self.attn(x);return self.fc(pooled),pooled
# class CrossModalAttentionFusion(nn.Module):
#     def __init__(self,tab_dim,cnn_dim,fusion_dim=64,heads=2):
#         super().__init__();self.tab_proj=nn.Linear(tab_dim,fusion_dim);self.cnn_proj=nn.Linear(cnn_dim,fusion_dim);self.attn=nn.MultiheadAttention(fusion_dim,heads,batch_first=True,dropout=0.1);self.fc=nn.Sequential(nn.Linear(fusion_dim*2,fusion_dim),nn.ReLU(),nn.Dropout(0.1))
#     def forward(self,tab,cnn):
#         tab,cnn=self.tab_proj(tab),self.cnn_proj(cnn);x=torch.stack([tab,cnn],dim=1);attn_out,_=self.attn(x,x,x);return self.fc(torch.cat([attn_out[:,0],attn_out[:,1]],dim=1))
# class HierarchicalGatingClassifier(nn.Module):
#     def __init__(self,lgb_dim,cnn_dim,fusion_dim,num_classes,dropout_prob=0.3):
#         super().__init__();self.fc=nn.Sequential(nn.Linear(lgb_dim+cnn_dim+fusion_dim,128),nn.ReLU(),nn.Dropout(dropout_prob),nn.Linear(128,64),nn.ReLU(),nn.Dropout(dropout_prob),nn.Linear(64,num_classes))
#     def forward(self,lgb,cnn,fused):return self.fc(torch.cat([lgb,cnn,fused],dim=1))


# # ===================================================================================
# # 2. LOAD ARTIFACTS
# # This code runs once when the API starts up.
# # ===================================================================================
# ARTIFACTS_PATH = "artifacts"
# try:
#     print("Loading deployment artifacts...")
#     scaler = joblib.load(os.path.join(ARTIFACTS_PATH, "scaler.pkl"))
#     class_names = joblib.load(os.path.join(ARTIFACTS_PATH, "class_names.pkl"))
#     feature_names = joblib.load(os.path.join(ARTIFACTS_PATH, "feature_names.pkl"))
#     with open(os.path.join(ARTIFACTS_PATH, "deployment_config.json"), 'r') as f:
#         config = json.load(f)
#     TIME_STEPS = config['time_steps']
#     NUM_CLASSES = len(class_names)
    
#     # Load Hybrid Model
#     hybrid_model = HybridGatedNet(len(feature_names), NUM_CLASSES).to(device)
#     hybrid_model.load_state_dict(torch.load(os.path.join(ARTIFACTS_PATH, "hybrid_model.pth"), map_location=device))
#     hybrid_model.eval()

#     # Load HGAE Models
#     hgae_lgbm = joblib.load(os.path.join(ARTIFACTS_PATH, "hgae_lgbm.txt"))
#     hgae_temporal = TemporalCNNExpert(len(feature_names), NUM_CLASSES).to(device)
#     hgae_temporal.load_state_dict(torch.load(os.path.join(ARTIFACTS_PATH, "hgae_temporal.pth"), map_location=device))
#     hgae_temporal.eval()
    
#     hgae_fusion = CrossModalAttentionFusion(NUM_CLASSES, 128).to(device)
#     hgae_fusion.load_state_dict(torch.load(os.path.join(ARTIFACTS_PATH, "hgae_fusion.pth"), map_location=device))
#     hgae_fusion.eval()
    
#     hgae_gating = HierarchicalGatingClassifier(NUM_CLASSES, NUM_CLASSES, 64, NUM_CLASSES).to(device)
#     hgae_gating.load_state_dict(torch.load(os.path.join(ARTIFACTS_PATH, "hgae_gating.pth"), map_location=device))
#     hgae_gating.eval()
    
#     print(" Artifacts loaded successfully.")

# except Exception as e:
#     print(f" Error loading artifacts: {e}")
#     # In a real app, you might want to exit or handle this more gracefully
#     scaler, class_names, feature_names, hybrid_model, hgae_lgbm, hgae_temporal, hgae_fusion, hgae_gating = [None]*8


# # ===================================================================================
# # 3. PREDICTION LOGIC
# # Replicating the prediction pipeline from the notebook.
# # ===================================================================================
# def get_probs_hybrid(model, X_seq_tensor):
#     with torch.no_grad():
#         logits, _ = model(X_seq_tensor.to(device))
#         probs = F.softmax(logits, dim=1).cpu().numpy()
#     return probs

# def get_probs_hgae(lgbm, temporal, fusion, gating, X_tab, X_seq):
#     lgb_probs = lgbm.predict_proba(X_tab)
    
#     with torch.no_grad():
#         X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
#         logits, feats = temporal(X_seq_tensor)
#         cnn_probs = F.softmax(logits, dim=1).cpu().numpy()
#         cnn_feats = feats.cpu().numpy()
    
#     with torch.no_grad():
#         lgbp_t = torch.tensor(lgb_probs, dtype=torch.float32).to(device)
#         cnp_t = torch.tensor(cnn_probs, dtype=torch.float32).to(device)
#         cnf_t = torch.tensor(cnn_feats, dtype=torch.float32).to(device)
        
#         fused = fusion(lgbp_t, cnf_t)
#         logits_final = gating(lgbp_t, cnp_t, fused)
#         final_probs = F.softmax(logits_final, dim=1).cpu().numpy()
#     return final_probs

# def predict_pipeline(input_df: pd.DataFrame):
#     """
#     Takes a raw dataframe and returns the final prediction and confidence.
#     """
#     # 1. Preprocessing
#     input_df_reordered = input_df[feature_names] # Ensure column order
#     X_scaled = scaler.transform(input_df_reordered)
    
#     if len(X_scaled) < TIME_STEPS:
#         raise ValueError(f"Input data must have at least {TIME_STEPS} rows.")
    
#     # Use only the last `TIME_STEPS` for prediction
#     X_seq = X_scaled[-TIME_STEPS:].reshape(1, TIME_STEPS, -1)
#     X_tab = X_scaled[-1].reshape(1, -1)
    
#     X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)

#     # 2. Get probabilities from both models
#     hybrid_probs = get_probs_hybrid(hybrid_model, X_seq_tensor)
#     hgae_probs = get_probs_hgae(hgae_lgbm, hgae_temporal, hgae_fusion, hgae_gating, X_tab, X_seq)

#     # 3. Apply the exact same ensembling logic from the notebook
#     weights = np.ones_like(hybrid_probs) * 0.5
#     if hybrid_probs.shape[1] > 2:
#         weights[:, 1] = 0.7
#         weights[:, 2] = 0.7
    
#     ensemble_probs = (1 - weights) * hybrid_probs + weights * hgae_probs
#     ensemble_probs[:, [1, 2]] *= 1.2
    
#     capped = np.clip(ensemble_probs, 1e-8, 0.97)
#     final_probs = capped / np.sum(capped, axis=1, keepdims=True)
    
#     # 4. Get final prediction and confidence
#     prediction_index = np.argmax(final_probs[0])
#     confidence = final_probs[0][prediction_index]
#     predicted_class = class_names[prediction_index]
    
#     return {
#         "predicted_class": predicted_class,
#         "confidence": float(confidence)
#     }

# # ===================================================================================
# # 4. API ENDPOINTS
# # ===================================================================================
# app = FastAPI(title="DDoS Detection API")

# # Define the input data model for a single row
# class InputRow(BaseModel):
#     # This model is flexible; it accepts any feature name as a key.
#     # In a stricter app, you'd list all feature names here.
#     __root__: Dict[str, float]

# @app.get("/", summary="Health check endpoint")
# def read_root():
#     return {"status": "DDoS Detection API is running"}

# @app.post("/predict", summary="Predict DDoS attack class from input data")
# def predict(data: List[Dict]):
#     """
#     Receives a list of data rows (as dictionaries) representing a time series.
#     - **data**: A JSON array of objects, where each object is a row of data.
#       Must contain at least `time_steps` (10) rows.
    
#     Returns the predicted class and confidence score.
#     """
#     if not all([scaler, class_names, feature_names]):
#         raise HTTPException(status_code=500, detail="Models or artifacts not loaded. API is not operational.")
    
#     try:
#         input_df = pd.DataFrame(data)
#         # Check for missing columns
#         missing_cols = set(feature_names) - set(input_df.columns)
#         if missing_cols:
#             raise HTTPException(status_code=400, detail=f"Missing input columns: {list(missing_cols)}")
            
#         result = predict_pipeline(input_df)
#         return result
#     except ValueError as ve:
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         # Catch-all for other errors
#         raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
