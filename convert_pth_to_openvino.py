# convert_pth_to_openvino.py
import torch
import torch.nn as nn
import openvino as ov
import numpy as np
import os
import json

# Trained action models live in models/actions/ alongside everything else a
# training run produces. A bare name here used to read and write in whatever
# directory the script was started from; the fallback keeps a checkpoint left
# in the repo root by an older run convertible.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_ACTIONS_DIR = os.path.join(_REPO_ROOT, "models", "actions")


def _action_model(name, for_write=False):
    managed = os.path.join(_ACTIONS_DIR, name)
    if for_write or os.path.exists(managed):
        return managed
    legacy = os.path.join(_REPO_ROOT, name)
    return legacy if os.path.exists(legacy) else managed


CHECKPOINT_PATH = _action_model("intel_finetuned_classifier_3d.pth")
MAPPING_PATH = _action_model("intel_finetuned_classifier_3d_mapping.json")
DECODER_XML_PATH = _action_model("action_classifier_3d.xml", for_write=True)
DECODER_ONNX_PATH = _action_model("action_classifier_3d.onnx", for_write=True)

class EncoderLSTM(nn.Module):
    """Enhanced classifier matching your checkpoint structure"""
    def __init__(self, feature_dim=512, hidden_dim=256, num_classes=31, 
                 num_layers=2, dropout=0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 2-layer bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(hidden_dim * 2)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Second normalization after attention
        self.ln2 = nn.LayerNorm(hidden_dim * 2)
        
        # Dropout before classification
        self.dropout = nn.Dropout(dropout)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, feature_dim)
        Returns: (batch_size, num_classes)
        """
        # LSTM with 2 layers
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Layer normalization
        lstm_out = self.ln1(lstm_out)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.nn.functional.softmax(attention_weights, dim=1)
        
        # Context vector: weighted sum of LSTM outputs
        context = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Second normalization
        context = self.ln2(context)
        
        # Dropout
        context = self.dropout(context)
        
        # Classification
        logits = self.classifier(context)
        
        return logits  # Only return logits for inference

def inspect_checkpoint():
    """Inspect the checkpoint to understand its structure"""
    print("🔍 正在检查检查点…")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    
    print("\n检查点键：")
    for key in checkpoint.keys():
        print(f"  - {key}: {checkpoint[key].shape}")
    
    return checkpoint

def convert_current_model():
    # First, inspect the checkpoint
    checkpoint = inspect_checkpoint()
    
    # Load the mapping file to get model parameters
    mapping_path = MAPPING_PATH
    
    if not os.path.exists(mapping_path):
        print(f"\n❌ 错误：未找到映射文件 {mapping_path}！")
        return
    
    with open(mapping_path, 'r') as f:
        mapping_data = json.load(f)
    
    # Get parameters from mapping file
    feature_dim = mapping_data['feature_dim']
    sequence_length = mapping_data['sequence_length'] 
    num_classes = mapping_data['num_classes']
    hidden_dim = mapping_data.get('hidden_dim', 256)
    num_layers = mapping_data.get('num_layers', 2)
    
    print("\n📋 映射文件中的模型参数：")
    print(f"  - 特征维度：{feature_dim}")
    print(f"  - 隐藏层维度：{hidden_dim}")
    print(f"  - 序列长度：{sequence_length}")
    print(f"  - 层数：{num_layers}")
    print(f"  - 类别数：{num_classes}")
    
    # Verify dimensions from checkpoint
    lstm_weight_shape = checkpoint['lstm.weight_ih_l0'].shape
    feature_dim_from_checkpoint = lstm_weight_shape[1]
    hidden_dim_from_checkpoint = lstm_weight_shape[0] // 4  # Divided by 4 (LSTM gates)
    
    classifier_weight_shape = checkpoint['classifier.3.weight'].shape
    output_dim = classifier_weight_shape[0]
    
    print("\n📊 检查点结构：")
    print(f"  - 特征维度：{feature_dim_from_checkpoint}")
    print(f"  - 隐藏层维度：{hidden_dim_from_checkpoint}")
    print(f"  - 输出类别数：{output_dim}")
    
    # Verify dimensions match
    if feature_dim_from_checkpoint != feature_dim:
        print(f"\n⚠️  警告：检查点 feature_dim（{feature_dim_from_checkpoint}）与映射文件（{feature_dim}）不一致")
        print(f"   将使用检查点 feature_dim：{feature_dim_from_checkpoint}")
        feature_dim = feature_dim_from_checkpoint
    
    if hidden_dim_from_checkpoint != hidden_dim:
        print(f"\n⚠️  警告：检查点 hidden_dim（{hidden_dim_from_checkpoint}）与映射文件（{hidden_dim}）不一致")
        print(f"   将使用检查点 hidden_dim：{hidden_dim_from_checkpoint}")
        hidden_dim = hidden_dim_from_checkpoint
    
    if output_dim != num_classes:
        print(f"\n⚠️  警告：检查点输出维度（{output_dim}）与 num_classes（{num_classes}）不一致")
        print(f"   将使用检查点输出维度：{output_dim}")
        num_classes = output_dim
    
    # Create the correct model architecture
    print("\n🔨 正在创建 EncoderLSTM 模型…")
    model = EncoderLSTM(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=0.3
    )
    
    # Load the checkpoint
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("✓ 模型加载成功！")
    print("\n模型结构：")
    print(model)
    
    # Create dummy input matching the checkpoint's expected input
    # Shape: [batch_size, sequence_length, feature_dim]
    dummy_input = torch.randn(1, sequence_length, feature_dim)
    
    print("\n🔄 正在转换为 ONNX…")
    print(f"  - 输入形状：[batch_size, {sequence_length}, {feature_dim}]")
    
    # The ONNX is kept, not deleted. It used to be a scratch file on the way to
    # OpenVINO IR, which is Intel-only; ONNX Runtime's DirectML provider runs
    # the same graph on any DX12 card and is the one accelerated runtime a
    # packaged build can carry (see modules/system/ort_directml.py). Writing both
    # costs a few MB and gives the AMD path a model to load.
    onnx_path = DECODER_ONNX_PATH
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # Convert to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},  
            'output': {0: 'batch_size'}
        },
        opset_version=13,
        verbose=False
    )
    
    # Convert ONNX to OpenVINO
    print("🔄 正在转换为 OpenVINO 格式…")
    ov_model = ov.convert_model(onnx_path)
    
    # Save the model
    output_path = DECODER_XML_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ov.save_model(ov_model, output_path)
    
    print("\n✅ 转换成功！")
    print("✓ 架构：双层带注意力机制的 BiLSTM")
    print(f"✓ 输入形状：[batch_size, {sequence_length}, {feature_dim}]")
    print(f"✓ 隐藏层维度：{hidden_dim}")
    print(f"✓ 类别数：{num_classes}")
    print(f"✓ 模型已保存为：{output_path}")
    print(f"✓ 已保留供 DirectML/ONNX Runtime 使用的 ONNX：{onnx_path}")
    
    # Update mapping file with correct dimensions
    mapping_data['model_feature_dim'] = feature_dim
    mapping_data['model_hidden_dim'] = hidden_dim
    mapping_data['model_sequence_length'] = sequence_length
    mapping_data['model_num_layers'] = num_layers
    with open(mapping_path, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print("✓ 已将模型维度写入映射文件")
    
    # Print class labels for reference
    print(f"\n📝 类别标签（{num_classes} 个类别）：")
    idx_to_label = mapping_data['idx_to_label']
    for idx in sorted([int(k) for k in idx_to_label.keys()]):
        label = idx_to_label[str(idx)]
        print(f"  {idx}: {label}")
    
    return feature_dim, sequence_length, num_classes

def test_converted_model(feature_dim, sequence_length, num_classes):
    """Test the converted OpenVINO model"""
    print("\n🧪 正在测试转换后的模型…")
    
    # Load the mapping file
    mapping_path = MAPPING_PATH
    with open(mapping_path, 'r') as f:
        mapping_data = json.load(f)
    
    # Load OpenVINO model
    core = ov.Core()
    compiled_model = core.compile_model(DECODER_XML_PATH, "CPU")
    
    # Create test input
    test_input = np.random.randn(1, sequence_length, feature_dim).astype(np.float32)
    
    # Run inference
    result = compiled_model([test_input])
    output = result[0]
    
    print("✓ 模型测试成功！")
    print(f"✓ 输入形状：{test_input.shape}")
    print(f"✓ 输出形状：{output.shape}")
    print(f"✓ 输出范围：[{output.min():.4f}, {output.max():.4f}]")
    
    # Apply softmax to get probabilities
    exp_output = np.exp(output - np.max(output))
    probs = exp_output / exp_output.sum()
    
    # Show predicted class
    predicted_class = np.argmax(output, axis=1)[0]
    class_name = mapping_data['idx_to_label'][str(predicted_class)]
    confidence = probs[0][predicted_class]
    
    print(f"✓ 测试预测：{class_name}（类别 {predicted_class}）")
    print(f"✓ 置信度：{confidence:.4f}")
    
    # Show top 3 predictions
    top3_indices = np.argsort(output[0])[-3:][::-1]
    print("\n🏆 预测结果 TOP 3：")
    for i, idx in enumerate(top3_indices, 1):
        label = mapping_data['idx_to_label'][str(idx)]
        score = probs[0][idx]
        print(f"  {i}. {label}: {score:.4f}")

if __name__ == "__main__":
    try:
        feature_dim, sequence_length, num_classes = convert_current_model()
        test_converted_model(feature_dim, sequence_length, num_classes)
        print("\n✅ 全部完成！模型已可使用。")
    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()