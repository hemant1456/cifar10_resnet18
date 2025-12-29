import torch
checkpoint = torch.load('./checkpoints/model_final.pt',map_location=torch.device('cpu'))

from models.cifar10_resnet_18 import resnet18

model = resnet18()

model.load_state_dict(checkpoint['model_params'])


from onnxruntime.quantization import quantize_dynamic, QuantType
onnx_program = torch.onnx.export(
    model = model,
    args = torch.randn(1,3,32,32),
    f = 'final_model_aws.onnx',
    input_names = ['input'],
    output_names= ['output'],
    dynamic_axes={
        'input': {0: 'batch_size'}, 
        'output': {0: 'batch_size'}
    },
    opset_version=14,
)

