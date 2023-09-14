


onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['output'],
    input_shape=None,
    optimize=True)

codebase_config = dict(type='mmseg', task='Segmentation', with_argmax=True)

backend_config = dict(type='onnxruntime')