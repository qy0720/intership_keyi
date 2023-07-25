import onnx
model = onnx.load("/home/xz/work/ultralytics/onnx_model_tools/best.onnx")

output_dir = '/home/xz/work/ultralytics/onnx_model_tools'
image_input_shape = [1, 3, 608, 800] # NCHW

# 找到head节点的名称
head_node_name = "model.22"
split_node_name = ["/model.15/cv2/act/Mul_output_0", 
                   "/model.18/cv2/act/Mul_output_0",
                   "/model.21/cv2/act/Mul_output_0"]
new_node_name = ["p3", "p4", "p5"]


# 修改输入图片尺寸
for i, v in enumerate(image_input_shape):
    model.graph.input[0].type.tensor_type.shape.dim[i].dim_value = v


# 定义backbone_graph head_graph
backbone_graph = onnx.GraphProto(name=model.graph.name)
head_graph = onnx.GraphProto(name=model.graph.name)


#model.22之后全部都是head ，初始化head_graph的node 
for node in model.graph.node:
    if head_node_name in node.name:
        head_graph.node.extend([node])
    else:
        backbone_graph.node.extend([node])

#model.22之后全部都是head ，初始化head_graph的initializer
for node in model.graph.initializer:
    if head_node_name in node.name:
        head_graph.initializer.extend([node])
    else:
        backbone_graph.initializer.extend([node])

#初始化输入 input ,input不算节点，需要重新初始化
for node in model.graph.input:
    backbone_graph.input.extend([node])

#初始化输出 ooutput , output不算节点，需要重新初始化

for node in model.graph.output:
    head_graph.output.extend([node])


# 建立节点从 "/model.15/cv2/act/Mul_output_0" 到  "p3"
for i, name in enumerate(new_node_name):
    output_node = onnx.helper.make_node(
        'Identity',
        inputs=[split_node_name[i]],
        outputs=[name]
    )
    backbone_graph.node.append(output_node)
    backbone_graph.output.extend([onnx.helper.make_tensor_value_info(
        name,
        onnx.TensorProto.FLOAT,
        None
    )])





info_model = onnx.helper.make_model(backbone_graph, opset_imports=model.opset_import, ir_version=model.ir_version)
onnx_model = onnx.shape_inference.infer_shapes(info_model)
onnx.checker.check_model(onnx_model)

onnx.save(onnx_model, output_dir + "/backbone.onnx")

# 
shapes = []
for node in onnx_model.graph.output:
    shape = []
    for i in range(4):
        shape.append(node.type.tensor_type.shape.dim[i].dim_value)
    shapes.append(shape)


for i, name in enumerate(new_node_name):
    head_graph.input.extend([onnx.helper.make_tensor_value_info(
        name,
        onnx.TensorProto.FLOAT,
        shapes[i]
    )])


for node in head_graph.node:
    if (len(node.input) == 0):
        continue
    for new_name, old_name in zip(new_node_name, split_node_name):
        if node.input[0] == old_name:
            node.input[0] = new_name

info_model = onnx.helper.make_model(head_graph, opset_imports=model.opset_import, ir_version=model.ir_version)
onnx_model = onnx.shape_inference.infer_shapes(info_model)
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, output_dir + "/head.onnx")
