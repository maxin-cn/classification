import os
import onnx
from onnx import helper, checker
from onnx import TensorProto
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
import argparse

def replace_dynamic_pad(model_name='twins_svt_small', onnx_path='./', input_size=224):

    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph

    id_list_dict = {"twins_svt_small":[113, 460, 807, 1113, 1395, 1677, 1959, 2282, 2564],
                    "twins_svt_base":[113, 460, 807, 1113, 1395, 1677, 1959, 2241, 2523, 2805, 3087, 3410],
                    "twins_svt_large":[113, 460, 807, 1113, 1395, 1677, 1959, 2241, 2523, 2805, 3087, 3410],
                    "twins_svt_exlarge":[113, 460, 807, 1113, 1395, 1677, 1959, 2241, 2523, 2805, 3087, 3410]}

    assert model_name in id_list_dict.keys(), "wrong model name provided" 

    id_list = id_list_dict[model_name]

    if input_size == 224:
        pads_list = [[0, 0, 0, 0]]*len(id_list)
    else:
        assert False, "input sizes other than 224 to be supported later"

    for i in range(len(id_list)):
        # get old pad node which has three inputs: x, pads, value 
        old_pad = graph.node[id_list[i]]

        print('input:', old_pad.input[1])

        # get shape
        shape_node = onnx.helper.make_node(
                'Shape',
                name='get_shape_'+str(id_list[i]),
                inputs=[old_pad.input[1]],
                outputs=[str(id_list[i]) + '07'],
        )

        # transpose NHWC to NCHW to support padding (TRT only supports last two dimensions)
        transpose_node = onnx.helper.make_node(
            'Transpose',
            name='transpose_' + str(id_list[i]),
            inputs=[old_pad.input[0]],
            outputs=[str(id_list[i]) + '001'],
            perm=[0,3,1,2],
        )

        # new pads tensor is fixed
        pads = onnx.helper.make_tensor(str(id_list[i]) + '05', onnx.TensorProto.INT64, [4], pads_list[i])
        graph.initializer.append(pads)

        new_pad = onnx.helper.make_node(
            'Pad',
            name=old_pad.name,
            inputs=[str(id_list[i]) + '001', str(id_list[i]) + '05', old_pad.input[2]],
            outputs=[str(id_list[i]) + '002'],
            mode='constant',
        )

        # transpose NCHW back to NHWC
        transpose_back_node = onnx.helper.make_node(
            'Transpose',
            name='transpose_back_' + str(id_list[i]),
            inputs=[str(id_list[i]) + '002'],
            outputs=[old_pad.output[0]],
            perm=[0,2,3,1],
        )

        graph.node.remove(old_pad)  # delete old node
        graph.node.insert(5506 + id_list[i], shape_node)  # insert shape
        graph.node.insert(55001 + id_list[i], transpose_node)
        graph.node.insert(55002 + id_list[i], transpose_back_node)
        graph.node.insert(id_list[i], new_pad)  # insert new pad

        print("{}-th Node:".format(i))
        print("modified:\n", shape_node)
        print("modified:\n", new_pad)
        print("modified:\n", transpose_node)

    onnx.save(onnx_model, onnx_path)
    print("Saved {}!".format(onnx_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fix Twins ONNX')
    parser.add_argument('--model', help='model name', default="twins_svt_small", 
                        choices=["twins_svt_small", "twins_svt_large", "twins_svt_base", "twins_svt_exlarge"])
    parser.add_argument('--onnx', help='onnx path')
    parser.add_argument('--input', default=224, help='input shape', type=int)
    args = parser.parse_args()
    
    replace_dynamic_pad(args.model, args.onnx, args.input)
