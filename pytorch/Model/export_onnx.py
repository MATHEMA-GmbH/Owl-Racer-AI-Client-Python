import torch.onnx




def export_to_onnx(torch_model, sample, path):
    # Input to the model
    sample = torch.randn(1, 1, 6)



    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      sample[0],                    # model input (or a tuple for multiple inputs)
                      path,        # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ["input"],   # the model's input names
                      output_names = ['output'] # the model's output names
                      )