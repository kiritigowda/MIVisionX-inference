# MIVisionX Python Inference Application

MIVisionX Inference Application using pre-trained ONNX/NNEF/Caffe models

````
usage: mivisionx_classifier.py  [-h] 
                                --model_format MODEL_FORMAT 
                                --model_name MODEL_NAME 
                                --model MODEL 
                                --model_input_dims MODEL_INPUT_DIMS 
                                --model_output_dims MODEL_OUTPUT_DIMS 
                                --label LABEL 
                                [--add ADD]
                                [--multiply MULTIPLY] 
                                [--video VIDEO]
                                [--capture CAPTURE] 
                                [--replace REPLACE]
                                [--verbose VERBOSE]
````
## Usage help
```
  -h, --help            show help message and exit
  --model_format        pre-trained model format, options:caffe/onnx/nnef     [required]
  --model_name          model name                                            [required]
  --model               pre_trained model file                                [required]
  --model_input_dims    c,h,w - channel,height,width                          [required]
  --model_output_dims   c,h,w - channel,height,width                          [required]
  --label               labels text file                                      [required]
  --add                 input preprocessing factor               [optional - default:0 ]
  --multiply            input preprocessing factor               [optional - default:1 ]
  --video               video file for classification            [optional - default:'']
  --capture             capture device id                        [optional - default:0 ]
  --replace             replace/overwrite model                  [optional - default:no]
  --verbose             verbose                                  [optional - default:no]
```
