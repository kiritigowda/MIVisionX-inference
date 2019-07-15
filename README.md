# MIVisionX Python Inference Application

[MIVisionX](https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/) Inference Application using pre-trained `ONNX`/`NNEF`/`Caffe` models.

Pre-trained models in [ONNX](https://onnx.ai/), [NNEF](https://www.khronos.org/nnef), & [Caffe](http://caffe.berkeleyvision.org/) formats are supported by MIVisionX. The app first converts the pre-trained models to AMD Neural Net Intermediate Representation (NNIR), once the model has been translated into AMD NNIR (AMD's internal open format), the Optimizer goes through the NNIR and applies various optimizations which would allow the model to be deployed on to target hardware most efficiently. Finally, AMD NNIR is converted into OpenVX C code, which is compiled and wrapped with a python API to run on any targeted hardware.

### Prerequisites

* Ubuntu `16.04`/`18.04` or CentOS `7.5`/`7.6`
* [ROCm supported hardware](https://rocm.github.io/ROCmInstall.html#hardware-support) 
	* AMD Radeon GPU or AMD APU required
* Latest [ROCm](https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories)
* Build & Install [MIVisionX](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX#linux-1)

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
