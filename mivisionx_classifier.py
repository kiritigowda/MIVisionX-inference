__author__      = "Kiriti Nagesh Gowda"
__copyright__   = "Copyright 2019, AMD MIVisionX"
__license__     = "MIT"
__version__     = "0.9.0"
__maintainer__  = "Kiriti Nagesh Gowda"
__email__       = "Kiriti.NageshGowda@amd.com"
__status__      = "ALPHA"
__script_name__ = "MIVisionX Classifier Application"

import argparse
import os
import ctypes
import cv2
import time
import numpy
import numpy as np
from numpy.ctypeslib import ndpointer

class AnnAPI:
	def __init__(self,library):
		self.lib = ctypes.cdll.LoadLibrary(library)
		self.annQueryInference = self.lib.annQueryInference
		self.annQueryInference.restype = ctypes.c_char_p
		self.annQueryInference.argtypes = []
		self.annCreateInference = self.lib.annCreateInference
		self.annCreateInference.restype = ctypes.c_void_p
		self.annCreateInference.argtypes = [ctypes.c_char_p]
		self.annReleaseInference = self.lib.annReleaseInference
		self.annReleaseInference.restype = ctypes.c_int
		self.annReleaseInference.argtypes = [ctypes.c_void_p]
		self.annCopyToInferenceInput = self.lib.annCopyToInferenceInput
		self.annCopyToInferenceInput.restype = ctypes.c_int
		self.annCopyToInferenceInput.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t, ctypes.c_bool]
		self.annCopyFromInferenceOutput = self.lib.annCopyFromInferenceOutput
		self.annCopyFromInferenceOutput.restype = ctypes.c_int
		self.annCopyFromInferenceOutput.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t]
		self.annRunInference = self.lib.annRunInference
		self.annRunInference.restype = ctypes.c_int
		self.annRunInference.argtypes = [ctypes.c_void_p, ctypes.c_int]
		print('OK: AnnAPI found "' + self.annQueryInference().decode("utf-8") + '" as configuration in ' + library)

class annieObjectWrapper():
	def __init__(self, annpythonlib, weightsfile):
		select = 1
		self.api = AnnAPI(annpythonlib)
		input_info,output_info,empty = self.api.annQueryInference().decode("utf-8").split(';')
		input,name,n_i,c_i,h_i,w_i = input_info.split(',')
		outputCount = output_info.split(",")
		stringcount = len(outputCount)
		if stringcount == 6:
			output,opName,n_o,c_o,h_o,w_o = output_info.split(',')
		else:
			output,opName,n_o,c_o= output_info.split(',')
			h_o = '1'; w_o  = '1';
		self.hdl = self.api.annCreateInference(weightsfile.encode('utf-8'))
		self.dim = (int(w_i),int(h_i))
		self.outputDim = (int(n_o),int(c_o),int(h_o),int(w_o))

	def __del__(self):
		self.api.annReleaseInference(self.hdl)

	def runInference(self, img, out):
		# create input.f32 file
		img_r = img[:,:,0]
		img_g = img[:,:,1]
		img_b = img[:,:,2]
		img_t = np.concatenate((img_r, img_g, img_b), 0)	
		# copy input.f32 to inference input
		status = self.api.annCopyToInferenceInput(self.hdl, np.ascontiguousarray(img_t, dtype=np.float32), (img.shape[0]*img.shape[1]*3*4), 0)
		# run inference
		status = self.api.annRunInference(self.hdl, 1)
		# copy output.f32
		status = self.api.annCopyFromInferenceOutput(self.hdl, np.ascontiguousarray(out, dtype=np.float32), out.nbytes)
		return out

	def classify(self, img):
		# create output.f32 buffer
		out_buf = bytearray(self.outputDim[0]*self.outputDim[1]*self.outputDim[2]*self.outputDim[3]*4)
		out = np.frombuffer(out_buf, dtype=numpy.float32)
		# run inference & receive output.f32
		output = self.runInference(img, out)
		return output

# global variables
labelNames = None
colors =[
        (0,153,0),        # Top1
        (153,153,0),      # Top2
        (153,76,0),       # Top3
        (0,128,255),      # Top4
        (255,102,102),    # Top5
        ];

def processClassificationOutput(inputImage, modelName, modelOutput, verbosePrint):
	# post process output file
	start = time.time()
	softmaxOutput = np.float32(modelOutput)
	topIndex = []
	topLabels = []
	topProb = []
	for x in softmaxOutput.argsort()[-5:]:
		topIndex.append(x)
		topLabels.append(labelNames[x])
		topProb.append(softmaxOutput[x])
	end = time.time()
	if(verbosePrint == 'yes'):
		print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms'

	# display output
	start = time.time()
	# initialize the result image
	resultImage = np.zeros(((9 * 25), 525, 3), dtype="uint8")
	resultImage.fill(255)
	cv2.putText(resultImage, 'MIVisionX Object Classification', (25,  25),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
	topK = 1   
	for i in reversed(range(5)):
		txt =  topLabels[i].decode('utf-8')[:-1]
		conf = topProb[i]
		txt = 'Top'+str(topK)+':'+txt+' '+str(int(round((conf*100), 0)))+'%' 
		size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
		t_width = size[0][0]
		t_height = size[0][1]
		textColor = (colors[topK - 1])
		cv2.putText(resultImage,txt,(15,t_height+(topK*30+40)),cv2.FONT_HERSHEY_SIMPLEX,0.7,textColor,2)
		topK = topK + 1
	end = time.time()
	if(verbosePrint == 'yes'):
		print '%30s' % 'Processed results image in ', str((end - start)*1000), 'ms'

	return resultImage

# MIVisionX Classifier
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_format',    	type=str, required=True,  help='pre-trained model format, options:caffe/onnx/nnef [required]')
	parser.add_argument('--model_name',  		type=str, required=True,  help='model name [required]')
	parser.add_argument('--model',    		 	type=str, required=True,  help='pre_trained model file [required]')
	parser.add_argument('--model_input_dims',   type=str, required=True,  help='c,h,w - channel,height,width [required]')
	parser.add_argument('--model_output_dims',  type=str, required=True,  help='c,h,w - channel,height,width [required]')
	parser.add_argument('--label',  			type=str, required=True,  help='labels text file [required]')
	parser.add_argument('--add', 				type=str, default='',     help='input preprocessing factor [optional - default:0]')
	parser.add_argument('--multiply', 			type=str, default='',     help='input preprocessing factor [optional - default:1]')
	parser.add_argument('--video', 				type=str, default='',  	  help='video file for classification [optional]')
	parser.add_argument('--capture', 			type=str, default='',  	  help='capture device id [optional]')
	parser.add_argument('--replace', 			type=str, default='no',   help='replace/overwrite model [optional - default:no]')
	parser.add_argument('--verbose', 			type=str, default='no',   help='verbose [optional - default:no]')
	args = parser.parse_args()

	# get arguments
	modelFormat = args.model_format
	modelName = args.model_name
	modelLocation = args.model
	modelInputDims = args.model_input_dims
	modelOutputDims = args.model_output_dims
	label = args.label
	inputAdd = args.add
	inputMultiply = args.multiply
	videoFile = args.video
	captureID = args.capture
	replaceModel = args.replace
	verbosePrint = args.verbose

	# set paths
	modelCompilerPath = '/opt/rocm/mivisionx/model_compiler/python'
	setupDir = '~/.mivisionx-classifier'
	analyzerDir = os.path.expanduser(setupDir)
	modelDir = analyzerDir+'/'+modelName+'_dir'
	nnirDir = modelDir+'/nnir-files'
	openvxDir = modelDir+'/openvx-files'
	modelBuildDir = modelDir+'/build'
	trainedModel = os.path.expanduser(modelLocation)
	labelText = os.path.expanduser(label)
	pythonLib = modelBuildDir+'/libannpython.so'
	weightsFile = openvxDir+'/weights.bin'

	# video or capture
	videoFileLocation = ''
	captureIDInt = 0
	if(captureID != ''):
		captureIDInt = int(captureID)
	elif(videoFile != ''):
		videoFileLocation = os.path.expanduser(videoFile)

	# get input & output dims
	str_c_i, str_h_i, str_w_i = modelInputDims.split(',')
	c_i = int(str_c_i); h_i = int(str_h_i); w_i = int(str_w_i)
	str_c_o, str_h_o, str_w_o = modelOutputDims.split(',')
	c_o = int(str_c_o); h_o = int(str_h_o); w_o = int(str_w_o)

	# input pre-processing values
	Ax=0
	if(inputAdd != ''):
		Ax = int(inputAdd)
	Mx=1
	if(inputMultiply != ''):
		Mx = float(inputMultiply)

	# check pre-trained model
	if(not os.path.isfile(trainedModel) and modelFormat != 'nnef' ):
		print("\nPre-Trained Model not found, check argument --model\n")
		quit()

	# check for label file
	if (not os.path.isfile(labelText)):
		print("\nlabels.txt not found, check argument --labeln")
		quit()
	else:
		fp = open(labelText, 'r')
		labelNames = fp.readlines()
		#labelNames = [x.strip() for x in labelNames]
		fp.close()

	# MIVisionX setup
	if(os.path.exists(analyzerDir)):
		print("\nMIVisionX Classifier\n")
		# replace old model or throw error
		if(replaceModel == 'yes'):
			os.system('rm -rf '+modelDir)
		elif(os.path.exists(modelDir)):
			print("ERROR: Model exists, use --replace yes option to overwrite or use a different name in --model_name")
			quit()
	else:
		print("\nMIVisionX Classifier Created\n")
		os.system('(cd ; mkdir .mivisionx-classifier)')

	# Compile Model and generate python .so files
	os.system('mkdir '+modelDir)
	if(os.path.exists(modelDir)):
		# convert to NNIR
		if(modelFormat == 'caffe'):
			os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/caffe_to_nnir.py '+trainedModel+' nnir-files --input-dims 1,'+modelInputDims+' )')
		elif(modelFormat == 'onnx'):
			os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/onnx_to_nnir.py '+trainedModel+' nnir-files --input-dims 1,'+modelInputDims+' )')
		elif(modelFormat == 'nnef'):
			os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/nnef_to_nnir.py '+trainedModel+' nnir-files )')
		else:
			print("ERROR: Neural Network Format Not supported, use caffe/onnx/nnef in arugment --model_format")
			quit()
		# convert to openvx
		if(os.path.exists(nnirDir)):
			os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/nnir_to_openvx.py nnir-files openvx-files)')
		else:
			print("ERROR: Converting Pre-Trained model to NNIR Failed")
			quit()
		# build model
		if(os.path.exists(openvxDir)):
			os.system('mkdir '+modelBuildDir)
			os.system('(cd '+modelBuildDir+'; cmake ../openvx-files; make; ./anntest ../openvx-files/weights.bin )')
			print("\nSUCCESS: Converting Pre-Trained model to MIVisionX Runtime successful\n")
		else:
			print("ERROR: Converting NNIR to OpenVX Failed")
			quit()
	else:
		print("ERROR: MIVisionX Classifier Failed")
		quit()

	# opencv display window
	windowInput = "MIVisionX Classifier"
	windowResult = "MIVisionX Object Classification - Results"
	cv2.namedWindow(windowInput, cv2.WINDOW_GUI_EXPANDED)

	# create inference classifier
	classifier = annieObjectWrapper(pythonLib, weightsFile)

	# video or live capture
	if (videoFileLocation == ''):
		cap = cv2.VideoCapture(captureIDInt)
	else:
		cap = cv2.VideoCapture(videoFileLocation)

	rval = True
	frame_number = 0
	while rval:
		# get frame from live or video
		start = time.time()
		rval, frame = cap.read()
		if rval == False:
			break
		end = time.time()
		if(verbosePrint == 'yes'):
			print '%30s' % 'Grabbed camera frame in ', str((end - start)*1000), 'ms'

		# resize and process frame
		start = time.time()
		frame = cv2.flip(frame, 1) 
		resizedFrame = cv2.resize(frame, (w_i,h_i))
		RGBframe = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2RGB)
		if(inputAdd != '' or inputMultiply != ''):
			RGBframe = ( RGBframe.copy() * Mx) + Ax
		end = time.time()
		if(verbosePrint == 'yes'):
			print '%30s' % 'Input pre-processed in ', str((end - start)*1000), 'ms'

		# run inference
		start = time.time()
		output = classifier.classify(RGBframe)
		end = time.time()
		if(verbosePrint == 'yes'):
			print '%30s' % 'Executed Model in ', str((end - start)*1000), 'ms'

		# process output and display
		resultImage = processClassificationOutput(resizedFrame, modelName, output, verbosePrint)
		start = time.time()
		cv2.imshow(windowInput, frame)
		cv2.imshow(windowResult, resultImage)
		end = time.time()
		if(verbosePrint == 'yes'):
			print '%30s' % 'Processed display in ', str((end - start)*1000), 'ms\n'

		# exit on ESC
		key = cv2.waitKey(2)
		if key == 27: 
			break

	cap.release()
	cv2.destroyAllWindows()