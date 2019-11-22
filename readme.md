<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>
-----------------

**B**ased on  **my other 3 projects**, [tensorflow-lite-apps-on-raspberry-pi-3](https://github.com/huaxiaozhong1/tensorflow-lite-apps-on-raspberry-pi-3),  [complete-procedure-to-train-and-recognize-you-by-raspberry-pi-3-and-tensorflow-lite](https://github.com/huaxiaozhong1/complete-procedure-to-train-and-recognize-you-by-raspberry-pi-3-and-tensorflow-lite) and [YourOwnModel-TfLite-RaspberryPi](https://github.com/huaxiaozhong1/YourOwnModel-TfLite-RaspberryPi), we have learned: **1)** how to develop a **Tensorflow-lite (Tf-lite)** app to run  **on device** (such as, Raspbrerry PI, RPI)with an existing Tf-lite model; **2)** how to re-train an existing Tf model for your own data, then run it on RPI; **3)** how to create/train/convert your own Tf model/data-set, then run it **on device**.   
**Those** methods are all not relevant to sequence (or say: time-sequence) if you have got an overlook on them :). If recalling some subjects that we learnt in university, these techniques to create AI models just correspond to "random variable" in Theory of Chances, or to "Fast Fourier Transformation" in Digital Signal Processing. It is nature to move our knowledge forward to create models relevant to sequence, which corresponds to "Stochastic Process" in Theory of Chances or "Infinite/Finite Implusing Response (IIR/FIR) Filter" in Digital Signal Process :) This is what we will reach at the end of this repository: a full life-cycle of Tf ConvNet for sequence, which is one of the most popular techniques to solve AI problems ralevant to time-sequence. 

**All the steps** of the technique includes: setting up a working Open Sourrce envieonement, collecting your own data, creating/training your own model, converting the model to the one running on **device**. 
In the repository, the device sample, which will be programmed, flashed and tested, is **[a micro-contaroller, SparskFun Edge development board](https://codelabs.developers.google.com/codelabs/sparkfun-tensorflow/#0)**). Its hardware is designed and released in [open source!](https://github.com/sparkfun/SparkFun_Edge). It is a micro-controller that our program trained for "audio recognition" will run in only 18.2 kbytes, and [be supported by a 3V CR2032 coin cell battery for up to 10 days](https://www.sparkfun.com/products/15170).

The reposibory utlizes [Simple Audio Recognition](https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md#simple-audio-recognition) as example. However, in theory, it can well work for many sequential applications "not sensitive to the order of the timesteps" (see section 6.4.5 of <<Deep-Learning-with-Python-François-Chollet>>), which we could consider as a "short term FIR filter" in Digital Signal Processing. 
Based the convnet with sequence, the "long term pattern" can be also recognized by combining the technique and RNN (recurrent neural network), which corresponds to IIR filter :)

### 1,  Setup environment.
#### 1.1, 
Looks like that only few Tf commits can work for the example as above. "[How to Train New TensorFlow Lite Micro Speech Models](https://www.digikey.com/en/maker/projects/how-to-train-new-tensorflow-lite-micro-speech-models/e9480d4a38264604a2bf0336ce11aa9e)" is a good guidance for reference. 
```
sudo docker run --runtime=nvidia –name <your-container-name> -it tensorflow/tensorflow:1.15.0-gpu-py3-jupyter bash
sudo docker container start <your-container-name>
sudo docker container exec -it <your-container-name> /bin/bash 
```
#### 1.2
Enter the container, get tools for building --
```
apt-get install -y curl zip git xxd

pip3 list | grep tensorflow
tensorflow-estimator 1.15.1             
tensorflow-gpu       1.15.0

pip3 uninstall tensorflow-gpu==1.15.0
pip3 install tf-nightly-gpu==1.15.0.dev20190729

pip3 uninstall tensorflow-estimator==1.15.1
pip3 install -I tensorflow_estimator==1.13.0

curl -O -L https://github.com/bazelbuild/bazel/releases/download/0.23.1/bazel-0.23.1-installer-linux-x86_64.sh
chmod +x bazel-0.23.1-installer-linux-x86_64.sh
./bazel-0.23.1-installer-linux-x86_64.sh 
```

#### 1.3,
Still in the container, git 2 commits of Tf --
```
git clone https://github.com/tensorflow/tensorflow.git

cp -rp  tensorflow  tensorflow-aa47072ff4e2b7735b0e0ef9ef52f68ffbf7ef54
cd  tensorflow-aa47072ff4e2b7735b0e0ef9ef52f68ffbf7ef54
git checkout aa47072ff4e2b7735b0e0ef9ef52f68ffbf7ef54

cd ..
mv tensorflow tensorflow-4a464440b2e8f382f442b6e952d64a56701ab045
cd tensorflow-4a464440b2e8f382f442b6e952d64a56701ab045
git checkout 4a464440b2e8f382f442b6e952d64a56701ab045

yes "" | ./configure

```
The commit of 4a464440b2e8f382f442b6e952d64a56701ab045 is prepared to train the model. 
So is the commit of aa47072ff4e2b7735b0e0ef9ef52f68ffbf7ef54  to flash the model and dependencies. 

### 2, Training model.

Still in the container --
```
bazel run -c opt --copt=-mavx2 --copt=-mfma tensorflow/examples/speech_commands:train -- --model_architecture=tiny_conv --window_stride=20 --preprocess=micro --wanted_words="yes,no" --silence_percentage=25 --unknown_percentage=25 –quantize=1
```
Freeze, convert the model, create its C array.
```

bazel run tensorflow/examples/speech_commands:freeze -- --model_architecture=tiny_conv --window_stride=20 --preprocess=micro --wanted_words="yes,no" --quantize=1 --output_file=/tmp/tiny_conv.pb –start_checkpoint=/tmp/speech_commands_train/tiny_conv.ckpt-18000

bazel run tensorflow/lite/toco:toco -- --input_file=/tmp/tiny_conv.pb --output_file=/tmp/tiny_conv.tflite --input_shapes=1,49,40,1 --input_arrays=Reshapexxd -i /tmp/tiny_conv.tflite > /tmp/tiny_conv_micro_features_model_data.cc_1 --output_arrays='labels_softmax' --inference_type=QUANTIZED_UINT8 --mean_values=0 --std_values=9.8077

xxd -i /tmp/tiny_conv.tflite > /tmp/tiny_conv_micro_features_model_data.cc
```
#### 2.3, Notes --
1) The sequence data-set is at /tmp/speech_dataset, in which there are word vioces in length of 1 second, like "Yes", "No". You could add sub-folder over here, and record your own voices into.

2) The source file to create the model is tensorflow-4a464440b2e8f382f442b6e952d64a56701ab045/tensorflow/examples/speech_commands/models.py. You could change any code over here, so that your own model is developed. 

3) Since micro-controller does usually not support file-system, Tf-lite file has to be converted to C array before the model is flashed to SparkFun Edge board. The array created through the steps above is /tmp/tiny_conv_micro_features_model_data.cc. Checking the last line of the array, you will be aware of the momery size that the model will occupy, which is 18200 bytes.

### 3, Flash, test the model on board.
#### 3.1,
Prepare the board imgage to flash, which the trained model is in.
Still in the container, change directory --
```
cd ../tensorflow-4a464440b2e8f382f442b6e952d64a56701ab045
```
Actually the commit provides an existing trained model, which is: tensorflow-4a464440b2e8f382f442b6e952d64a56701ab045/tensorflow/lite/experimental/micro/examples/micro_speech/tiny_conv_micro_features_model_data.cc. Change its name as tiny_conv_micro_features_model_data.ori.cc and keep it in case you may need to test it as comparison.

Regarding to the current "confidence score" that can be reached with the 2 commits, I would suggest to change value of "[detection_threshold](tensorflow-4a464440b2e8f382f442b6e952d64a56701ab045/tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/recognize_commands.h)" to 150.
```
mv tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.cc 
tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.ori.cc
cp -p  /tmp/tiny_conv_micro_features_model_data.cc tensorflow/lite/experimental/micro/examples/micro_speech/micro_features
```
Following "[Convert to a C array](https://www.tensorflow.org/lite/microcontrollers/build_convert#convert_to_a_c_array)", copy the head and tail of tensorflow-4a464440b2e8f382f442b6e952d64a56701ab045/tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.ori.cc, paste them to tiny_conv_micro_featurestensorflow-4a464440b2e8f382f442b6e952d64a56701ab045/tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.cc.

Build borad image including the model and all depedencies --
```
make -f tensorflow/lite/experimental/micro/tools/make/Makefile clean

make -f tensorflow/lite/experimental/micro/tools/make/Makefile TARGET=sparkfun_edge TAGS="CMSIS" micro_speech_bin

cp tensorflow/lite/experimental/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/keys_info0.py \
tensorflow/lite/experimental/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/keys_info.py

python3 tensorflow/lite/experimental/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/create_cust_image_blob.py \
--bin tensorflow/lite/experimental/micro/tools/make/gen/sparkfun_edge_cortex-m4/bin/micro_speech.bin \
--load-address 0xC000 \
--magic-num 0xCB \
-o main_nonsecure_ota \
--version 0x0

python3 tensorflow/lite/experimental/micro/tools/make/downloads/AmbiqSuite-Rel2.0.0/tools/apollo3_scripts/create_cust_wireupdate_blob.py \
--load-address 0x20000 \
--bin main_nonsecure_ota.bin \
-i 6 \detection_threshold
-o main_nonsecure_wire \
--options 0x1
```

#### 3.2, Flash the board
Following "[Get ready to flash thdetection_thresholde binary](https://codelabs.developers.google.com/codetection_thresholddelabs/sparkfun-tensorflow/#4)" and "[Flash the binary](httpdetection_thresholds://codelabs.developers.google.com/codetection_thresholddelabs/sparkfun-tensorflow/#5)", flash the board.

#### 3.3, Test on the board
Following "[Read the debug outputdetection_threshold](https://codelabs.developers.google.com/codelabs/sparkfun-tensorflow/#8)", get debug printed out when the board hears "Yes". My log is [here](model-trained-by-the-commits.png).

### 4, Todo...
1) If using tensorflow-4a464440b2e8f382f442b6e952d64a56701ab045/tensorflow/lite/experimental/micro/examples/micro_speech/tiny_conv_micro_features_model_data.ori.cc as trained model, you could keep "detection_threshold" as it (200) in tensorflow-4a464440b2e8f382f442b6e952d64a56701ab045/tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/recognize_commands.h. On the case, the board could recognize almost all "Yes" it hears.
But I do still not know what steps we should walk through to create the C array model. A discussion with Tf team about the issue is [here](https://github.com/tensorflow/tensorflow/issues/33778). So, it is still interesting to put a little more efforts at :)

2) Some places are still considered to try in my mind, in order to promote the final "confidence score" :)




