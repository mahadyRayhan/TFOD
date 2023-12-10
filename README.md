# Training Tensorflow Objecte Detection

The most convenient way to train a TensorFlow object detection model is to use verified Tensorflow models architectures provided by TensorFlow. you can find the GitHub repo at this link [TensorFlow official](https://github.com/tensorflow/models).

In this section, I train an object detection model (EfficientDet D3) in a virtual environment. The reason for using a virtual environment is to make the whole process separate so that it won't affect other model implements or environment setups.

I use an anaconda virtual environment to train this model. You can use any kind of virtual environment setup to train the model. the benefit of using an anaconda environment is, when I install TensorFlow GPU using conda commands, it automatically install compatible Cuda and Cudnn libraries. if you use other environments, you may or may not have to set up these two necessary libraries.

### Install anaconda virtual environment

installing anaconda is fairly easy. Open a terminal copy-paste the following commands.

<pre>
sudo apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

bash /tmp/Anaconda3-2020.02-Linux-x86_64.sh

source ~/.bashrc
</pre>

This process will ask for license approval. just approve the license and you are ready to go.

After installing an anaconda-navigator, close the terminal and open a new one. Now create a new anaconda virtual environment. Simple type -
<pre>conda create -n tensorflow pip python=3.6</pre>
where "tensorflow" is the name of your environment. You can choose any python version. I use python 3.6 as I know this version is one of the most stable versions.

After creating the virtual environment activate the environment. Simply type **source activate tensorflow**. The terminal will show you this command once the environment installation is successfully finished.

Next, install TensorFlow-GPU (as we will use GPU to train) on our system. simply type -

<pre>conda install -c anaconda tensorflow-gpu</pre>

This will install the TensorFlow-GPU version to our system as well as compatible Cuda and Cudnn in this virtual environment.

That's all for an anaconda virtual environment setup.<br>
**Note: all these commands should be run in a terminal, not in the Jupyter notebook.**

## Install TensorFlow Object Detection API

Now open a jupyter notebook for the next installation process that is TensorFlow Object Detection API. Simply type -
<pre> jupyter notebook</pre> to open a jupyter notebook.

Now you may face some errors as jupyter notebook is not always installed with the new environment setup. If this happens for you simply type - <pre> pip install notebook </pre> This will install a new jupyter notebook for this virtual environment.<br>

Now open jupyter notebook.

First check the installed version of Tensorflow. I installed Tensorflow 2.4.1. Your version might be different.


```python
import tensorflow as tf
tf.__version__
```

    2022-01-26 13:49:37.407858: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.10.1
    '2.4.1'



### Downloading the TensorFlow Model Garden

First create a folder where you like to store all your files(pre-trained model, trained model, data, etc) related to object detection. then go to that directory and execute the following command.


```python
pwd
```
    '/home/ubuntu-20/Desktop/TensorFlow'

```python
!git clone https://github.com/tensorflow/models.git
```

    Cloning into 'models'...
    remote: Enumerating objects: 68544, done.[K
    remote: Counting objects: 100% (10/10), done.[K
    remote: Compressing objects: 100% (8/8), done.[K
    remote: Total 68544 (delta 2), reused 9 (delta 2), pack-reused 68534[K
    Receiving objects: 100% (68544/68544), 576.95 MiB | 14.26 MiB/s, done.
    Resolving deltas: 100% (48277/48277), done.


This will download Tensorflow model garden in your folder. I named my model as **"TensorFlow"**. You can name anything you want.

### Installat Protobuf 

Tensorflow Object Detection API uses Protobufs to configure model and training parameters. This library must be installed inside the **"research"** folder, which is inside the **"models"** folder. BTW, the **"models"** folder is the Tensorflow model garden repo which we pulled earlier.


```python
cd models/research/
```
    /home/ubuntu-20/Desktop/TensorFlow/models/research
```python
pwd
```
    '/home/ubuntu-20/Desktop/TensorFlow/models/research'
Now execute the Protobuf installation command here.


```python
!protoc object_detection/protos/*.proto --python_out=.
```

### Install COCO API

"pycocotools" is another dependency needed for Tensorflow object detection API. You can simply pull the library from GitHub and install/make it. But before doing that, you need "cython" library to be installed. Install "cython" as follows -


```python
!pip install cython
```


Next clone "pycocotools" repo from GitHub and execute the following commands one by one.


```python
!git clone https://github.com/cocodataset/cocoapi.git
```

```python
cd cocoapi/PythonAPI
```
    /home/ubuntu-20/Desktop/TensorFlow/models/research/cocoapi/PythonAPI

```python
!make
```

```python
cp -r pycocotools /home/ubuntu-20/Desktop/TensorFlow/models/research
```

So,"pycocotools" library is installed successfully.

### Install the Object Detection API

Now all the dependencies necessary for installing Object Detection API are done. So, we can install Object Detection API now. Object Detection API should be installed inside the **"research"** directory. Check the present current directory and go to the **"research"** directory


```python
pwd
```
    '/home/ubuntu-20/Desktop/TensorFlow/models/research/cocoapi/PythonAPI'

```python
cd ..
```
    /home/ubuntu-20/Desktop/TensorFlow/models/research/cocoapi

```python
cd ..
```
    /home/ubuntu-20/Desktop/TensorFlow/models/research

```python
pwd
```
    '/home/ubuntu-20/Desktop/TensorFlow/models/research'

Next run the following two commands to install **"Object Detection API"**


```python
cp object_detection/packages/tf2/setup.py .
```


```python
!python -m pip install .
```
    Successfully installed apache-beam-2.35.0 avro-python3-1.10.2 colorama-0.4.4 contextlib2-21.6.0 crcmod-1.7 cycler-0.11.0 dill-0.3.1.1 dm-tree-0.1.6 docopt-0.6.2 fastavro-1.4.9 fonttools-4.29.0 future-0.18.2 gin-config-0.5.0 google-api-core-2.4.0 google-api-python-client-2.36.0 google-auth-1.35.0 google-auth-httplib2-0.1.0 googleapis-common-protos-1.54.0 hdfs-2.6.0 httplib2-0.19.1 joblib-1.1.0 kaggle-1.5.12 keras-2.7.0 kiwisolver-1.3.2 libclang-12.0.0 lvis-0.5.3 lxml-4.7.1 matplotlib-3.5.1 numpy-1.20.3 oauth2client-4.1.3 object-detection-0.1 opencv-python-4.5.5.62 opencv-python-headless-4.5.5.62 orjson-3.6.6 pandas-1.4.0 pillow-9.0.0 portalocker-2.3.2 promise-2.3 proto-plus-1.19.9 psutil-5.9.0 py-cpuinfo-8.0.0 pyarrow-6.0.1 pycocotools-2.0.4 pydot-1.4.2 pymongo-3.12.3 pyparsing-2.4.7 python-slugify-5.0.2 pytz-2021.3 pyyaml-6.0 regex-2022.1.18 sacrebleu-2.0.0 scikit-learn-1.0.2 sentencepiece-0.1.96 seqeval-1.2.2 tabulate-0.8.9 tensorflow-2.7.0 tensorflow-addons-0.15.0 tensorflow-datasets-4.4.0 tensorflow-estimator-2.7.0 tensorflow-hub-0.12.0 tensorflow-io-0.23.1 tensorflow-io-gcs-filesystem-0.23.1 tensorflow-metadata-1.6.0 tensorflow-model-optimization-0.7.0 tensorflow-text-2.7.3 text-unidecode-1.3 tf-models-official-2.7.0 tf-slim-1.1.0 threadpoolctl-3.0.0 tqdm-4.62.3 typeguard-2.13.3 typing-extensions-3.10.0.2 uritemplate-4.1.1


Now object detection setup is done. You can verify the installation by executing the following command.


```python
!python object_detection/builders/model_builder_tf2_test.py
```
    Ran 24 tests in 28.079s
    OK (skipped=1)

## Train Object Detection

To train an Object Detection model, we need some assets like some images for both train and test, image annotations, pre-trained object detection model, trained resources, and exported model to use for prediction. To keep things clear I create some directories. All the new folders are created inside the named **"training_demo"**.


```python
pwd
```




    '/home/ubuntu-20/Desktop/TensorFlow/models/research'



**"training_demo"** for organizing all the resources


```python
!mkdir ../../training_demo
```

**"annotations"** for ".record" and ".pbtxt" files. ".record" files are used as annotations and ".pbtxt" file for class names.


```python
!mkdir ../../training_demo/annotations
```

**"exported-models"** for final exported models. This model (.pb file and .config file is used for final prediction)


```python
!mkdir ../../training_demo/exported-models
```

**"images"** folder for store train and validation images


```python
!mkdir ../../training_demo/images
```


```python
!mkdir ../../training_demo/images/train
```


```python
!mkdir ../../training_demo/images/test
```

**"models"** directory to store models and training informations. 


```python
!mkdir ../../training_demo/models
```

**"pre-trained-models"** for storing "pre-trained-models" models. in our case "efficientdetD3"


```python
!mkdir ../../training_demo/pre-trained-models
```

### Download pre-trained model


```python
pwd
```




    '/home/ubuntu-20/Desktop/TensorFlow/models/research'



change directory to "pre-trained-models" (which I create earlier). Then downlaod pre-train model from tensorflow model zoo. To download any model go to this [GitHub repo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Copy any model's link address and download it using wget


```python
cd ../../training_demo/pre-trained-models
```

    /home/ubuntu-20/Desktop/TensorFlow/training_demo/pre-trained-models



```python
pwd
```




    '/home/ubuntu-20/Desktop/TensorFlow/training_demo/pre-trained-models'




```python
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz
```

    --2022-01-26 16:12:25--  http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz
    Resolving download.tensorflow.org (download.tensorflow.org)... 2404:6800:4004:810::2010, 172.217.174.112
    Connecting to download.tensorflow.org (download.tensorflow.org)|2404:6800:4004:810::2010|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 92858658 (89M) [application/x-tar]
    Saving to: ‘efficientdet_d3_coco17_tpu-32.tar.gz’
    
    efficientdet_d3_coc 100%[===================>]  88.56M  27.6MB/s    in 3.5s    
    
    2022-01-26 16:12:29 (25.6 MB/s) - ‘efficientdet_d3_coco17_tpu-32.tar.gz’ saved [92858658/92858658]
    


unpack downloaded model inside "pre-trained-models"


```python
!tar -xvf efficientdet_d3_coco17_tpu-32.tar.gz
```

    efficientdet_d3_coco17_tpu-32/
    efficientdet_d3_coco17_tpu-32/checkpoint/
    efficientdet_d3_coco17_tpu-32/checkpoint/ckpt-0.data-00000-of-00001
    efficientdet_d3_coco17_tpu-32/checkpoint/checkpoint
    efficientdet_d3_coco17_tpu-32/checkpoint/ckpt-0.index
    efficientdet_d3_coco17_tpu-32/pipeline.config
    efficientdet_d3_coco17_tpu-32/saved_model/
    efficientdet_d3_coco17_tpu-32/saved_model/saved_model.pb
    efficientdet_d3_coco17_tpu-32/saved_model/assets/
    efficientdet_d3_coco17_tpu-32/saved_model/variables/
    efficientdet_d3_coco17_tpu-32/saved_model/variables/variables.data-00000-of-00001
    efficientdet_d3_coco17_tpu-32/saved_model/variables/variables.index



```python
pwd
```




    '/home/ubuntu-20/Desktop/TensorFlow/training_demo/pre-trained-models'



### Generate .record file

Now we have a pre-trained model. Let's create training and test files. We have some training images along with annotation XML files both in the train and test folder. There is a **"generate_tfrecord.py"** file inside the "training_demo" folder. This file can be used to generate train and test the ".record" file.<br>

simply call "generate_tfrecord.py" and pass train images folder path, label_map.pbtxt file path, and directory path to save train.record file path. Paths can be anywhere. it completely depends on you.


```python
cd ..
```

    /home/ubuntu-20/Desktop/TensorFlow/training_demo



```python
# Create train data:
!python generate_tfrecord.py -x images/train -l annotations/label_map.pbtxt -o annotations/train.record

#Create test data:
!python generate_tfrecord.py -x images/test -l annotations/label_map.pbtxt -o annotations/test.record
```

    Successfully created the TFRecord file: annotations/train.record
    Successfully created the TFRecord file: annotations/test.record


Now we have necessary training files (train.record, test.record and label_map.pbtxt)and necessary pre-traind models. One last thing is needed is to configer **"pipeline.config"** file. Inside this file we have to configer

- train.record file path
- test.record file path
- label_map.pbtxt file path
- batch_size
- number of classes
- fine_tune_checkpoint_type
- use_bfloat16
- fine_tune_checkpoint_type

You can find "pipeline.config" file inside the downloaded pre-trained model folder

To make everything clean and reusable, I coppied the "pipeline.config" file from "pre-trained-models/efficientdet_d3_coco17_tpu-32" to "exported_models/efficientdet" folder and edit it. Open "pipeline.config" and searche for 
1. num_classes
<pre>As we are focusing just on finding the object, our number of classes is 1. Now change num_classes to 1.</pre>
2. batch_size
<pre>Depending of the GPU, we can use any batch size. So use any number suitable for the training</pre>
3. total_steps/num_steps
<pre>Defines how long we like to train the model. Note that total_steps & num_steps are two different parameters. You have to assign the same value for both of them. Another important point is "warmup_steps". Both total_steps & num_steps must be higher than warmup_steps. Otherwise, you will get some errors.</pre>
4. fine_tune_checkpoint
<pre>You may find it just before the "num_steps" parameter. Basically, you have to specify the downloaded pre-trained model's checkpoint path to start training using that pre-trained model. Currently, we are using efficientdetD3. So, go to the pre-trained-models -> efficientdet_d3_coco17_tpu-32 -> checkpoint folder  and copy the relative path of "ckpt-0.index" and paste it here. As I am currently inside "training_demo" folder, so relative path from "training_demo" would be "./pre-trained-models/efficientdet_d3_coco17_tpu-32/checkpoint/ckpt-0". <strong>Note that, you have to remove the extension from "ckpt-0.index" file</strong>. The model already knows the file extension. <strong>BTW, all folder names are changeable. If you want to change the folder name or path, you can do that. Just make sure to specify the file path correctly</strong></pre>
5. fine_tune_checkpoint_type
<pre>Change fine_tune_checkpoint_type to <strong>"detection"</strong> as we are trying to detect objects.</pre>
6. use_bfloat16
<pre>Change "use_bfloat16" to <strong>flase</strong> as we are not using TPU's</pre>
7. train_input_reader
<pre>Inside "train_input_reader" section, you have to modify 2 parameters, "label_map_path" & "input_path".  The "label_map.pbtxt" can be found inside "annotations" folder (as i put it there. You can put it anywhere you like). As we are still inside "training_demo" folder, so relative path would be "./annotations/label_map.pbtxt".<br>
Next "input_path". As this "input_path" is inside "train_input_reader" section, so we have to specify traing input file which is <strong>train.record</strong> file. The "train.record" file can also be found inside "annotations" folder. As we are still inside "training_demo" folder, so relative path would be "./annotations/train.record".</pre>

8. eval_input_reader
<pre>Inside "eval_input_reader" section, you also have to modify 2 parameters, "label_map_path" & "input_path".  The "label_map.pbtxt" can be found inside "annotations" folder (as i put it there. You can put it anywhere you like). As we are still inside "training_demo" folder, so relative path would be "./annotations/label_map.pbtxt".<br>
Next "input_path". As this "input_path" is inside the "eval_input_reader" section, so we have to specify the training input file which is <strong>test.record</strong> file (I named validation set as test. You can use val or validation. Just make sure to change the path accordingly). The "test.record" file can also be found inside the "annotations" folder. As we are still inside the "training_demo" folder, so the relative path would be "./annotations/test.record".</pre>

### Training

Now we are done with all installation and configurations. Let's train the model. Currently we are inside **"training_demo"** folder. If you check, you will find a file named **"model_main_tf2.py"**. This is the file we are going to use to train the object detection model. This file is also provided by Tensorflow. Check out this file. There are also some configurations inside this file. You can change/pass the argument depending on your objective.


```python
pwd
```




    '/home/ubuntu-20/Desktop/TensorFlow/training_demo'




```python
ls
```

    [0m[01;34mannotations[0m/         export_tflite_graph_tf2.py  model_main_tf2.py
    [01;34mexported-models[0m/     generate_tfrecord.py        [01;34mmodels[0m/
    exporter_main_v2.py  [01;34mimages[0m/                     [01;34mpre-trained-models[0m/


To train the model, you have to pass two parameters, **"model_dir"** & **"pipeline_config_path"**. "model_dir" is the folder where you want to save your model training progress (this folder will be needed later for visualization of the training graphs) and "pipeline_config_path" is the folder where we want to save our trained model configuration. (Basically, the configuration file which will be saved in this training process is the same one we configured earlier. Just to make this separate, I saved this config file in the same folder where the trained models are going to save). 


```python
!python model_main_tf2.py --model_dir=models/EfficientDet-D3-custom-trained --pipeline_config_path=models/EfficientDet-D3-custom-trained/pipeline.config
```

    2022-01-26 19:18:16.924165: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:939] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-01-26 19:18:16.927108: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcusolver.so.11'; dlerror: libcusolver.so.11: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/cv2/../../lib64:/usr/local/cuda/lib64
    2022-01-26 19:18:16.927694: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1850] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
    Skipping registering GPU devices...
    2022-01-26 19:18:16.928442: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    WARNING:tensorflow:There are non-GPU devices in `tf.distribute.Strategy`, not using nccl allreduce.
    W0126 19:18:16.930691 139684560966016 cross_device_ops.py:1387] There are non-GPU devices in `tf.distribute.Strategy`, not using nccl allreduce.
    INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:CPU:0',)
    I0126 19:18:16.931914 139684560966016 mirrored_strategy.py:376] Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:CPU:0',)
    INFO:tensorflow:Maybe overwriting train_steps: None
    I0126 19:18:16.935007 139684560966016 config_util.py:552] Maybe overwriting train_steps: None
    INFO:tensorflow:Maybe overwriting use_bfloat16: False
    I0126 19:18:16.935164 139684560966016 config_util.py:552] Maybe overwriting use_bfloat16: False
    I0126 19:18:16.944221 139684560966016 ssd_efficientnet_bifpn_feature_extractor.py:145] EfficientDet EfficientNet backbone version: efficientnet-b3
    I0126 19:18:16.944335 139684560966016 ssd_efficientnet_bifpn_feature_extractor.py:147] EfficientDet BiFPN num filters: 160
    I0126 19:18:16.944365 139684560966016 ssd_efficientnet_bifpn_feature_extractor.py:148] EfficientDet BiFPN num iterations: 6
    I0126 19:18:16.947594 139684560966016 efficientnet_model.py:147] round_filter input=32 output=40
    I0126 19:18:16.979430 139684560966016 efficientnet_model.py:147] round_filter input=32 output=40
    I0126 19:18:16.979545 139684560966016 efficientnet_model.py:147] round_filter input=16 output=24
    I0126 19:18:17.123809 139684560966016 efficientnet_model.py:147] round_filter input=16 output=24
    I0126 19:18:17.123928 139684560966016 efficientnet_model.py:147] round_filter input=24 output=32
    I0126 19:18:17.395811 139684560966016 efficientnet_model.py:147] round_filter input=24 output=32
    I0126 19:18:17.395925 139684560966016 efficientnet_model.py:147] round_filter input=40 output=48
    I0126 19:18:17.663366 139684560966016 efficientnet_model.py:147] round_filter input=40 output=48
    I0126 19:18:17.663570 139684560966016 efficientnet_model.py:147] round_filter input=80 output=96
    I0126 19:18:18.200055 139684560966016 efficientnet_model.py:147] round_filter input=80 output=96
    I0126 19:18:18.200171 139684560966016 efficientnet_model.py:147] round_filter input=112 output=136
    I0126 19:18:18.719847 139684560966016 efficientnet_model.py:147] round_filter input=112 output=136
    I0126 19:18:18.719960 139684560966016 efficientnet_model.py:147] round_filter input=192 output=232
    I0126 19:18:19.291749 139684560966016 efficientnet_model.py:147] round_filter input=192 output=232
    I0126 19:18:19.291912 139684560966016 efficientnet_model.py:147] round_filter input=320 output=384
    I0126 19:18:19.552504 139684560966016 efficientnet_model.py:147] round_filter input=1280 output=1536
    I0126 19:18:19.620378 139684560966016 efficientnet_model.py:457] Building model efficientnet with params ModelConfig(width_coefficient=1.2, depth_coefficient=1.4, resolution=300, dropout_rate=0.3, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='simple_swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=False, data_format='channels_last', dtype='float32')
    WARNING:tensorflow:From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/object_detection/model_lib_v2.py:563: StrategyBase.experimental_distribute_datasets_from_function (from tensorflow.python.distribute.distribute_lib) is deprecated and will be removed in a future version.
    Instructions for updating:
    rename to distribute_datasets_from_function
    W0126 19:18:19.679734 139684560966016 deprecation.py:341] From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/object_detection/model_lib_v2.py:563: StrategyBase.experimental_distribute_datasets_from_function (from tensorflow.python.distribute.distribute_lib) is deprecated and will be removed in a future version.
    Instructions for updating:
    rename to distribute_datasets_from_function
    INFO:tensorflow:Reading unweighted datasets: ['./annotations/train.record']
    I0126 19:18:19.692135 139684560966016 dataset_builder.py:163] Reading unweighted datasets: ['./annotations/train.record']
    INFO:tensorflow:Reading record datasets for input file: ['./annotations/train.record']
    I0126 19:18:19.692332 139684560966016 dataset_builder.py:80] Reading record datasets for input file: ['./annotations/train.record']
    INFO:tensorflow:Number of filenames to read: 1
    I0126 19:18:19.692425 139684560966016 dataset_builder.py:81] Number of filenames to read: 1
    WARNING:tensorflow:num_readers has been reduced to 1 to match input file shards.
    W0126 19:18:19.692487 139684560966016 dataset_builder.py:87] num_readers has been reduced to 1 to match input file shards.
    WARNING:tensorflow:From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/object_detection/builders/dataset_builder.py:101: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.deterministic`.
    W0126 19:18:19.694516 139684560966016 deprecation.py:341] From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/object_detection/builders/dataset_builder.py:101: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.deterministic`.
    WARNING:tensorflow:From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/object_detection/builders/dataset_builder.py:236: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.map()
    W0126 19:18:19.851195 139684560966016 deprecation.py:341] From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/object_detection/builders/dataset_builder.py:236: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.map()
    WARNING:tensorflow:From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/tensorflow/python/util/dispatch.py:1096: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
    W0126 19:18:24.580934 139684560966016 deprecation.py:341] From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/tensorflow/python/util/dispatch.py:1096: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
    WARNING:tensorflow:From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/tensorflow/python/autograph/impl/api.py:465: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    W0126 19:18:27.232981 139684560966016 deprecation.py:341] From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/tensorflow/python/autograph/impl/api.py:465: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    2022-01-26 19:18:29.275193: W tensorflow/core/framework/dataset.cc:744] Input of GeneratorDatasetOp::Dataset will not be optimized because the dataset does not implement the AsGraphDefInternal() method needed to apply optimizations.
    2022-01-26 19:18:29.802818: W tensorflow/core/framework/op_kernel.cc:1745] OP_REQUIRES failed at multi_device_iterator_ops.cc:789 : NOT_FOUND: Resource AnonymousMultiDeviceIterator/AnonymousMultiDeviceIterator0/N10tensorflow4data12_GLOBAL__N_119MultiDeviceIteratorE does not exist.
    /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/keras/backend.py:414: UserWarning: `tf.keras.backend.set_learning_phase` is deprecated and will be removed after 2020-10-11. To update it, simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.
      warnings.warn('`tf.keras.backend.set_learning_phase` is deprecated and '
    WARNING:tensorflow:From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/tensorflow/python/util/deprecation.py:620: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Use fn_output_signature instead
    W0126 19:18:58.426820 139676748863232 deprecation.py:545] From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/tensorflow/python/util/deprecation.py:620: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Use fn_output_signature instead
    WARNING:tensorflow:Gradients do not exist for variables ['stack_6/block_1/expand_bn/gamma:0', 'stack_6/block_1/expand_bn/beta:0', 'stack_6/block_1/depthwise_conv2d/depthwise_kernel:0', 'stack_6/block_1/depthwise_bn/gamma:0', 'stack_6/block_1/depthwise_bn/beta:0', 'stack_6/block_1/project_bn/gamma:0', 'stack_6/block_1/project_bn/beta:0', 'top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    W0126 19:19:06.249039 139676748863232 utils.py:76] Gradients do not exist for variables ['stack_6/block_1/expand_bn/gamma:0', 'stack_6/block_1/expand_bn/beta:0', 'stack_6/block_1/depthwise_conv2d/depthwise_kernel:0', 'stack_6/block_1/depthwise_bn/gamma:0', 'stack_6/block_1/depthwise_bn/beta:0', 'stack_6/block_1/project_bn/gamma:0', 'stack_6/block_1/project_bn/beta:0', 'top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    WARNING:tensorflow:Gradients do not exist for variables ['stack_6/block_1/expand_bn/gamma:0', 'stack_6/block_1/expand_bn/beta:0', 'stack_6/block_1/depthwise_conv2d/depthwise_kernel:0', 'stack_6/block_1/depthwise_bn/gamma:0', 'stack_6/block_1/depthwise_bn/beta:0', 'stack_6/block_1/project_bn/gamma:0', 'stack_6/block_1/project_bn/beta:0', 'top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    W0126 19:19:17.842705 139676748863232 utils.py:76] Gradients do not exist for variables ['stack_6/block_1/expand_bn/gamma:0', 'stack_6/block_1/expand_bn/beta:0', 'stack_6/block_1/depthwise_conv2d/depthwise_kernel:0', 'stack_6/block_1/depthwise_bn/gamma:0', 'stack_6/block_1/depthwise_bn/beta:0', 'stack_6/block_1/project_bn/gamma:0', 'stack_6/block_1/project_bn/beta:0', 'top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    WARNING:tensorflow:Gradients do not exist for variables ['stack_6/block_1/expand_bn/gamma:0', 'stack_6/block_1/expand_bn/beta:0', 'stack_6/block_1/depthwise_conv2d/depthwise_kernel:0', 'stack_6/block_1/depthwise_bn/gamma:0', 'stack_6/block_1/depthwise_bn/beta:0', 'stack_6/block_1/project_bn/gamma:0', 'stack_6/block_1/project_bn/beta:0', 'top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    W0126 19:19:27.477190 139676748863232 utils.py:76] Gradients do not exist for variables ['stack_6/block_1/expand_bn/gamma:0', 'stack_6/block_1/expand_bn/beta:0', 'stack_6/block_1/depthwise_conv2d/depthwise_kernel:0', 'stack_6/block_1/depthwise_bn/gamma:0', 'stack_6/block_1/depthwise_bn/beta:0', 'stack_6/block_1/project_bn/gamma:0', 'stack_6/block_1/project_bn/beta:0', 'top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    WARNING:tensorflow:Gradients do not exist for variables ['stack_6/block_1/expand_bn/gamma:0', 'stack_6/block_1/expand_bn/beta:0', 'stack_6/block_1/depthwise_conv2d/depthwise_kernel:0', 'stack_6/block_1/depthwise_bn/gamma:0', 'stack_6/block_1/depthwise_bn/beta:0', 'stack_6/block_1/project_bn/gamma:0', 'stack_6/block_1/project_bn/beta:0', 'top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    W0126 19:19:38.556267 139676748863232 utils.py:76] Gradients do not exist for variables ['stack_6/block_1/expand_bn/gamma:0', 'stack_6/block_1/expand_bn/beta:0', 'stack_6/block_1/depthwise_conv2d/depthwise_kernel:0', 'stack_6/block_1/depthwise_bn/gamma:0', 'stack_6/block_1/depthwise_bn/beta:0', 'stack_6/block_1/project_bn/gamma:0', 'stack_6/block_1/project_bn/beta:0', 'top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    INFO:tensorflow:Step 100 per-step time 6.912s
    I0126 19:30:29.503887 139684560966016 model_lib_v2.py:705] Step 100 per-step time 6.912s
    INFO:tensorflow:{'Loss/classification_loss': 2.3749464,
     'Loss/localization_loss': 0.0,
     'Loss/regularization_loss': 0.038565468,
     'Loss/total_loss': 2.4135118,
     'learning_rate': 0.0}
    I0126 19:30:29.526928 139684560966016 model_lib_v2.py:708] {'Loss/classification_loss': 2.3749464,
     'Loss/localization_loss': 0.0,
     'Loss/regularization_loss': 0.038565468,
     'Loss/total_loss': 2.4135118,
     'learning_rate': 0.0}


If you to train on multiple GPU, then you have to add **"num_clones"**. "num_clones" defines how many GPUs you like to use. I have only 1 GPU. So i assign **--num_clones=1**


```python
!python model_main_tf2.py  --model_dir=models/EfficientDet-D3-custom-trained --pipeline_config_path=models/EfficientDet-D3-custom-trained/pipeline.config --num_clones=1 --ps_tasks=1
```

    2022-01-26 19:51:11.329579: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:939] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-01-26 19:51:11.332355: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcusolver.so.11'; dlerror: libcusolver.so.11: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/cv2/../../lib64:/usr/local/cuda/lib64
    2022-01-26 19:51:11.332701: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1850] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
    Skipping registering GPU devices...
    2022-01-26 19:51:11.333297: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    WARNING:tensorflow:There are non-GPU devices in `tf.distribute.Strategy`, not using nccl allreduce.
    W0126 19:51:11.335279 140671546012032 cross_device_ops.py:1387] There are non-GPU devices in `tf.distribute.Strategy`, not using nccl allreduce.
    INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:CPU:0',)
    I0126 19:51:11.336194 140671546012032 mirrored_strategy.py:376] Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:CPU:0',)
    INFO:tensorflow:Maybe overwriting train_steps: None
    I0126 19:51:11.338904 140671546012032 config_util.py:552] Maybe overwriting train_steps: None
    INFO:tensorflow:Maybe overwriting use_bfloat16: False
    I0126 19:51:11.339073 140671546012032 config_util.py:552] Maybe overwriting use_bfloat16: False
    I0126 19:51:11.347533 140671546012032 ssd_efficientnet_bifpn_feature_extractor.py:145] EfficientDet EfficientNet backbone version: efficientnet-b3
    I0126 19:51:11.347657 140671546012032 ssd_efficientnet_bifpn_feature_extractor.py:147] EfficientDet BiFPN num filters: 160
    I0126 19:51:11.347728 140671546012032 ssd_efficientnet_bifpn_feature_extractor.py:148] EfficientDet BiFPN num iterations: 6
    I0126 19:51:11.350413 140671546012032 efficientnet_model.py:147] round_filter input=32 output=40
    I0126 19:51:11.381894 140671546012032 efficientnet_model.py:147] round_filter input=32 output=40
    I0126 19:51:11.382004 140671546012032 efficientnet_model.py:147] round_filter input=16 output=24
    I0126 19:51:11.517031 140671546012032 efficientnet_model.py:147] round_filter input=16 output=24
    I0126 19:51:11.517138 140671546012032 efficientnet_model.py:147] round_filter input=24 output=32
    I0126 19:51:11.772451 140671546012032 efficientnet_model.py:147] round_filter input=24 output=32
    I0126 19:51:11.772562 140671546012032 efficientnet_model.py:147] round_filter input=40 output=48
    I0126 19:51:12.061040 140671546012032 efficientnet_model.py:147] round_filter input=40 output=48
    I0126 19:51:12.061159 140671546012032 efficientnet_model.py:147] round_filter input=80 output=96
    I0126 19:51:12.604596 140671546012032 efficientnet_model.py:147] round_filter input=80 output=96
    I0126 19:51:12.604715 140671546012032 efficientnet_model.py:147] round_filter input=112 output=136
    I0126 19:51:13.175338 140671546012032 efficientnet_model.py:147] round_filter input=112 output=136
    I0126 19:51:13.175463 140671546012032 efficientnet_model.py:147] round_filter input=192 output=232
    I0126 19:51:13.943156 140671546012032 efficientnet_model.py:147] round_filter input=192 output=232
    I0126 19:51:13.943384 140671546012032 efficientnet_model.py:147] round_filter input=320 output=384
    I0126 19:51:14.294837 140671546012032 efficientnet_model.py:147] round_filter input=1280 output=1536
    I0126 19:51:14.345150 140671546012032 efficientnet_model.py:457] Building model efficientnet with params ModelConfig(width_coefficient=1.2, depth_coefficient=1.4, resolution=300, dropout_rate=0.3, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='simple_swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=False, data_format='channels_last', dtype='float32')
    WARNING:tensorflow:From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/object_detection/model_lib_v2.py:563: StrategyBase.experimental_distribute_datasets_from_function (from tensorflow.python.distribute.distribute_lib) is deprecated and will be removed in a future version.
    Instructions for updating:
    rename to distribute_datasets_from_function
    W0126 19:51:14.389445 140671546012032 deprecation.py:341] From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/object_detection/model_lib_v2.py:563: StrategyBase.experimental_distribute_datasets_from_function (from tensorflow.python.distribute.distribute_lib) is deprecated and will be removed in a future version.
    Instructions for updating:
    rename to distribute_datasets_from_function
    INFO:tensorflow:Reading unweighted datasets: ['./annotations/train.record']
    I0126 19:51:14.401006 140671546012032 dataset_builder.py:163] Reading unweighted datasets: ['./annotations/train.record']
    INFO:tensorflow:Reading record datasets for input file: ['./annotations/train.record']
    I0126 19:51:14.401212 140671546012032 dataset_builder.py:80] Reading record datasets for input file: ['./annotations/train.record']
    INFO:tensorflow:Number of filenames to read: 1
    I0126 19:51:14.401286 140671546012032 dataset_builder.py:81] Number of filenames to read: 1
    WARNING:tensorflow:num_readers has been reduced to 1 to match input file shards.
    W0126 19:51:14.401366 140671546012032 dataset_builder.py:87] num_readers has been reduced to 1 to match input file shards.
    WARNING:tensorflow:From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/object_detection/builders/dataset_builder.py:101: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.deterministic`.
    W0126 19:51:14.403273 140671546012032 deprecation.py:341] From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/object_detection/builders/dataset_builder.py:101: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.deterministic`.
    WARNING:tensorflow:From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/object_detection/builders/dataset_builder.py:236: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.map()
    W0126 19:51:14.562267 140671546012032 deprecation.py:341] From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/object_detection/builders/dataset_builder.py:236: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.map()
    WARNING:tensorflow:From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/tensorflow/python/util/dispatch.py:1096: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
    W0126 19:51:19.616015 140671546012032 deprecation.py:341] From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/tensorflow/python/util/dispatch.py:1096: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
    WARNING:tensorflow:From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/tensorflow/python/autograph/impl/api.py:465: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    W0126 19:51:22.496867 140671546012032 deprecation.py:341] From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/tensorflow/python/autograph/impl/api.py:465: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    2022-01-26 19:51:24.571085: W tensorflow/core/framework/dataset.cc:744] Input of GeneratorDatasetOp::Dataset will not be optimized because the dataset does not implement the AsGraphDefInternal() method needed to apply optimizations.
    /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/keras/backend.py:414: UserWarning: `tf.keras.backend.set_learning_phase` is deprecated and will be removed after 2020-10-11. To update it, simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.
      warnings.warn('`tf.keras.backend.set_learning_phase` is deprecated and '
    WARNING:tensorflow:From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/tensorflow/python/util/deprecation.py:620: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Use fn_output_signature instead
    W0126 19:51:55.055413 140663467276032 deprecation.py:545] From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/tensorflow/python/util/deprecation.py:620: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Use fn_output_signature instead
    WARNING:tensorflow:Gradients do not exist for variables ['stack_6/block_1/expand_bn/gamma:0', 'stack_6/block_1/expand_bn/beta:0', 'stack_6/block_1/depthwise_conv2d/depthwise_kernel:0', 'stack_6/block_1/depthwise_bn/gamma:0', 'stack_6/block_1/depthwise_bn/beta:0', 'stack_6/block_1/project_bn/gamma:0', 'stack_6/block_1/project_bn/beta:0', 'top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    W0126 19:52:03.805298 140663467276032 utils.py:76] Gradients do not exist for variables ['stack_6/block_1/expand_bn/gamma:0', 'stack_6/block_1/expand_bn/beta:0', 'stack_6/block_1/depthwise_conv2d/depthwise_kernel:0', 'stack_6/block_1/depthwise_bn/gamma:0', 'stack_6/block_1/depthwise_bn/beta:0', 'stack_6/block_1/project_bn/gamma:0', 'stack_6/block_1/project_bn/beta:0', 'top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    WARNING:tensorflow:Gradients do not exist for variables ['stack_6/block_1/expand_bn/gamma:0', 'stack_6/block_1/expand_bn/beta:0', 'stack_6/block_1/depthwise_conv2d/depthwise_kernel:0', 'stack_6/block_1/depthwise_bn/gamma:0', 'stack_6/block_1/depthwise_bn/beta:0', 'stack_6/block_1/project_bn/gamma:0', 'stack_6/block_1/project_bn/beta:0', 'top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    W0126 19:52:17.698312 140663467276032 utils.py:76] Gradients do not exist for variables ['stack_6/block_1/expand_bn/gamma:0', 'stack_6/block_1/expand_bn/beta:0', 'stack_6/block_1/depthwise_conv2d/depthwise_kernel:0', 'stack_6/block_1/depthwise_bn/gamma:0', 'stack_6/block_1/depthwise_bn/beta:0', 'stack_6/block_1/project_bn/gamma:0', 'stack_6/block_1/project_bn/beta:0', 'top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    WARNING:tensorflow:Gradients do not exist for variables ['stack_6/block_1/expand_bn/gamma:0', 'stack_6/block_1/expand_bn/beta:0', 'stack_6/block_1/depthwise_conv2d/depthwise_kernel:0', 'stack_6/block_1/depthwise_bn/gamma:0', 'stack_6/block_1/depthwise_bn/beta:0', 'stack_6/block_1/project_bn/gamma:0', 'stack_6/block_1/project_bn/beta:0', 'top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    W0126 19:52:28.856673 140663467276032 utils.py:76] Gradients do not exist for variables ['stack_6/block_1/expand_bn/gamma:0', 'stack_6/block_1/expand_bn/beta:0', 'stack_6/block_1/depthwise_conv2d/depthwise_kernel:0', 'stack_6/block_1/depthwise_bn/gamma:0', 'stack_6/block_1/depthwise_bn/beta:0', 'stack_6/block_1/project_bn/gamma:0', 'stack_6/block_1/project_bn/beta:0', 'top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    WARNING:tensorflow:Gradients do not exist for variables ['stack_6/block_1/expand_bn/gamma:0', 'stack_6/block_1/expand_bn/beta:0', 'stack_6/block_1/depthwise_conv2d/depthwise_kernel:0', 'stack_6/block_1/depthwise_bn/gamma:0', 'stack_6/block_1/depthwise_bn/beta:0', 'stack_6/block_1/project_bn/gamma:0', 'stack_6/block_1/project_bn/beta:0', 'top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    W0126 19:52:40.214063 140663467276032 utils.py:76] Gradients do not exist for variables ['stack_6/block_1/expand_bn/gamma:0', 'stack_6/block_1/expand_bn/beta:0', 'stack_6/block_1/depthwise_conv2d/depthwise_kernel:0', 'stack_6/block_1/depthwise_bn/gamma:0', 'stack_6/block_1/depthwise_bn/beta:0', 'stack_6/block_1/project_bn/gamma:0', 'stack_6/block_1/project_bn/beta:0', 'top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?
    INFO:tensorflow:Step 100 per-step time 6.888s
    I0126 20:03:23.654749 140671546012032 model_lib_v2.py:705] Step 100 per-step time 6.888s
    INFO:tensorflow:{'Loss/classification_loss': 1.0755744,
     'Loss/localization_loss': 0.024037467,
     'Loss/regularization_loss': 0.038565367,
     'Loss/total_loss': 1.1381773,
     'learning_rate': 0.0}
    I0126 20:03:23.673841 140671546012032 model_lib_v2.py:708] {'Loss/classification_loss': 1.0755744,
     'Loss/localization_loss': 0.024037467,
     'Loss/regularization_loss': 0.038565367,
     'Loss/total_loss': 1.1381773,
     'learning_rate': 0.0}



```python
pwd
```




    '/home/ubuntu-20/Desktop/TensorFlow/training_demo'



# Visualizing the training process

Tensorflow has a nice tool to visulize the praining progress named "Tensorbord". This tool is automatically installed when you install tensorflow using the conda comand. To visulize training progress: <br>

- Open a new terminal.
- activate the virtual environment by trping **source activate tensorflow** (which we have done before in a different terminal)
- navigate to **"training_demo"** folder
- now type **tensorboard --logdir=models/EfficientDet-D3-custom-trained** as we saved our trained network in "EfficientDet-D3-custom-trained" folder.

Once this is done, terminal will show you something like this **"TensorBoard 2.6.0 at http://localhost:6006/ (Press CTRL+C to quit)"**<br>

Go to **http://localhost:6006/** and you can see different graphs related to your training progress.(Port number might be different)

# Export the trained model

We already trained our model. But Tensorflow Object Detection API didn't save the final model. Instead, it saves the checkpoints after every 1000 steps(you can change this step number from model_main_tf2.py). Now we have to export the final model from these checkpoints. <br> <br>
**Note:** the benefit of having these checkpoint files is, you can restart your training process from any of these checkpoint files later if you want.<br><br>
Now to export the final model, we will use "exporter_main_v2.py". This file is also provided by Tensorflow.<br><br>

Execute the following command with appropriate parameters. We already discussed all the folders necessary for this command.


```python
!python exporter_main_v2.py --input_type image_tensor --pipeline_config_path models/EfficientDet-D3-custom-trained/pipeline.config --trained_checkpoint_dir models/EfficientDet-D3-custom-trained --output_directory exported_models/efficientdet
```

    2022-01-26 19:38:08.131745: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:939] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-01-26 19:38:08.134366: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcusolver.so.11'; dlerror: libcusolver.so.11: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/lib64
    2022-01-26 19:38:08.134966: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1850] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
    Skipping registering GPU devices...
    2022-01-26 19:38:08.140429: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    I0126 19:38:08.147075 140401355592064 ssd_efficientnet_bifpn_feature_extractor.py:145] EfficientDet EfficientNet backbone version: efficientnet-b3
    I0126 19:38:08.147200 140401355592064 ssd_efficientnet_bifpn_feature_extractor.py:147] EfficientDet BiFPN num filters: 160
    I0126 19:38:08.147293 140401355592064 ssd_efficientnet_bifpn_feature_extractor.py:148] EfficientDet BiFPN num iterations: 6
    I0126 19:38:08.150027 140401355592064 efficientnet_model.py:147] round_filter input=32 output=40
    I0126 19:38:08.166386 140401355592064 efficientnet_model.py:147] round_filter input=32 output=40
    I0126 19:38:08.166506 140401355592064 efficientnet_model.py:147] round_filter input=16 output=24
    I0126 19:38:08.273832 140401355592064 efficientnet_model.py:147] round_filter input=16 output=24
    I0126 19:38:08.273949 140401355592064 efficientnet_model.py:147] round_filter input=24 output=32
    I0126 19:38:08.459990 140401355592064 efficientnet_model.py:147] round_filter input=24 output=32
    I0126 19:38:08.460103 140401355592064 efficientnet_model.py:147] round_filter input=40 output=48
    I0126 19:38:08.650922 140401355592064 efficientnet_model.py:147] round_filter input=40 output=48
    I0126 19:38:08.651039 140401355592064 efficientnet_model.py:147] round_filter input=80 output=96
    I0126 19:38:08.969503 140401355592064 efficientnet_model.py:147] round_filter input=80 output=96
    I0126 19:38:08.969624 140401355592064 efficientnet_model.py:147] round_filter input=112 output=136
    I0126 19:38:09.477416 140401355592064 efficientnet_model.py:147] round_filter input=112 output=136
    I0126 19:38:09.477530 140401355592064 efficientnet_model.py:147] round_filter input=192 output=232
    I0126 19:38:09.952585 140401355592064 efficientnet_model.py:147] round_filter input=192 output=232
    I0126 19:38:09.952699 140401355592064 efficientnet_model.py:147] round_filter input=320 output=384
    I0126 19:38:10.110199 140401355592064 efficientnet_model.py:147] round_filter input=1280 output=1536
    I0126 19:38:10.148629 140401355592064 efficientnet_model.py:457] Building model efficientnet with params ModelConfig(width_coefficient=1.2, depth_coefficient=1.4, resolution=300, dropout_rate=0.3, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='simple_swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=False, data_format='channels_last', dtype='float32')
    WARNING:tensorflow:From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/tensorflow/python/autograph/impl/api.py:464: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with back_prop=False is deprecated and will be removed in a future version.
    Instructions for updating:
    back_prop=False is deprecated. Consider using tf.stop_gradient instead.
    Instead of:
    results = tf.map_fn(fn, elems, back_prop=False)
    Use:
    results = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(fn, elems))
    W0126 19:38:13.148314 140401355592064 deprecation.py:614] From /home/ubuntu-20/anaconda3/envs/tensorflow/lib/python3.9/site-packages/tensorflow/python/autograph/impl/api.py:464: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with back_prop=False is deprecated and will be removed in a future version.
    Instructions for updating:
    back_prop=False is deprecated. Consider using tf.stop_gradient instead.
    Instead of:
    results = tf.map_fn(fn, elems, back_prop=False)
    Use:
    results = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(fn, elems))
    WARNING:tensorflow:Skipping full serialization of Keras layer <object_detection.meta_architectures.ssd_meta_arch.SSDMetaArch object at 0x7fb13074bca0>, because it is not built.
    W0126 19:38:28.202083 140401355592064 save_impl.py:71] Skipping full serialization of Keras layer <object_detection.meta_architectures.ssd_meta_arch.SSDMetaArch object at 0x7fb13074bca0>, because it is not built.
    2022-01-26 19:38:59.669341: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
    W0126 19:39:36.125263 140401355592064 save.py:263] Found untraced functions such as WeightSharedConvolutionalBoxPredictor_layer_call_fn, WeightSharedConvolutionalBoxPredictor_layer_call_and_return_conditional_losses, WeightSharedConvolutionalBoxHead_layer_call_fn, WeightSharedConvolutionalBoxHead_layer_call_and_return_conditional_losses, WeightSharedConvolutionalBoxPredictor_layer_call_fn while saving (showing 5 of 1315). These functions will not be directly callable after loading.
    INFO:tensorflow:Assets written to: exported_models/efficientdet/saved_model/assets
    I0126 19:39:58.307252 140401355592064 builder_impl.py:783] Assets written to: exported_models/efficientdet/saved_model/assets
    INFO:tensorflow:Writing pipeline config file to exported_models/efficientdet/pipeline.config
    I0126 19:39:59.810654 140401355592064 config_util.py:253] Writing pipeline config file to exported_models/efficientdet/pipeline.config



```python

```
