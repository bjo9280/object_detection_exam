
# coding: utf-8

# In[1]:


import os
import sys
# sys.path.insert(0,os.path.realpath("../slim"))
# sys.path.insert(0,os.path.realpath(".."))


# In[2]:


# from collections import defaultdict
# from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


# In[3]:


import numpy as np
import tensorflow as tf

assert tf.__version__ >= '1.4.0',ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

tag_constants = tf.saved_model.tag_constants


# In[4]:


#get_ipython().run_line_magic('matplotlib', 'inline')


# ## model export

# In[5]:


# %%bash
# OBJECT_DETECTION_CONFIG=./faster_rcnn_resnet101_TTA.config
# YOUR_LOCAL_CHK_DIR=.
# CHECKPOINT_NUMBER=72232
# YOUR_LOCAL_EXPORT_DIR=./export   ## export_dir become ==> ./export/saved_model
# env PYTHONPATH=$(pwd)/..:$(pwd)/../slim:$PYTHONPATH \
# python ../object_detection/export_inference_graph.py \
#     --input_type image_tensor \
#     --pipeline_config_path ${OBJECT_DETECTION_CONFIG} \
#     --trained_checkpoint_prefix ${YOUR_LOCAL_CHK_DIR}/model.ckpt-${CHECKPOINT_NUMBER} \
#     --output_directory ${YOUR_LOCAL_EXPORT_DIR}


# ## Object detection imports
# Here are the imports from the object detection module.

# In[6]:


export_dir = 'export/saved_model' # 'export/Servo/1533281229'


# In[7]:


from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils
from tensorflow.python.tools import saved_model_utils


# In[8]:


saved_model_dir = export_dir
tag_set = tag_constants.SERVING


# In[9]:


meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir,tag_set)
signature_def = meta_graph_def.signature_def['serving_default']


# In[10]:


inputs_map = {k: input_.name for k,input_ in signature_def.inputs.items()}
inputs_map


# In[11]:


outputs_map = {k: output_.name for k,output_ in signature_def.outputs.items()}
outputs_map


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[12]:


PATH_TO_LABELS = './datasetname_label_map.pbtxt'
NUM_CLASSES = 40


# In[13]:


# from object_detection.utils import label_map_util
import label_map_util

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[14]:


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[15]:


tf.reset_default_graph()
config = tf.ConfigProto(gpu_options={'allow_growth':True})
sess = tf.Session(config=config)
tf.saved_model.loader.load(sess, [tag_constants.SERVING], export_dir);
image_tensor_ = tf.get_default_graph().get_tensor_by_name(inputs_map['inputs'])
tensor_dict = {k: tf.get_default_graph().get_tensor_by_name(v) for k, v in outputs_map.items()}


# In[16]:


image_tensor_


# In[17]:


import glob
TEST_IMAGE_PATHS = glob.glob('Test_ImageSet_datasetname/Val_tf/*')


# In[18]:


import visualization_utils as vis_util


# In[19]:


import hashlib
def image_cache_name(filepath):
    return hashlib.md5(filepath.encode('utf-8')).hexdigest() + '.png'
if not os.path.isdir('cache'):
    os.makedirs('cache')


# In[20]:


# dry-run
image_path = TEST_IMAGE_PATHS[0]
image_np = load_image_into_numpy_array(Image.open(image_path))
image_np_expanded = np.expand_dims(image_np, axis=0)
output_dict = sess.run(tensor_dict,feed_dict={image_tensor_: image_np_expanded})


# In[21]:


import time

im_names = TEST_IMAGE_PATHS[:20]

t_start = time.time()
t_inference_elapsed_sum = 0.0
for image_path in im_names:
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(Image.open(image_path))
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Run inference
    t_inference_start = time.time()
    output_dict = sess.run(tensor_dict,feed_dict={image_tensor_: image_np_expanded})
    t_inference_elapsed = (time.time() - t_inference_start)
    print(('Image:',image_path,'t_inference_elapsed',t_inference_elapsed))
    t_inference_elapsed_sum += t_inference_elapsed
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np_expanded[0],
        output_dict['detection_boxes'][0],
        output_dict['detection_classes'][0].astype(int),
        output_dict['detection_scores'][0],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    plt.savefig('cache/'+image_cache_name(image_path))
    plt.close()
t_elapsed = time.time() - t_start
t_elapsed /= len(im_names)
t_inference_elapsed_sum /= len(im_names)
print(('mean t_elapsed',t_elapsed,'mean t_inference_elapsed',t_inference_elapsed_sum))


### # In[22]:
### 
### 
### im_name=TEST_IMAGE_PATHS[0]
### image_cache_name(im_name)
### 
### 
### # In[23]:
### 
### 
### from ipywidgets import interact,IntSlider
### from IPython.display import Image
### 
### @interact(num=IntSlider(0,0,len(im_names)-1,continuous_update=False))
### def show_result(num):
###     im_name = im_names[num]
###     display(Image('cache/' + image_cache_name(im_name)))
### 
