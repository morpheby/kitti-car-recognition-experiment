
# coding: utf-8

import tensorflow as tf
import skimage.draw
import skimage.filters
import numpy as np
import random
import functools
import math
import sklearn
import sklearn.model_selection
import scipy
import bokeh.plotting as p
import bokeh.charts as c
import bokeh.layouts as bl
import tensorflow.contrib
import tensorflow.contrib.losses
from tensorflow.python.client import timeline


# ## 1. Setup task
# 
# We are going to create a set of geometric figures, and the task of the neural net is to correctly identify positions of the circles. Input is grayscale 2D image data, and the output is a 2D matrix with probabilities of a circle being in corresponding rectangular section of the input image.

# In[28]:


def create_dataset(count, f, seed = 42):
    random.seed(seed)
    
    dataset = []
    
    for _ in range(count):
        dataset += [f()]
    
    return dataset

def create_canvas(element_count, size = (512,512), only_circles = False, use_ellipse = False, apply_gauss = False):
    canvas = np.zeros(size)
    circles = []
    if only_circles:
        figures = ['circle']
    elif use_ellipse:
        figures = ['circle', 'rectangle', 'ellipse']
    else:
        figures = ['circle', 'rectangle']
        
    for _ in range(element_count):
        figure = random.choice(figures)
        fill_color = random.random() * 0.8 + 0.2
        border_color = 1.0
        if figure == 'circle':
            radius = random.randint(5, min(size)//3)
            loc_x = random.randint(0, size[0])
            loc_y = random.randint(0, size[1])
            shape_perimeter = skimage.draw.circle_perimeter(loc_y, loc_x, radius, shape = size)
            shape_figure = skimage.draw.circle(loc_y, loc_x, radius, shape = size)
            
            circles += [(loc_x, loc_y, radius)]
        elif figure == 'rectangle':
            size_x = random.randint(5, min(size)//3)
            size_y = random.randint(5, min(size)//3)
            loc_x = random.randint(0, size[0])
            loc_y = random.randint(0, size[1])
            vertices_row = [loc_y-size_y, loc_y-size_y, loc_y+size_y, loc_y+size_y]
            vertices_column = [loc_x-size_x, loc_x+size_x, loc_x+size_x, loc_x-size_x]
            
            shape_perimeter = skimage.draw.polygon_perimeter(vertices_row, vertices_column, shape = size)
            shape_figure = skimage.draw.polygon(vertices_row, vertices_column, shape = size)
        elif figure == 'triangle':
            shape_figure = []
            shape_perimeter = []
            pass
        elif figure == 'ellipse':
            shape_figure = []
            shape_perimeter = []
            pass
        else:
            raise RuntimeError
            
        canvas[shape_figure] = fill_color
        canvas[shape_perimeter] = border_color
    
    if apply_gauss:
        canvas = skimage.filters.gaussian(canvas, sigma=0.5)
            
    return (canvas, circles)
    


# Build a dataset of 500 images.

# In[29]:

dataset_image_shape = (64,64)

dataset_creator_f = functools.partial(create_canvas, 5, size = dataset_image_shape, only_circles = False, apply_gauss = True)
image_dataset = create_dataset(500, dataset_creator_f)


# Prepare result matrix for the data.

# In[30]:


def create_markers_matrix(image_dataset):
    image_dataset_markers = []
    for img,markers in image_dataset:
        factor = 4.0
        target_shape = (int(round(img.shape[0]/factor)), int(round(img.shape[1]/factor)))
        markers_matrix = np.zeros(target_shape)
        for loc_x,loc_y,radius in markers:
            loc_x = loc_x / factor
            loc_y = loc_y / factor
            radius = radius / factor
            def value_at(x,y):
                value = 1.0 - math.sqrt((loc_x-x)**2 + (loc_y-y)**2)/radius
                return value if value > 0 else 0
            rr,cc = skimage.draw.circle(loc_y, loc_x, radius, shape = target_shape)
            for r,c in zip(rr, cc):
                markers_matrix[r,c] += value_at(c,r)
            markers_matrix /= np.max(markers_matrix)
        image_dataset_markers += [markers_matrix]
    return image_dataset_markers

image_dataset_markers = create_markers_matrix(image_dataset)
dataset_output_shape = image_dataset_markers[0].shape


# Split dataset into train and test.

# In[31]:

random.seed(1234)
np.random.seed(1234)

full_dataset = [(img[0].flatten(), matrix.flatten()) for img, matrix in zip(image_dataset, image_dataset_markers)]
full_dataset_train, full_dataset_test =     sklearn.model_selection.train_test_split(full_dataset, test_size = 0.3, random_state = 42)
print(len(full_dataset_train), len(full_dataset_test))

full_dataset_train = zip(*full_dataset_train)
full_dataset_test = zip(*full_dataset_test)

full_dataset_train = [np.array(x) for x in full_dataset_train]
full_dataset_test = [np.array(x) for x in full_dataset_test]


# ## 3. Build Class 1 Neural Network and experiment

# ### Test creating index mapping
# 
# We need to build an index suitable for `tf.SparseTensor` function that will have random non-repeating values.

# In[ ]:


def nn_layer_sizes(factor, input_size, output_size, depth):
    """
        Creates sizes for neural network with reducing sizes specified by factor. Input
    """
    try:
        len(factor)
        spaced_factors = np.array(factor)
    except:
        spaced_factors = np.linspace(1, factor, num=depth, dtype = np.float64)
    layer_sizes = np.asarray(([input_size]*depth) / spaced_factors, dtype=np.int)
    if output_size != None:
        layer_sizes[-1] = output_size
    return layer_sizes

def weight_matrices_shapes_for_layers(layer_sizes):
    shapes = []
    for i in range(1, len(layer_sizes)):
        shape = (layer_sizes[i-1], layer_sizes[i])
        shapes += [shape]
        
    return shapes

def weights_for_shapes(shapes):
    return [np.prod(s) for s in shapes]
    

def sparse_from_dense_factored(dense_shape, fraction):
    """
        Creates indices for sparse matrix from the dense_shape with randomized
        elements. Count of elements present is fraction * prod(dense_shape)
    """
    out_count = math.floor(np.prod(dense_shape) * fraction)
    res_indices = make_random_sparse(out_count, dense_shape)
        
    return res_indices

def weights_sparse_indices(weights_shapes, fractions):
    return [sparse_from_dense_factored(shape, fraction) if fraction != None else None for shape,fraction in zip(weights_shapes,fractions)]

def weights_size_shapes_indices(shapes, indices):
    return [len(idx) if None != idx else np.prod(shape) for shape,idx in zip(shapes,indices)]


# Now let's create a special class to hold configuration that will be easy to use and see.

# In[37]:


class StructMeta(type):
    def __new__(cls, name, bases, attrs):
        return super(StructMeta, cls).__new__(cls, name, bases, attrs)
    
    def __init__(self, name, bases, attrs):
        super(StructMeta, self).__init__(name, bases, attrs)
        
    def _rawlist(self):
        attrs = self._allattrs()
        return [
            f"{attrname}: {attrvalue}" if type(attrvalue) != StructMeta else
            (attrname, attrvalue._rawlist())
            for attrname, attrvalue in attrs.items()
        ]
        
    def __str__(self):
        ls = self._rawlist()
        return self._linetab_str(ls)
    
    def __repr__(self):
        ls = self._rawlist()
        return self._linetab_repr(ls) + "\n}"
    
    def _allattrs(self):
        builtin = ['__module__', '__qualname__', '__doc__']
        return dict([(n,a) for n,a in self.__dict__.items() if n not in builtin])
    
    @staticmethod
    def _linetab_repr(lines, tab = '   ', true_tab = None):
        if true_tab == None: true_tab = tab
        s = "{\n" + true_tab
        s += (', \n' + true_tab).join([
            l if type(l) == str
            else f"{l[0]}: " + StructMeta._linetab_repr(l[1], tab=tab, true_tab=tab+true_tab) + "\n" + true_tab + "}"
            for l in lines
        ])
        return s
    
    @staticmethod
    def _linetab_str(lines, tab = ''):
        return "\n".join([
            tab + l if type(l) == str
            else StructMeta._linetab_str(l[1], tab=f"{tab}{l[0]}.",)
            for l in lines
        ])

class ReprStruct(object, metaclass = StructMeta): pass
    


# ### Class 1A + Gaussian Selectrick Implementation
# 
# Ok, here we intend to add a special layer between layer-1 and layer-2 networoks: gaussian selectrick. We limit the size of the layer-1 network output to a certain number and map it on layer-1 weights using gaussian distribution. $\sigma$ and $\mu$ are declared as variables and will be updated during the training process, yielding a (hopefully) smart enough choice of what weights will need to be mapped.

# In[41]:

class ClassOneSTNN(object):
    
    DTYPE_INT = tf.int32
    DTYPE_FLOAT = tf.float32
    
    def __init__(self, config):
        super(ClassOneSTNN, self).__init__()
        
        self.config = config
    
    def setup_nn(self):
        config = self.config
        
        input_tensor = tf.placeholder(self.DTYPE_FLOAT, shape=(None, config.L1.LAYER_SZ[0]))
        
        sample_size = tf.gather(tf.shape(input_tensor, out_type=self.DTYPE_INT), 0)
        
        with tf.name_scope("layer-1"):
            
            l1_layers = [input_tensor]
            
            for i in range(config.L1.DEPTH-1):
                weights = self.nn_weights_for_layer(config.L1.WEIGHTS_SHAPES[i])
                biases = self.nn_biases_for_layer(config.L1.LAYER_SZ[i+1])
                layer_out, layer_pre = self.nn_layer(l1_layers[-1], weights, biases, act = config.L1.ACTS[i])
                l1_layers += [layer_out]
            
            l1_out = l1_layers[-1]
            l1_preout = layer_pre
        
        with tf.name_scope("selectrick"):
#             lst_out = tf.zeros(tf.stack([sample_size, config.ST.OUTPUT_SIZE], axis=0), dtype=self.DTYPE_FLOAT)
#             lst_map_base = tf.constant(config.ST.MAP, dtype=self.DTYPE_FLOAT)
#             lst_outs = []
            t_adder = tf.transpose(tf.expand_dims(tf.range(0, sample_size), axis=0))
            
            lst_out_idxs = []
            lst_out_vals = []
            lst_out_ds = tf.to_int64(tf.stack([config.ST.INPUT_SIZE, sample_size, config.ST.OUTPUT_SIZE]))
            for i in range(config.ST.INPUT_SIZE):
                weights = self.nn_weights_for_layer(config.ST.SHAPE)
                biases = self.nn_biases_for_layer(config.ST.MEDIAL_SIZE)
                layer_out, layer_pre = self.nn_layer(tf.slice(l1_out, [0,i], [-1, 1]), weights, biases, act = config.ST.ACT)
                
                lst_out_idxs += [
                    tf.to_int64(
                        tf.transpose(tf.stack([
                            tf.tile(tf.constant([i], dtype=self.DTYPE_INT), tf.expand_dims(sample_size*config.ST.MEDIAL_SIZE,0)),
                            tf.tile(tf.constant(config.ST.MAP[i], dtype=self.DTYPE_INT), tf.expand_dims(sample_size, axis=0)),
                            tf.reshape(tf.tile(tf.expand_dims(tf.range(sample_size), axis=1), [1, config.ST.MEDIAL_SIZE]), [-1])
                        ]))
                    )
                ]
                
                lst_out_vals += [tf.reshape(layer_out, [-1])]
                
#                 rev_idx = np.zeros((config.ST.OUTPUT_SIZE), dtype=np.int32) + config.ST.MEDIAL_SIZE
#                 rev_idx[config.ST.MAP[i]] = np.arange(0, config.ST.MEDIAL_SIZE, dtype=np.int32)
#                 fixed_zero = tf.zeros([sample_size, 1], dtype=self.DTYPE_FLOAT)
                
#                 t = tf.concat([layer_out, fixed_zero], axis=1)
#                 old_shape = tf.shape(t)
#                 t = tf.reshape(t, [-1])
#                 idx_reshape = tf.reshape(tf.tile(tf.expand_dims(rev_idx, axis=0), [sample_size, 1]) + t_adder, [-1])
#                 lst_out += tf.reshape(tf.gather(t, idx_reshape), [sample_size, -1])
#                 lst_outs += [lst_out]
#                 lst_preout = layer_out @ lst_map_base
#                 shift_pre = config.ST.MAP_SHIFTS[i]
#                 shift_post = config.ST.OUTPUT_SIZE - (shift_pre + config.ST.PREOUT_SIZE)
#                 if shift_post < 0:
#                     shift_gap = shift_post + shift_pre
#                     lst_o_mid = tf.zeros([sample_size, shift_gap], dtype=self.DTYPE_FLOAT)
#                     lst_o_start = tf.slice(lst_preout, [0, config.ST.PREOUT_SIZE + shift_post], [-1, -1])
#                     lst_o_end = tf.slice(lst_preout, [0, 0], [-1, config.ST.PREOUT_SIZE + shift_post])
#                 else:
#                     lst_o_mid = lst_preout
#                     lst_o_start = tf.zeros([sample_size, shift_pre], dtype=self.DTYPE_FLOAT)
#                     lst_o_end = tf.zeros([sample_size, shift_post], dtype=self.DTYPE_FLOAT)
#                 lst_outs += [tf.concat([lst_o_start, lst_o_mid, lst_o_end], axis=1)]
#                 lst_out += tf.concat([lst_o_start, lst_o_mid, lst_o_end], axis=1)
#             lst_out = tf.add_n(lst_outs)
            lst_outs = tf.SparseTensor(
                tf.concat(lst_out_idxs, axis = 0),
                tf.concat(lst_out_vals, axis = 0),
                lst_out_ds
            )
            lst_out = tf.sparse_reduce_sum(lst_outs, axis=0)
        
        with tf.name_scope("layer-0"):
            l0_layers = [input_tensor]
            
            t = 0
            for i in range(config.L0.DEPTH-1):
                weights_shape = tf.concat([tf.expand_dims(sample_size,0),
                                           tf.constant(config.L0.WEIGHTS_SHAPES[i], dtype=self.DTYPE_INT)],
                                          0)
                weights = tf.reshape(tf.slice(lst_out, (0,t), (-1,config.L0.WEIGHTS_SZ[i])),
                                   weights_shape)
                biases = self.nn_biases_for_layer(config.L0.LAYER_SZ[i+1])
                layer_out, layer_pre = self.nn_layer(tf.expand_dims(l0_layers[-1],1), weights, biases,
                                                act = config.L0.ACTS[i], mul = tf.matmul)
                layer_out = tf.squeeze(layer_out)
                layer_pre = tf.squeeze(layer_pre)
                l0_layers += [layer_out]

                t += config.L0.WEIGHTS_SZ[i]
            
            l0_out = l0_layers[-1]
            l0_preout = layer_pre
            
            self.input_tensor = input_tensor
            self.l1_out = l1_out
            self.l1_preout = l1_preout
            self.l1_layers = l1_layers
            self.lst_out = lst_out
            self.l0_out = l0_out
            self.l0_preout = l0_preout
            self.l0_layers = l0_layers
            
    def setup_error(self, f_error, optimizer, optimizer_args):
        with tf.name_scope("train"):
            train_tensor = tf.placeholder(self.DTYPE_FLOAT, shape=(None, self.config.OUTPUT_SIZE))
            error_tensor = f_error(train_tensor, self.l0_preout)
            minimizer = optimizer(*optimizer_args).minimize(error_tensor)
            
        self.train_tensor = train_tensor
        self.minimizer_tensor = minimizer
        self.error_tensor = error_tensor

    @staticmethod
    def nn_weights_for_layer(shape):
        with tf.name_scope("weights"):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.001, dtype=ClassOneNN.DTYPE_FLOAT))

    @staticmethod
    def nn_biases_for_layer(layer_size):
        with tf.name_scope("biases"):
            return tf.Variable(tf.constant(0.0, shape=[layer_size], dtype=ClassOneNN.DTYPE_FLOAT))

    @staticmethod
    def nn_layer(input_tensor, weights_tensor, biases_tensor, act = tf.sigmoid, mul = tf.matmul):
        preact = mul(input_tensor, weights_tensor) + biases_tensor
        outputs = act(preact)
        return (outputs, preact)


# ### Class 1A + Gaussian Seltrick on MNIST dataset

# In[42]:

random.seed(1234)
np.random.seed(1234)

class NNClassOneST_Cfg_0(ReprStruct):
    
    INPUT_SIZE = 28*28
    OUTPUT_SIZE = 10
    
    class L0(ReprStruct): pass
    class ST(ReprStruct): pass
    class L1(ReprStruct): pass
    
        
    L0.INPUT_SIZE = INPUT_SIZE
    L0.OUTPUT_SIZE = OUTPUT_SIZE
    L0.DEPTH = 2
#     L0.DEPTH = 3
    L0.FACTOR = 1
#     L0.FACTOR = [1, 4, 1]
    L0.LAYER_SZ = nn_layer_sizes(L0.FACTOR, L0.INPUT_SIZE, L0.OUTPUT_SIZE, L0.DEPTH)
    L0.WEIGHTS_SHAPES = weight_matrices_shapes_for_layers(L0.LAYER_SZ)
    L0.ACTS = [tf.identity]
#     L0.ACTS = [tf.nn.relu, tf.identity]
    
    L0.WEIGHTS_SZ = weights_for_shapes(L0.WEIGHTS_SHAPES)
    
    L1.INPUT_SIZE = INPUT_SIZE
    L1.OUTPUT_SIZE = int(sum(L0.WEIGHTS_SZ) * 0.10)
    L1.DEPTH = 2
#     L1.DEPTH = 3
#     L1.DEPTH = 4
    L1.FACTOR = [1, 1]
#     L1.FACTOR = [1, 1./4, 1]
#     L1.FACTOR = [1, 1./4, 1./4, 1]
    L1.LAYER_SZ = nn_layer_sizes(L1.FACTOR, L1.INPUT_SIZE, L1.OUTPUT_SIZE, L1.DEPTH)
    L1.WEIGHTS_SHAPES = weight_matrices_shapes_for_layers(L1.LAYER_SZ)
    L1.ACTS = [tf.identity]
#     L1.ACTS = [tf.nn.sigmoid, tf.identity]
#     L1.ACTS = [tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.tanh]

    ST.INPUT_SIZE = L1.OUTPUT_SIZE
    ST.OUTPUT_SIZE = sum(L0.WEIGHTS_SZ)
    ST.MEDIAL_SIZE = 128
    ST.PREOUT_SIZE = 256
    
    @classmethod
    def _stmapgen(self):
        t = [
            random.sample(range(self.ST.OUTPUT_SIZE), k=self.ST.MEDIAL_SIZE)
            for _ in range(self.ST.INPUT_SIZE)
        ]
        return t
    
    ST.MAP = []
    
#     ST.MAP[range(ST.MEDIAL_SIZE),] = 1
#     ST.MAP_SHIFTS = random.sample(range(ST.OUTPUT_SIZE), k=ST.INPUT_SIZE)
    
    ST.ACT = tf.identity
    ST.SHAPE = (1,ST.MEDIAL_SIZE)

NNClassOneST_Cfg_0.ST.MAP = NNClassOneST_Cfg_0._stmapgen()
    


# In[53]:

NNClassOneST_Cfg_0


# In[45]:

# Reset TF graph environment
try:
    del nn0
    del accuracy
    session.close()
    del session
except:
    pass

tf.reset_default_graph()
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

session = tf.InteractiveSession()


# In[46]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[47]:

def f_error_softmax(train, result):
    return tf.nn.softmax_cross_entropy_with_logits(labels = train, logits = result)

nn1 = ClassOneSTNN(NNClassOneST_Cfg_0)
nn1.setup_nn()
nn1.setup_error(f_error_softmax, tf.train.GradientDescentOptimizer, [0.1])

correct_prediction = tf.equal(tf.argmax(nn1.l0_out,1), tf.argmax(nn1.train_tensor,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[ ]:




# In[ ]:

tf.global_variables_initializer().run()


# In[ ]:

acc_vec = []

for i in range(100):
    if i % 2 == 0:
        print(f"{i//2}%", end=' ')
    if i % 20 == 0:
        acc_vec += [(
            accuracy.eval(feed_dict={nn1.input_tensor: mnist.train.images[:100], nn1.train_tensor: mnist.train.labels[:100]}),
            accuracy.eval(feed_dict={nn1.input_tensor: mnist.test.images[:100], nn1.train_tensor: mnist.test.labels[:100]})
        )]
        print('.', end=' ')
    batch = mnist.train.next_batch(100)
    nn1.minimizer_tensor.run(feed_dict={nn1.input_tensor: batch[0], nn1.train_tensor: batch[1]})

acc_vec += [(
    accuracy.eval(feed_dict={nn1.input_tensor: mnist.train.images[:100], nn1.train_tensor: mnist.train.labels[:100]}),
    accuracy.eval(feed_dict={nn1.input_tensor: mnist.test.images[:100], nn1.train_tensor: mnist.test.labels[:100]})
)]
print('. 100%')


# In[ ]:

run_metadata = tf.RunMetadata()
with open('timeline.ctf.json', 'w') as trace_file:
    batch = mnist.train.next_batch(100)
    session.run(nn1.minimizer_tensor, feed_dict={nn1.input_tensor: batch[0], nn1.train_tensor: batch[1]},
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                run_metadata=run_metadata)

    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
    trace_file.write(trace.generate_chrome_trace_format())



# In[ ]:

res = accuracy.eval(feed_dict={nn1.input_tensor: mnist.test.images[:100], nn1.train_tensor: mnist.test.labels[:100]})
res

print(res)
print(acc_vec)
