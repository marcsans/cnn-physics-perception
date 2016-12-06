# cnn-physics-perception

python requirement:
  * theano
  * cv2
  * numpy
  * scipy
  * keras
  * PIL, can not be installed with pip, to be downloaded from [here](http://www.pythonware.com/products/pil/)

Configure keras to use theano : write following in file `.keras/keras.json` :
```
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```

