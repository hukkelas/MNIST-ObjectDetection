# MNIST Object Detection dataset

![](example.png)

A simple to generate a dataset for object detection on MNIST numbers.
By default, the script generates a dataset with the following attributes:

- 10,000 images in train. 10,000 images in validation/test
- 10 Classes
- Maximum size of each digit: 100 pixels
- Minimum size of each digit: 15 pixels
- Between 1 and 20 digits per image.


### Installation
Requires python>= 3.6
```bash
pip install -r requirements.txt
```

### Generate dataset 
```bash
python3 generate_dataset.py
```

You can also change a bunch of settings by writing:
```bash
python3 generate_dataset.py -h
```

### Visualize dataset
```bash
python3 visualize_dataset.py data/mnist_detection/train/
.py
```