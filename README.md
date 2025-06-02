<h1 align="center">Intels Natural Scenes Image Classification</h1>

<h2>Description:</h2>

<p>
After the failure of my 2048 Q-Learning project https://github.com/Bloodaxe90/2048-Q-Learning, I realized it finally needed to start learning about neural networks.
</p>

<p>
This is the first project I worked on after learning PyTorch. I built a deep convolutional neural network for image classification, trained on the <a href="https://www.kaggle.com/datasets/puneet6060/intel-image-classification">Intel Image Classification</a> dataset from Kaggle. The model learns to classify natural scene images into the following six categories:
</p>

<ol>
  <li>Buildings</li>
  <li>Forests</li>
  <li>Glaciers</li>
  <li>Mountains</li>
  <li>Oceans</li>
  <li>Streets</li>
</ol>

<p>
In addition to the model, I implemented early stopping, a custom dataset class, and inference methods like confusion matrices and basic hyperparameter tuning.
</p>

<h2>Usage:</h2>
<ol>
  <li>Activate a virtual environment.</li>
  <li>Run <code>pip install -r requirements.txt</code> to install the dependencies.</li>
  <li>Download the <a href="https://www.kaggle.com/datasets/puneet6060/intel-image-classification">Intel Image Classification</a> dataset from Kaggle and place it in the main directory of the project.</li>
  <li>Run <code>initial_setup.py</code> to prepare the dataset and directories.</li>
  <li>Run <code>main.py</code> to train and evaluate the model.</li>
</ol>


<h2>Hyperparameters:</h2>
<p>All hyperparameters can be found in <code>main.py</code>.</p>
<ul>
  <li><code>INFERENCE</code> (bool): Enables inference mode. (I hadn't yet discovered Jupyter Notebooks.)</li>
  <li><code>EPOCHS</code> (int): Number of training epochs.</li>
  <li>
    <code>NEURONS_PER_HIDDEN_LAYER</code> (list[int]): Defines the number of neurons in each hidden layer. 
    The number of hidden layers is <code>len(NEURONS_PER_HIDDEN_LAYER) - 1</code>. 
    For example, <code>[128, 64, 32]</code> gives two hidden layers: the first with 128 input and 64 output neurons, 
    and the second with 64 input and 32 output neurons.
  </li>
  <li><code>LEARNING_RATE</code> (float): Learning rate used by the optimizer.</li>
  <li><code>PATIENCE</code> (int): Number of epochs to wait for a significant improvement (as defined by <code>MIN_DELTA</code>) before early stopping is triggered.</li>
  <li><code>MIN_DELTA</code> (float): Minimum improvement in validation loss required to be seen as progress.</li>
  <li><code>NEW_MODEL_NAME</code> (str): Name of a new/existing model you want to train.</li>
  <li><code>LOAD_MODEL_NAME</code> (str): Name of an existing model to load for inference.</li>
</ul>

<h2>Results:</h2>
<p>
After training with early stopping for 30 epochs the model learnt to successfully classify the test data.
</p>
<p>
<strong>Below are some statistics of the models training results:</strong>
<ul>
  <li>

![myplot](https://github.com/user-attachments/assets/02ebbda2-2be8-4520-a322-8dbba2541679)
  </br> Some of the trained models predictions on 9 random images from the test data
  </li>
  <li>

![plot_2025-06-02 15-00-50_2](https://github.com/user-attachments/assets/a526b11b-f743-4844-93c6-6ecbeeebff35)
  </br> Confusion matrix of the trained models perfomance on the test data
  </li>
  <li>  

![plot_2025-06-02 15-00-50_0](https://github.com/user-attachments/assets/851a50d3-3d5c-499d-b3d9-ba4c333ea431)
  </br> The train and test loss and accuracy during training 
  </li>
</ul>
</p>




