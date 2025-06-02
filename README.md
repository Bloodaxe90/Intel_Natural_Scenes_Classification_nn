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
  <li>Download the <a href="https://www.kaggle.com/datasets/puneet6060/intel-image-classification">Intel Image Classification</a> dataset from kaggle and add it to the main directory of the project</li>
  <li>Run <code>Initial</code>.</li>
</ol>

<h2>Hyperparameters:</h2>
<p>All can be found in <code>QLearning</code></p>
<ul>
  <li><code>GAMMA</code> (float): Discount factor</li>
  <li><code>MAX_ALPHA</code> (float): Initial learning rate</li>
  <li><code>MIN_ALPHA</code> (float): Minimum learning rate</li>
  <li><code>MAX_EPSILON</code> (float): Initial exploration rate</li>
  <li><code>MIN_EPSILON</code> (float): Minimum exploration rate</li>
  <li><code>LAMBDA</code> (float): Decay rate for eligibility trace</li>
</ul>

<h2>Controls:</h2>
<p>
  Its TicTacToe!
</p>
<ul>
    <li><strong>Mouse Click:</strong> Places a piece</li>
    <li><strong>Radio Buttons:</strong>
      <ol>
        <li>Playing against another player </li>
        <li>Playing against the Minimax AI </li>
        <li>Playing against the Q-Learning Agent</li>
      </ol>
    </li>
    <li><strong>Train Button:</strong> Trains the AI for the input number of episodes (Disables everything until training is over (Multi-processing not implemented))</li>
</ul>

<h2>Results:</h2>
<p>
After training for 1 million episodes (which I understand is an unnecessary amount, but I wanted to make 100% sure the agents learned the target policy) the Q-Learning agent successfully learned how to consistently win or draw against me player.
</p>
<p>
<strong>Below are some of the statistics from training:</strong>
<ul>
  <li>  
    
  ![image](https://github.com/user-attachments/assets/17981ef4-e450-458b-8a56-c2cbd325f7ef)
  </br> In the above image you can see that eventually the agents began drawing more than they won against one another
  </li>
  <li>
    
  ![image](https://github.com/user-attachments/assets/3c124e79-783b-4603-9e02-942ef915e6a2)
  </br> In the above image you can see how the learning rate and exploration rate decay with each cumulative eisode
  </li>
  <li>
    
  ![image](https://github.com/user-attachments/assets/59207f7c-db7e-4c4c-9ec9-d1d1f091c79e)
  </br> In the above image you can see that very quickly the agents discover close to all possible state-action pairs which is expected due to TicTacToes small state-action space
  </li>
</ul>
</p>

Finally here is a screenshot of the final UI
![image](https://github.com/user-attachments/assets/662e8790-f892-4a0b-8b14-23d18383277a)


