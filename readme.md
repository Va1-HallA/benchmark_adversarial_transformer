<h1>NOTICE:LIBRARY HAS BEEN DELETED, PLEASE SET UP EVERYTHING AGAIN BEFORE USE!</h1>

<h1>Introduction</h1>
<p>This project is a experimental framework that evaluate 
the robustness of transformer models against adversarial 
attacks.</p>
<p>Details of how to use this project is explained below.</p>

<h1>Set Up</h1>
<p>This project is based on Python 3.9</p>
<p>The main materials of this project are: </p>
<ol>
<li>PyTorch Image Models (TIMM)</li>
<li>Adversarial Robustness Toolbox (ART)</li>
<li>MNIST dataset (included in ART)</li>
<li>ILSVRC2012 validation dataset</li>
</ol>
<p>For full requiremnets, please refer to requirements.txt.
The full guidance of setting up this project is as follows: </p>

<ol>
<li>In project root directory, run: pip install requirements.txt</li>
<li>In art.attacks.pixel_threshold.py, change "from scipy.optimize.optimize import _status_message"
 to "from scipy.optimize._optimize import _status_message"</li>
<li>Move "projected_gradient_descent_pytorch.py" in "modified_art_attacks" to "/venv_location/Lib/site-packages/art/attacks/evasion/projected_gradient_descent", 
replace the original file (assuming the user uses virtual environment, if not, move to where the libraries stored)</li>
<li>Move the other two files in "modified_art_attacks" to "/venv_location/Lib/site-packages/art/attacks/evasion", 
replace the original files (assuming the user uses virtual environment, if not, move to where the libraries stored)</li>
<li>Go to the imagenet official website (URL: https://www.image-net.org/)
, download the source images of ILSVRC2012 validation dataset. Move the 
images to "/data/validation/images" directory. All other required files are
ready to use.</li>
</ol>

<h1>How To Run the Experiment</h1>
<p>To reproduce the result in my report, do the following steps: </p>
<ol>
<li>Simply run the experiment.py file to get the experimental results</li>
<li>Run main.ipynb to get the visualized results</li>
<li>Note that the runtime for each attack is in generating_time.txt in the root directory</li>
</ol>