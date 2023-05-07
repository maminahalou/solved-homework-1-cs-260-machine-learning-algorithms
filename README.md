Download Link: https://assignmentchef.com/product/solved-homework-1-cs-260-machine-learning-algorithms
<br>
<h1>1           Sequence of Coin Flips</h1>

Suppose you have a <em>biased </em>coin with probability of heads equal to <em>p</em>. Imagine that you flip this coin until observing the first heads. Let <em>X </em>denote the number of flips needed to observe the first heads.

<ol>

 <li>Describe P[<em>X </em>= <em>k</em>] as a function of <em>p</em>.</li>

 <li>For any <em>x</em><sub>0 </sub>1, show that the probability P[<em>X    x</em><sub>0</sub>] = (1  <em>p</em>)<em><sup>x</sup></em><sup>0 1</sup>.</li>

 <li>Suppose we have a prior belief that <em>p </em>is uniformly distributed in the [0<em>,</em>1] interval. Assuming that the first coin flip equals heads, compute the probability that <em>p &gt; </em>1<em>/</em>2, i.e., compute P[<em>p &gt; </em>1<em>/</em>2|<em>X </em>= 1]. Does our belief about the probability of the event {<em>p &gt; </em>1<em>/</em>2} increase or decrease after observing the head in the first flip. Hint: Use Bayes Rule.</li>

</ol>

<h1>2           Convex Functions and Information Theory</h1>

<ol>

 <li>Show that the function <em>f</em>(<em>x</em>) = |<em>x</em>| + exp(<em>x</em>) is convex.</li>

 <li>Suppose the random variable <em>X </em>is distributed according to a <em>k</em>-class multi-nominal distributions with class probabilities <em>p</em><sub>1</sub><em>,p</em><sub>2</sub><em>,…,p<sub>k</sub></em>, such that = 1. Find the values of <em>p<sub>i</sub>,i </em>= 1<em>,…,k </em>such that the <a href="https://en.wikipedia.org/wiki/Entropy_(information_theory)#Definition">entropy</a> of <em>X </em>is maximized.</li>

</ol>

<h1>3           Linear Algebra</h1>

<ol>

 <li>The covariance matrix <strong>⌃ </strong>of a random vector <em>X </em>is defined as <strong>⌃ </strong>= E[(<em>X </em>E<em>X</em>)(<em>X </em>E<em>X</em>)<sup>T</sup>], where E<em>X </em>is the expectation of <em>X</em>. Is <strong>⌃ </strong><a href="https://en.wikipedia.org/wiki/Positive-semidefinite_matrix#Negative-definite.2C_semidefinite_and_indefinite_matrices">positive-semidefinite</a><a href="https://en.wikipedia.org/wiki/Positive-semidefinite_matrix#Negative-definite.2C_semidefinite_and_indefinite_matrices">?</a></li>

 <li>Let <em>A </em>and <em>B </em>be two R<sup>D</sup><sup>⇥</sup><sup>D </sup>symmetric matrices. Suppose <em>A </em>and <em>B </em>have the exact same set of <a href="https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors">eigen</a><a href="https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors">vectors</a> <em>u</em><sub>1</sub><em>,</em><em>u</em><sub>2</sub><em>,</em>·· <em>,</em><em>u</em><sub>D </sub>with the corresponding eigenvalues <em>↵</em><sub>1</sub><em>,↵</em><sub>2</sub><em>,</em>··· <em>,↵</em><sub>D </sub>for <em>A</em>, and <sub>1</sub><em>, </em><sub>2</sub><em>,</em>··· <em>, </em><sub>D </sub>for <em>B</em>. Please write down the eigenvectors and their corresponding eigenvalues for the following matrices:

  <ul>

   <li><em>C </em>= <em>A </em>+ <em>B</em></li>

   <li><em>D </em>= <em>A B</em></li>

   <li><em>E </em>= <em>AB</em></li>

   <li><em>F </em>= <em>A </em><sup>1</sup><em>B </em>(assume <em>A </em>is invertible)</li>

  </ul></li>

</ol>

1

<h1>4           KNN Classification in MATLAB/Octave</h1>

In this problem, you will implement a KNN classifier and deploy it on a real-world dataset. Below, we describe the steps that you need to take to accomplish this programming assignment.

You will work with a preprocessed version of the <em>Car Evaluation Dataset </em>from UCI’s machine learning data repository. The training/validation/test sets are provided along with the assignment as cars train.data, cars valid.data, and cars test.data. For a description of the dataset and to determine which field corresponds to the label, please refer to <a href="https://archive.ics.uci.edu/ml/datasets/Car+Evaluation">http://archive.ics.uci.edu/ml/datasets/Car+Evaluation</a><a href="https://archive.ics.uci.edu/ml/datasets/Car+Evaluation">.</a>

<ol>

 <li>The first step in every data analysis experiment involves inspecting the data and to make sure it is properly formatted. You will find that the features in the provided dataset are categorical. However, KNN requires the features to be real-valued numbers. To convert a categorical feature with <em>K </em>categories to a real-valued number, you can create <em>K </em>new <em>binary </em> The <em>i</em>th binary feature indicates whether the original feature belongs to the <em>i</em>th category or not. This strategy is called ‘one-hot encoding.’</li>

 <li>Please fill in the function knn classify in knn m. The inputs of this function are training data, new data (either validation or testing data) and <em>k</em>. The function needs to output the accuracy on both training and new data (either validation or testing).</li>

 <li>Consider <em>k </em>= 1<em>,</em>3<em>,</em>5<em>,</em>·· <em>,</em>23. For each <em>k</em>, report the training and validation accuracy. Identify the <em>k </em>with the highest validation accuracy, and report the test accuracy with this choice of <em>k</em>. Note: if multiple values of <em>k </em>result in the highest validation accuracy, then report test accuracies for all such values of <em>k</em>.</li>

 <li>Apply <em>k</em>NN on the mat dataset which is a binary classification dataset with only two features. You need to run <em>k</em>NN with <em>k </em>= 1<em>,</em>5<em>,</em>15<em>,</em>20 and examine the decision boundary. A simple way to visualize the decision boundary is to draw 10000 data points on a uniform 100 ⇥ 100 grid in the square (<em>x,y</em>) 2 [0<em>,</em>1] ⇥ [0<em>,</em>1] and classify them using the <em>k</em>NN classifier. Then, plot the data points with di↵erent markers corresponding to di↵erent classes. Repeat this process for all <em>k </em>and discuss the smoothness of the decision boundaries as <em>k </em>increases.</li>

</ol>


