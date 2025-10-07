# AI and Machine Learning Jargon

A playful, practitioner-first glossary of AI/ML terms I actually use. Plain English, punchy definitions, and quick mental models—sprinkled with Rick & Morty asides when things get weird. Built for real work: code reviews, design docs, and late‑night debugging with a portal gun in one hand and coffee in the other.

Minimal fluff, maximum signal. When it helps, I drop tiny Python snippets, gotchas, and the occasional “don’t overfit, Morty” so you can apply ideas immediately.

<img width="1024" height="1024" alt="Gemini_Generated_Image_o1sm9bo1sm9bo1sm" src="https://github.com/user-attachments/assets/8ee1fffd-4056-4cce-a54f-c18d72e360c4" />


__Table of Contents__
<!-- RM(noparent,notop) -->

* [A/B Test / Canary / Shadow](#ab-test--canary--shadow)
* [Accuracy, Precision, Recall, F1](#accuracy-precision-recall-f1)
* [Activation Function](#activation-function)
* [Adversarial Examples / Robustness](#adversarial-examples--robustness)
* [Attention](#attention)
* [Autoencoder](#autoencoder)
* [Backpropagation](#backpropagation)
* [Batch Norm / Layer Norm](#batch-norm--layer-norm)
* [Batch / Batch Size](#batch--batch-size)
* [Bayesian Inference](#bayesian-inference)
* [Beam Search](#beam-search)
* [Bias-Variance Trade-off](#bias-variance-trade-off)
* [Code Examples](#code-examples)
* [Confusion Matrix](#confusion-matrix)
* [Context Window](#context-window)
* [Cross-Validation](#cross-validation)
* [Data Augmentation](#data-augmentation)
* [Dataset](#dataset)
* [Dimensionality Reduction](#dimensionality-reduction)
* [Diffusion Model](#diffusion-model)
* [Dropout](#dropout)
* [Embeddings](#embeddings)
* [Entropy / KL Divergence](#entropy--kl-divergence)
* [Epoch](#epoch)
* [Exploration vs. Exploitation](#exploration-vs-exploitation)
* [Fairness / Bias / Explainability](#fairness--bias--explainability)
* [Feature](#feature)
* [Feature Engineering](#feature-engineering)
* [Feature Store](#feature-store)
* [Fine-Tuning](#fine-tuning)
* [GAN](#gan)
* [Gradient](#gradient)
* [Gradient Descent](#gradient-descent)
* [Hyperparameters](#hyperparameters)
* [Inference](#inference)
* [Label](#label)
* [Latency / Throughput](#latency--throughput)
* [Learning Rate](#learning-rate)
* [Linear / Logistic Regression](#linear--logistic-regression)
* [LLM](#llm)
* [Logits](#logits)
* [Loss Function](#loss-function)
* [MAP / MLE](#map--mle)
* [MCMC / Variational Inference](#mcmc--variational-inference)
* [Markov Chains / HMM](#markov-chains--hmm)
* [MLOps](#mlops)
* [Model](#model)
* [Model Distillation](#model-distillation)
* [Model Registry](#model-registry)
* [Naive Bayes](#naive-bayes)
* [Normalization / Standardization](#normalization--standardization)
* [Objective Function](#objective-function)
* [One-Hot Encoding](#one-hot-encoding)
* [Optimizer](#optimizer)
* [Overfitting](#overfitting)
* [PCA / t-SNE / UMAP](#pca--t-sne--umap)
* [Perplexity](#perplexity)
* [Policy / Value Function](#policy--value-function)
* [Prompt](#prompt)
* [Pruning / Quantization](#pruning--quantization)
* [Q-Learning / TD Learning](#q-learning--td-learning)
* [Random Forest / XGBoost](#random-forest--xgboost)
* [ROC / AUC](#roc--auc)
* [SVM / k-NN / Decision Trees](#svm--k-nn--decision-trees)
* [Softmax](#softmax)
* [Supervised Learning](#supervised-learning)
* [Temperature / Top-p](#temperature--top-p)
* [Tokenization](#tokenization)
* [Transfer Learning](#transfer-learning)
* [Transformer](#transformer)
* [Train/Validation/Test Split](#trainvalidationtest-split)
* [Underfitting](#underfitting)
* [Unsupervised Learning](#unsupervised-learning)
* [VAE](#vae)
* [Weight Decay](#weight-decay)

<!-- /RM -->

## Supervised Learning

Learning with labeled data: the model maps inputs to known target outputs (classification, regression). The goal is to minimize a loss that measures prediction error.

<details>
<summary>Rick & Morty: What's this?</summary>

Morty: "So supervised learning is like having the answers on the back of the book?"<br>

Rick: "Yeah, Morty. Labeled inputs, known targets, minimize a loss. Training wheels for pattern‑matching—now let's not flunk the cosmos."<br>

Morty: "Okay, so supervised means we already know the answers?"<br>

Rick: "Labeled input–output pairs, Morty. The model learns a mapping that minimizes a loss. Think of it like teaching someone to recognize cats by showing them thousands of pictures labeled 'cat' and 'not cat'. The algorithm finds patterns in the pixels that correlate with the labels."<br>

Morty: "And the loss is like a score of how wrong we are?"<br>

Rick: "Exactly. We adjust weights to make that score smaller until generalization stops improving. Lower loss means better predictions, but don't get cocky—what matters is how well it works on new, unseen data."<br>

Morty: "Any gotchas?"<br>

Rick: "Overfitting. Regularize, validate, and stop before you memorize the homework key. The model might learn to recognize the exact training examples instead of the underlying patterns. It's like memorizing answers without understanding the concepts—works great on the practice test, fails spectacularly on the real exam."<br>

Morty: "So we need to test it on data it's never seen?"<br>

Rick: "That's the validation set, Morty. Keep some data hidden during training, then see if the model can handle surprises. If training accuracy is high but validation accuracy tanks, you've got yourself a classic overfitting situation."<br>

</details>


## Unsupervised Learning

Learning patterns from unlabeled data (e.g., clustering, density modeling, dimensionality reduction). Often used for exploration or as preprocessing.

<details>
<summary>Rick & Morty: What's this?</summary>

Morty: "Uh, so… no labels?"<br>

Rick: "Right. We toss data into a void and let it find structure—clusters, manifolds, whatever. Curiosity without supervision, Morty."<br>

Morty: "No labels… so what are we learning?"<br>

Rick: "Structure, Morty. Groups, manifolds, directions of variance—whatever patterns float to the top. Imagine you have a bunch of customer data but no idea what makes them different. Unsupervised learning might discover that some customers buy luxury items while others are bargain hunters, even though nobody told it to look for that."<br>

Morty: "How do we check if it's good?"<br>

Rick: "Qualitative checks, downstream performance, or metrics like silhouette. Don't expect a single 'right' answer. Unlike supervised learning where you can check against ground truth, here you're exploring the unknown. Maybe the clusters make business sense, maybe they reveal hidden customer segments, or maybe they're just mathematical artifacts."<br>

Morty: "So it's like exploring a new dimension?"<br>

Rick: "Yeah, and trying not to get eaten by your own assumptions. The algorithm might find patterns that are real but useless, or useful but not obvious. It's like being an explorer without a map—you might discover treasure or just end up lost in a swamp of irrelevant correlations."<br>

Morty: "When would we actually use this?"<br>

Rick: "Data exploration, feature engineering, anomaly detection, or when you need to understand your data before building supervised models. It's reconnaissance for your data science mission, Morty."<br>

</details>

 

## Reinforcement Learning

Learning through trial and error by receiving rewards for actions taken in an environment; aims to learn a policy that maximizes expected return.

<details>
<summary>Rick & Morty: What's this?</summary>

Morty: "We get points for doing good stuff?"<br>

Rick: "Rewards, Morty. Agent learns a policy to maximize return by trial and error. It's like arcade tokens but with Bellman equations."<br>

Morty: "We just try stuff and see what pays off?"<br>

Rick: "Agent, environment, rewards. Learn a policy that maximizes expected return, Morty. Think of it like learning to play a video game—you don't know the rules at first, but you get points for good moves and lose points for bad ones. Eventually you figure out the strategy that gets you the highest score."<br>

Morty: "Do we plan ahead or just react?"<br>

Rick: "Both—value functions evaluate, policies act. Bootstrapping stitches it together. The value function is like your inner voice saying 'this situation looks promising' or 'this is probably a bad idea.' The policy is your actual decision-making process. Sometimes you plan several moves ahead, sometimes you just react."<br>

Morty: "And exploration?"<br>

Rick: "Essential. Otherwise you get stuck milking mediocre rewards forever. It's the classic explore-exploit dilemma—do you keep doing what you know works, or try something new that might work better? Most RL algorithms have some mechanism to encourage trying new things, otherwise they get stuck in local optima like a rat hitting the same lever."<br>

Morty: "So it's like learning to ride a bike?"<br>

Rick: "Exactly, but the bike is on fire and the road keeps changing. You start terrible, crash a lot, but gradually learn what actions lead to staying upright. The reward signal is your main teacher—pain when you fall, satisfaction when you succeed."<br>

</details>

 

## Model

A function with learnable parameters that transforms inputs into predictions or decisions. Can be parametric (fixed-size, e.g., linear regression) or nonparametric (flexible size, e.g., k-NN).

<details>
<summary>Rick & Morty: What's this?</summary>

Morty: "A model is like a brain?"<br>

Rick: "A function, Morty. Parameters in, predictions out. Some are simple, some are monstrosities. All are hungry for data."<br>

Morty: "So a model is like a brain-in-a-jar?"<br>

Rick: "It's a function with tunable parameters, Morty. Some tiny, some so big they write papers. A linear regression model might have just a few parameters—maybe one for each input feature plus a bias term. A large language model might have billions of parameters, each one a tiny weight that helps the model understand language."<br>

Morty: "How do we pick one?"<br>

Rick: "Start simple, watch validation, escalate complexity only if it earns its keep. Begin with something like logistic regression—it's interpretable and fast. If it's not performing well enough, maybe try a random forest. Still not good enough? Neural networks. But each step up the complexity ladder means more data requirements, longer training times, and harder debugging."<br>

Morty: "And k‑NN isn't even training?"<br>

Rick: "Lazy learning. You store data and suffer at query time. k-NN doesn't build a model during training—it just memorizes all the examples. When you ask for a prediction, it searches through all stored examples to find the k nearest neighbors and averages their labels. Fast training, slow inference."<br>

Morty: "So bigger models are always better?"<br>

Rick: "Hell no, Morty. Bigger models overfit easier, cost more to run, and are harder to understand. Sometimes a simple model that captures the main pattern is better than a complex one that memorizes noise. It's about finding the sweet spot between underfitting and overfitting."<br>

</details>

## Parameters

Internal values learned from data (e.g., weights in neural networks, coefficients in linear models). Adjusted by optimizers using gradients.

<details>
<summary>Rick & Morty: What's this?</summary>

Morty: "Are parameters the knobs?"<br>

Rick: "They're the weights, Morty. The numbers the optimizer cranks so the model stops embarrassing itself."<br>

Morty: "Parameters change during training, right?"<br>

Rick: "Yeah. Gradients nudge them toward lower loss each update. Think of parameters as the model's memory—they encode everything the model has learned about the patterns in your data. In a neural network, they're the connection strengths between neurons."<br>

Morty: "Do all parameters get updated the same?"<br>

Rick: "Not necessarily—different layers, schedules, and weight decay can treat them differently. Early layers in a neural network might learn slowly with small learning rates, while the final layer learns faster. Some parameters might be frozen during transfer learning, others might have different regularization."<br>

Morty: "So they're the dials the model learns to twist?"<br>

Rick: "Exactly, Morty. Imagine the model as a complex machine with millions of tiny dials. During training, the optimizer figures out how to adjust each dial to make better predictions. Some dials barely move, others get cranked hard. The art is in getting them all to work together."<br>

Morty: "How many parameters do models usually have?"<br>

Rick: "Ranges from dozens to trillions. A simple linear model might have one parameter per feature. GPT-3 has 175 billion parameters. More parameters mean more capacity to learn complex patterns, but also more risk of overfitting and higher computational costs."<br>

</details>

## Hyperparameters

External configuration values set before training (e.g., learning rate, tree depth, regularization strength). Tuned via validation or cross-validation.

<details>
<summary>Rick & Morty: What's this?</summary>

Morty: "So these are settings we pick?"<br>

Rick: "Yeah. Learning rate, depth, regularization. You tune 'em or watch your model crash like a rickety portal gun."<br>

Morty: "We pick hyperparameters before we start?"<br>

Rick: "Yep. They shape the learning process—too spicy or too bland ruins dinner. Think of hyperparameters as the recipe settings for cooking your model. Learning rate is how fast you turn up the heat, batch size is how much you cook at once, number of layers is how complex your dish gets."<br>

Morty: "How do we choose them?"<br>

Rick: "Search: grid, random, Bayesian, or bandits. Validate honestly, avoid leakage. Grid search tries every combination like a methodical scientist. Random search is like throwing darts—surprisingly effective and way faster. Bayesian optimization is the smart approach that learns from previous attempts."<br>

Morty: "And automate it?"<br>

Rick: "If you like compute bills, sure. Automated hyperparameter tuning can burn through cloud credits faster than Jerry burns through excuses. But it's often worth it—the difference between good and great hyperparameters can make or break your model."<br>

Morty: "What happens if we get them wrong?"<br>

Rick: "Learning rate too high? Your loss function bounces around like a ping-pong ball. Too low? Training takes forever and might never converge. Wrong architecture depth? Either underfitting or overfitting hell. It's like tuning a musical instrument—everything has to be just right for harmony."<br>

</details>

## Feature

An input variable describing aspects of the data (e.g., age, pixels). Features can be raw or engineered; quality strongly affects model performance.

<details>
<summary>Rick & Morty: What's this?</summary>

Morty: "Features are like clues?"<br>

Rick: "Inputs, Morty. Pixels, ages, frequencies. Better clues, better models—unless you leak future info and blow up reality."<br>

Morty: "Features are our clues to the answer?"<br>

Rick: "Clues and context. Good ones make learning easy; bad ones make models hallucinate. Think of features as the sensory inputs for your model—they're how the algorithm perceives the world. In image recognition, features might be pixel values, edges, or textures. For predicting house prices, features could be square footage, neighborhood, and number of bedrooms."<br>

Morty: "So engineering matters?"<br>

Rick: "Often more than model choice, Morty. But don't leak the future. A brilliant feature engineer can make a simple linear model outperform a fancy neural network with bad features. Features are where domain expertise meets machine learning—knowing what matters in your problem space."<br>

Morty: "Leak the… future?"<br>

Rick: "Using info unavailable at prediction time. Multiverse-breaking mistake. Like using tomorrow's stock price to predict today's stock price—technically perfect accuracy, completely useless in practice. Always ask: 'Would I have this information when making real predictions?'"<br>

Morty: "What makes a good feature?"<br>

Rick: "Relevance, availability, and stability, Morty. It should be correlated with your target, available when you need predictions, and not change meaning over time. A feature that's perfect in your training data but unavailable in production is worse than useless—it's a trap."<br>

</details>

## Label

The target variable the model is trained to predict. Can be numeric (regression) or categorical (classification).

<details>
<summary>Rick & Morty: What's this?</summary>

Morty: "Labels are answers?"<br>

Rick: "Targets, Morty. The thing you're trying to predict. Numbers or categories. No label, no supervision."<br>

Morty: "Labels are the answers we compare to?"<br>

Rick: "Exactly. Noisy labels mislead, so curate. Garbage in, garbage universes out. Labels are your ground truth—what you want the model to learn to predict. In spam detection, labels are 'spam' or 'not spam'. In house pricing, the label is the actual sale price."<br>

Morty: "What about class imbalance?"<br>

Rick: "Reweight, resample, or tune thresholds—metrics must match reality. If 99% of your emails are not spam, your model will just predict 'not spam' for everything and be 99% accurate but completely useless. You need to balance the training data or adjust how you evaluate performance."<br>

Morty: "So labels can be the bottleneck?"<br>

Rick: "Frequently, Morty. Getting high-quality labels is often the hardest part of machine learning. Think medical diagnosis—you need expert doctors to label X-rays, which is expensive and time-consuming. Bad labels create bad models, period."<br>

Morty: "How do we get better labels?"<br>

Rick: "Multiple annotators, clear guidelines, quality checks, and sometimes active learning where the model asks for labels on the most uncertain examples. And always remember—your model can only be as good as your labels allow it to be."<br>

</details>

## Dataset

Collection of examples used to train and evaluate models, typically split into train/validation/test sets to measure generalization.

<details>
<summary>Rick & Morty: What's this?</summary>

Morty: "A dataset is just… a bunch of examples?"<br>

Rick: "Collections of reality slices, Morty. Train, validate, test—don't mix 'em or you'll contaminate timelines."<br>

Morty: "Dataset equals rows of reality?"<br>

Rick: "Rows and context. Quality, coverage, and drift resistance matter more than size alone. A dataset is your model's entire understanding of the world. If it's biased, incomplete, or outdated, your model will be too. Think of it as a representative sample of all possible situations your model might encounter."<br>

Morty: "How do we keep it clean?"<br>

Rick: "Version it, audit it, and track provenance like it's plutonium. Every time you update your dataset, version it. Know where each data point came from, when it was collected, and how it was processed. Data lineage is crucial for debugging and compliance."<br>

Morty: "And splits?"<br>

Rick: "Disjoint, time‑aware when needed, consistent preprocessing across them. Your train, validation, and test sets should never overlap—that's cheating. For time series data, respect temporal order. And whatever preprocessing you do to training data, do exactly the same to validation and test data."<br>

Morty: "How big should datasets be?"<br>

Rick: "Depends on the complexity, Morty. Simple problems might need hundreds of examples, complex deep learning might need millions. But remember: 1000 high-quality, relevant examples often beat 100,000 noisy, irrelevant ones."<br>

</details>

## Train/Validation/Test Split

Common split to train models, tune hyperparameters, and estimate generalization. Avoid leakage by ensuring disjoint splits and consistent preprocessing.

<details>
<summary>Rick & Morty: What's this?</summary>

Morty: "We divide the data into three piles?"<br>

Rick: "Train to learn, validation to tune, test to judge. Keep 'em disjoint or you're just flattering yourself, Morty."<br>

Morty: "Three piles, three purposes?"<br>

Rick: "Train learns, validation tunes, test judges. Keep them separate or you flatter yourself. Training data teaches the model patterns, validation data helps you pick the best hyperparameters, and test data gives you an honest assessment of real-world performance."<br>

Morty: "Time series, too?"<br>

Rick: "Use temporal splits. Random shuffles can lie to you, Morty. With time series, you can't randomly mix past and future—always split chronologically. Train on older data, validate on recent data, test on the most recent data. Otherwise you're cheating by using future information."<br>

Morty: "And never peek?"<br>

Rick: "Never. The test set is sacred, Morty. Look at it once, at the very end, after everything else is finalized. If you keep peeking and adjusting based on test performance, you're essentially training on your test set."<br>

Morty: "What about the split ratios?"<br>

Rick: "Common rule is 60/20/20 or 70/15/15, but it depends on your data size. With millions of examples, you can get away with smaller validation and test sets. With hundreds, you might need larger portions to get reliable estimates."<br>

</details>

## Overfitting

Model fits noise or spurious patterns in training data and performs poorly on unseen data. Mitigate with regularization, augmentation, and early stopping.

<details>
<summary>Rick & Morty: What's this?</summary>

Morty: "Overfitting is memorizing, right?"<br>

Rick: "Yeah. The model writes the answers on its arm and fails the real exam. Regularize, augment, stop early, Morty."<br>

Morty: "Memorizing the homework key, huh?"<br>

Rick: "Yeah. Low training error, high test regret. The model learns the training data so well that it memorizes irrelevant details and noise instead of the underlying patterns. It's like a student who memorizes the exact wording of practice problems but can't solve new ones with different wording."<br>

Morty: "How do we fix it?"<br>

Rick: "Simplify, regularize, augment, or get more data—and watch validation like a hawk. Use a simpler model, add regularization penalties, augment your training data with variations, or collect more diverse examples. The key is monitoring validation performance—when it starts getting worse while training improves, you're overfitting."<br>

Morty: "Early stopping?"<br>

Rick: "Classic and effective. Stop training when validation performance plateaus or starts degrading, even if training performance could still improve. Save the model state from when validation was best, not when training finished."<br>

Morty: "How do we spot it?"<br>

Rick: "Large gap between training and validation performance, Morty. If your model gets 99% accuracy on training data but only 70% on validation, that's a classic overfitting red flag. The model is too complex for the amount of data you have."<br>

</details>

## Underfitting

Model is too simple or undertrained to capture underlying patterns. Mitigate by increasing capacity, training longer, or improving features.

<details>
<summary>Rick & Morty: What's this?</summary>

Morty: "Underfitting means it's too simple?"<br>

Rick: "Too weak to capture the signal. Give it capacity, time, or smarter features."<br>

Morty: "The model's just not smart enough?"<br>

Rick: "Or not trained enough. Add capacity, train longer, or build better features. Underfitting is the opposite of overfitting—the model is too simple to learn even the basic patterns in your data. It's like trying to fit a curved line with a straight line—it just can't capture the complexity."<br>

Morty: "Any risks?"<br>

Rick: "Swing too far and you overfit. It's a bias‑variance seesaw. The art is finding the sweet spot between too simple (underfitting) and too complex (overfitting). You want just enough model complexity to capture the real patterns without memorizing noise."<br>

Morty: "So tune, don't blindly crank."<br>

Rick: "Bingo. Start simple and gradually increase complexity while monitoring validation performance. Add layers, parameters, or training time incrementally. Stop when validation performance plateaus or starts degrading."<br>

Morty: "How do we recognize underfitting?"<br>

Rick: "Both training and validation performance are poor, Morty. If your model can't even learn the training data well, it's probably underfitting. The validation performance will be bad but close to training performance—they're both struggling."<br>

</details>

## Bias-Variance Trade-off
Balance between error from overly simple assumptions (bias) and sensitivity to noise (variance). Proper model/regularization choice aims to minimize total error.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Two kinds of being wrong?"<br>
Rick: "Bias is systematic error; variance is sensitivity to noise."<br>
Morty: "Can't we squash both?"<br>
Rick: "We minimize total error, Morty—trade bias for variance or vice versa."<br>
Morty: "How do we shift the balance?"<br>
Rick: "Regularization, data size, and model capacity—all nudge the seesaw."<br>
Morty: "So there's no free lunch?"<br>
Rick: "Only trade‑offs. Measure, don't guess."<br>
</details>

## Regularization
Techniques to prevent overfitting by penalizing complexity or adding noise (e.g., L1/L2, dropout, early stopping). Encourages simpler, more generalizable models.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Regularization keeps models humble?"<br>
Rick: "Penalize big weights, drop neurons, stop early—anything to tame variance."<br>
Morty: "Does it hurt training loss?"<br>
Rick: "Usually. But it helps test error, which is what matters."<br>
Morty: "So it's a controlled handicap?"<br>
Rick: "Exactly."<br>
</details>

 

## Loss Function

Quantifies prediction error for optimization (e.g., MSE for regression, cross-entropy for classification). Lower loss indicates better fit to data.

<details>
<summary>Rick & Morty: What's this?</summary>

Morty: "It's how wrong we are?"<br>

Rick: "A scalar shame-meter. MSE, cross-entropy—lower it and maybe you'll impress one timeline. The loss function is how you teach the model what 'wrong' means. It's the feedback signal that drives learning."<br>

Morty: "Why different loss functions?"<br>

Rick: "Different problems, different measures of wrongness, Morty. Mean Squared Error for regression penalizes big mistakes heavily. Cross-entropy for classification cares about probability distributions. Huber loss is robust to outliers. The choice shapes how your model learns."<br>

Morty: "How does it actually work?"<br>

Rick: "Compare predictions to truth, compute badness, backpropagate to update weights. The optimizer uses the loss gradient to figure out which direction to nudge each parameter. Lower loss means better predictions, in theory."<br>

Morty: "Any gotchas?"<br>

Rick: "Loss can be gamed, Morty. A model might minimize training loss by memorizing noise. That's why we use validation loss to judge real performance. Also, some losses don't match real-world objectives—optimizing for accuracy might not optimize for fairness."<br>

</details>

 

## Objective Function
The function optimized during training (often the loss plus regularization).

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Objective equals target of training?"<br>
Rick: "Yep. Loss plus any penalties you tack on."<br>
Morty: "Why add penalties?"<br>
Rick: "To keep models from overfitting—simplicity tax, Morty."<br>
Morty: "So the optimizer chases this function?"<br>
Rick: "Relentlessly, with your compute bill screaming."<br>
</details>

## Gradient
Vector of partial derivatives indicating the direction of steepest increase of a function.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Gradients tell us how to change?"<br>
Rick: "Direction and magnitude for each parameter."<br>
Morty: "And we go downhill to minimize?"<br>
Rick: "Negative gradient, Morty—gravity for math."<br>
Morty: "Any issues?"<br>
Rick: "Vanishing or exploding. Normalize, clip, or redesign."<br>
</details>

## Gradient Descent
Optimization algorithm that updates parameters in the negative gradient direction. Variants include SGD with momentum, Adam, RMSProp.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "We take steps guided by gradients?"<br>
Rick: "SGD is the basic stride; momentum and Adam smooth or adapt it."<br>
Morty: "Which one should we use?"<br>
Rick: "Start with Adam; switch to SGD for fine polish if needed."<br>
Morty: "And tune learning rate?"<br>
Rick: "Always."<br>
</details>

 

## Learning Rate
Step size controlling how much parameters change per update.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Too big and we overshoot?"<br>
Rick: "Too small and we crawl. Schedules help—warmup, decay."<br>
Morty: "Adaptive optimizers fix it?"<br>
Rick: "They help, but you still tune."<br>
Morty: "So LR is the most important knob?"<br>
Rick: "Usually, Morty."<br>
</details>

## Epoch
One full pass through the training dataset.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "A season of training, basically?"<br>
Rick: "Complete pass. Count them, but watch validation—not just episode numbers."<br>
Morty: "Early stopping watches validation?"<br>
Rick: "And stops the show before it jumps the shark."<br>
</details>

## Batch / Batch Size
Subset of the dataset used per gradient update; size controls memory and stability.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Bigger batches, smoother updates?"<br>
Rick: "Smoother but costlier. Small batches add noise that sometimes helps."<br>
Morty: "Any rule of thumb?"<br>
Rick: "Fit your memory and keep throughput high."<br>
Morty: "Gradient accumulation?"<br>
Rick: "A hack to pretend the batch is bigger."<br>
</details>

## Optimizer
Algorithm to update parameters (e.g., SGD, Adam, RMSProp). Choice affects convergence speed and stability.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Optimizers are update strategies?"<br>
Rick: "Exactly. They decide how to move in parameter space."<br>
Morty: "And hyperparameters matter here too?"<br>
Rick: "Learning rate, betas, weight decay—tune 'em or suffer."<br>
Morty: "So optimizer choice isn't magic?"<br>
Rick: "It's taste plus evidence, Morty."<br>
</details>

 

## Backpropagation
Method to compute gradients in neural networks via the chain rule.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "We backtrack errors through layers?"<br>
Rick: "Chain rule lets each layer know how it messed up."<br>
Morty: "Why do gradients vanish?"<br>
Rick: "Deep chains with saturating activations. Use residuals, norms, and better activations."<br>
Morty: "So architecture fights calculus?"<br>
Rick: "It negotiates, Morty."<br>
</details>

## Activation Function
Nonlinear function applied to neuron outputs (e.g., ReLU, sigmoid, tanh).

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Nonlinearity gives power?"<br>
Rick: "Otherwise it's just a linear stack. ReLU's the workhorse."<br>
Morty: "Sigmoid and tanh?"<br>
Rick: "Squashers—use carefully or drown in saturation."<br>
Morty: "Newer ones?"<br>
Rick: "GELU, Swish—smoother vibes."<br>
</details>

## Softmax
Normalizes logits into a probability distribution.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Exponentiate then normalize?"<br>
Rick: "Yep. Softmax turns raw scores into a proper distribution."<br>
Morty: "Temperature ties in?"<br>
Rick: "Divide logits before softmax to control sharpness."<br>
Morty: "Calibration?"<br>
Rick: "Check it—confidence isn't accuracy."<br>
</details>

 

## Logits
Pre-activation scores output by a model before normalization (e.g., before softmax).

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Raw scores, unnormalized?"<br>
Rick: "Logits. Big differences → confident softmax; tiny ones → indecision."<br>
Morty: "We can inspect them?"<br>
Rick: "For debugging or margin tricks, yeah."<br>
Morty: "So they’re pre‑probabilities?"<br>
Rick: "Exactly."<br>
</details>

## Embeddings
Dense, low-dimensional vector representations of discrete items (e.g., words, products, users) that capture semantic relationships. Learned via tasks like next-token prediction or matrix factorization.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Vectors that capture meaning?"<br>
Rick: "Geometry that encodes relationships—neighbors mean similar."<br>
Morty: "How do we train them?"<br>
Rick: "Self‑supervised tasks or downstream fine‑tuning."<br>
Morty: "And we reuse them?"<br>
Rick: "Transfer them across tasks like a portal pass."<br>
</details>

## Tokenization
Converting raw text into tokens (words, subwords, characters) for modeling. Modern LLMs use subword tokenizers (e.g., BPE, WordPiece) to balance vocabulary size and expressivity.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Breaking text into chunks?"<br>
Rick: "Subwords balance vocabulary size and flexibility."<br>
Morty: "Why not characters?"<br>
Rick: "Longer sequences; models cry."<br>
Morty: "And words?"<br>
Rick: "Too many, rare ones explode."<br>
</details>

 

## Attention

Mechanism allowing models to focus on relevant parts of the input when producing outputs by computing weighted combinations of values based on query–key similarity.

<details>
<summary>Rick & Morty: What's this?</summary>

Morty: "Paying attention—literally?"<br>

Rick: "Queries, keys, values. We weight what's relevant so the model stops staring into space, Morty."<br>

Morty: "Queries point to keys?"<br>

Rick: "Scores pick values to mix—attention directs computation. Think of it like this: you have a question (query), a bunch of topics (keys), and detailed answers (values). The attention mechanism figures out which topics are most relevant to your question, then mixes together the corresponding answers based on relevance."<br>

Morty: "Multi‑head?"<br>

Rick: "Parallel focuses for different patterns. Each head learns to pay attention to different aspects—one might focus on grammar, another on meaning, another on relationships between words. It's like having multiple experts each with their own specialty, then combining their insights."<br>

Morty: "Self vs cross?"<br>

Rick: "Self looks within; cross looks across modalities or sequences. Self-attention lets words in a sentence pay attention to other words in the same sentence—like 'it' referring back to 'cat'. Cross-attention looks between different sources, like when you're translating and need to figure out which English word corresponds to which French word."<br>

Morty: "Why is this such a big deal?"<br>

Rick: "Because it solved the long-range dependency problem, Morty. Before attention, models would forget important context from earlier in the sequence. Now they can directly connect to any relevant information, regardless of distance. It's what made transformers possible and kicked off the modern AI revolution."<br>

Morty: "So it's like having perfect memory?"<br>

Rick: "Perfect memory with smart indexing. The model doesn't just remember everything—it learns what to remember and when it's relevant. It's the difference between a cluttered attic and a well-organized library."<br>

</details>

 

## Transformer

Neural architecture built on attention mechanisms (self-attention, cross-attention) and feed-forward layers, often with residual connections and normalization. Dominant in NLP, vision, and multimodal tasks.

<details>
<summary>Rick & Morty: What's this?</summary>

Morty: "The big attention architecture?"<br>

Rick: "Stacks of attention and feed-forward layers with skips and norms. Dominates NLP, vision—pretty much your homework, Morty."<br>

Morty: "Stacks of attention blocks?"<br>

Rick: "Plus feed‑forwards, skips, and norms. Each transformer block has a self-attention layer that lets tokens talk to each other, followed by a feed-forward network that processes each token independently. Residual connections and layer normalization keep gradients flowing during training."<br>

Morty: "Why so dominant?"<br>

Rick: "Scales well and models long‑range dependencies. Unlike RNNs that process sequences step-by-step, transformers can look at all positions simultaneously. This parallelization makes training much faster, and the attention mechanism captures relationships between distant tokens that RNNs often forget."<br>

Morty: "Any downsides?"<br>

Rick: "Context window limits and compute hunger. Attention complexity scales quadratically with sequence length, so transformers hit memory and compute walls with very long sequences. They also need lots of data and computation to train effectively."<br>

Morty: "What made them revolutionary?"<br>

Rick: "The 'Attention is All You Need' paper, Morty. They showed you could ditch recurrence and convolution entirely, just use attention. This unlocked massive parallelization and led to the current AI revolution—GPT, BERT, Vision Transformers, everything."<br>

</details>

 

## LLM

Large Language Model: a transformer-based model with many parameters trained on large text corpora to model next-token probabilities. Supports prompting and fine-tuning for downstream tasks.

<details>
<summary>Rick & Morty: What's this?</summary>

Morty: "A huge text-predictor brain?"<br>

Rick: "Gigantic transformers trained on oceans of text to guess the next token. Prompt it right or it rambles like Jerry."<br>

Morty: "Huge text predictors?"<br>

Rick: "Autoregressive token guessers. Prompt savvy makes them useful, Morty. Think of them as incredibly sophisticated autocomplete systems—they've read basically everything on the internet and learned to predict what word comes next in any context."<br>

Morty: "Fine‑tuning or prompts?"<br>

Rick: "Both—choose based on data, control, and cost. Prompting is like asking a really smart friend for help—you describe what you want and hope they understand. Fine-tuning is like hiring that friend and training them specifically for your exact job. Prompting is cheap and flexible, fine-tuning gives you more control but costs more."<br>

Morty: "Safety?"<br>

Rick: "Guardrails or you'll get cosmic nonsense. These models will confidently tell you that sharks are mammals or help you make explosives if you ask nicely enough. They need safety training, content filters, and careful prompt engineering to behave responsibly."<br>

Morty: "How do they actually work?"<br>

Rick: "They break text into tokens, encode them as vectors, then use attention mechanisms to understand relationships between all the tokens. When generating, they sample from a probability distribution over all possible next tokens. The magic is in the training—they learn patterns from billions of examples."<br>

Morty: "Why are they so good at everything?"<br>

Rick: "Scale and emergent abilities, Morty. Train a big enough model on enough data, and it starts showing capabilities nobody explicitly taught it—like reasoning, coding, and creative writing. It's like crossing a threshold where quantity becomes quality."<br>

</details>

 

## Prompt

Text or structured input used to guide an LLM's output. Good prompts provide context, constraints, and examples (few-shot) to steer model behavior.

<details>
<summary>Rick & Morty: What's this?</summary>

Morty: "We tell it what to do?"<br>

Rick: "Context, constraints, examples. Good prompts steer; bad ones summon gibberish from the abyss, Morty."<br>

Morty: "We steer with text?"<br>

Rick: "Context + constraints + examples—give it rails to run on. Think of prompting as programming with natural language. You're not coding logic, you're describing the task, providing context, and showing examples of what good output looks like."<br>

Morty: "Few‑shot helps?"<br>

Rick: "Shows patterns to imitate. Zero-shot is 'translate this French text'. Few-shot is 'here are three examples of French-to-English translations, now translate this new one'. The examples teach the model the specific style and format you want."<br>

Morty: "And system messages?"<br>

Rick: "Set behavior—like telling me to be nice. System prompts define the model's role and personality. 'You are a helpful assistant' vs 'You are a cynical scientist' will produce very different responses to the same user question."<br>

Morty: "What makes prompts work well?"<br>

Rick: "Clarity, specificity, and structure, Morty. Be explicit about what you want, provide relevant context, use clear formatting, and give examples when needed. The model can't read your mind—it only has your prompt to work with."<br>

Morty: "Any tricks?"<br>

Rick: "Chain of thought prompting—ask the model to think step by step. Role playing—have it act as an expert. Temperature tuning—adjust randomness. And always test variations to see what works best for your use case."<br>

</details>

## Context Window
Maximum number of tokens (input + generated) an LLM can process at once; exceeding it truncates inputs or requires special strategies (e.g., chunking, retrieval).

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Memory limit in tokens?"<br>
Rick: "Context window—go past it, you lose information."<br>
Morty: "Workarounds?"<br>
Rick: "Chunking, retrieval, or models with bigger windows."<br>
Morty: "So planning matters?"<br>
Rick: "Always."<br>
</details>

## Temperature / Top-p
Sampling parameters controlling randomness (temperature) and nucleus sampling (top-p) during generation. Lower temperature/top-p increases determinism; higher values increase diversity.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Knobs for chaos?"<br>
Rick: "Temperature sets noise; top‑p keeps only the likely mass."<br>
Morty: "Defaults?"<br>
Rick: "Start modest—then tune for creativity or reliability."<br>
Morty: "Combine both?"<br>
Rick: "Sure, but don't over‑randomize."<br>
</details>

 

## Beam Search
Search strategy that explores multiple candidate sequences in parallel and keeps the best beams by cumulative log-probability. Useful for translation and structured generation.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Parallel guesses with pruning?"<br>
Rick: "Keep top beams by score, expand until done."<br>
Morty: "Any trade‑offs?"<br>
Rick: "Diversity drops; add penalties or sampling for variety."<br>
Morty: "Use cases?"<br>
Rick: "Translation, constrained generation, planning."<br>
</details>

## Accuracy, Precision, Recall, F1
Common classification metrics; F1 balances precision and recall.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Accuracy vs precision vs recall?"<br>
Rick: "Accuracy is overall rightness; precision avoids false positives; recall avoids false negatives."<br>
Morty: "And F1?"<br>
Rick: "Harmonic mediator—use when you need balance."<br>
Morty: "Pick based on stakes?"<br>
Rick: "Always, Morty."<br>
</details>

 

## Confusion Matrix
Table showing counts of true/false positives/negatives.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Four boxes, huh?"<br>
Rick: "TP, FP, TN, FN—see where the model fails."<br>
Morty: "Threshold changes it?"<br>
Rick: "Yes—move it and the boxes reshuffle."<br>
Morty: "So context matters."<br>
Rick: "Yep."<br>
</details>

 

## ROC / AUC
Receiver Operating Characteristic curve and Area Under the Curve; evaluate ranking quality across thresholds. Use predicted probabilities/scores, not hard labels.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Plot TPR vs FPR?"<br>
Rick: "Across thresholds—AUC summarizes ranking performance."<br>
Morty: "Use scores, not labels?"<br>
Rick: "Right—labels are binary; we need the continuum."<br>
Morty: "And PR curves?"<br>
Rick: "For imbalanced data, often more telling."<br>
</details>

 

## Perplexity
Exponentiated average negative log-likelihood; measures language model uncertainty (lower is better). Sensitive to tokenization and dataset domain.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Perplexity is confusion?"<br>
Rick: "Lower means the model predicts well."<br>
Morty: "But tokenization matters?"<br>
Rick: "Change tokens, change numbers—compare apples to apples."<br>
Morty: "So domain shifts break comparisons."<br>
Rick: "Exactly."<br>
</details>

## Cross-Validation
Resampling technique to estimate generalization by training/validating across multiple splits (e.g., k-fold, stratified). Helps reduce variance in performance estimates.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "We rotate the validation set?"<br>
Rick: "Train on folds, validate on the held‑out one—repeat and average."<br>
Morty: "Stratify for class balance?"<br>
Rick: "Yep, or risk misleading scores."<br>
Morty: "Expensive?"<br>
Rick: "Computationally, yes."<br>
</details>

 

## Data Augmentation
Synthetic transformations to expand training data (e.g., flips, crops, noise); improves robustness and reduces overfitting, especially in vision/audio.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "We mutate images and audio?"<br>
Rick: "And text—paraphrases, masking. Toughen models against reality."<br>
Morty: "Any pitfalls?"<br>
Rick: "Don't change labels or inject artifacts."<br>
Morty: "So realistic transforms only."<br>
Rick: "Right."<br>
</details>

 

## Normalization / Standardization
Scaling features to comparable ranges (e.g., min–max, z-score) to stabilize training and accelerate convergence. Fit on training data, apply consistently to validation/test.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "We scale for stability?"<br>
Rick: "And speed. Fit on train, reuse on val/test to avoid leakage."<br>
Morty: "Per‑feature or per‑batch?"<br>
Rick: "Depends—preprocessing vs BatchNorm/LayerNorm."<br>
Morty: "Keep pipelines consistent."<br>
Rick: "Always."<br>
</details>

 

## Dropout
Randomly zeroing activations during training to reduce overfitting by preventing co-adaptation of features. Disabled at inference.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "We drop neurons randomly?"<br>
Rick: "Force redundancy so features don't collude."<br>
Morty: "At inference?"<br>
Rick: "Turn it off; scale weights accordingly."<br>
Morty: "Rates?"<br>
Rick: "Tune 0.1–0.5, context‑dependent."<br>
</details>

 

## Weight Decay
L2 regularization applied to weights during optimization, discouraging large weights and smoothing solutions. Implemented via optimizer `weight_decay`.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Shrink weights to smooth?"<br>
Rick: "Adds a penalty—curbs complexity and improves generalization."<br>
Morty: "Same as L2?"<br>
Rick: "Equivalent effect within the optimizer—mind exceptions like AdamW's decoupling."<br>
Morty: "So use AdamW."<br>
Rick: "Often, yes."<br>
</details>

## Batch Norm / Layer Norm
Normalization strategies that stabilize and accelerate training in deep nets: BatchNorm normalizes over batch statistics; LayerNorm normalizes over feature dimensions and is common in transformers.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Normalize to keep training sane?"<br>
Rick: "BatchNorm uses batch stats; LayerNorm uses per‑feature."<br>
Morty: "When do we pick which?"<br>
Rick: "Transformers love LayerNorm; conv nets often use BatchNorm."<br>
Morty: "Any caveats?"<br>
Rick: "Batch size sensitivity and inference behavior—mind the stats."<br>
</details>

 

## Transfer Learning
Using a pretrained model as a starting point for a new task. Typically freeze early layers, replace task head, then fine-tune selectively.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Shortcut to good features?"<br>
Rick: "Borrow general representations, adapt the head to your task."<br>
Morty: "Freeze or not?"<br>
Rick: "Freeze early layers, unfreeze later when you have data."<br>
Morty: "Watch for forgetting?"<br>
Rick: "Regularize and use small LRs."<br>
</details>

 

 

## Fine-Tuning
Further training of a pretrained model on task-specific data using lower learning rates and careful regularization to avoid catastrophic forgetting.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "We tune gently?"<br>
Rick: "Small steps, strong priors—retain the useful generality."<br>
Morty: "Layer‑wise learning rates?"<br>
Rick: "Lower for early layers, higher for task head."<br>
Morty: "Checkpoint often?"<br>
Rick: "And validate constantly."<br>
</details>

 

## Model Distillation
Training a small “student” model to mimic a large “teacher” model.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Shrink without losing brains?"<br>
Rick: "Distill softened targets/logits and sometimes features."<br>
Morty: "Why softened?"<br>
Rick: "They carry dark knowledge—fine‑grained class relations."<br>
Morty: "Deploy the student, retire the teacher?"<br>
Rick: "If metrics hold, yes."<br>
</details>

## Pruning / Quantization
Compressing models by removing weights (pruning) or using lower precision (quantization). Reduces model size and latency at some accuracy cost.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Trade size for accuracy?"<br>
Rick: "Sparse weights or fewer bits—measure the hit and the speedup."<br>
Morty: "Post‑training or during?"<br>
Rick: "Both exist—calibrate carefully for quantization."<br>
Morty: "Edge devices?"<br>
Rick: "These tricks are their lifeline."<br>
</details>

 

## Feature Engineering
The process of transforming raw data into meaningful, machine‑learnable inputs using domain knowledge and statistical techniques. It includes handling missing values, encoding categorical variables, scaling, creating interaction or aggregate features, and time‑aware features. Good feature engineering often yields bigger gains than model changes and must avoid leakage by using only information available at prediction time.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Make inputs smarter?"<br>
Rick: "Clean, encode, aggregate, and respect time—no peeking ahead."<br>
Morty: "Why so powerful?"<br>
Rick: "Signal quality beats model complexity nine times out of ten."<br>
Morty: "Share features?"<br>
Rick: "Use a feature store to avoid chaos."<br>
</details>

## One-Hot Encoding
A method to represent categorical variables as sparse binary vectors, one column per category with a 1 for the observed category and 0 otherwise. It preserves distance relationships for algorithms that assume numeric inputs without imposing arbitrary ordering. Beware of high‑cardinality explosion; consider target encoding or embeddings when categories are numerous.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Binary flags per category?"<br>
Rick: "Yep—simple and reliable until cardinality explodes."<br>
Morty: "Alternatives?"<br>
Rick: "Target encoding or learn embeddings."<br>
Morty: "Watch leakage?"<br>
Rick: "Always, especially with target encoding."<br>
</details>

## Dimensionality Reduction
Techniques that reduce the number of input variables while preserving as much information as possible. Linear methods (e.g., PCA) capture variance along orthogonal directions; nonlinear methods (e.g., t‑SNE, UMAP) capture manifold structure for visualization or preprocessing. Benefits include noise reduction, speedups, and mitigation of the curse of dimensionality.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Fewer features, same gist?"<br>
Rick: "Compress dimensions to keep signal and drop noise."<br>
Morty: "For modeling or visuals?"<br>
Rick: "Both—PCA for models, t‑SNE/UMAP for plots."<br>
Morty: "Beware distortions?"<br>
Rick: "Especially with t‑SNE globally."<br>
</details>

## PCA / t-SNE / UMAP
PCA is a linear projection maximizing variance and enabling fast compression and whitening. t‑SNE preserves local neighbor relationships for high‑quality visualizations but distorts global distances and is not ideal for downstream learning. UMAP models data as a fuzzy topological graph to preserve local/global structure better than t‑SNE in many cases and scales well.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Pick the right reducer?"<br>
Rick: "PCA for linear variance; UMAP for structure with speed; t‑SNE for pretty plots."<br>
Morty: "Use UMAP over t‑SNE?"<br>
Rick: "Often for scalability and global coherence."<br>
Morty: "Downstream learning?"<br>
Rick: "Prefer PCA embeddings there."<br>
</details>

## Linear / Logistic Regression
Linear regression models a continuous target as a weighted sum of features under assumptions like linearity and homoscedastic errors. Logistic regression models the log‑odds of class membership to produce calibrated probabilities for binary or multiclass tasks. Both are interpretable, support regularization (L1/L2), and serve as strong baselines.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Numbers vs probabilities?"<br>
Rick: "Linear for continuous; logistic for classes with odds."<br>
Morty: "Regularization?"<br>
Rick: "L1 sparsifies, L2 smooths—pick your poison."<br>
Morty: "Calibration?"<br>
Rick: "Logistic often does it well."<br>
</details>

## SVM / k-NN / Decision Trees
SVMs find a maximum‑margin separator (with kernels for nonlinearity) and can be robust in high dimensions but require tuning C/kernel parameters. k‑NN classifies by majority vote among nearest neighbors and is simple yet sensitive to scaling and k choice. Decision trees learn hierarchical rules; they are interpretable but prone to overfitting without pruning.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Three classics, three vibes?"<br>
Rick: "Margins (SVM), neighbors (k‑NN), rules (trees)."<br>
Morty: "Scaling matters?"<br>
Rick: "For k‑NN and SVM—normalize features."<br>
Morty: "Trees overfit?"<br>
Rick: "Prune or bag 'em."<br>
</details>

## Random Forest / XGBoost
Random Forest averages many decorrelated decision trees (bagging) to reduce variance and improve generalization with minimal tuning. XGBoost (gradient boosting) builds trees sequentially to correct residual errors, often delivering state‑of‑the‑art tabular performance with careful regularization. Boosting is powerful but more sensitive to hyperparameters than bagging.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Bag vs boost?"<br>
Rick: "Forest reduces variance; boosting reduces bias—tune more carefully."<br>
Morty: "When to use which?"<br>
Rick: "Forest for quick baselines; XGBoost when squeezing leaderboard points."<br>
Morty: "Feature importance?"<br>
Rick: "Mind biases—permutation importance helps."<br>
</details>

## Naive Bayes
A family of probabilistic classifiers that assume conditional independence of features given the class (e.g., Gaussian, Multinomial, Bernoulli variants). Despite the strong assumption, they work surprisingly well for text and other high‑dimensional sparse data. They are fast to train, require little data, and yield calibrated posteriors under model correctness.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Cheap and cheerful?"<br>
Rick: "Fast, decent for text, naive in assumptions—know when to stop."<br>
Morty: "Feature independence is false, right?"<br>
Rick: "Often, but it still works."<br>
Morty: "Use as baseline?"<br>
Rick: "Always a good start."<br>
</details>

## Entropy / KL Divergence
Entropy quantifies the uncertainty of a random variable; higher entropy means more unpredictability. KL divergence measures how one probability distribution diverges from a reference distribution and is asymmetric. They are foundational in information theory, variational inference, and regularization of probabilistic models.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Surprise and mismatch?"<br>
Rick: "Entropy is average surprise; KL is directed difference."<br>
Morty: "Symmetric?"<br>
Rick: "No—KL(P||Q) ≠ KL(Q||P)."<br>
Morty: "Use cases?"<br>
Rick: "VI, regularization, and diagnostics."<br>
</details>

## Bayesian Inference
A principled framework that combines prior beliefs with data likelihood to produce a posterior distribution over parameters or latent variables. Exact posteriors are rare, so conjugacy, MCMC, or variational inference are used for approximation. Bayesian methods enable uncertainty quantification and coherent decision‑making under uncertainty.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Beliefs updated by evidence?"<br>
Rick: "Prior × likelihood → posterior."<br>
Morty: "Exact answers?"<br>
Rick: "Rare—approximate with MCMC or VI."<br>
Morty: "Why bother?"<br>
Rick: "Uncertainty you can reason about."<br>
</details>

## MAP / MLE
MLE chooses parameters that maximize the likelihood of observed data, often yielding unbiased estimates with large samples. MAP incorporates a prior and maximizes the posterior, acting like regularized MLE (e.g., Gaussian prior → L2 penalty). They coincide when the prior is uniform and differ when prior beliefs meaningfully constrain parameters.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Likelihood vs prior influence?"<br>
Rick: "MLE ignores priors; MAP bakes them in."<br>
Morty: "Regularization link?"<br>
Rick: "MAP with Gaussian prior looks like L2."<br>
Morty: "Pick based on beliefs?"<br>
Rick: "And data scarcity."<br>
</details>

## Markov Chains / HMM
Markov chains model sequences where the next state depends only on the current state (memoryless property). Hidden Markov Models add latent states emitting observations with state‑dependent probabilities, enabling speech, bioinformatics, and time‑series modeling. Inference typically uses the Forward‑Backward and Viterbi algorithms.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Memoryless steps?"<br>
Rick: "Markov property—next depends on now."<br>
Morty: "Hidden states?"<br>
Rick: "HMMs infer unseen causes of observations."<br>
Morty: "Algorithms?"<br>
Rick: "Forward‑Backward and Viterbi."<br>
</details>

## MCMC / Variational Inference
MCMC constructs a Markov chain whose stationary distribution is the target posterior, producing asymptotically exact samples at the cost of compute and mixing diagnostics. Variational inference turns inference into optimization over a tractable family, trading bias for speed and scalability. Modern practice often mixes both, e.g., using VI for initialization and MCMC for refinement.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Exact by wandering vs fast approximations?"<br>
Rick: "MCMC samples; VI optimizes an approximation."<br>
Morty: "Which to use?"<br>
Rick: "Start with VI for scale; refine with MCMC if needed."<br>
Morty: "Diagnostics?"<br>
Rick: "Check mixing, ELBO, and autocorrelation."<br>
</details>

## Autoencoder
A neural network trained to reconstruct inputs through a bottleneck, forcing a compact latent representation. The encoder maps inputs to a latent code; the decoder reconstructs inputs from that code, with reconstruction loss guiding learning. Uses include denoising, dimensionality reduction, pretraining, and anomaly detection.

## VAE
A probabilistic autoencoder that learns a distribution over latent variables and decodes samples to data space. Trained by maximizing the ELBO, it balances reconstruction accuracy with a KL penalty that regularizes the latent space toward a prior (often standard normal). VAEs enable interpolation, sampling, and controlled generation with continuous latents.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Probabilistic latent spaces?"<br>
Rick: "Encode to distributions, sample with reparameterization, decode—learn a smooth latent map."<br>
Morty: "Why the KL term?"<br>
Rick: "Regularizes latents toward a prior so you can sample and interpolate sensibly."<br>
Morty: "Tuning tips?"<br>
Rick: "Balance reconstruction vs KL—β‑VAE trades disentanglement for sharpness."<br>
</details>

## GAN
An adversarial framework where a generator produces samples and a discriminator distinguishes real from fake, trained in a minimax game. GANs can generate sharp images but are prone to instability, mode collapse, and require careful architecture, normalization, and loss choices. Variants (WGAN, StyleGAN) improve training dynamics and controllability.

<details>
<summary>Rick & Morty: What’s this?</summary>

Morty: "Adversaries that teach each other?"<br>

Rick: "Generator learns to fool; discriminator learns to detect—training is a knife‑edge balance."<br>

Morty: "Stability hacks?"<br>

Rick: "Spectral norm, gradient penalties, better losses like WGAN."<br>

Morty: "Mode collapse?"<br>

Rick: "Diversify with techniques like minibatch discrimination."<br>
</details>

## Diffusion Model
A generative model that learns to invert a forward noising process via a sequence of denoising steps. Training fits a noise predictor across timesteps; sampling iteratively refines from noise to data, yielding high‑fidelity, diverse outputs. They are compute‑intensive at inference but amenable to acceleration (DDIM, distillation).

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "We teach the model to un‑noise?"<br>
Rick: "Learn a noise predictor across timesteps, then sample by gradually denoising from pure noise."<br>
Morty: "Why so slow?"<br>
Rick: "Many steps—accelerate with schedulers, DDIM, or distilled samplers."<br>
Morty: "Quality vs speed?"<br>
Rick: "Trade‑off central—pick your sampler and schedule wisely."<br>
</details>

## Policy / Value Function

In RL, a policy maps states to actions (stochastic or deterministic), while value functions estimate expected returns for states or state‑action pairs. Policies can be learned directly (policy gradient) or derived from value estimates (e.g., greedy w.r.t. Q). The interplay between acting and evaluating underpins most RL algorithms.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Decide vs judge?"<br>
Rick: "Policy acts, value estimates reward."<br>
Morty: "Learn both?"<br>
Rick: "Actor‑critic pairs them nicely."<br>
Morty: "Exploration still needed?"<br>
Rick: "Always."<br>
</details>

## Exploration vs. Exploitation

The tension between gathering information (exploration) and maximizing reward using current knowledge (exploitation). Practical strategies include ε‑greedy, softmax over action values, optimism/UCB, and intrinsic motivation bonuses. Effective exploration reduces regret and avoids premature convergence to suboptimal policies.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Try new, or farm known?"<br>
Rick: "Balance. ε‑greedy is simple; UCB is clever."<br>
Morty: "Intrinsic bonuses?"<br>
Rick: "Curiosity signals to seek novelty."<br>
Morty: "Measure regret?"<br>
Rick: "Lower is better, Morty."<br>
</details>

## Q-Learning / TD Learning

Q‑learning learns optimal action‑values off‑policy by bootstrapping from estimated future returns, enabling learning from replayed experiences. Temporal‑difference methods update estimates using a mix of observed rewards and bootstrap predictions, balancing bias and variance. Stability often relies on target networks, experience replay, and careful learning‑rate schedules.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Update values from experience?"<br>
Rick: "TD methods blend observed rewards with predictions."<br>
Morty: "Why target networks?"<br>
Rick: "Stability—reduce moving‑target chaos."<br>
Morty: "Replay buffers?"<br>
Rick: "Decorrelate and reuse data."<br>
</details>

## Replay Buffer / Actor-Critic

A replay buffer stores past transitions to decorrelate updates and improve sample efficiency by reusing data. Actor‑critic methods pair a policy (actor) with a value estimator (critic), combining low‑variance value updates with flexible policy optimization. Modern variants (A2C/A3C, PPO, SAC) add stability through constraints or entropy regularization.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Save experiences and split duties?"<br>
Rick: "Buffer for reuse; actor chooses, critic evaluates."<br>
Morty: "Variants?"<br>
Rick: "PPO clips updates; SAC adds entropy."<br>
Morty: "Why?"<br>
Rick: "Stability and exploration."<br>
</details>

## Inference

The deployment‑time phase where a trained model processes new inputs to produce outputs under latency, memory, and cost constraints. Optimizations include batching, quantization, graph compilation, and hardware acceleration. Observability and correctness (schema validation, canaries) are critical to safe operation.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Use the model in production?"<br>
Rick: "Serve predictions fast and correctly—optimize and observe."<br>
Morty: "Batching helps throughput?"<br>
Rick: "And hurts latency—trade‑offs everywhere."<br>
Morty: "Ship safely?"<br>
Rick: "Validate schemas and canary changes."<br>
</details>

## Latency / Throughput

Latency measures time per prediction, while throughput measures predictions per unit time for a system. They trade off via batching and parallelism, and both are constrained by model size, I/O, and hardware. SLOs commonly set tail‑latency targets; monitoring captures warm vs cold start and queuing effects.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Fast vs many?"<br>
Rick: "Latency is speed; throughput is volume—tune batching and parallelism."<br>
Morty: "Watch tails?"<br>
Rick: "Tail latency ruins SLOs—optimize cold starts and queues."<br>
Morty: "Hardware matters?"<br>
Rick: "Always—CPU/GPU/TPU change the game."<br>
</details>

## MLOps

Engineering practices to reliably build, train, evaluate, deploy, and operate ML systems at scale. It emphasizes reproducibility (data/versioning), automated pipelines (CI/CD for ML), governance (approvals/audit), and monitoring (performance, drift, fairness) across the lifecycle. Collaboration between data science and platform teams is central.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Ops but for ML?"<br>
Rick: "Pipelines, registries, monitoring—make models repeatable and observable."<br>
Morty: "Governance?"<br>
Rick: "Approvals, audit trails—keep regulators off your back."<br>
Morty: "Teams?"<br>
Rick: "Data science plus platform—no silos, Morty."<br>
</details>

## Feature Store

A centralized system that defines, computes, and serves features consistently to training and online inference. Key capabilities include point‑in‑time correctness, offline/online parity, and low‑latency retrieval keyed by entity IDs. It reduces leakage bugs and duplication while enabling feature reuse across teams.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "One source of feature truth?"<br>
Rick: "Consistent definitions online/offline—avoid leakage and mismatches."<br>
Morty: "Keys?"<br>
Rick: "Entity IDs for fast lookups."<br>
Morty: "Reuse?"<br>
Rick: "Share across teams without chaos."<br>
</details>

## Model Registry

A catalog that tracks models, versions, lineage, metrics, and deployment stages (e.g., staging, production, archived). It supports approvals, rollbacks, and governance by linking artifacts to code, data, and evaluations. Registries integrate with CI/CD to automate promotion and deployment.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Where we keep model history?"<br>
Rick: "Versions, metrics, lineage—know what runs and why."<br>
Morty: "Rollbacks?"<br>
Rick: "Push a button and undo the bad."<br>
Morty: "CI/CD?"<br>
Rick: "Automate promotions safely."<br>
</details>

## A/B Test / Canary / Shadow

A/B testing splits traffic to compare candidate vs control models with statistical rigor. Canary gradually shifts a small fraction of live traffic to a new model to detect issues before full rollout; shadow sends mirrored traffic to a model without affecting users to gather metrics safely. Choice depends on risk tolerance, evaluation time, and observability.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Split, trickle, or mirror?"<br>
Rick: "A/B compares; canary tests safely; shadow observes silently."<br>
Morty: "Pick one?"<br>
Rick: "Based on risk and how fast you need answers."<br>
Morty: "Metrics?"<br>
Rick: "Collect thoroughly—no surprises."<br>
</details>

## Monitoring / Drift

Production monitoring tracks prediction quality, data quality, fairness, and system health, triggering alerts on anomalies. Data drift (input distribution change) and concept drift (target relationship change) degrade performance if unaddressed. Mitigations include retraining, feature recalibration, and adaptive thresholds.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Keep an eye on models?"<br>
Rick: "Metrics and alerts—catch drift early."<br>
Morty: "Data vs concept drift?"<br>
Rick: "Inputs shift vs relationships shift—both hurt."<br>
Morty: "Fixes?"<br>
Rick: "Retrain or adapt features/thresholds."<br>
</details>

## Fairness / Bias / Explainability

Fairness assesses whether outcomes are equitable across groups via metrics like demographic parity, equalized odds, or calibration. Bias can stem from data, labels, or modeling choices; mitigation techniques include reweighting, debiasing, and constraint‑aware training. Explainability tools (SHAP, LIME, saliency) provide local/global insight to support trust and compliance.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Fair and explainable outcomes?"<br>
Rick: "Measure across groups; mitigate bias; explain decisions."<br>
Morty: "Which metrics?"<br>
Rick: "Depends on policy—parity, odds, calibration."<br>
Morty: "Tools?"<br>
Rick: "SHAP, LIME, and saliency maps."<br>
</details>

## Adversarial Examples / Robustness

Adversarial examples are inputs with small, targeted perturbations that cause large model errors; threat models range from white‑box to black‑box. Robustness techniques include adversarial training, certified defenses, and randomized smoothing, with trade‑offs in accuracy and compute. Robust evaluation requires adaptive attacks and realistic constraints.

<details>
<summary>Rick & Morty: What’s this?</summary>
Morty: "Tiny tweaks, big mistakes?"<br>
Rick: "Attacks craft perturbations; defenses toughen models—at a cost."<br>
Morty: "White vs black box?"<br>
Rick: "Access to internals vs just outputs—changes attack strength."<br>
Morty: "Evaluate how?"<br>
Rick: "Adaptive attacks and real constraints."<br>
</details>

## Code Examples


<details>
<summary>Toggle Code Examples</summary>

Practical snippets grouped by concept. Use these alongside the glossary for hands-on intuition.<br>

```python<br>
# Supervised Learning<br>
from sklearn.linear_model import LogisticRegression<br>
model = LogisticRegression()<br>
model.fit(X_train, y_train)<br>
preds = model.predict(X_val)<br>
```<br>

```python<br>
# Unsupervised Learning<br>
from sklearn.cluster import KMeans<br>
km = KMeans(n_clusters=3)<br>
labels = km.fit_predict(X)<br>
```<br>

```python<br>
# Reinforcement Learning (Q-learning)<br>
Q[s, a] = Q[s, a] + alpha * (r + gamma * max(Q[s_next, a_]) - Q[s, a])<br>
```<br>

```python<br>
# Regularization (L2 / Weight Decay)<br>
import torch<br>
opt = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-4)<br>
```<br>

```python<br>
# Loss Function (Cross-Entropy)<br>
import torch.nn.functional as F<br>
loss = F.cross_entropy(logits, targets)<br>
```<br>

```python<br>
# Gradient Descent<br>
w = w - lr * grad_w<br>
```<br>

```python<br>
# Optimizer (Adam)<br>
import torch<br>
opt = torch.optim.Adam(model.parameters(), lr=1e-3)<br>
```<br>

```python<br>
# Softmax<br>
import numpy as np<br>
def softmax(z):<br>
    e = np.exp(z - np.max(z))<br>
    return e / e.sum()<br>
```<br>

```python<br>
# Classification Metrics<br>
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score<br>
precision = precision_score(y_true, y_pred)<br>
recall   = recall_score(y_true, y_pred)<br>
f1       = f1_score(y_true, y_pred)<br>
acc      = accuracy_score(y_true, y_pred)<br>
```<br>

```python<br>
# Confusion Matrix<br>
from sklearn.metrics import ConfusionMatrixDisplay<br>
ConfusionMatrixDisplay.from_predictions(y_true, y_pred)<br>
```<br>

```python<br>
# ROC / AUC<br>
from sklearn.metrics import RocCurveDisplay, roc_auc_score<br>
RocCurveDisplay.from_predictions(y_true, y_score)<br>
auc = roc_auc_score(y_true, y_score)<br>
```<br>

```python<br>
# Tokenization (Transformer tokenizers)<br>
from transformers import AutoTokenizer<br>
tok = AutoTokenizer.from_pretrained("bert-base-uncased")<br>
batch = tok(["Hello world", "ML is fun"], padding=True, return_tensors="pt")<br>
```<br>

```python<br>
# Attention (Scaled Dot-Product)<br>
import numpy as np<br>
def scaled_dot_product_attention(Q, K, V):<br>
    scores = (Q @ K.T) / np.sqrt(Q.shape[-1])<br>
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))<br>
    weights = weights / weights.sum(axis=-1, keepdims=True)<br>
    return weights @ V<br>
```<br>

```python<br>
# Transformer (PyTorch built-in)<br>
import torch<br>
import torch.nn as nn<br>
model = nn.Transformer(d_model=128, nhead=8, num_encoder_layers=2, num_decoder_layers=2)<br>
src = torch.randn(10, 32, 128)<br>
tgt = torch.randn(9,  32, 128)<br>
out = model(src, tgt)<br>
```<br>

```python<br>
# LLM Generation (Transformers)<br>
from transformers import AutoModelForCausalLM, AutoTokenizer<br>
tok = AutoTokenizer.from_pretrained("gpt2")<br>
lm  = AutoModelForCausalLM.from_pretrained("gpt2")<br>
inp = tok("Hello, I'm a language model", return_tensors="pt")<br>
gen = lm.generate(**inp, max_length=50, temperature=0.8, top_p=0.9)<br>
print(tok.decode(gen[0], skip_special_tokens=True))<br>
```<br>

```python<br>
# Beam Search / Sampling Controls<br>
lm.generate(**inp, num_beams=5)<br>
lm.generate(**inp, temperature=0.7, top_p=0.9)<br>
```<br>

```python<br>
# Cross-Validation<br>
from sklearn.model_selection import cross_val_score<br>
from sklearn.linear_model import LogisticRegression<br>
scores = cross_val_score(LogisticRegression(), X, y, cv=5)<br>
```<br>

```python<br>
# Data Augmentation (Vision)<br>
from torchvision import transforms<br>
aug = transforms.Compose([<br>
    transforms.RandomResizedCrop(224),<br>
    transforms.RandomHorizontalFlip(),<br>
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),<br>
])<br>
```<br>

```python<br>
# Normalization / Standardization<br>
from sklearn.preprocessing import StandardScaler<br>
scaler = StandardScaler().fit(X_train)<br>
X_train_std = scaler.transform(X_train)<br>
X_val_std   = scaler.transform(X_val)<br>
```<br>

```python<br>
# Dropout / BatchNorm / LayerNorm<br>
import torch.nn as nn<br>
layer = nn.Sequential(<br>
    nn.Linear(128, 128),<br>
    nn.ReLU(),<br>
    nn.Dropout(p=0.5),<br>
    nn.LayerNorm(128),<br>
)<br>
```<br>

```python<br>
# Transfer Learning / Fine-Tuning<br>
import torch<br>
import torchvision.models as models<br>
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)<br>
for p in model.parameters():<br>
    p.requires_grad = False<br>
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)<br>
for p in model.layer4.parameters():<br>
    p.requires_grad = True<br>
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)<br>
```<br>

```python<br>
# One-Hot Encoding<br>
import pandas as pd<br>
X = pd.DataFrame({"color": ["red", "green", "blue"]})<br>
X_oh = pd.get_dummies(X, columns=["color"])  # one-hot columns<br>
```<br>

```python<br>
# PCA / Dimensionality Reduction<br>
from sklearn.decomposition import PCA<br>
pca = PCA(n_components=2)<br>
X2 = pca.fit_transform(X)<br>
```<br>

```python<br>
# Linear / Logistic Regression<br>
from sklearn.linear_model import LinearRegression, LogisticRegression<br>
reg = LinearRegression().fit(X, y_cont)<br>
clf = LogisticRegression().fit(X, y_bin)<br>
```<br>

```python<br>
# SVM / k-NN / Decision Trees<br>
from sklearn.svm import SVC<br>
from sklearn.neighbors import KNeighborsClassifier<br>
from sklearn.tree import DecisionTreeClassifier<br>
svm = SVC().fit(X, y)<br>
knn = KNeighborsClassifier(n_neighbors=5).fit(X, y)<br>
dt  = DecisionTreeClassifier().fit(X, y)<br>
```<br>

```python<br>
# Random Forest<br>
from sklearn.ensemble import RandomForestClassifier<br>
rf = RandomForestClassifier(n_estimators=200, max_depth=10).fit(X, y)<br>
```<br>

```python<br>
# Pruning / Quantization<br>
import torch<br>
quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)<br>
import torch.nn.utils.prune as prune<br>
for module in model.modules():<br>
    if isinstance(module, torch.nn.Linear):<br>
        prune.l1_unstructured(module, name="weight", amount=0.2)<br>
        prune.remove(module, "weight")<br>
```<br>

```python<br>
# Naive Bayes<br>
from sklearn.naive_bayes import GaussianNB<br>
nb = GaussianNB().fit(X, y)<br>
```<br>

```python<br>
# Autoencoder (Skeleton)<br>
import torch.nn as nn<br>
class AE(nn.Module):<br>
    def __init__(self, d):<br>
        super().__init__()<br>
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16))<br>
        self.dec = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d))<br>
    def forward(self, x):<br>
        z = self.enc(x)<br>
        return self.dec(z)<br>
```<br>

```python<br>
# VAE (Reparameterization Trick)<br>
import torch<br>
def reparam(mu, logvar):<br>
    std = torch.exp(0.5 * logvar)<br>
    eps = torch.randn_like(std)<br>
    return mu + eps * std<br>
```<br>

```python<br>
# GAN (Training Loop Sketch)<br>
for x in dataloader:<br>
    d_loss = D_loss(D(x), D(G(z)))<br>
    d_opt.zero_grad(); d_loss.backward(); d_opt.step()<br>
    g_loss = G_loss(D(G(z)))<br>
    g_opt.zero_grad(); g_loss.backward(); g_opt.step()<br>
```<br>

```python<br>
# Diffusion (Denoising Step Sketch)<br>
x_{t-1} = denoise(x_t, t, eps_theta)<br>
```<br>

```python<br>
# Inference / Latency<br>
import time<br>
start = time.perf_counter()<br>
_ = model(x)<br>
latency_ms = (time.perf_counter() - start) * 1000<br>
```<br>

</details>
