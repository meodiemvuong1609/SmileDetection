# SmileDetectionBKNet
This source code using availabel BKNet Network and based on paper: [Multitask Learning BKNet](https://www.researchgate.net/publication/328586470_Effective_Deep_Multi-source_Multi-task_Learning_Frameworks_for_Smile_Detection_Emotion_Recognition_and_Gender_Classification?fbclid=IwAR0Mw11DfcFSOfpqFLp4rcHuVG06TC7KG6C9mrOHXktH_8slFvSCsBMtlMk) of Dr. Dinh Viet Sang from SoICT (HUST)

### Install requirements

<ul>
<li>Create virtualenv and install requirements </li>
</ul>

```bash
virtualenv env
env\Scripts\activate
pip install -r requirements.txt
```
### Preprocess data
<ul>
<li>Converts the detected face regions to grayscale, and resizes them to a fixed size of 28x28 pixels</li>
<li>Run preprocessing using command below:</li>
</ul>

```bash
python preprocess.py
```

### Prepare  data
<ul>
<li> Prepare data for training and testing</li>
<li> Run prepare data using command below:</li>

</ul>
    
```bash
python prepare_data.py
```
### Train model
<ul>
<li>Change path to model if your model path is different</li>
<li>Run training using command below:</li>
</ul>

```bash
python train.py -num_epochs 200 --save "/content/drive/MyDrive/CV/last/"
```

### Run demo
<ul>
<li>Change path to model if your model path is different</li>
<li>Run main programe using command below:</li>
</ul>

```bash
python main.py
```
