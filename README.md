



# HW2 ― Object Detection
In this assignment, you are given a dataset of aerial images. Your task is to detect and classify the objects present in the images by determining their bounding boxes.

![enter image description here](https://lh3.googleusercontent.com/jUokHJn3aphsNTopJSh_tMxOvCTHK65EJLCVV-RBW-2LRxSIla7aS8KmbtKn05mcwUxDuIxF8b4)

For more details, please click [this link](https://docs.google.com/presentation/d/1CiO0rZzYbPabMjcgDGfRS6V85bRTLvR5cY3jiEngeLc/edit?usp=sharing) to view the slides of HW2.

# Usage
To start working on this assignment, you should clone this repository into your local machine by using the following command.

    git clone https://github.com/dlcv-spring-2019/hw2-<username>.git
Note that you should replace `<username>` with your own GitHub username.

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `hw2_train_val`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://www.dropbox.com/s/7wnulnv1y1s67qr/hw2_train_val.zip?dl=1) and unzip the compressed file manually.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `hw2_train_val` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

### Evaluation
To evaluate your model, you can run the provided evaluation script provided in the starter code by using the following command.

    python3 hw2_evaluation_task.py <PredictionDir> <AnnotationDir>

 - `<PredictionDir>` should be the directory to output your prediction files (e.g. `hw2_train_val/val1500/labelTxt_hbb_pred/`)
 - `<AnnotationDir>` should be the directory of ground truth (e.g. `hw2_train_val/val1500/labelTxt_hbb/`)

Note that your predicted label file should have the same filename as that of its corresponding ground truth label file (both of extension ``.txt``).

### Visualization
To visualization the ground truth or predicted bounding boxes in an image, you can run the provided visualization script provided in the starter code by using the following command.

    python3 visualize_bbox.py <image.jpg> <label.txt>

# Submission Rules
### Deadline
108/04/17 (Wed.) 01:00 AM

### Late Submission Policy
You have a three-day delay quota for the whole semester. Once you have exceeded your quota, the credit of any late submission will be deducted by 30% each day.

Note that while it is possible to continue your work in this repository after the deadline, **we will by default grade your last commit before the deadline** specified above. If you wish to use your quota or submit an earlier version of your repository, please contact the TAs and let them know which commit to grade.

### Academic Honesty
-   Taking any unfair advantages over other class members (or letting anyone do so) is strictly prohibited. Violating university policy would result in an **F** grade for this course (**NOT** negotiable).    
-   If you refer to some parts of the public code, you are required to specify the references in your report (e.g. URL to GitHub repositories).      
-   You are encouraged to discuss homework assignments with your fellow class members, but you must complete the assignment by yourself. TAs will compare the similarity of everyone’s submission. Any form of cheating or plagiarism will not be tolerated and will also result in an **F** grade for students with such misconduct.

### Submission Format
Aside from your own Python scripts and model files, you should make sure that your submission includes *at least* the following files in the root directory of this repository:
 1.   `hw2_<StudentID>.pdf`  
The report of your homework assignment. Refer to the "*Grading Policy*" section in the slides for what you should include in the report. Note that you should replace `<StudentID>` with your student ID, **NOT** your GitHub username.
 2.   `hw2.sh`  
The shell script file for running your `YoloV1-vgg16bn` model.
 3.   `hw2_best.sh`  
The shell script file for running your improved model.

We will run your code in the following manner:

    bash ./hw2.sh $1 $2
    bash ./hw2_best.sh $1 $2
where `$1` is the testing images directory (e.g. `test/images`), and `$2` is the output prediction directory (e.g. `test/labelTxt_hbb_pred/` ).

### Packages
Below is a list of packages you are allowed to import in this assignment:

> [`python`](https://www.python.org/): 3.5+  
> [`tensorflow`](https://www.tensorflow.org/): 1.13  
> [`keras`](https://keras.io/): 2.2+  
> [`torch`](https://pytorch.org/): 1.0  
> [`h5py`](https://www.h5py.org/): 2.9.0  
> [`numpy`](http://www.numpy.org/): 1.16.2  
> [`pandas`](https://pandas.pydata.org/): 0.24.0  
> [`torchvision`](https://pypi.org/project/torchvision/), [`cv2`](https://pypi.org/project/opencv-python/), [`matplotlib`](https://matplotlib.org/), [`skimage`](https://scikit-image.org/), [`Pillow`](https://pillow.readthedocs.io/en/stable/), [`scipy`](https://www.scipy.org/)  
> [The Python Standard Library](https://docs.python.org/3/library/)

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

### Remarks
- If your model is larger than GitHub’s maximum capacity (100MB), you can upload your model to another cloud service (e.g. Dropbox). However, your shell script files should be able to download the model automatically. For a tutorial on how to do this using Dropbox, please click [this link](https://goo.gl/XvCaLR).
- **DO NOT** hard code any path in your file or script, and the execution time of your testing code should not exceed an allowed maximum of **10 minutes**.
- **If we fail to run your code due to not following the submission rules, you will receive 0 credit for this assignment.**

# Q&A
If you have any problems related to HW2, you may
- Use TA hours (please check [course website](http://vllab.ee.ntu.edu.tw/dlcv.html) for time/location)
- Contact TAs by e-mail ([ntudlcvta2019@gmail.com](mailto:ntudlcvta2019@gmail.com))
- Post your question in the comment section of [this post](https://www.facebook.com/notes/dlcv-spring-2019/hw2-q-a/318555458854838/)
