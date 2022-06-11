# FIN2017 Term Project - US Baby Names Dashboard

## Environment Setup

### System Requirements

+ [`Python 3.7`](https://www.python.org/downloads/) or above
+ C++ Compiler in order to run WordCloud successfully
+ Tested on macOS 12.3.1 (macOS Monetary)

### (Recommended) Virtual Environment 

Running on a virtual environment is easier and faster for this project. I used [`pipenv`](https://pipenv.pypa.io/en/latest/) to create a virtual environment for this project. 

```sh
pip install pipenv
pipenv install
pipenv shell
```
### Package Requirements

Then run the following command to install the required packages for this project:

```sh
pip install -r requirements.txt
```

## Getting Started 

I use `Streamlit` to visualize the dashboard. Running Streamlit is easy. Once you activate your environment by running `pipenv shell`, run the following command:

```sh
streamlit run TermProject.py
```

### Local Build

Though I have deployed this project on the Streamlit server, I recommend you to build the project locally to allow faster and smoother experience. 

There are two steps in order to build the project locally:


Run the following command to create a csv file for the data (which will be stored under the `Data/` directory with the name "`all_states.csv`"):

```sh
python DataSetup.py
```

Then on `Line 10`, you may see a `DEPLOY` variable, set it to `False` if you wish to run the dashboard locally.

That's it! You can now run the dashboard locally. Enjoy!


## References

+ <https://medium.com/@chihsuan/pipenv-%E6%9B%B4%E7%B0%A1%E5%96%AE-%E6%9B%B4%E5%BF%AB%E9%80%9F%E7%9A%84-python-%E5%A5%97%E4%BB%B6%E7%AE%A1%E7%90%86%E5%B7%A5%E5%85%B7-135a47e504f4>