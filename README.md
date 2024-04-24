# q-and-a

This is a test program for answering questions about the context of a PDF document.

## Installation
*NOTE:* The program was developed on Ubuntu 22.04. Launch on older versions is unpredictable. Using [GPT4ALL](https://gpt4all.io/index.html), you may have to build from source if you are not running the latest version of Ubuntu.

- git pull master
- pip install -r requirements.txt
- [Download gpt4all-falcon-newbpe-q4_0 model](https://gpt4all.io/models/gguf/gpt4all-falcon-newbpe-q4_0.gguf) to the root of this project. You can also use a *downloading_the_model.sh* for this.
- python3 answer_the_question.py -s [SOURCE] -q [QUESTION] -d [DEVICE]

You can use python3 answer_the_question.py -h for more information.

## Example
Answering questions in the context of a [file](https://lingua.com/pdf/english-text-washington.pdf)
![alt text](http://kappa.cs.petrsu.ru/~bakanov/example1.png)

![alt text](http://kappa.cs.petrsu.ru/~bakanov/example2.png)

