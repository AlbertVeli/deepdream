This is my clone of the google deepdream repository. I added albert.py and frames2mov.sh. Otherwise it is unchanged.

albert.py contains mainly code from dream.ipynb. I translated it to regular python and played around with it a bit. It generates frames into the subdirectory frames and the frames2mov.sh script combines the frames into a video using avconv.

Below is the original readme.

# deepdream

This repository contains IPython Notebook with sample code, complementing 
Google Research [blog post](http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html) about Neural Network art.
See [original gallery](https://photos.google.com/share/AF1QipPX0SCl7OzWilt9LnuQliattX4OUCj_8EP65_cTVnBmS1jnYgsGQAieQUc1VQWdgQ?key=aVBxWjhwSzg2RjJWLWRuVFBBZEN1d205bUdEMnhB) for more examples.

You can view "dream.ipynb" directly on github, or clone the repository, 
install dependencies listed in the notebook and play with code locally.

It'll be interesting to see what imagery people are able to generate using the described technique. If you post images to Google+, Facebook, or Twitter, be sure to tag them with [#deepdream](https://twitter.com/hashtag/deepdream) so other researchers can check them out too.

* [Alexander Mordvintsev](mailto:moralex@google.com)
* [Michael Tyka](https://www.twitter.com/mtyka)
* [Christopher Olah](mailto:colah@google.com)
