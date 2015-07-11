#!/bin/sh

# Combine images under frames directory into video using avconv.
# You could also use something like ffmpeg or mencoder for this.
#
# Albert Veli
# Boomtime, the 46th day of Confusion in the YOLD 3181

avconv -framerate 30 -f image2 -i frames/%04d.jpg -c:v h264 -crf 1 out.mov
