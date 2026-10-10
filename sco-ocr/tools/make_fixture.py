"""Write the synthetic frame to disk so tools can be tried without the live stream."""

import sys

import cv2

from tests.synthetic import make_frame

cv2.imwrite(
    sys.argv[1] if len(sys.argv) > 1 else "synthetic.png",
    make_frame({3: [("500G GRAN CHOC-AMA", "3.35"), ("MELK 1L", "1.19")]}),
)
