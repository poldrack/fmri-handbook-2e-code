# show pdf from local file or url in jupyter notebook

from IPython.display import IFrame
import IPython.display
from PyPDF2 import PdfFileReader
import urllib.request
import tempfile
import numpy
import os

def showPDF(infile,showLabel=True):
    if showLabel:
        print(infile)
    if infile.find('http')==0:
        with urllib.request.urlopen(infile) as response:
            tmpfile = tempfile.NamedTemporaryFile(delete=False,suffix='.pdf')
            tmpfile.write(response.read())
            tmpfile.close()
        input1 = PdfFileReader(open(tmpfile.name,'rb'))
    else:
        input1 = PdfFileReader(open(infile,'rb'))
    img=IFrame(infile,
           width=numpy.round(input1.getPage(0).mediaBox[2]*1.1).astype('int'),
           height=numpy.round(input1.getPage(0).mediaBox[3]*1.2).astype('int'))
    IPython.display.display(img)
    try:
        os.remove(tmpfile.name)
    except OSError:
        pass
